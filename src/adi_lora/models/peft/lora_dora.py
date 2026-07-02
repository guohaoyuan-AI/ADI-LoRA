"""LoRA and DoRA modules for ViT linear layers.

The implementation is intentionally explicit and small.  It wraps selected
`nn.Linear` modules, freezes the original linear projection, and exposes a
`delta_scale` field for adapter-delta interpolation.  No merge/unmerge side
 effects are required during evaluation; alpha selection simply changes
`delta_scale` before forward passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PeftInjectionReport:
    peft_type: str
    target_modules: List[str]
    wrapped_modules: List[str]
    rank: int
    alpha: float
    dropout: float


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for a frozen linear layer."""

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        self.delta_scale = 1.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # PEFT convention: A random, B zero, so the wrapped model starts at the frozen model.
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield self.lora_A
        yield self.lora_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return base_out + float(self.delta_scale) * self.scaling * delta

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.lora_B @ self.lora_A)


class DoRALinear(nn.Module):
    """Weight-Decomposed LoRA wrapper.

    This implements the PEFT-relevant DoRA behavior for linear projections:
    low-rank direction update plus a trainable row-wise magnitude.  During
    adapter-delta interpolation, both the direction delta and magnitude delta are
    scaled by `delta_scale`; alpha=0 recovers the frozen base direction and base
    magnitude, while alpha=1 recovers the trained DoRA layer.
    """

    def __init__(self, base: nn.Linear, rank: int = 16, alpha: float = 16.0, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("DoRALinear can only wrap nn.Linear")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.in_features = int(base.in_features)
        self.out_features = int(base.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(rank)
        # Current DoRA implementation intentionally uses exact dense-weight DoRA.
        # Dropout in the LoRA direction update is not algebraically representable as
        # a single dense W_eff and would break alpha-equivalence tests. Keep DoRA
        # dropout disabled unless a separate stochastic DoRA derivation is added.
        if float(dropout) > 0:
            raise ValueError("DoRALinear currently requires dropout=0.0 for exact DoRA/ADI-DoRA semantics.")
        self.dropout = nn.Identity()
        self.eps = float(eps)
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))
        base_magnitude = self.base.weight.detach().float().norm(dim=1).clamp_min(self.eps)
        self.register_buffer("base_magnitude", base_magnitude)
        self.magnitude = nn.Parameter(base_magnitude.clone())
        self.delta_scale = 1.0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        with torch.no_grad():
            self.magnitude.copy_(self.base_magnitude)

    def adapter_parameters(self) -> Iterator[nn.Parameter]:
        yield self.lora_A
        yield self.lora_B
        yield self.magnitude

    def _dora_weight(self) -> torch.Tensor:
        """Return the standard trained DoRA effective weight at alpha=1.

        W_DoRA = m * (W0 + DeltaW) / ||W0 + DeltaW||_row
        """
        base_w = self.base.weight
        delta_w = self.scaling * (self.lora_B @ self.lora_A).to(dtype=base_w.dtype)
        direction = base_w + delta_w
        direction_norm = direction.float().norm(dim=1, keepdim=True).clamp_min(self.eps).to(dtype=direction.dtype)
        mag = self.magnitude.to(device=direction.device, dtype=direction.dtype)
        return direction / direction_norm * mag.view(-1, 1)

    def _interpolated_weight(self) -> torch.Tensor:
        """Return exact adapter-delta interpolation for DoRA.

        alpha=0: W0
        alpha=1: W_DoRA
        0<alpha<1: W0 + alpha * (W_DoRA - W0)

        This is intentionally different from scaling only the low-rank direction
        update before DoRA row normalization. The latter is nonlinear in alpha
        and is not strict ADI-DoRA.
        """
        base_w = self.base.weight
        scale = float(self.delta_scale)
        if abs(scale) < 1e-12:
            return base_w
        trained_w = self._dora_weight()
        if abs(scale - 1.0) < 1e-12:
            return trained_w
        return base_w + scale * (trained_w - base_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self._interpolated_weight(), self.base.bias)

    @torch.no_grad()
    def delta_weight(self) -> torch.Tensor:
        # For DoRA, the adapter delta must include both direction and magnitude
        # effects after row normalization. This diagnostic is not used for training.
        return self._dora_weight() - self.base.weight


PeftLinear = LoRALinear | DoRALinear


def _get_parent_and_name(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _matches_target(name: str, target_suffixes: Sequence[str]) -> bool:
    return any(name.endswith(suffix) for suffix in target_suffixes)


def inject_lora_dora(
    model: nn.Module,
    peft_type: str = "lora",
    target_modules: Sequence[str] = ("attn.qkv", "attn.proj"),
    target_blocks: str | Iterable[int] = "all",
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> PeftInjectionReport:
    """Wrap selected ViT attention projections with LoRA or DoRA.

    Args:
        model: timm ViT-like model containing `blocks`.
        peft_type: `lora` or `dora`.
        target_modules: suffixes such as `attn.qkv` and `attn.proj`.
        target_blocks: `all`, `last6`, or an iterable of block indices.
    """
    peft_type = str(peft_type).lower()
    if peft_type not in {"lora", "dora"}:
        raise ValueError("peft_type must be 'lora' or 'dora'.")

    if target_blocks == "all":
        allowed_blocks = None
    elif isinstance(target_blocks, str) and target_blocks.startswith("last"):
        if not hasattr(model, "blocks"):
            raise AttributeError("target_blocks='lastK' requires model.blocks")
        k = int(target_blocks.replace("last", ""))
        n = len(model.blocks)
        allowed_blocks = set(range(max(0, n - k), n))
    else:
        allowed_blocks = {int(i) for i in target_blocks}

    candidates: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not _matches_target(name, list(target_modules)):
            continue
        if allowed_blocks is not None:
            # Expected name format: blocks.6.attn.qkv
            parts = name.split(".")
            if len(parts) < 2 or parts[0] != "blocks":
                continue
            if int(parts[1]) not in allowed_blocks:
                continue
        candidates.append((name, module))

    if not candidates:
        raise RuntimeError(
            f"No target Linear modules matched target_modules={target_modules} target_blocks={target_blocks}."
        )

    wrapped_names: list[str] = []
    wrapper_cls = LoRALinear if peft_type == "lora" else DoRALinear
    for name, module in candidates:
        parent, child_name = _get_parent_and_name(model, name)
        setattr(parent, child_name, wrapper_cls(module, rank=rank, alpha=alpha, dropout=dropout))
        wrapped_names.append(name)

    return PeftInjectionReport(
        peft_type=peft_type,
        target_modules=list(target_modules),
        wrapped_modules=wrapped_names,
        rank=int(rank),
        alpha=float(alpha),
        dropout=float(dropout),
    )


def iter_peft_modules(model: nn.Module) -> Iterator[LoRALinear | DoRALinear]:
    for module in model.modules():
        if isinstance(module, (LoRALinear, DoRALinear)):
            yield module


def set_delta_scale(model: nn.Module, alpha: float) -> None:
    for module in iter_peft_modules(model):
        module.delta_scale = float(alpha)


def mark_only_peft_and_head_trainable(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for module in iter_peft_modules(model):
        for p in module.adapter_parameters():
            p.requires_grad = True
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True


def mark_head_only_trainable(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if not hasattr(model, "head"):
        raise AttributeError("model has no .head attribute")
    for p in model.head.parameters():
        p.requires_grad = True


@torch.no_grad()
def peft_delta_norm(model: nn.Module) -> float:
    sq_sum = 0.0
    for module in iter_peft_modules(model):
        delta = module.delta_weight().detach().float()
        sq_sum += float(delta.pow(2).sum().item())
    return float(sq_sum ** 0.5)


def peft_module_count(model: nn.Module) -> int:
    return sum(1 for _ in iter_peft_modules(model))
