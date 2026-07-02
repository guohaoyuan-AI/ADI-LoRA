"""Training utilities for FR-PEFT."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from adi_lora.models.peft import peft_delta_norm
from adi_lora.utils.params import (
    count_parameters,
    global_grad_norm,
    grad_norm_by_name,
)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == targets).float().mean().item())


def build_optimizer(model: nn.Module, cfg: dict, stage: str) -> torch.optim.Optimizer:
    train_cfg = cfg.get("train", {})
    if stage == "lp":
        lr = float(train_cfg.get("lr_head", train_cfg.get("lr", 1e-3)))
    else:
        lr = float(train_cfg.get("lr_peft", train_cfg.get("lr", 1e-3)))

    wd = float(train_cfg.get("weight_decay", 0.05))
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_params:
        raise RuntimeError(f"No trainable parameters for stage={stage}.")

    # DoRA's row-wise magnitude is itself the learned norm of the effective
    # weight. Applying AdamW decay to it can systematically shrink the whole
    # projection norm and create a false DoRA collapse. Keep other historical
    # behavior unchanged unless additional no-decay rules are explicitly added.
    decay_params = [p for n, p in named_params if "magnitude" not in n]
    no_decay_params = [p for n, p in named_params if "magnitude" in n]

    opt_name = str(train_cfg.get("optimizer", "adamw")).lower()
    if opt_name == "adamw":
        groups = []
        if decay_params:
            groups.append({"params": decay_params, "weight_decay": wd})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, lr=lr)
    if opt_name == "sgd":
        groups = []
        if decay_params:
            groups.append({"params": decay_params, "weight_decay": wd})
        if no_decay_params:
            groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)

    raise ValueError(f"Unsupported optimizer={opt_name}")


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict, epochs: int):
    sched = str(cfg.get("train", {}).get("scheduler", "cosine")).lower()
    if epochs <= 0:
        return None
    if sched == "none":
        return None
    if sched == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs))
    raise ValueError(f"Unsupported scheduler={sched}")


def _finite_trainable_parameters(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.requires_grad and not torch.isfinite(p.detach()).all().item():
            return False
    return True


def _finite_effective_gradients(model: nn.Module) -> bool:
    """Only check gradients that actually exist. Missing gradients are not NaN."""
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            if not torch.isfinite(p.grad.detach()).all().item():
                return False
    return True


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    stage: str,
    cfg: dict,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, Any]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    amp_enabled = bool(cfg.get("train", {}).get("amp", True)) and device.type == "cuda"
    grad_clip_norm = cfg.get("train", {}).get("grad_clip_norm", None)
    if grad_clip_norm is not None:
        grad_clip_norm = float(grad_clip_norm)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    # These flags now only mean real numerical instability in tensor computation.
    # They do NOT inspect placeholder NaN fields in CSV logs.
    nan_flag = 0
    inf_flag = 0
    divergence_flag = 0

    start_time = time.time()

    pbar = tqdm(loader, desc=f"train/{stage} epoch={epoch}", leave=False)

    for images, targets in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        loss_is_finite = torch.isfinite(loss).all().item()
        logits_are_finite = torch.isfinite(logits).all().item()

        if not loss_is_finite or not logits_are_finite:
            nan_flag = int(torch.isnan(loss).any().item() or torch.isnan(logits).any().item())
            inf_flag = int(torch.isinf(loss).any().item() or torch.isinf(logits).any().item())
            divergence_flag = 1
            raise FloatingPointError(
                f"Non-finite tensor at epoch={epoch}, stage={stage}, "
                f"loss_finite={loss_is_finite}, logits_finite={logits_are_finite}"
            )

        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()

            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip_norm,
                )

            if not _finite_effective_gradients(model):
                nan_flag = 1
                divergence_flag = 1
                raise FloatingPointError(f"Non-finite gradients at epoch={epoch}, stage={stage}")

            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip_norm,
                )

            if not _finite_effective_gradients(model):
                nan_flag = 1
                divergence_flag = 1
                raise FloatingPointError(f"Non-finite gradients at epoch={epoch}, stage={stage}")

            optimizer.step()

        if not _finite_trainable_parameters(model):
            nan_flag = 1
            divergence_flag = 1
            raise FloatingPointError(f"Non-finite trainable parameters at epoch={epoch}, stage={stage}")

        batch_size = int(targets.shape[0])
        total_seen += batch_size
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum().item())

        pbar.set_postfix(
            loss=total_loss / max(total_seen, 1),
            acc=total_correct / max(total_seen, 1),
        )

    elapsed = max(time.time() - start_time, 1e-6)
    train_loss_epoch = total_loss / max(total_seen, 1)

    # Divergence is now a conservative real-training flag.
    if train_loss_epoch > 1e3:
        divergence_flag = 1

    trainable, total = count_parameters(model)
    lr = float(optimizer.param_groups[0]["lr"])
    peak_mem_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else 0.0
    )

    # Grad norm functions return 0.0 if no matching gradient exists.
    grad_norm_global = global_grad_norm(model)
    if not torch.isfinite(torch.tensor(grad_norm_global)).item():
        grad_norm_global = 0.0
        nan_flag = 1
        divergence_flag = 1

    return {
        "stage": stage,
        "epoch": int(epoch),
        "train_loss": float(train_loss_epoch),
        "train_acc": float(total_correct / max(total_seen, 1)),
        "lr": lr,
        "num_samples": int(total_seen),
        "epoch_time_sec": float(elapsed),
        "throughput_img_s": float(total_seen / elapsed),
        "peak_mem_mb": peak_mem_mb,
        "trainable_params": trainable,
        "total_params": total,
        "adapter_delta_norm": float(peft_delta_norm(model)),
        "grad_norm_global": float(grad_norm_global),
        "grad_norm_head": float(grad_norm_by_name(model, "head")),
        "grad_norm_lora_A": float(grad_norm_by_name(model, "lora_A")),
        "grad_norm_lora_B": float(grad_norm_by_name(model, "lora_B")),
        "grad_norm_dora_magnitude": float(grad_norm_by_name(model, "magnitude")),
        "nan_flag": int(nan_flag),
        "inf_flag": int(inf_flag),
        "divergence_flag": int(divergence_flag),
    }
