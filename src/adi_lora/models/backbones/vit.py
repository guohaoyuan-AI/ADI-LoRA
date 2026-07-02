"""timm ViT construction and feature extraction helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _load_local_checkpoint(model: nn.Module, checkpoint_path: str) -> dict:
    """Load a local timm/HF checkpoint while dropping incompatible classifier keys."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint_path does not exist: {path}")

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError("safetensors is required to load .safetensors checkpoints.") from exc
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(str(path), map_location="cpu")
        if isinstance(state, dict):
            for key in ("model", "state_dict", "module"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break

    # Strip common prefixes.
    cleaned = {}
    for k, v in state.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v

    model_state = model.state_dict()
    compatible = {}
    skipped = []

    for k, v in cleaned.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            compatible[k] = v
        else:
            skipped.append(k)

    missing, unexpected = model.load_state_dict(compatible, strict=False)

    report = {
        "checkpoint_path": str(path),
        "loaded_keys": len(compatible),
        "skipped_keys": len(skipped),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "skipped_examples": skipped[:10],
        "missing_examples": list(missing)[:10],
    }
    print("[build_vit_backbone] Loaded local checkpoint:", report)
    return report


def build_vit_backbone(
    name: str = "vit_base_patch16_224.augreg_in21k_ft_in1k",
    num_classes: int = 10,
    pretrained: bool = True,
    checkpoint_path: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    """Build a timm ViT classification model.

    If checkpoint_path is provided, timm is created with pretrained=False to avoid
    network access, then compatible local weights are loaded manually.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for ViT backbone construction. Run: pip install timm") from exc

    use_timm_pretrained = bool(pretrained) and not checkpoint_path

    model = timm.create_model(
        name,
        pretrained=use_timm_pretrained,
        num_classes=int(num_classes),
        **kwargs,
    )

    if checkpoint_path:
        _load_local_checkpoint(model, checkpoint_path)

    return model


def freeze_backbone_keep_head(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True


def extract_feature_tokens(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return final ViT token features, normally [B, 197, C]."""
    features = model.forward_features(x)
    if isinstance(features, dict):
        if "x" in features:
            features = features["x"]
        elif "features" in features:
            features = features["features"]
        else:
            raise TypeError(f"Unsupported forward_features dict keys: {list(features.keys())}")
    if features.ndim == 2:
        features = features.unsqueeze(1)
    if features.ndim != 3:
        raise ValueError(f"Expected ViT token features [B,N,C], got shape={tuple(features.shape)}")
    return features


def infer_vit_spatial_size(model: nn.Module) -> tuple[int, int]:
    """Infer patch grid size for ViT-B/16 at 224x224 -> (14,14)."""
    patch_embed = getattr(model, "patch_embed", None)
    if patch_embed is not None:
        grid_size = getattr(patch_embed, "grid_size", None)
        if grid_size is not None:
            return int(grid_size[0]), int(grid_size[1])
        num_patches = getattr(patch_embed, "num_patches", None)
        if num_patches is not None:
            h = int(num_patches ** 0.5)
            if h * h == int(num_patches):
                return h, h
    return (14, 14)
