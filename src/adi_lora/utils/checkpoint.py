"""Checkpoint helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def save_head_checkpoint(path: str | Path, model: nn.Module, metadata: Dict[str, Any] | None = None) -> None:
    if not hasattr(model, "head"):
        raise AttributeError("model has no .head attribute; cannot save LP head.")
    save_checkpoint(path, {"head_state_dict": model.head.state_dict(), "metadata": metadata or {}})


def load_head_checkpoint(path: str | Path, model: nn.Module, strict: bool = True) -> None:
    payload = load_checkpoint(path, map_location="cpu")
    if "head_state_dict" not in payload:
        raise KeyError(f"{path} does not contain head_state_dict")
    model.head.load_state_dict(payload["head_state_dict"], strict=strict)
