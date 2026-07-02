"""Parameter and gradient diagnostics."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(trainable), int(total)


def global_grad_norm(model: nn.Module) -> float:
    sq_sum = 0.0
    for p in model.parameters():
        if p.grad is not None:
            norm = float(p.grad.detach().float().norm().item())
            sq_sum += norm * norm
    return float(sq_sum ** 0.5)


def grad_norm_by_name(model: nn.Module, keyword: str) -> float:
    sq_sum = 0.0
    for name, p in model.named_parameters():
        if keyword in name and p.grad is not None:
            norm = float(p.grad.detach().float().norm().item())
            sq_sum += norm * norm
    return float(sq_sum ** 0.5)


def finite_parameters(model: nn.Module) -> bool:
    return all(torch.isfinite(p.detach()).all().item() for p in model.parameters())


def finite_gradients(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad.detach()).all().item():
            return False
    return True


def parameter_delta_norm(current: Dict[str, torch.Tensor], reference: Dict[str, torch.Tensor]) -> float:
    sq_sum = 0.0
    for key, value in current.items():
        if key in reference and value.shape == reference[key].shape:
            diff = value.detach().float().cpu() - reference[key].detach().float().cpu()
            sq_sum += float(diff.pow(2).sum().item())
    return float(sq_sum ** 0.5)
