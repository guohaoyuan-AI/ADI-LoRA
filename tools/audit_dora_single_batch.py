#!/usr/bin/env python3
"""Single-batch DoRA forward/backward audit for the current FR-PEFT project.

Example:
  export PYTHONPATH=$PWD/src
  python tools/audit_dora_single_batch.py --config configs/fr_peft_cifar10_dora.yaml --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from adi_lora.data import build_cifar10_loaders, build_cifar100_loaders  # noqa: E402
from adi_lora.engine.run_fr_peft import build_model_from_config, inject_peft_from_config  # noqa: E402
from adi_lora.models.peft import iter_peft_modules, mark_only_peft_and_head_trainable, set_delta_scale  # noqa: E402
from adi_lora.models.peft.lora_dora import DoRALinear  # noqa: E402
from adi_lora.utils.config import load_config  # noqa: E402
from adi_lora.utils.params import count_parameters, grad_norm_by_name  # noqa: E402
from adi_lora.utils.seed import set_seed  # noqa: E402


def as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "y"}
    return bool(v)


def grad_norm_param(p: torch.nn.Parameter) -> float:
    if p.grad is None:
        return 0.0
    return float(p.grad.detach().float().norm().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs/dora_audit/single_batch_audit.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["seed"] = args.seed
    set_seed(args.seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    dataset = str(data_cfg.get("name", "CIFAR10")).lower()
    loader_builder = build_cifar100_loaders if dataset in {"cifar100", "cifar-100"} else build_cifar10_loaders
    loaders = loader_builder(
        root=data_cfg.get("root", "./outputs/data"),
        image_size=int(data_cfg.get("image_size", 224)),
        train_interpolation=data_cfg.get("train_interpolation", "bicubic"),
        val_interpolations=data_cfg.get("val_interpolations", ["bicubic", "bilinear"]),
        test_interpolations=data_cfg.get("test_interpolations", ["bicubic", "bilinear", "nearest"]),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=args.seed,
        download=as_bool(data_cfg.get("download", False)),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model_from_config(cfg).to(device)
    report = inject_peft_from_config(model, cfg)
    model = model.to(device)
    mark_only_peft_and_head_trainable(model)

    modules = list(iter_peft_modules(model))
    dora_modules = [m for m in modules if isinstance(m, DoRALinear)]
    if not dora_modules:
        raise RuntimeError("No DoRALinear modules found. Check method.peft_type=dora.")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=0.0)
    images, targets = next(iter(loaders.train))
    images = images.to(device)
    targets = targets.to(device)

    model.train()
    set_delta_scale(model, 1.0)
    opt.zero_grad(set_to_none=True)
    logits = model(images)
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    first = dora_modules[0]
    stats = {
        "config": args.config,
        "seed": args.seed,
        "device": str(device),
        "injection_report": getattr(report, "__dict__", str(report)),
        "num_peft_modules": len(modules),
        "num_dora_modules": len(dora_modules),
        "trainable_params": count_parameters(model)[0],
        "total_params": count_parameters(model)[1],
        "loss": float(loss.detach().cpu().item()),
        "logits_finite": bool(torch.isfinite(logits).all().item()),
        "loss_finite": bool(torch.isfinite(loss).all().item()),
        "grad_norm_lora_A": grad_norm_by_name(model, "lora_A"),
        "grad_norm_lora_B": grad_norm_by_name(model, "lora_B"),
        "grad_norm_magnitude": grad_norm_by_name(model, "magnitude"),
        "first_module": {
            "base_weight_device": str(first.base.weight.device),
            "lora_A_device": str(first.lora_A.device),
            "lora_B_device": str(first.lora_B.device),
            "magnitude_device": str(first.magnitude.device),
            "base_requires_grad": bool(first.base.weight.requires_grad),
            "lora_A_requires_grad": bool(first.lora_A.requires_grad),
            "lora_B_requires_grad": bool(first.lora_B.requires_grad),
            "magnitude_requires_grad": bool(first.magnitude.requires_grad),
            "lora_A_grad_norm": grad_norm_param(first.lora_A),
            "lora_B_grad_norm": grad_norm_param(first.lora_B),
            "magnitude_grad_norm": grad_norm_param(first.magnitude),
            "base_magnitude_buffer": bool("base_magnitude" in dict(first.named_buffers())),
        },
    }
    assert stats["loss_finite"] and stats["logits_finite"], stats
    assert stats["grad_norm_lora_B"] > 0, stats
    assert stats["grad_norm_magnitude"] > 0, stats
    assert stats["first_module"]["magnitude_requires_grad"], stats
    assert not stats["first_module"]["base_requires_grad"], stats
    assert all(math.isfinite(float(v)) for v in [stats["loss"], stats["grad_norm_lora_B"], stats["grad_norm_magnitude"]])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
