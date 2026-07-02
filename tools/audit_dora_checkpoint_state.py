#!/usr/bin/env python3
"""Inspect whether a saved checkpoint contains finite DoRA params and buffers.

Example:
  python tools/audit_dora_checkpoint_state.py --checkpoint outputs/.../checkpoints/final_dora_seed42.pth
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def tensor_stats(t: torch.Tensor) -> dict:
    x = t.detach().float().cpu()
    return {
        "shape": list(x.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0,
        "min": float(x.min().item()),
        "max": float(x.max().item()),
        "norm": float(x.norm().item()),
        "finite": bool(torch.isfinite(x).all().item()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt if isinstance(ckpt, dict) else {})
    wanted = {k: tensor_stats(v) for k, v in state.items() if any(s in k for s in ["lora_A", "lora_B", "magnitude", "base_magnitude"])}
    n_mag = sum(k.endswith(".magnitude") for k in wanted)
    n_base_mag = sum(k.endswith(".base_magnitude") for k in wanted)
    finite_all = all(v["finite"] for v in wanted.values()) if wanted else False
    report = {
        "checkpoint": args.checkpoint,
        "num_lora_or_dora_tensors": len(wanted),
        "num_magnitude_params": n_mag,
        "num_base_magnitude_buffers": n_base_mag,
        "finite_all": finite_all,
        "missing_base_magnitude_risk": n_mag != n_base_mag,
        "tensors": wanted,
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
