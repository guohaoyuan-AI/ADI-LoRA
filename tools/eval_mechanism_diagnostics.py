#!/usr/bin/env python3
"""Evaluate CKA/SpecDist diagnostics for alpha=1.0 and validation-selected alpha."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from adi_lora.data import build_cifar10_loaders, build_cifar100_loaders
from adi_lora.engine.evaluate import evaluate
from adi_lora.engine.run_fr_peft import build_model_from_config, inject_peft_from_config, _as_bool
from adi_lora.models.peft import set_delta_scale
from adi_lora.utils.seed import set_seed


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def build_loaders(cfg: dict, seed: int):
    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    name = str(data_cfg.get("name", "CIFAR100")).lower()
    builder = build_cifar100_loaders if name in {"cifar100", "cifar-100"} else build_cifar10_loaders
    return builder(
        root=data_cfg.get("root", "./outputs/data"),
        image_size=int(data_cfg.get("image_size", 224)),
        train_interpolation=data_cfg.get("train_interpolation", "bicubic"),
        val_interpolations=data_cfg.get("val_interpolations", ["bicubic", "bilinear"]),
        test_interpolations=data_cfg.get("test_interpolations", ["bicubic", "bilinear", "nearest"]),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        download=_as_bool(data_cfg.get("download", False)),
        pin_memory=torch.cuda.is_available(),
    )


def find_final_ckpt(run_dir: Path) -> Path:
    for pattern in ["checkpoints/final_*seed*.pth", "checkpoints/final_*.pth", "checkpoints/*.pth"]:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")


def infer_seed_and_alpha(run_dir: Path, cfg: dict, fallback_alpha: float | None) -> tuple[int, float]:
    summary_path = run_dir / "summary.csv"
    if summary_path.exists():
        row = read_csv(summary_path)[-1]
        seed = int(float(row.get("seed", cfg.get("seed", 42))))
        alpha = float(row.get("selected_alpha", fallback_alpha if fallback_alpha is not None else 1.0))
        return seed, alpha
    alpha_path = run_dir / "alpha_selection.csv"
    if alpha_path.exists():
        rows = read_csv(alpha_path)
        selected = [r for r in rows if int(float(r.get("selected", 0))) == 1]
        alpha = float((selected or rows)[-1]["alpha"])
        seed = int(float((selected or rows)[-1].get("seed", cfg.get("seed", 42))))
        return seed, alpha
    if fallback_alpha is None:
        raise ValueError(f"Cannot infer selected alpha for {run_dir}; pass --selected-alpha.")
    return int(cfg.get("seed", 42)), float(fallback_alpha)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--selected-alpha", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--method-label", default=None, help="LoRA or DoRA; default inferred from config")
    ap.add_argument("--max-eval-batches", type=int, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    inferred_seed, selected_alpha = infer_seed_and_alpha(run_dir, cfg, args.selected_alpha)
    seed = int(args.seed if args.seed is not None else inferred_seed)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else find_final_ckpt(run_dir)
    method = args.method_label or str(cfg.get("method", {}).get("peft_type", cfg.get("method", {}).get("name", ""))).upper()

    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(cfg, seed)
    base_loader = loaders.test["nearest"]

    def limited_loader():
        if args.max_eval_batches is None or args.max_eval_batches <= 0:
            return base_loader
        def gen():
            for idx, batch in enumerate(base_loader):
                if idx >= args.max_eval_batches:
                    break
                yield batch
        return gen()

    model = build_model_from_config(cfg).to(device)
    inject_peft_from_config(model, cfg)
    model = model.to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()

    reference_model = build_model_from_config(cfg).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    rows = []
    for alpha_name, alpha in [("alpha1", 1.0), ("selected", selected_alpha)]:
        set_delta_scale(model, alpha)
        result = evaluate(model, limited_loader(), device, cfg, split="test", interpolation="nearest", alpha=alpha, reference_model=reference_model)
        result.update({
            "dataset": cfg.get("dataset", {}).get("name", "CIFAR-100"),
            "method": method,
            "seed": seed,
            "alpha_name": alpha_name,
            "alpha": alpha,
            "selected_alpha": selected_alpha,
            "checkpoint": str(ckpt_path),
            "load_missing_keys": len(missing),
            "load_unexpected_keys": len(unexpected),
            "alpha_selection_source": "val_bicubic_and_val_bilinear",
            "nearest_used_for_alpha_selection": 0,
            "corruption_used_for_alpha_selection": 0,
            "checkpoint_selection_rule": "final_checkpoint_only",
        })
        rows.append(result)

    write_csv(Path(args.out_csv), rows)
    print(f"[SAVED] {args.out_csv}")


if __name__ == "__main__":
    main()
