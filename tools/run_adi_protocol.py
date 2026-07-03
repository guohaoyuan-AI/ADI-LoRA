#!/usr/bin/env python3
"""Unified ADI protocol runner.

This entry point trains one final checkpoint, selects alpha from validation
Bicubic/Bilinear only, and then evaluates alpha=1.0 and alpha* on the same final
checkpoint. It writes protocol audit files that make the no-leakage boundary
explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from adi_lora.data import build_cifar10_loaders, build_cifar100_loaders
from adi_lora.engine.evaluate import evaluate
from adi_lora.engine.run_fr_peft import build_model_from_config, inject_peft_from_config, run, _as_bool
from adi_lora.models.peft import set_delta_scale
from adi_lora.utils.config import load_config
from adi_lora.utils.seed import set_seed


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


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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


def find_final_checkpoint(run_dir: Path, seed: int) -> Path:
    patterns = [f"checkpoints/final_*seed{seed}.pth", "checkpoints/final_*.pth", "checkpoints/*.pth"]
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[-1]
    raise FileNotFoundError(f"No final checkpoint found under {run_dir / 'checkpoints'}")


@torch.no_grad()
def eval_alpha(model, loaders, device, cfg: dict, alpha: float, reference_model=None) -> list[dict]:
    set_delta_scale(model, alpha)
    rows: list[dict] = []
    for interp, loader in loaders.test.items():
        row = evaluate(
            model=model,
            loader=loader,
            device=device,
            cfg=cfg,
            split="test",
            interpolation=interp,
            alpha=alpha,
            reference_model=reference_model,
        )
        row["alpha"] = alpha
        row["checkpoint_selection_rule"] = "final_checkpoint_only"
        row["nearest_used_for_alpha_selection"] = 0
        row["corruption_used_for_alpha_selection"] = 0
        rows.append(row)
    return rows


def metric(rows: list[dict], interp: str, key: str = "acc") -> float:
    for row in rows:
        if row.get("interpolation") == interp:
            return float(row.get(key, "nan"))
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-eval-batches", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    summary = run(
        cfg,
        config_path=args.config,
        seed_override=args.seed,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )

    run_dir = Path(summary["output_dir"])
    seed = int(summary["seed"])
    selected_alpha = float(summary["selected_alpha"])
    ckpt_path = find_final_checkpoint(run_dir, seed)

    resolved_cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    set_seed(seed, deterministic=bool(resolved_cfg.get("deterministic", False)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loaders = build_loaders(resolved_cfg, seed)

    split_manifest = {
        "dataset": resolved_cfg.get("dataset", {}).get("name", "CIFAR100"),
        "seed": seed,
        "val_ratio": float(resolved_cfg.get("dataset", {}).get("val_ratio", 0.1)),
        "stratified": True,
        "train_size": len(loaders.train_indices),
        "val_size": len(loaders.val_indices),
        "test_size": sum(len(loader.dataset) for loader in loaders.test.values()) // max(len(loaders.test), 1),
        "train_indices": loaders.train_indices,
        "val_indices": loaders.val_indices,
        "alpha_selection_splits": ["val_bicubic", "val_bilinear"],
        "heldout_splits": ["test_nearest", "corruption_tests"],
        "nearest_used_for_alpha_selection": 0,
        "corruption_used_for_alpha_selection": 0,
        "checkpoint_selection_rule": "final_checkpoint_only",
        "nearest_used_for_checkpoint_selection": 0,
    }
    (run_dir / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    model = build_model_from_config(resolved_cfg).to(device)
    inject_peft_from_config(model, resolved_cfg)
    model = model.to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    reference_model = build_model_from_config(resolved_cfg).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    alpha1_rows = eval_alpha(model, loaders, device, resolved_cfg, alpha=1.0, reference_model=reference_model)
    selected_rows = eval_alpha(model, loaders, device, resolved_cfg, alpha=selected_alpha, reference_model=reference_model)
    write_csv(run_dir / "test_eval_alpha1.csv", alpha1_rows)
    write_csv(run_dir / "test_eval_selected.csv", selected_rows)

    comparison = [{
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "seed": seed,
        "selected_alpha": selected_alpha,
        "alpha_selection_source": "val_bicubic_and_val_bilinear",
        "alpha1_iid": metric(alpha1_rows, "bicubic"),
        "alpha1_bilinear": metric(alpha1_rows, "bilinear"),
        "alpha1_nearest": metric(alpha1_rows, "nearest"),
        "selected_iid": metric(selected_rows, "bicubic"),
        "selected_bilinear": metric(selected_rows, "bilinear"),
        "selected_nearest": metric(selected_rows, "nearest"),
        "nearest_gain": metric(selected_rows, "nearest") - metric(alpha1_rows, "nearest"),
        "nearest_used_for_alpha_selection": 0,
        "corruption_used_for_alpha_selection": 0,
        "checkpoint_selection_rule": "final_checkpoint_only",
        "nearest_used_for_checkpoint_selection": 0,
    }]
    write_csv(run_dir / "same_checkpoint_comparison.csv", comparison)

    alpha_rows = read_csv(run_dir / "alpha_selection.csv") if (run_dir / "alpha_selection.csv").exists() else []
    audit = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "load_missing_keys": len(missing),
        "load_unexpected_keys": len(unexpected),
        "seed": seed,
        "selected_alpha": selected_alpha,
        "alpha_candidates": [float(r["alpha"]) for r in alpha_rows if "alpha" in r],
        "alpha_selection_source": "val_bicubic_and_val_bilinear",
        "nearest_used_for_alpha_selection": 0,
        "corruption_used_for_alpha_selection": 0,
        "checkpoint_selection_rule": "final_checkpoint_only",
        "nearest_used_for_checkpoint_selection": 0,
        "same_checkpoint_alpha1_and_selected": True,
        "extra_trainable_params_for_adi": 0,
        "extra_inference_modules_for_adi": 0,
    }
    (run_dir / "protocol_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(run_dir), "selected_alpha": selected_alpha}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
