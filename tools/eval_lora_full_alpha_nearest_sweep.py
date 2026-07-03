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


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def build_loaders(cfg: dict, seed: int):
    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    name = str(data_cfg.get("name", "CIFAR100")).lower()

    if name in {"cifar100", "cifar-100"}:
        builder = build_cifar100_loaders
    elif name in {"cifar10", "cifar-10"}:
        builder = build_cifar10_loaders
    else:
        raise ValueError(f"Unsupported dataset.name={data_cfg.get('name')}")

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
    ckpts = sorted((run_dir / "checkpoints").glob("final_*seed*.pth"))
    if not ckpts:
        ckpts = sorted((run_dir / "checkpoints").glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found under {run_dir / 'checkpoints'}")
    return ckpts[-1]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--alphas", default="0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    project = Path.cwd()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project / run_dir

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))
    summary_rows = read_csv(run_dir / "summary.csv")
    summary = summary_rows[-1]

    seed = args.seed if args.seed is not None else int(float(summary.get("seed", cfg.get("seed", 42))))
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    ckpt_path = find_final_ckpt(run_dir)

    alpha_selection_path = run_dir / "alpha_selection.csv"
    alpha_rows = read_csv(alpha_selection_path)
    alpha_meta = {float(r["alpha"]): r for r in alpha_rows}

    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loaders = build_loaders(cfg, seed)
    nearest_loader = loaders.test["nearest"]

    model = build_model_from_config(cfg).to(device)
    inject_peft_from_config(model, cfg)
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[LOAD]", run_dir, "missing", len(missing), "unexpected", len(unexpected))
    model.eval()

    rows = []
    for alpha in alphas:
        set_delta_scale(model, alpha)
        result = evaluate(
            model=model,
            loader=nearest_loader,
            device=device,
            cfg=cfg,
            split="test",
            interpolation="nearest",
            alpha=alpha,
            reference_model=None,
        )

        meta = alpha_meta.get(alpha, {})
        rows.append({
            "dataset": "CIFAR-100",
            "method": "ADI-LoRA",
            "peft_type": "lora",
            "seed": seed,
            "alpha": alpha,
            "checkpoint_type": "final",
            "checkpoint_id": str(ckpt_path),
            "val_bicubic_acc_pct": float(meta.get("val_bicubic_acc", "nan")) * 100,
            "val_bilinear_acc_pct": float(meta.get("val_bilinear_acc", "nan")) * 100,
            "test_nearest_acc_pct": float(result["acc"]) * 100,
            "test_nearest_loss_ce": float(result.get("loss_ce", 0.0)),
            "loss_ce": float(result.get("loss_ce", 0.0)),
            "selected": int(float(meta.get("selected", 0))),
            "alpha_selection_source": "val_bicubic_and_val_bilinear",
            "nearest_used_for_selection": int(float(meta.get("nearest_used_for_selection", 0))),
            "nearest_used_for_alpha_selection": 0,
            "corruption_used_for_selection": 0,
            "corruption_used_for_alpha_selection": 0,
            "checkpoint_selection_rule": "final_checkpoint_only",
            "nearest_used_for_checkpoint_selection": 0,
            "diagnostic_nearest_full_alpha_sweep": 1,
            "selection_rule": meta.get("selection_rule", "val_bilinear_acc_with_bicubic_guard"),
            "source_run_dir": str(run_dir),
        })

    write_csv(out_dir / "lora_full_alpha_nearest_sweep.csv", rows)
    print("[SAVED]", out_dir / "lora_full_alpha_nearest_sweep.csv")


if __name__ == "__main__":
    main()
