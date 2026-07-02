"""FR-PEFT CIFAR-10 runner.

This module implements the first-stage minimal validation loop:
LoRA / DoRA / LP-LoRA / LP-DoRA / Delta-LoRA / Delta-DoRA /
LP-Delta-LoRA / LP-Delta-DoRA.

Hard rule: alpha selection uses validation Bicubic/Bilinear only.  Test Nearest
is evaluated only after the validation-selected alpha has been fixed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from torch.cuda.amp import GradScaler

from adi_lora.data import build_cifar10_loaders, build_cifar100_loaders
from adi_lora.engine.evaluate import evaluate
from adi_lora.engine.train_one_epoch import build_optimizer, build_scheduler, train_one_epoch
from adi_lora.models.backbones import build_vit_backbone
from adi_lora.models.peft import (
    inject_lora_dora,
    mark_head_only_trainable,
    mark_only_peft_and_head_trainable,
    peft_delta_norm,
    peft_module_count,
    set_delta_scale,
)
from adi_lora.utils.checkpoint import save_checkpoint, save_head_checkpoint
from adi_lora.utils.config import load_config
from adi_lora.utils.csv_logger import append_csv
from adi_lora.utils.params import count_parameters
from adi_lora.utils.seed import set_seed


TRAIN_LOG_FIELDS = [
    "run_id", "method", "peft_type", "seed", "stage", "epoch", "global_step", "train_loss", "train_acc", "lr",
    "batch_size", "num_samples", "epoch_time_sec", "throughput_img_s", "peak_mem_mb", "trainable_params", "total_params",
    "adapter_delta_norm", "head_delta_norm", "grad_norm_global", "grad_norm_head", "grad_norm_lora_A", "grad_norm_lora_B",
    "grad_norm_dora_magnitude", "nan_flag", "inf_flag", "divergence_flag", "checkpoint_path",
]

EVAL_LOG_FIELDS = [
    "run_id", "method", "peft_type", "seed", "stage", "epoch", "checkpoint_type", "split", "interpolation", "alpha",
    "acc", "loss_ce", "lfer", "cka_to_frozen", "specdist_to_frozen", "high_specdist_to_frozen", "num_samples",
    "eval_time_sec", "throughput_img_s", "peak_mem_mb",
]

ALPHA_FIELDS = [
    "run_id", "method", "peft_type", "seed", "checkpoint_type", "alpha", "val_bicubic_acc", "val_bilinear_acc",
    "val_bilinear_drop", "selection_score", "bicubic_guard_pass", "selected", "selection_rule", "nearest_used_for_selection",
]

SUMMARY_FIELDS = [
    "run_id", "method", "peft_type", "seed", "use_lp", "use_delta_interpolation", "lp_epochs", "peft_epochs",
    "final_iid_acc", "final_bilinear_acc", "final_nearest_acc", "bilinear_drop", "nearest_drop", "bilinear_rrr", "nearest_rrr",
    "lfer_iid", "lfer_nearest", "cka_nearest", "specdist_nearest", "high_specdist_nearest", "trainable_params", "total_params",
    "throughput", "peak_mem_mb", "training_time_total_min", "training_time_per_epoch_sec", "selected_alpha", "alpha_selection_metric",
    "alpha_selection_rule", "nearest_used_for_alpha_selection", "checkpoint_selection_rule", "nearest_used_for_checkpoint_selection",
    "nan_flag", "divergence_flag", "config_path", "output_dir",
]


def _subset_loader(loader, max_batches: int | None):
    if max_batches is None or max_batches <= 0:
        return loader
    # Generator wrapper for quick smoke tests.
    def gen():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            yield batch
    return gen()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def build_model_from_config(cfg: dict) -> torch.nn.Module:
    bcfg = cfg.get("backbone", {})
    return build_vit_backbone(
        name=bcfg.get("name", "vit_base_patch16_224.augreg_in21k_ft_in1k"),
        num_classes=int(bcfg.get("num_classes", 10)),
        pretrained=_as_bool(bcfg.get("pretrained", True)),
        checkpoint_path=bcfg.get("checkpoint_path", None),
    )


def inject_peft_from_config(model: torch.nn.Module, cfg: dict):
    mcfg = cfg.get("method", {})
    return inject_lora_dora(
        model,
        peft_type=mcfg.get("peft_type", mcfg.get("name", "lora")),
        target_modules=mcfg.get("target_modules", ["attn.qkv", "attn.proj"]),
        target_blocks=mcfg.get("target_blocks", "all"),
        rank=int(mcfg.get("rank", 16)),
        alpha=float(mcfg.get("lora_alpha", 16)),
        dropout=float(mcfg.get("dropout", 0.0)),
    )


def _evaluate_and_log(
    model,
    loader,
    device,
    cfg,
    output_dir: Path,
    run_id: str,
    method: str,
    peft_type: str,
    seed: int,
    stage: str,
    epoch: int,
    checkpoint_type: str,
    split: str,
    interpolation: str,
    alpha: float,
    reference_model=None,
):
    set_delta_scale(model, alpha)
    row = evaluate(
        model=model,
        loader=loader,
        device=device,
        cfg=cfg,
        split=split,
        interpolation=interpolation,
        alpha=alpha,
        reference_model=reference_model,
    )
    row.update(
        {
            "run_id": run_id,
            "method": method,
            "peft_type": peft_type,
            "seed": int(seed),
            "stage": stage,
            "epoch": int(epoch),
            "checkpoint_type": checkpoint_type,
        }
    )
    append_csv(output_dir / "eval_log.csv", row, EVAL_LOG_FIELDS)
    return row


def select_alpha(alpha_rows: List[dict], cfg: dict) -> tuple[float, str]:
    """Select alpha from validation Bicubic/Bilinear only."""
    rule = cfg.get("delta_interpolation", {}).get("selection_rule", "val_bilinear_acc_with_bicubic_guard")
    guard_pp = float(cfg.get("delta_interpolation", {}).get("bicubic_guard_pp", 0.5))
    if not alpha_rows:
        return 1.0, rule
    max_bicubic = max(float(r["val_bicubic_acc"]) for r in alpha_rows)
    for r in alpha_rows:
        r["bicubic_guard_pass"] = int(float(r["val_bicubic_acc"]) >= max_bicubic - guard_pp / 100.0)
        r["selection_score"] = float(r["val_bilinear_acc"])
    candidates = [r for r in alpha_rows if int(r["bicubic_guard_pass"]) == 1]
    if not candidates:
        candidates = alpha_rows
    # Max bilinear acc, min bilinear drop, then smaller alpha for stronger retention.
    candidates = sorted(
        candidates,
        key=lambda r: (-float(r["val_bilinear_acc"]), float(r["val_bilinear_drop"]), float(r["alpha"])),
    )
    return float(candidates[0]["alpha"]), rule


def run(cfg: dict, config_path: str, seed_override: int | None = None, max_train_batches: int | None = None, max_eval_batches: int | None = None) -> dict:
    if seed_override is not None:
        cfg["seed"] = int(seed_override)
    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic=bool(cfg.get("deterministic", False)))

    method_cfg = cfg.get("method", {})
    method = str(method_cfg.get("name", "lora"))
    peft_type = str(method_cfg.get("peft_type", "lora")).lower()
    use_lp = _as_bool(method_cfg.get("use_lp", False))
    use_delta = _as_bool(method_cfg.get("use_delta_interpolation", False))

    project = cfg.get("project", {})
    output_root = Path(project.get("output_root", "outputs/fr_peft_cifar10"))
    run_id = f"{method}_seed{seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = output_root / run_id
    ckpt_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    data_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("train", {})
    dataset_name = str(data_cfg.get("name", "CIFAR10")).lower()
    if dataset_name in {"cifar100", "cifar-100"}:
        loader_builder = build_cifar100_loaders
    elif dataset_name in {"cifar10", "cifar-10"}:
        loader_builder = build_cifar10_loaders
    else:
        raise ValueError(f"Unsupported dataset.name={data_cfg.get('name')}")

    loaders = loader_builder(
        root=data_cfg.get("root", "./outputs/data"),
        image_size=int(data_cfg.get("image_size", 224)),
        train_interpolation=data_cfg.get("train_interpolation", "bicubic"),
        val_interpolations=data_cfg.get("val_interpolations", ["bicubic", "bilinear"]),
        test_interpolations=data_cfg.get("test_interpolations", ["bicubic", "bilinear", "nearest"]),
        batch_size=int(train_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        val_ratio=float(data_cfg.get("val_ratio", 0.1)),
        seed=seed,
        download=_as_bool(data_cfg.get("download", True)),
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device)

    total_train_start = time.time()
    nan_flag = 0
    divergence_flag = 0
    global_step = 0

    lp_epochs = int(train_cfg.get("lp_epochs", 0)) if use_lp else 0
    peft_epochs = int(train_cfg.get("peft_epochs", 15 if not use_lp else 10))
    save_weights = _as_bool(project.get("save_weights", True))

    if use_lp and lp_epochs > 0:
        mark_head_only_trainable(model)
        optimizer = build_optimizer(model, cfg, stage="lp")
        scheduler = build_scheduler(optimizer, cfg, epochs=lp_epochs)
        scaler = GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")
        for ep in range(1, lp_epochs + 1):
            row = train_one_epoch(
                model,
                _subset_loader(loaders.train, max_train_batches),
                optimizer,
                device,
                epoch=ep,
                stage="lp",
                cfg=cfg,
                scaler=scaler,
            )
            global_step += int(row["num_samples"])
            row.update({"run_id": run_id, "method": method, "peft_type": peft_type, "seed": seed, "global_step": global_step, "batch_size": int(train_cfg.get("batch_size", 64)), "head_delta_norm": "", "checkpoint_path": ""})
            append_csv(output_dir / "train_log.csv", row, TRAIN_LOG_FIELDS)
            nan_flag = max(nan_flag, int(row.get("nan_flag", 0)))
            divergence_flag = max(divergence_flag, int(row.get("divergence_flag", 0)))
            if scheduler is not None:
                scheduler.step()
        lp_head_path = ckpt_dir / f"lp_head_seed{seed}.pth"
        if save_weights:
            save_head_checkpoint(lp_head_path, model, metadata={"method": method, "seed": seed, "lp_epochs": lp_epochs})

    # Inject PEFT after LP so adapter delta starts from zero around the LP head.
    injection_report = inject_peft_from_config(model, cfg)

    # Important: PEFT wrappers are created after the base model has already been
    # moved to device. Newly created LoRA/DoRA parameters start on CPU unless the
    # whole model is moved to device again.
    model = model.to(device)

    mark_only_peft_and_head_trainable(model)
    optimizer = build_optimizer(model, cfg, stage="peft")
    scheduler = build_scheduler(optimizer, cfg, epochs=peft_epochs)
    scaler = GradScaler(enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    for ep in range(1, peft_epochs + 1):
        row = train_one_epoch(
            model,
            _subset_loader(loaders.train, max_train_batches),
            optimizer,
            device,
            epoch=ep,
            stage="peft",
            cfg=cfg,
            scaler=scaler,
        )
        global_step += int(row["num_samples"])
        row.update({"run_id": run_id, "method": method, "peft_type": peft_type, "seed": seed, "global_step": global_step, "batch_size": int(train_cfg.get("batch_size", 64)), "head_delta_norm": "", "checkpoint_path": ""})
        append_csv(output_dir / "train_log.csv", row, TRAIN_LOG_FIELDS)
        nan_flag = max(nan_flag, int(row.get("nan_flag", 0)))
        divergence_flag = max(divergence_flag, int(row.get("divergence_flag", 0)))
        if scheduler is not None:
            scheduler.step()

    final_ckpt_path = ckpt_dir / f"final_{method}_seed{seed}.pth"
    if save_weights:
        save_checkpoint(
            final_ckpt_path,
            {
                "model_state_dict": model.state_dict(),
                "method": method,
                "peft_type": peft_type,
                "seed": seed,
                "injection_report": injection_report.__dict__,
                "config": cfg,
            },
        )

    # Frozen reference for feature-retention metrics.  It is never used for alpha selection.
    reference_model = None
    if any(bool(cfg.get("metrics", {}).get(k, True)) for k in ["compute_lfer", "compute_cka", "compute_specdist", "compute_high_specdist"]):
        reference_model = build_model_from_config(cfg).to(device)
        reference_model.eval()
        for p in reference_model.parameters():
            p.requires_grad = False

    # Alpha selection on validation Bicubic/Bilinear only.
    alpha_candidates = cfg.get("delta_interpolation", {}).get("alpha_candidates", [1.0]) if use_delta else [1.0]
    alpha_candidates = [float(a) for a in alpha_candidates]
    alpha_rows: list[dict] = []
    for alpha in alpha_candidates:
        val_bicubic = _evaluate_and_log(
            model,
            _subset_loader(loaders.val["bicubic"], max_eval_batches),
            device,
            cfg,
            output_dir,
            run_id,
            method,
            peft_type,
            seed,
            stage="alpha_select",
            epoch=peft_epochs,
            checkpoint_type="final",
            split="val",
            interpolation="bicubic",
            alpha=alpha,
            reference_model=None,
        )
        val_bilinear = _evaluate_and_log(
            model,
            _subset_loader(loaders.val["bilinear"], max_eval_batches),
            device,
            cfg,
            output_dir,
            run_id,
            method,
            peft_type,
            seed,
            stage="alpha_select",
            epoch=peft_epochs,
            checkpoint_type="final",
            split="val",
            interpolation="bilinear",
            alpha=alpha,
            reference_model=None,
        )
        alpha_rows.append(
            {
                "run_id": run_id,
                "method": method,
                "peft_type": peft_type,
                "seed": seed,
                "checkpoint_type": "final",
                "alpha": alpha,
                "val_bicubic_acc": float(val_bicubic["acc"]),
                "val_bilinear_acc": float(val_bilinear["acc"]),
                "val_bilinear_drop": float(val_bicubic["acc"]) - float(val_bilinear["acc"]),
                "selection_score": float(val_bilinear["acc"]),
                "bicubic_guard_pass": 0,
                "selected": 0,
                "selection_rule": cfg.get("delta_interpolation", {}).get("selection_rule", "val_bilinear_acc_with_bicubic_guard"),
                "nearest_used_for_selection": 0,
            }
        )

    selected_alpha, selection_rule = select_alpha(alpha_rows, cfg)
    for r in alpha_rows:
        r["selected"] = int(abs(float(r["alpha"]) - selected_alpha) < 1e-12)
        r["selection_rule"] = selection_rule
        append_csv(output_dir / "alpha_selection.csv", r, ALPHA_FIELDS)

    # Held-out test evaluation after alpha is fixed.  Nearest is not used before this point.
    test_rows: dict[str, dict] = {}
    for interp, loader in loaders.test.items():
        test_rows[interp] = _evaluate_and_log(
            model,
            _subset_loader(loader, max_eval_batches),
            device,
            cfg,
            output_dir,
            run_id,
            method,
            peft_type,
            seed,
            stage="test",
            epoch=peft_epochs,
            checkpoint_type="final_val_alpha_selected",
            split="test",
            interpolation=interp,
            alpha=selected_alpha,
            reference_model=reference_model,
        )

    trainable, total = count_parameters(model)
    total_elapsed = max(time.time() - total_train_start, 1e-6)
    iid = float(test_rows.get("bicubic", {}).get("acc", 0.0))
    bilinear = float(test_rows.get("bilinear", {}).get("acc", 0.0))
    nearest = float(test_rows.get("nearest", {}).get("acc", 0.0))
    nearest_row = test_rows.get("nearest", {})
    iid_row = test_rows.get("bicubic", {})
    summary = {
        "run_id": run_id,
        "method": method,
        "peft_type": peft_type,
        "seed": seed,
        "use_lp": int(use_lp),
        "use_delta_interpolation": int(use_delta),
        "lp_epochs": lp_epochs,
        "peft_epochs": peft_epochs,
        "final_iid_acc": iid,
        "final_bilinear_acc": bilinear,
        "final_nearest_acc": nearest,
        "bilinear_drop": iid - bilinear,
        "nearest_drop": iid - nearest,
        "bilinear_rrr": bilinear / iid if iid > 0 else 0.0,
        "nearest_rrr": nearest / iid if iid > 0 else 0.0,
        "lfer_iid": iid_row.get("lfer", ""),
        "lfer_nearest": nearest_row.get("lfer", ""),
        "cka_nearest": nearest_row.get("cka_to_frozen", ""),
        "specdist_nearest": nearest_row.get("specdist_to_frozen", ""),
        "high_specdist_nearest": nearest_row.get("high_specdist_to_frozen", ""),
        "trainable_params": trainable,
        "total_params": total,
        "throughput": nearest_row.get("throughput_img_s", ""),
        "peak_mem_mb": nearest_row.get("peak_mem_mb", ""),
        "training_time_total_min": total_elapsed / 60.0,
        "training_time_per_epoch_sec": total_elapsed / max(lp_epochs + peft_epochs, 1),
        "selected_alpha": selected_alpha,
        "alpha_selection_metric": "val_bilinear_acc_with_bicubic_guard",
        "alpha_selection_rule": selection_rule,
        "nearest_used_for_alpha_selection": 0,
        "checkpoint_selection_rule": "final_checkpoint_only",
        "nearest_used_for_checkpoint_selection": 0,
        "nan_flag": int(nan_flag),
        "divergence_flag": int(divergence_flag),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
    }
    append_csv(output_dir / "summary.csv", summary, SUMMARY_FIELDS)
    # Also append a project-level summary for easy aggregation.
    append_csv(output_root / "summary_all.csv", summary, SUMMARY_FIELDS)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FR-PEFT CIFAR-10 experiment.")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None, help="Debug only: limit train batches per epoch")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Debug only: limit eval batches")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    summary = run(
        cfg,
        config_path=args.config,
        seed_override=args.seed,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
