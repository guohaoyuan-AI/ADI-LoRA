"""Evaluation and feature-diagnostic utilities for FR-PEFT."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from adi_lora.metrics.frequency_metrics import linear_cka, low_frequency_energy_ratio, spectral_log_amplitude_distance
from adi_lora.models.backbones import extract_feature_tokens, infer_vit_spatial_size


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: dict,
    split: str,
    interpolation: str,
    alpha: float = 1.0,
    reference_model: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    model.eval()
    if reference_model is not None:
        reference_model.eval()
    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(cfg.get("train", {}).get("amp", True)) and device.type == "cuda"
    metrics_cfg = cfg.get("metrics", {})
    cutoff = float(metrics_cfg.get("freq_cutoff_ratio", 0.35))

    spatial_size = tuple(metrics_cfg.get("spatial_size", []) or infer_vit_spatial_size(model))

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    lfer_sum = 0.0
    cka_sum = 0.0
    specdist_sum = 0.0
    high_specdist_sum = 0.0
    metric_batches = 0
    start = time.time()

    for images, targets in tqdm(loader, desc=f"eval/{split}/{interpolation}/a={alpha}", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with autocast(enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)
        batch_size = int(targets.shape[0])
        total_seen += batch_size
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum().item())

        if reference_model is not None and any(
            bool(metrics_cfg.get(key, True))
            for key in ["compute_lfer", "compute_cka", "compute_specdist", "compute_high_specdist"]
        ):
            with autocast(enabled=False):
                feat = extract_feature_tokens(model, images).float()
                ref_feat = extract_feature_tokens(reference_model, images).float()
            metric_batches += 1
            if bool(metrics_cfg.get("compute_lfer", True)):
                lfer_sum += float(low_frequency_energy_ratio(feat, spatial_size=spatial_size, cutoff_ratio=cutoff).cpu().item())
            if bool(metrics_cfg.get("compute_cka", True)):
                cka_sum += float(linear_cka(feat, ref_feat).cpu().item())
            if bool(metrics_cfg.get("compute_specdist", True)):
                specdist_sum += float(
                    spectral_log_amplitude_distance(feat, ref_feat, spatial_size=spatial_size, cutoff_ratio=cutoff).cpu().item()
                )
            if bool(metrics_cfg.get("compute_high_specdist", True)):
                high_specdist_sum += float(
                    spectral_log_amplitude_distance(
                        feat,
                        ref_feat,
                        spatial_size=spatial_size,
                        cutoff_ratio=cutoff,
                        high_pass_only=True,
                    ).cpu().item()
                )

    elapsed = max(time.time() - start, 1e-6)
    peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    denom = max(metric_batches, 1)
    return {
        "split": split,
        "interpolation": interpolation,
        "alpha": float(alpha),
        "acc": total_correct / max(total_seen, 1),
        "loss_ce": total_loss / max(total_seen, 1),
        "lfer": lfer_sum / denom if metric_batches else "",
        "cka_to_frozen": cka_sum / denom if metric_batches else "",
        "specdist_to_frozen": specdist_sum / denom if metric_batches else "",
        "high_specdist_to_frozen": high_specdist_sum / denom if metric_batches else "",
        "num_samples": int(total_seen),
        "eval_time_sec": elapsed,
        "throughput_img_s": float(total_seen / elapsed),
        "peak_mem_mb": peak_mem_mb,
    }
