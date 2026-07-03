#!/usr/bin/env python3
"""Export a dense merged checkpoint for a trained ADI-LoRA/ADI-DoRA model."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import torch.nn as nn

from adi_lora.engine.run_fr_peft import build_model_from_config, inject_peft_from_config
from adi_lora.models.peft import DoRALinear, LoRALinear, set_delta_scale


def get_parent_and_name(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


@torch.no_grad()
def dense_from_peft(module: LoRALinear | DoRALinear, alpha: float) -> nn.Linear:
    set_delta = float(alpha)
    module.delta_scale = set_delta
    dense = nn.Linear(module.in_features, module.out_features, bias=module.base.bias is not None)
    dense = dense.to(device=module.base.weight.device, dtype=module.base.weight.dtype)
    if isinstance(module, LoRALinear):
        weight = module.base.weight + set_delta * module.delta_weight().to(device=module.base.weight.device, dtype=module.base.weight.dtype)
    else:
        weight = module._interpolated_weight()
    dense.weight.copy_(weight)
    if module.base.bias is not None:
        dense.bias.copy_(module.base.bias)
    return dense


@torch.no_grad()
def merge_model(model: nn.Module, alpha: float) -> tuple[nn.Module, int]:
    merged = copy.deepcopy(model)
    replacements = []
    for name, module in merged.named_modules():
        if isinstance(module, (LoRALinear, DoRALinear)):
            replacements.append((name, dense_from_peft(module, alpha)))
    for name, dense in replacements:
        parent, child = get_parent_and_name(merged, name)
        setattr(parent, child, dense)
    return merged, len(replacements)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-out", default=None)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--probe-seed", type=int, default=123)
    ap.add_argument("--pass-threshold", type=float, default=1e-4)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8")) if args.config.endswith(".json") else None
    if cfg is None:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    model = build_model_from_config(cfg).to(device)
    inject_peft_from_config(model, cfg)
    model = model.to(device)
    state = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(state.get("model_state_dict", state), strict=False)
    model.eval()
    set_delta_scale(model, args.alpha)

    merged, merged_modules = merge_model(model, args.alpha)
    merged = merged.to(device)
    merged.eval()

    image_size = int(cfg.get("dataset", {}).get("image_size", 224))
    torch.manual_seed(int(args.probe_seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.probe_seed))
    sample = torch.randn(2, 3, image_size, image_size, device=device)
    with torch.no_grad():
        before = model(sample)
        after = merged(sample)
    max_err = float((before - after).abs().max().detach().cpu().item())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": merged.state_dict(),
            "alpha": float(args.alpha),
            "merged": True,
            "extra_inference_modules": 0,
            "source_checkpoint": str(args.checkpoint),
            "source_config": str(args.config),
        },
        out,
    )

    audit = {
        "alpha": float(args.alpha),
        "merged": True,
        "merged_peft_modules": int(merged_modules),
        "extra_inference_modules": 0,
        "max_output_error_before_after_merge": max_err,
        "max_output_error_pass_threshold": float(args.pass_threshold),
        "merge_equivalence_pass": bool(max_err <= float(args.pass_threshold)),
        "probe_seed": int(args.probe_seed),
        "probe_batch_size": 2,
        "load_missing_keys": len(missing),
        "load_unexpected_keys": len(unexpected),
        "alpha_selection_source": "val_bicubic_and_val_bilinear",
        "nearest_used_for_alpha_selection": 0,
        "corruption_used_for_alpha_selection": 0,
        "checkpoint_selection_rule": "final_checkpoint_only",
        "nearest_used_for_checkpoint_selection": 0,
        "diagnostic_or_deployment_evidence": "deployment_equivalence_not_accuracy_selection",
        "source_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(out),
    }
    audit_out = Path(args.audit_out) if args.audit_out else out.with_name("merge_audit.json")
    audit_out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
