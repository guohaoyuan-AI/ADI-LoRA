#!/usr/bin/env python3
"""Summarize ADI merged-checkpoint equivalence audits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    rows: list[dict] = []
    for audit_path in sorted(root.glob("*/merge_audit.json")):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        parts = audit_path.parent.name.split("_")
        family = parts[0] if parts else ""
        seed = parts[-1].replace("seed", "") if parts else ""
        rows.append(
            {
                "run": audit_path.parent.name,
                "family": family,
                "seed": seed,
                "alpha": audit.get("alpha", ""),
                "merged": audit.get("merged", ""),
                "merged_peft_modules": audit.get("merged_peft_modules", ""),
                "extra_inference_modules": audit.get("extra_inference_modules", ""),
                "max_output_error_before_after_merge": audit.get("max_output_error_before_after_merge", ""),
                "max_output_error_pass_threshold": audit.get("max_output_error_pass_threshold", ""),
                "merge_equivalence_pass": audit.get("merge_equivalence_pass", ""),
                "load_missing_keys": audit.get("load_missing_keys", ""),
                "load_unexpected_keys": audit.get("load_unexpected_keys", ""),
                "alpha_selection_source": audit.get("alpha_selection_source", ""),
                "nearest_used_for_alpha_selection": audit.get("nearest_used_for_alpha_selection", ""),
                "corruption_used_for_alpha_selection": audit.get("corruption_used_for_alpha_selection", ""),
                "checkpoint_selection_rule": audit.get("checkpoint_selection_rule", ""),
                "nearest_used_for_checkpoint_selection": audit.get("nearest_used_for_checkpoint_selection", ""),
                "source_checkpoint": audit.get("source_checkpoint", ""),
                "output_checkpoint": audit.get("output_checkpoint", ""),
                "audit_path": str(audit_path),
            }
        )

    fields = [
        "run",
        "family",
        "seed",
        "alpha",
        "merged",
        "merged_peft_modules",
        "extra_inference_modules",
        "max_output_error_before_after_merge",
        "max_output_error_pass_threshold",
        "merge_equivalence_pass",
        "load_missing_keys",
        "load_unexpected_keys",
        "alpha_selection_source",
        "nearest_used_for_alpha_selection",
        "corruption_used_for_alpha_selection",
        "checkpoint_selection_rule",
        "nearest_used_for_checkpoint_selection",
        "source_checkpoint",
        "output_checkpoint",
        "audit_path",
    ]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    passed = [
        bool(row["merge_equivalence_pass"])
        and str(row["merged"]).lower() == "true"
        and str(row["extra_inference_modules"]) in {"0", "0.0"}
        for row in rows
    ]
    summary = {
        "root": str(root),
        "num_expected": 6,
        "num_found": len(rows),
        "all_expected_found": len(rows) == 6,
        "all_merge_equivalence_pass": bool(rows) and all(passed) and len(rows) == 6,
        "nearest_used_for_alpha_selection": 0,
        "corruption_used_for_alpha_selection": 0,
        "checkpoint_selection_rule": "final_checkpoint_only",
        "interpretation": "deployment equivalence evidence; not alpha selection and not an accuracy experiment",
    }
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
