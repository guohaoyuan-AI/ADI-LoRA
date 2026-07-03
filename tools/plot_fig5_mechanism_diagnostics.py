#!/usr/bin/env python3
"""Plot Fig. 5 mechanism diagnostics from same-checkpoint diagnostic CSV rows."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    ("cka_to_frozen", "CKA to frozen backbone"),
    ("specdist_to_frozen", "SpecDist to frozen backbone"),
    ("high_specdist_to_frozen", "HighSpecDist to frozen backbone"),
]


def load_inputs(patterns: list[str]) -> pd.DataFrame:
    frames = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            df = pd.read_csv(path)
            df["source_file"] = path
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No mechanism diagnostic CSV files matched: {patterns}")
    data = pd.concat(frames, ignore_index=True)
    required = {"method", "seed", "alpha_name", "nearest_used_for_alpha_selection", "corruption_used_for_alpha_selection", "checkpoint_selection_rule"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns for Fig. 5: {sorted(missing)}")
    bad = data[
        (data["nearest_used_for_alpha_selection"].astype(int) != 0)
        | (data["corruption_used_for_alpha_selection"].astype(int) != 0)
        | (data["checkpoint_selection_rule"].astype(str) != "final_checkpoint_only")
    ]
    if not bad.empty:
        raise ValueError("Fig. 5 input contains protocol-invalid rows.")
    return data


def label_method_alpha(row: pd.Series) -> str:
    method = str(row["method"]).upper()
    family = "LoRA" if "LORA" in method and "DORA" not in method else "DoRA"
    alpha_name = str(row["alpha_name"]).lower()
    return f"{family} alpha=1.0" if alpha_name == "alpha1" else f"ADI-{family} alpha*"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="figures/generated/fig5_mechanism_diagnostics.pdf")
    args = ap.parse_args()

    data = load_inputs(args.inputs)
    data = data.copy()
    data["group"] = data.apply(label_method_alpha, axis=1)
    order = ["LoRA alpha=1.0", "ADI-LoRA alpha*", "DoRA alpha=1.0", "ADI-DoRA alpha*"]
    colors = ["#9ECAE9", "#3182BD", "#FDBE85", "#E6550D"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
    for ax, (metric, title) in zip(axes, METRICS):
        if metric not in data.columns:
            ax.set_title(title + " (missing)")
            continue
        stats = data.groupby("group")[metric].agg(["mean", "std"]).reindex(order)
        x = range(len(order))
        ax.bar(x, stats["mean"], yerr=stats["std"].fillna(0.0), color=colors, edgecolor="#222222", linewidth=0.5, capsize=2)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)
    fig.suptitle("Representation and spectral diagnostics on held-out Nearest", y=0.98, fontsize=10)
    fig.text(0.5, 0.01, "Diagnostics suggest reduced drift from the pretrained visual anchor; they are not causal proof.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 0.91])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
