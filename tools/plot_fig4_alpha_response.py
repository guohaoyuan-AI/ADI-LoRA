#!/usr/bin/env python3
"""Plot Fig. 4 alpha-response curves from protocol-valid sweep CSV files."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_inputs(patterns: list[str]) -> pd.DataFrame:
    frames = []
    for pattern in patterns:
        for path in glob.glob(pattern):
            df = pd.read_csv(path)
            df["source_file"] = path
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No CSV files matched: {patterns}")
    data = pd.concat(frames, ignore_index=True)
    required = {
        "method", "seed", "alpha", "val_bicubic_acc_pct", "val_bilinear_acc_pct",
        "test_nearest_acc_pct", "selected", "nearest_used_for_alpha_selection",
        "corruption_used_for_alpha_selection", "checkpoint_selection_rule",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns for Fig. 4: {sorted(missing)}")
    bad = data[
        (data["nearest_used_for_alpha_selection"].astype(int) != 0)
        | (data["corruption_used_for_alpha_selection"].astype(int) != 0)
        | (data["checkpoint_selection_rule"].astype(str) != "final_checkpoint_only")
    ]
    if not bad.empty:
        raise ValueError("Fig. 4 input contains protocol-invalid rows.")
    return data


def panel(ax, data: pd.DataFrame, method: str, title: str) -> None:
    method_rows = data[data["method"].astype(str).str.upper().str.contains(method.upper())].copy()
    if method_rows.empty:
        ax.set_title(title + " (missing)")
        return
    metrics = [
        ("val_bicubic_acc_pct", "Val Bicubic", "#4C78A8"),
        ("val_bilinear_acc_pct", "Val Bilinear", "#F58518"),
        ("test_nearest_acc_pct", "Held-out Test Nearest", "#54A24B"),
    ]
    grouped = method_rows.groupby("alpha")
    for col, label, color in metrics:
        mean = grouped[col].mean().sort_index()
        std = grouped[col].std().reindex(mean.index).fillna(0.0)
        ax.plot(mean.index, mean.values, marker="o", linewidth=1.8, label=label, color=color)
        ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values, color=color, alpha=0.14, linewidth=0)
    selected_alphas = sorted(method_rows.loc[method_rows["selected"].astype(int) == 1, "alpha"].astype(float).unique())
    for alpha in selected_alphas:
        y = method_rows[method_rows["alpha"].astype(float).eq(alpha)]["test_nearest_acc_pct"].mean()
        ax.scatter([alpha], [y], marker="D", s=42, color="#222222", zorder=5)
    ax.axvline(1.0, linestyle="--", color="#666666", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Delta scale alpha")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="y", color="#dddddd", linewidth=0.7)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="CSV paths/globs from LoRA and DoRA full-alpha sweeps")
    ap.add_argument("--out", default="figures/generated/fig4_alpha_response.pdf")
    args = ap.parse_args()

    data = load_inputs(args.inputs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
    panel(axes[0], data, "LORA", "(a) ADI-LoRA")
    panel(axes[1], data, "DORA", "(b) ADI-DoRA")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Alpha-response curve on CIFAR-100", y=0.98, fontsize=10)
    fig.text(0.5, 0.01, "Diamonds mark validation-selected alpha*. Nearest is diagnostic held-out evaluation only.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0.0, 0.10, 1.0, 0.93])
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
