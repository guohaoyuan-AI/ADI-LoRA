#!/usr/bin/env python3
"""Generate figure drafts from checked CSV data.

Run from the LaTeX project root:

    python scripts/plot_figures.py

Outputs are written to figures/generated/.
Publication-polished drafts use the *_v2.pdf/png/svg suffix.

Dependency behavior:
  - If matplotlib is installed, the script writes PDF and PNG.
  - If matplotlib is not installed, the script still writes SVG using only the
    Python standard library. This makes it runnable in a fresh VSCode setup.

The LaTeX manuscript does not include these figures automatically. Inspect and
tune the generated files before inserting them with \\includegraphics.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figure_data"
OUT = ROOT / "figures" / "generated"


def read_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">{text}</text>'
    )


def svg_multiline_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle", weight: str = "normal", line_gap: int = 14) -> str:
    lines = text.split("\n")
    tspans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else line_gap
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{line}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}">'
        + "".join(tspans)
        + "</text>"
    )


def write_svg_bar_chart(
    path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    ymax: float,
    notes: Iterable[str] = (),
) -> None:
    width, height = 760, 430
    left, right, top, bottom = 88, 30, 54, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    zero_y = top + plot_h

    def y_pos(v: float) -> float:
        return top + plot_h - (v / ymax) * plot_h

    n_groups = len(labels)
    n_series = len(series)
    group_w = plot_w / n_groups
    bar_w = min(52, group_w / (n_series + 1.5))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, title, size=16, weight="bold"),
        svg_text(18, top + plot_h / 2, ylabel, size=13, anchor="middle"),
        f'<line x1="{left}" y1="{zero_y}" x2="{width-right}" y2="{zero_y}" stroke="#222" stroke-width="1.1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{zero_y}" stroke="#222" stroke-width="1.1"/>',
    ]

    for tick in range(0, int(ymax) + 1, 2):
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd" stroke-width="0.8"/>')
        parts.append(svg_text(left - 10, y + 4, str(tick), size=11, anchor="end"))

    for gi, label in enumerate(labels):
        cx = left + group_w * (gi + 0.5)
        parts.append(svg_text(cx, zero_y + 28, label, size=12))
        for si, (_, values, color) in enumerate(series):
            v = values[gi]
            x = cx - (n_series * bar_w) / 2 + si * bar_w + 4
            y = y_pos(v)
            h = zero_y - y
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-8:.1f}" height="{h:.1f}" '
                f'fill="{color}" stroke="#222" stroke-width="0.7"/>'
            )
            parts.append(svg_text(x + (bar_w - 8) / 2, y - 7, f"+{v:.2f}", size=11))

    legend_x = left + 8
    legend_y = height - 34
    for i, (name, _, color) in enumerate(series):
        lx = legend_x + i * 170
        parts.append(f'<rect x="{lx}" y="{legend_y-11}" width="14" height="14" fill="{color}" stroke="#222" stroke-width="0.6"/>')
        parts.append(svg_text(lx + 20, legend_y, name, size=11, anchor="start"))

    for i, note in enumerate(notes):
        parts.append(svg_text(width / 2, height - 12 - i * 16, note, size=10))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_svg_bar_chart_v2(
    path: Path,
    title: str,
    ylabel: str,
    labels: list[str],
    series: list[dict[str, object]],
    ymax: float,
    tick_step: int = 2,
) -> None:
    width, height = 720, 420
    left, right, top, bottom = 78, 28, 62, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    zero_y = top + plot_h

    def y_pos(v: float) -> float:
        return top + plot_h - (v / ymax) * plot_h

    n_groups = len(labels)
    n_series = len(series)
    group_w = plot_w / n_groups
    bar_w = min(42, group_w / (n_series + 1.7))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 28, title, size=12, weight="bold"),
        svg_text(left, 49, ylabel, size=10, anchor="start"),
        f'<line x1="{left}" y1="{zero_y}" x2="{width-right}" y2="{zero_y}" stroke="#222" stroke-width="1.0"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{zero_y}" stroke="#222" stroke-width="1.0"/>',
    ]

    tick = 0
    while tick <= ymax:
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#dddddd" stroke-width="0.7"/>')
        parts.append(svg_text(left - 9, y + 4, f"{tick:g}", size=9, anchor="end"))
        tick += tick_step

    for gi, label in enumerate(labels):
        cx = left + group_w * (gi + 0.5)
        parts.append(svg_multiline_text(cx, zero_y + 28, label, size=10, line_gap=13))
        for si, item in enumerate(series):
            values = item["values"]
            color = item["color"]
            errors = item.get("errors")
            v = float(values[gi])
            x = cx - (n_series * bar_w) / 2 + si * bar_w + 4
            y = y_pos(v)
            h = zero_y - y
            bar_center = x + (bar_w - 8) / 2
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w-8:.1f}" height="{h:.1f}" '
                f'fill="{color}" stroke="#222" stroke-width="0.6"/>'
            )
            if errors is not None:
                err = float(errors[gi])
                y_top = y_pos(min(ymax, v + err))
                y_bottom = y_pos(max(0, v - err))
                parts.append(f'<line x1="{bar_center:.1f}" y1="{y_top:.1f}" x2="{bar_center:.1f}" y2="{y_bottom:.1f}" stroke="#222" stroke-width="0.7"/>')
                parts.append(f'<line x1="{bar_center-5:.1f}" y1="{y_top:.1f}" x2="{bar_center+5:.1f}" y2="{y_top:.1f}" stroke="#222" stroke-width="0.7"/>')
                parts.append(f'<line x1="{bar_center-5:.1f}" y1="{y_bottom:.1f}" x2="{bar_center+5:.1f}" y2="{y_bottom:.1f}" stroke="#222" stroke-width="0.7"/>')
            parts.append(svg_text(bar_center, y - 7, f"+{v:.2f}", size=9))

    legend_x = left + 4
    legend_y = height - 24
    legend_gap = 170 if len(series) > 1 else 0
    for i, item in enumerate(series):
        lx = legend_x + i * legend_gap
        parts.append(f'<rect x="{lx}" y="{legend_y-11}" width="13" height="13" fill="{item["color"]}" stroke="#222" stroke-width="0.5"/>')
        parts.append(svg_text(lx + 19, legend_y, str(item["name"]), size=9, anchor="start"))

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def generate_svg() -> None:
    ensure_out()

    clean = [r for r in read_rows("fig_clean_nearest_gains.csv") if r["dataset"] == "CIFAR-100"]
    write_svg_bar_chart(
        OUT / "fig_clean_nearest_gains.svg",
        "CIFAR-100 clean interpolation robustness",
        "Gain / reduction (pp)",
        [r["method"] for r in clean],
        [
            ("Nearest gain", [float(r["mean_nearest_gain_pp"]) for r in clean], "#4C78A8"),
            ("Drop reduction", [float(r["mean_drop_reduction_pp"]) for r in clean], "#F58518"),
        ],
        ymax=12,
        notes=["Alpha is selected without Nearest access."],
    )

    c100c = read_rows("fig_cifar100c_summary.csv")
    write_svg_bar_chart(
        OUT / "fig_cifar100c_summary.svg",
        "CIFAR-100-C ADI-DoRA robustness",
        "Mean Acc gain (pp)",
        ["Subset\n3 seeds", "Full19 s=3\nseed42"],
        [
            ("Mean Acc gain", [float(r["mean_acc_gain_pp"]) for r in c100c], "#54A24B"),
        ],
        ymax=6,
        notes=["Corruption sets are not used for alpha selection."],
    )

    write_svg_bar_chart_v2(
        OUT / "fig_clean_nearest_gains_v2.svg",
        "CIFAR-100 clean interpolation",
        "Gain / reduction (pp)",
        [r["method"] for r in clean],
        [
            {
                "name": "Nearest gain",
                "values": [float(r["mean_nearest_gain_pp"]) for r in clean],
                "errors": [float(r["std_nearest_gain_pp"]) for r in clean],
                "color": "#0072B2",
            },
            {
                "name": "Drop reduction",
                "values": [float(r["mean_drop_reduction_pp"]) for r in clean],
                "color": "#E69F00",
            },
        ],
        ymax=12,
    )

    write_svg_bar_chart_v2(
        OUT / "fig_cifar100c_summary_v2.svg",
        "CIFAR-100-C ADI-DoRA",
        "Accuracy gain (pp)",
        ["Subset\n3 seeds, 36 cases\n34/36 positive", "Full19 s=3\nseed42, 19 cases\n19/19 positive"],
        [
            {
                "name": "Mean accuracy gain",
                "values": [float(r["mean_acc_gain_pp"]) for r in c100c],
                "errors": [float(r["std_acc_gain_pp"]) for r in c100c],
                "color": "#009E73",
            },
        ],
        ymax=6,
    )


def generate_matplotlib() -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return False

    ensure_out()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )

    clean = [r for r in read_rows("fig_clean_nearest_gains.csv") if r["dataset"] == "CIFAR-100"]
    labels = [r["method"] for r in clean]
    gains = [float(r["mean_nearest_gain_pp"]) for r in clean]
    errors = [float(r["std_nearest_gain_pp"]) for r in clean]
    drops = [float(r["mean_drop_reduction_pp"]) for r in clean]

    fig, ax = plt.subplots(figsize=(4.3, 2.65))
    x = list(range(len(labels)))
    width = 0.34
    ax.bar([i - width / 2 for i in x], gains, width, yerr=errors, capsize=3, label="Nearest gain", color="#4C78A8", edgecolor="black", linewidth=0.5)
    ax.bar([i + width / 2 for i in x], drops, width, label="Drop reduction", color="#F58518", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gain / reduction (pp)")
    ax.set_title("CIFAR-100 clean interpolation robustness")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    for i, (g, d) in enumerate(zip(gains, drops)):
        ax.text(i - width / 2, g + 0.35, f"+{g:.2f}", ha="center")
        ax.text(i + width / 2, d + 0.35, f"+{d:.2f}", ha="center")
    fig.savefig(OUT / "fig_clean_nearest_gains.pdf")
    fig.savefig(OUT / "fig_clean_nearest_gains.png")
    plt.close(fig)

    c100c = read_rows("fig_cifar100c_summary.csv")
    labels = ["Subset\n3 seeds", "Full19 s=3\nseed42"]
    gains = [float(r["mean_acc_gain_pp"]) for r in c100c]
    errors = [float(r["std_acc_gain_pp"]) for r in c100c]
    positive = [f'{r["positive_acc_cases"]}/{r["total_acc_cases"]} positive' for r in c100c]

    fig, ax = plt.subplots(figsize=(4.1, 2.65))
    bars = ax.bar(labels, gains, yerr=errors, capsize=3, color=["#54A24B", "#9ACB88"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean accuracy gain (pp)")
    ax.set_title("CIFAR-100-C ADI-DoRA robustness")
    ax.set_ylim(0, max(g + e for g, e in zip(gains, errors)) + 1.0)
    for bar, gain, pos in zip(bars, gains, positive):
        x_pos = bar.get_x() + bar.get_width() / 2
        ax.text(x_pos, bar.get_height() + 0.12, f"+{gain:.2f}", ha="center")
        ax.text(x_pos, 0.15, pos, ha="center", fontsize=8)
    ax.text(0.5, -0.28, "Corruption sets are not used for alpha selection.", transform=ax.transAxes, ha="center", va="top", fontsize=8)
    fig.savefig(OUT / "fig_cifar100c_summary.pdf")
    fig.savefig(OUT / "fig_cifar100c_summary.png")
    plt.close(fig)

    generate_matplotlib_v2(plt)
    return True


def annotate_bar(ax, x: float, y: float, text: str, dy: float = 0.18) -> None:
    ax.text(x, y + dy, text, ha="center", va="bottom", fontsize=8)


def generate_matplotlib_v2(plt) -> None:
    """Generate publication-oriented figure drafts for visual QA.

    The v2 figures avoid in-plot explanatory prose and leave protocol details to
    the captions. They are not inserted automatically into the manuscript.
    """

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )

    clean = [r for r in read_rows("fig_clean_nearest_gains.csv") if r["dataset"] == "CIFAR-100"]
    labels = [r["method"] for r in clean]
    gains = [float(r["mean_nearest_gain_pp"]) for r in clean]
    errors = [float(r["std_nearest_gain_pp"]) for r in clean]
    drops = [float(r["mean_drop_reduction_pp"]) for r in clean]

    fig, ax = plt.subplots(figsize=(3.7, 2.45))
    fig.subplots_adjust(left=0.18, right=0.985, top=0.84, bottom=0.23)
    x = list(range(len(labels)))
    width = 0.32
    ax.bar(
        [i - width / 2 for i in x],
        gains,
        width,
        yerr=errors,
        capsize=2.5,
        label="Nearest gain",
        color="#0072B2",
        edgecolor="black",
        linewidth=0.45,
        error_kw={"elinewidth": 0.6, "capthick": 0.6},
    )
    ax.bar(
        [i + width / 2 for i in x],
        drops,
        width,
        label="Drop reduction",
        color="#E69F00",
        edgecolor="black",
        linewidth=0.45,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gain / reduction (pp)")
    ax.set_ylim(0, 12)
    ax.set_title("CIFAR-100 clean interpolation", pad=5)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=1.2)
    for i, (g, d) in enumerate(zip(gains, drops)):
        annotate_bar(ax, i - width / 2, g, f"+{g:.2f}")
        annotate_bar(ax, i + width / 2, d, f"+{d:.2f}")
    fig.savefig(OUT / "fig_clean_nearest_gains_v2.pdf")
    fig.savefig(OUT / "fig_clean_nearest_gains_v2.png")
    fig.savefig(OUT / "fig_clean_nearest_gains_v2.svg")
    plt.close(fig)

    c100c = read_rows("fig_cifar100c_summary.csv")
    labels = ["Subset\n3 seeds, 36 cases", "Full19 s=3\nseed42, 19 cases"]
    gains = [float(r["mean_acc_gain_pp"]) for r in c100c]
    errors = [float(r["std_acc_gain_pp"]) for r in c100c]
    positive = [f'{r["positive_acc_cases"]}/{r["total_acc_cases"]}' for r in c100c]
    tiers = ["Main", "Supp."]

    fig, ax = plt.subplots(figsize=(3.7, 2.45))
    fig.subplots_adjust(left=0.18, right=0.985, top=0.84, bottom=0.27)
    x = list(range(len(labels)))
    bars = ax.bar(
        x,
        gains,
        yerr=errors,
        capsize=2.5,
        color=["#009E73", "#8CC9B0"],
        edgecolor="black",
        linewidth=0.45,
        error_kw={"elinewidth": 0.6, "capthick": 0.6},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy gain (pp)")
    ax.set_ylim(0, 6)
    ax.set_title("CIFAR-100-C ADI-DoRA", pad=5)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.45, alpha=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for i, (bar, gain, pos, tier) in enumerate(zip(bars, gains, positive, tiers)):
        x_pos = bar.get_x() + bar.get_width() / 2
        annotate_bar(ax, x_pos, gain, f"+{gain:.2f}")
        ax.text(x_pos, 0.18, f"{pos} positive\n{tier}", ha="center", va="bottom", fontsize=7)
    fig.savefig(OUT / "fig_cifar100c_summary_v2.pdf")
    fig.savefig(OUT / "fig_cifar100c_summary_v2.png")
    fig.savefig(OUT / "fig_cifar100c_summary_v2.svg")
    plt.close(fig)


def convert_svg_with_inkscape() -> bool:
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        return False
    ok = True
    for svg_path in OUT.glob("*.svg"):
        pdf_path = svg_path.with_suffix(".pdf")
        png_path = svg_path.with_suffix(".png")
        for out_path, args in [
            (pdf_path, ["--export-type=pdf", f"--export-filename={pdf_path}"]),
            (png_path, ["--export-type=png", "--export-dpi=300", f"--export-filename={png_path}"]),
        ]:
            try:
                subprocess.run([inkscape, str(svg_path), *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                ok = False
    return ok


def main() -> None:
    generate_svg()
    has_mpl = generate_matplotlib()
    if has_mpl:
        print(f"Wrote SVG, PDF, and PNG figures to {OUT}")
    else:
        has_inkscape = convert_svg_with_inkscape()
        if has_inkscape:
            print(f"Wrote SVG, PDF, and PNG figures to {OUT} using Inkscape conversion")
        else:
            print(f"Wrote SVG figures to {OUT}")
            print("To also export PDF/PNG, install matplotlib or Inkscape:")
            print("  python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib")


if __name__ == "__main__":
    main()
