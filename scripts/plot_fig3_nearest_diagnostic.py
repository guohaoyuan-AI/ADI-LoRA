#!/usr/bin/env python3
"""Generate the CIFAR-100 held-out Nearest alpha-response diagnostic figure.

The figure is diagnostic only: Nearest accuracy is plotted after alpha has
already been selected from Bicubic/Bilinear validation signals.
"""

from __future__ import annotations

import csv
import html
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figure_data"
OUT = ROOT / "figures" / "generated"
ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text(x: float, y: float, value: object, size: int = 10, anchor: str = "middle", weight: str = "normal", fill: str = "#111", rotate: float | None = None) -> str:
    transform = f' transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}"{transform}>{esc(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = "#222", width: float = 1.0, dash: str | None = None, opacity: float = 1.0) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.2f}"{dash_attr}/>'


def circle(x: float, y: float, r: float, fill: str, stroke: str = "#222", width: float = 0.8, opacity: float = 1.0) -> str:
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}" opacity="{opacity:.2f}"/>'


def diamond(x: float, y: float, r: float, fill: str, stroke: str = "#222") -> str:
    pts = f"{x:.1f},{y-r:.1f} {x+r:.1f},{y:.1f} {x:.1f},{y+r:.1f} {x-r:.1f},{y:.1f}"
    return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>'


def read_rows() -> list[dict[str, str]]:
    path = DATA / "fig3_full_alpha_nearest_diagnostic.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def save_svg(name: str, width: int, height: int, parts: list[str]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        *parts,
        "</svg>",
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


def convert_svg(svg_path: Path) -> None:
    inkscape = shutil.which("inkscape")
    if not inkscape:
        return
    subprocess.run([inkscape, str(svg_path), "--export-type=pdf", f"--export-filename={svg_path.with_suffix('.pdf')}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([inkscape, str(svg_path), "--export-type=png", "--export-dpi=400", f"--export-filename={svg_path.with_suffix('.png')}"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def plot_panel(parts: list[str], rows: list[dict[str, str]], method: str, x0: float, y0: float, w: float, h: float, color: str, title: str, subtitle: str, show_ylabel: bool) -> None:
    ymin, ymax = 44.0, 66.0

    def xmap(alpha: float) -> float:
        return x0 + (alpha - 0.2) / 0.8 * w

    def ymap(value: float) -> float:
        return y0 + h - (value - ymin) / (ymax - ymin) * h

    parts.append(text(x0 + w / 2, y0 - 30, title, 12, weight="bold"))
    parts.append(text(x0 + w / 2, y0 - 14, subtitle, 9, fill="#555"))
    parts.append(line(x0, y0 + h, x0 + w, y0 + h, "#222", 0.9))
    parts.append(line(x0, y0, x0, y0 + h, "#222", 0.9))

    for tick in [45, 50, 55, 60, 65]:
        y = ymap(tick)
        parts.append(line(x0, y, x0 + w, y, "#D8D8D8", 0.55, opacity=0.9))
        parts.append(text(x0 - 8, y + 4, tick, 9, "end"))
    for alpha in ALPHAS:
        parts.append(text(xmap(alpha), y0 + h + 20, f"{alpha:.1f}", 9))
    if show_ylabel:
        parts.append(text(x0 - 45, y0 + h / 2, "Nearest accuracy (%)", 10, rotate=-90))

    parts.append(line(xmap(1.0), y0, xmap(1.0), y0 + h, "#555", 1.1, dash="5,4", opacity=0.9))
    parts.append(text(xmap(1.0) - 6, y0 + 14, "alpha=1.0", 8, "end", fill="#555", rotate=-90))

    by_seed: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            by_seed[row["seed"]].append(row)

    for seed, seed_rows in sorted(by_seed.items()):
        seed_rows = sorted(seed_rows, key=lambda r: float(r["alpha"]))
        pts = [(xmap(float(r["alpha"])), ymap(float(r["test_nearest_acc_pct"]))) for r in seed_rows]
        if len(by_seed) > 1:
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                parts.append(line(x1, y1, x2, y2, "#B8B8B8", 1.0, opacity=0.75))
            for x, y in pts:
                parts.append(circle(x, y, 2.6, "#F4F4F4", stroke="#8A8A8A", width=0.65))
        else:
            for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
                parts.append(line(x1, y1, x2, y2, color, 2.2, opacity=0.95))
            for x, y in pts:
                parts.append(circle(x, y, 3.6, color, stroke="#222", width=0.7))

    mean_points: list[tuple[float, float]] = []
    method_rows = [r for r in rows if r["method"] == method]
    for alpha in ALPHAS:
        vals = [float(r["test_nearest_acc_pct"]) for r in method_rows if abs(float(r["alpha"]) - alpha) < 1e-9]
        if vals:
            mean_points.append((xmap(alpha), ymap(mean(vals))))
    if len(by_seed) > 1:
        for (x1, y1), (x2, y2) in zip(mean_points, mean_points[1:]):
            parts.append(line(x1, y1, x2, y2, color, 2.4))
        for x, y in mean_points:
            parts.append(circle(x, y, 4.3, color, stroke="#222", width=0.7))

    selected_alphas = sorted({float(row["alpha"]) for row in method_rows if row["selected"] == "1"})
    for alpha in selected_alphas:
        x = xmap(alpha)
        parts.append(line(x, y0, x, y0 + h, color, 1.1, dash="3,3", opacity=0.75))
        parts.append(text(x + 5, y0 + 16, "alpha*", 8, "start", fill=color, rotate=-90))
    parts.append(text(x0 + w / 2, y0 + h + 44, "Delta scale alpha", 10))


def main() -> None:
    rows = read_rows()
    width, height = 920, 360
    parts: list[str] = [
        text(width / 2, 28, "Held-out Nearest alpha response diagnostic", 14, weight="bold"),
        text(width / 2, 48, "Nearest is plotted only after Bicubic/Bilinear validation selects alpha", 9, fill="#555"),
    ]
    plot_panel(parts, rows, "ADI-LoRA", 75, 92, 355, 190, "#0072B2", "ADI-LoRA", "three seeds; gray lines show seeds", True)
    plot_panel(parts, rows, "ADI-DoRA", 535, 92, 310, 190, "#D55E00", "ADI-DoRA", "seed42 only; diagnostic", False)
    parts.append(circle(90, 335, 4.3, "#0072B2"))
    parts.append(text(103, 339, "LoRA mean", 9, "start"))
    parts.append(circle(188, 335, 2.6, "#F4F4F4", stroke="#8A8A8A", width=0.65))
    parts.append(text(201, 339, "LoRA seed", 9, "start"))
    parts.append(circle(302, 335, 4.0, "#D55E00"))
    parts.append(text(315, 339, "DoRA seed42", 9, "start"))
    parts.append(line(418, 334, 440, 334, "#555", 1.1, dash="3,3"))
    parts.append(text(450, 339, "validation-selected alpha*", 9, "start"))
    svg = save_svg("fig3_full_alpha_nearest_diagnostic_v1.svg", width, height, parts)
    convert_svg(svg)


if __name__ == "__main__":
    main()
