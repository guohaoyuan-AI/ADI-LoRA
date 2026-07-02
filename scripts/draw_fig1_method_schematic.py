#!/usr/bin/env python3
"""Draw Fig. 1 method schematic for ADI.

Run from the LaTeX project root:

    python scripts/draw_fig1_method_schematic.py

Outputs:
    figures/generated/fig1_adi_method_schematic.svg
    figures/generated/fig1_adi_method_schematic.pdf  (if Inkscape is available)
    figures/generated/fig1_adi_method_schematic.png  (if Inkscape is available)
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"


BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRAY = "#777777"
DARK = "#222222"
LIGHT = "#F7F7F7"
PANEL = "#FBFBFB"


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def text(x: float, y: float, content: str, size: int = 14, anchor: str = "middle", weight: str = "normal", fill: str = DARK) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{esc(content)}</text>'
    )


def multiline(x: float, y: float, content: str, size: int = 13, anchor: str = "middle", weight: str = "normal", fill: str = DARK, gap: int = 16) -> str:
    lines = content.split("\n")
    tspans = []
    for i, line in enumerate(lines):
        dy = 0 if i == 0 else gap
        tspans.append(f'<tspan x="{x:.1f}" dy="{dy}">{esc(line)}</tspan>')
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">'
        + "".join(tspans)
        + "</text>"
    )


def rect(x: float, y: float, w: float, h: float, fill: str = "white", stroke: str = DARK, sw: float = 1.0, rx: float = 6) -> str:
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'


def line(x1: float, y1: float, x2: float, y2: float, stroke: str = DARK, sw: float = 1.2, arrow: bool = False, dash: str | None = None) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw:.1f}"{marker}{dash_attr}/>'


def circle(cx: float, cy: float, r: float, fill: str, stroke: str = DARK, sw: float = 1.0) -> str:
    return f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:.1f}"/>'


def panel_title(x: float, y: float, title: str) -> str:
    return multiline(x, y, title, size=16, anchor="middle", weight="bold", gap=18)


def draw() -> str:
    width, height = 1500, 570
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">',
        f'<path d="M0,0 L0,6 L9,3 z" fill="{DARK}"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        text(width / 2, 34, "Adapter Delta Interpolation (ADI): post-hoc delta calibration", size=20, weight="bold"),
    ]

    panels = [(36, 68, 444, 464), (528, 68, 444, 464), (1020, 68, 444, 464)]
    for x, y, w, h in panels:
        parts.append(rect(x, y, w, h, fill=PANEL, stroke="#D2D2D2", sw=1.0, rx=8))

    # Panel A
    x0, y0, _, _ = panels[0]
    parts.append(panel_title(x0 + 222, y0 + 34, "(a) Standard PEFT training"))
    parts.append(rect(x0 + 36, y0 + 88, 104, 54, fill="white", stroke="#BDBDBD"))
    parts.append(multiline(x0 + 88, y0 + 111, "Image\ninput", size=12, gap=14))
    parts.append(line(x0 + 142, y0 + 115, x0 + 185, y0 + 115, arrow=True))
    parts.append(rect(x0 + 188, y0 + 84, 198, 62, fill="#EFEFEF", stroke=GRAY))
    parts.append(multiline(x0 + 287, y0 + 105, "Frozen ViT weight\nW0", size=13, weight="bold", gap=16))
    parts.append(rect(x0 + 210, y0 + 174, 154, 52, fill="#FFF4DC", stroke=ORANGE))
    parts.append(multiline(x0 + 287, y0 + 195, "LoRA / DoRA\nadapter", size=12, gap=15))
    parts.append(line(x0 + 287, y0 + 146, x0 + 287, y0 + 172, arrow=True, stroke=ORANGE))
    parts.append(line(x0 + 287, y0 + 226, x0 + 287, y0 + 258, arrow=True))
    parts.append(rect(x0 + 136, y0 + 260, 302, 58, fill="white", stroke=DARK))
    parts.append(multiline(x0 + 287, y0 + 282, "Final PEFT checkpoint (α = 1.0)\nW_eff = W0 + ΔW", size=13, gap=16))
    parts.append(rect(x0 + 56, y0 + 344, 332, 88, fill=LIGHT, stroke="#CCCCCC"))
    parts.append(multiline(x0 + 222, y0 + 370, "Train once | final checkpoint only\nNo ADI during training", size=12, gap=18))

    # Panel B
    x1, y1, _, _ = panels[1]
    parts.append(panel_title(x1 + 222, y1 + 34, "(b) Adapter Delta Interpolation"))
    axis_y = y1 + 190
    left_x, mid_x, right_x = x1 + 66, x1 + 222, x1 + 378
    parts.append(line(left_x, axis_y, right_x, axis_y, stroke=DARK, sw=2.0, arrow=True))
    parts.append(circle(left_x, axis_y, 9, fill="#E0E0E0"))
    parts.append(circle(mid_x, axis_y, 10, fill=BLUE))
    parts.append(circle(right_x, axis_y, 9, fill=ORANGE))
    parts.append(multiline(left_x, axis_y + 36, "W0\npretrained anchor", size=12, gap=15))
    parts.append(multiline(mid_x, axis_y + 36, "W0 + αΔW\nADI calibrated model", size=12, weight="bold", fill=BLUE, gap=15))
    parts.append(multiline(right_x, axis_y + 36, "W0 + ΔW\nstandard PEFT", size=12, gap=15))
    parts.append(rect(x1 + 54, y1 + 274, 336, 58, fill="white", stroke=BLUE))
    parts.append(multiline(x1 + 222, y1 + 297, "W_ADI(α) = W0 + αΔW,   0 < α ≤ 1", size=13, weight="bold", fill=BLUE))
    parts.append(rect(x1 + 54, y1 + 348, 336, 58, fill=LIGHT, stroke="#CCCCCC"))
    parts.append(multiline(x1 + 222, y1 + 371, "DoRA: W_ADI(α) = W0 + α(W_DoRA - W0)", size=12, gap=16))
    parts.append(multiline(x1 + 222, y1 + 112, "Attenuate the task-specific delta\nretain the pretrained visual prior", size=13, weight="bold", fill=GREEN, gap=18))

    # Panel C
    x2, y2, _, _ = panels[2]
    parts.append(panel_title(x2 + 222, y2 + 34, "(c) Alpha selection and tests"))
    bx, bw, bh = x2 + 74, 296, 50
    ys = [y2 + 84, y2 + 154, y2 + 224, y2 + 310]
    labels = [
        "Candidate α grid\n{0.2, 0.4, 0.6, 0.8, 1.0}",
        "Bicubic / Bilinear validation",
        "Select α*",
        "Held-out evaluation\nBicubic | Bilinear | Nearest | CIFAR-C",
    ]
    fills = ["white", "#EAF3FB", "#EAF3FB", "#F1F1F1"]
    strokes = [DARK, BLUE, BLUE, GRAY]
    for i, (yy, label, fill, stroke) in enumerate(zip(ys, labels, fills, strokes)):
        parts.append(rect(bx, yy, bw, bh if i != 3 else 62, fill=fill, stroke=stroke))
        parts.append(multiline(bx + bw / 2, yy + 21, label, size=12, weight="bold" if i == 2 else "normal", gap=15))
        if i < 3:
            parts.append(line(bx + bw / 2, yy + (bh if i != 3 else 62) + 4, bx + bw / 2, ys[i + 1] - 6, arrow=True))
    parts.append(rect(x2 + 58, y2 + 386, 328, 74, fill="#FFF7F7", stroke="#CC7777"))
    parts.append(multiline(x2 + 222, y2 + 409, "Nearest / corruption not used for α selection\nFinal checkpoint only | same checkpoint\nNo extra params | no extra module", size=10, fill="#8A1F1F", gap=14))

    # Inter-panel arrows
    parts.append(line(486, 278, 522, 278, arrow=True, stroke="#666666"))
    parts.append(line(978, 278, 1014, 278, arrow=True, stroke="#666666"))

    parts.append("</svg>")
    return "\n".join(parts)


def convert(svg_path: Path) -> None:
    inkscape = shutil.which("inkscape")
    if inkscape is None:
        print(f"Wrote SVG only: {svg_path}")
        return
    pdf_path = svg_path.with_suffix(".pdf")
    png_path = svg_path.with_suffix(".png")
    subprocess.run([inkscape, str(svg_path), "--export-type=pdf", f"--export-filename={pdf_path}"], check=True)
    subprocess.run([inkscape, str(svg_path), "--export-type=png", "--export-dpi=300", f"--export-filename={png_path}"], check=True)
    print(f"Wrote {svg_path}, {pdf_path}, and {png_path}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svg_path = OUT / "fig1_adi_method_schematic.svg"
    svg_path.write_text(draw(), encoding="utf-8")
    convert(svg_path)


if __name__ == "__main__":
    main()
