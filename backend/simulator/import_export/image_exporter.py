from __future__ import annotations

from typing import Dict, List

from ..layers import LayerConfig


def export_svg(layers: List[LayerConfig], width: int = 1200, height: int = 400) -> Dict:
    padding = 40
    spacing = 120
    x = padding
    y = height // 2
    rect_w = 100
    rect_h = 50
    svg_parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]
    svg_parts.append("<rect width='100%' height='100%' fill='#0b1220' />")

    for idx, layer in enumerate(layers):
        label = layer.layer_type
        svg_parts.append(
            f"<rect x='{x}' y='{y - rect_h // 2}' width='{rect_w}' height='{rect_h}' fill='#1e293b' stroke='#38bdf8' stroke-width='2' rx='8' />"
        )
        svg_parts.append(
            f"<text x='{x + rect_w / 2}' y='{y + 5}' font-size='12' fill='#e2e8f0' text-anchor='middle'>{label}</text>"
        )
        if idx < len(layers) - 1:
            x2 = x + rect_w
            svg_parts.append(
                f"<line x1='{x2}' y1='{y}' x2='{x2 + spacing - 20}' y2='{y}' stroke='#94a3b8' stroke-width='2' />"
            )
        x += spacing

    svg_parts.append("</svg>")
    svg = "".join(svg_parts)
    return {"image_base64": svg, "format": "svg"}
