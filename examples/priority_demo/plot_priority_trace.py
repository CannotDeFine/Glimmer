#!/usr/bin/env python3

"""Render the mixed priority workload CSV traces as a standalone SVG."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


WIDTH = 1120
HEIGHT = 500
LEFT = 130
RIGHT = 35
TOP = 35
BOTTOM = 78
PLOT_WIDTH = WIDTH - LEFT - RIGHT
PLOT_HEIGHT = HEIGHT - TOP - BOTTOM
COLORS = {"inference": "#174a68", "training": "#8296a3"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                role = row.get("role", "")
                if role not in COLORS:
                    raise ValueError(f"unsupported role in {path}: {role}")
                rows.append(
                    {
                        "role": role,
                        "iteration": int(row["iteration"]),
                        "launch_requested_us": int(row["launch_requested_us"]),
                        "completed_us": int(row["completed_us"]),
                    }
                )
    if not rows:
        raise ValueError("input traces contain no rows")
    return rows


def scale(value: int, start: int, end: int) -> float:
    if end == start:
        return LEFT + PLOT_WIDTH / 2
    return LEFT + (value - start) * PLOT_WIDTH / (end - start)


def text(x: float, y: float, value: str, size: int = 14, anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="DejaVu Sans, sans-serif" '
        f'font-size="{size}px" fill="#111820" text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def render(rows: list[dict[str, float | int | str]]) -> str:
    start = min(int(row["launch_requested_us"]) for row in rows)
    end = max(int(row["completed_us"]) for row in rows)
    if end <= start:
        end = start + 1
    lane_height = PLOT_HEIGHT / 2
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="{LEFT}" y="{TOP}" width="{PLOT_WIDTH}" height="{PLOT_HEIGHT}" '
        'fill="#f4f7f9" stroke="#111820" stroke-width="1.5"/>',
    ]

    for index, role in enumerate(("inference", "training")):
        y = TOP + lane_height * index
        if index == 0:
            parts.append(
                f'<rect x="{LEFT}" y="{y:.1f}" width="{PLOT_WIDTH}" height="{lane_height:.1f}" '
                'fill="#e9f0f4" opacity="0.8"/>'
            )
        parts.append(text(LEFT - 16, y + lane_height / 2 + 5, role, 15, "end"))
        parts.append(
            f'<line x1="{LEFT}" y1="{y + lane_height:.1f}" x2="{LEFT + PLOT_WIDTH}" '
            f'y2="{y + lane_height:.1f}" stroke="#b5c0c8" stroke-width="1"/>'
        )

    bar_height = 27
    for row in sorted(rows, key=lambda item: int(item["launch_requested_us"])):
        role = str(row["role"])
        lane = 0 if role == "inference" else 1
        x_start = scale(int(row["launch_requested_us"]), start, end)
        x_end = scale(int(row["completed_us"]), start, end)
        width = max(2.0, x_end - x_start)
        y = TOP + lane_height * lane + (lane_height - bar_height) / 2
        parts.append(
            f'<rect x="{x_start:.1f}" y="{y:.1f}" width="{width:.1f}" height="{bar_height}" '
            f'rx="2" fill="{COLORS[role]}" opacity="0.9"/>'
        )

    axis_y = TOP + PLOT_HEIGHT
    parts.append(
        f'<line x1="{LEFT}" y1="{axis_y}" x2="{LEFT + PLOT_WIDTH}" y2="{axis_y}" '
        'stroke="#111820" stroke-width="1.5"/>'
    )
    for fraction in (0.0, 0.5, 1.0):
        x = LEFT + PLOT_WIDTH * fraction
        value_ms = (end - start) * fraction / 1000.0
        parts.append(
            f'<line x1="{x:.1f}" y1="{axis_y}" x2="{x:.1f}" y2="{axis_y + 6}" '
            'stroke="#111820"/>'
        )
        parts.append(text(x, axis_y + 25, f"{value_ms:.1f}", 13, "middle"))
    parts.append(text(LEFT + PLOT_WIDTH / 2, HEIGHT - 21, "time since first launch (ms)", 15, "middle"))

    legend_y = HEIGHT - 52
    legend_x = LEFT + PLOT_WIDTH / 2 - 120
    for offset, role in enumerate(("inference", "training")):
        x = legend_x + offset * 150
        parts.append(
            f'<rect x="{x:.1f}" y="{legend_y - 12}" width="18" height="18" fill="{COLORS[role]}"/>'
        )
        parts.append(text(x + 27, legend_y + 3, role, 14))
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    args = parse_args()
    try:
        svg = render(read_rows(args.input))
    except (OSError, KeyError, ValueError) as error:
        print(f"plot_priority_trace.py: {error}")
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
