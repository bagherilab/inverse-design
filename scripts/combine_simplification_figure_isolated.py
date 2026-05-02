#!/usr/bin/env python3
"""Simplification figure: isolated peak layout for the MI table plus metric bars."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from combine_simplification_metrics import (  # noqa: E402
    DEFAULT_CLUSTER_DIR,
    SIMP_LEVELS_R,
    SIMP_LEVELS_T,
    plot_metric_section,
)
from combine_simplification_mi_table import (  # noqa: E402
    ARCADE_OUTPUT_DIR,
    CLUSTER_N,
    DEFAULT_MI_SUMMARY_JSON,
    DEFAULT_N1024_DIR,
    DEFAULT_SUMMARY_JSON,
    export_mi_summary_to_json,
    plot_mi_table_panel,
)
from inverse_design.vis.utils import (  # noqa: E402
    peak_colors,
    peak_marker_markersize,
    peak_markers,
)
from inverse_design.plotting.style import (  # noqa: E402
    DPI,
    FONT_BASE,
    FONT_LABEL,
    apply_style,
    savefig_both,
)

DEFAULT_PEAK_LABELS = ("P1", "P2", "P3", "P4")
PEAK_LABEL_TO_INDEX = {label: idx for idx, label in enumerate(DEFAULT_PEAK_LABELS, start=1)}
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "simplification"
_SINGLE_PEAK_TOP_HEIGHT = 2.30
_SINGLE_PEAK_BOTTOM_HEIGHT = 1.95


def _draw_isolated_title(fig, peak_idx: int, top_y: float) -> None:
    color_idx = (peak_idx - 1 + 2) % len(peak_colors)
    cluster_marker = peak_markers[color_idx]
    cluster_color = peak_colors[color_idx]
    title_ax = fig.add_axes((0.05, top_y - 0.11, 0.22, 0.04), zorder=10)
    title_ax.set_axis_off()
    title_ax.plot(
        0.08,
        0.5,
        marker=cluster_marker,
        markersize=peak_marker_markersize(cluster_marker),
        markerfacecolor=cluster_color,
        markeredgecolor="black",
        markeredgewidth=0.4,
        linestyle="none",
        transform=title_ax.transAxes,
        clip_on=False,
    )
    title_ax.text(
        0.18,
        0.5,
        f"Peak {peak_idx} (n={CLUSTER_N[peak_idx]})",
        ha="left",
        va="center",
        transform=title_ax.transAxes,
        fontsize=FONT_LABEL,
        color="#000000",
    )


def _parse_peak_labels(peaks: list[str] | None) -> tuple[int, ...]:
    if not peaks:
        return (3,)
    normalized: list[int] = []
    for peak in peaks:
        token = peak.strip().upper()
        if token.isdigit():
            token = f"P{token}"
        if token not in PEAK_LABEL_TO_INDEX:
            raise ValueError(f"Unknown peak label: {peak}")
        peak_idx = PEAK_LABEL_TO_INDEX[token]
        if peak_idx not in normalized:
            normalized.append(peak_idx)
    return tuple(normalized)


def _resolve_output_path(output_dir: Path, out: Path | None) -> Path:
    if out is None:
        return output_dir / "simplified_figure_p3.png"
    return out if out.is_absolute() else output_dir / out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Isolated simplification figure: one peak with pairwise and hierarchical sections."
    )
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--n1024-dir", type=Path, default=DEFAULT_N1024_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--lr-arcade-root",
        type=Path,
        default=None,
        help=(
            "Where to find *linear_<r>* runs for per-column r MI masking (default "
            f"{ARCADE_OUTPUT_DIR})."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--peaks",
        nargs="+",
        default=["P3"],
        metavar="PEAK",
        help="Single peak label to render, e.g. --peaks P3.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output filename or absolute path for the isolated figure.",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--font-size", type=int, default=FONT_LABEL)
    parser.add_argument(
        "--export-mi-json",
        type=Path,
        default=DEFAULT_MI_SUMMARY_JSON,
        help=f"Write MI summary JSON (default: {DEFAULT_MI_SUMMARY_JSON}).",
    )
    parser.add_argument(
        "--no-export-mi-json",
        action="store_true",
        help="Skip writing data/summary_MI.json after the figure.",
    )
    return parser.parse_args(argv)


def render(
    args: argparse.Namespace,
    *,
    peaks: list[str] | tuple[str, ...] | None = None,
    out: str | Path | None = None,
) -> Path:
    render_args = copy.copy(args)
    render_args.peaks_to_plot = _parse_peak_labels(list(peaks) if peaks is not None else args.peaks)
    if len(render_args.peaks_to_plot) != 1:
        raise ValueError("combine_simplification_figure_isolated.py only supports one peak at a time")
    output_path = _resolve_output_path(render_args.output_dir, Path(out) if isinstance(out, str) else out)

    apply_style()
    render_args.output_dir.mkdir(parents=True, exist_ok=True)

    fig_height = _SINGLE_PEAK_TOP_HEIGHT + _SINGLE_PEAK_BOTTOM_HEIGHT
    fig = plt.figure(figsize=(8.00, fig_height), dpi=render_args.dpi, facecolor="white")
    fig.subplots_adjust(left=0.035, right=1.0, top=0.90, bottom=0.03)
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.04)
    _draw_isolated_title(fig, render_args.peaks_to_plot[0], 0.965)

    mi_args = copy.copy(render_args)
    mi_args.font_size = FONT_BASE
    mi_args.show_peak_header = False

    pairwise_outer = outer[0, 0].subgridspec(
        2,
        3,
        height_ratios=[1.50, 0.30],
        width_ratios=[0.06, 0.88, 0.06],
        hspace=0.10,
        wspace=0.0,
    )
    dendrogram_outer = outer[0, 1].subgridspec(
        2,
        3,
        height_ratios=[1.50, 0.30],
        width_ratios=[0.06, 0.88, 0.06],
        hspace=0.10,
        wspace=0.0,
    )

    plot_mi_table_panel(
        fig,
        pairwise_outer[0, 1],
        mi_args,
        panel_label="A",
        section_levels=SIMP_LEVELS_R,
        section_title="Pairwise regression",
        show_param_labels=True,
    )
    plot_metric_section(
        fig,
        pairwise_outer[1, 1],
        render_args,
        section="r",
        show_section_title=False,
        show_row_markers=False,
        title_strip_ratio=0.03,
    )
    plot_mi_table_panel(
        fig,
        dendrogram_outer[0, 1],
        mi_args,
        panel_label="B",
        section_levels=SIMP_LEVELS_T,
        section_title="Hierarchical clustering",
        show_param_labels=False,
        label_w_override=0.0,
        content_x1_override=0.72,
    )
    plot_metric_section(
        fig,
        dendrogram_outer[1, 1],
        render_args,
        section="t",
        show_section_title=False,
        show_row_markers=False,
        title_strip_ratio=0.03,
    )

    savefig_both(fig, output_path)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render(args, peaks=args.peaks, out=args.out)
    if not args.no_export_mi_json:
        export_mi_summary_to_json(args.export_mi_json, args)


if __name__ == "__main__":
    main()
