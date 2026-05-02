"""Generate the ARCADE model simplification figure."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from inverse_design.plotting.plot_model_simplification import (  # noqa: E402
    draw_dendrogram_panel,
    draw_scatter_panels,
    load_lr_data,
    load_peak_posterior,
    load_posterior,
)
from inverse_design.plotting.style import (  # noqa: E402
    DPI,
    FONT_LABEL,
    FONT_TITLE,
    WIDTH_1P5,
    apply_style,
    panel_letter,
    savefig_both,
)
from inverse_design.vis.utils import (  # noqa: E402
    peak_colors,
    peak_marker_markersize,
    peak_markers,
)

ARCADE_ROOT = Path(__file__).resolve().parents[2] / "ARCADE_OUTPUT"
# Same peak marker/color cycling as ``combine_simplification_mi_table`` / fit-exp figures.
_CLUSTER_COLOR_OFFSET = 2
# Shared figure y (transFigure) for panel A letter and Peak marker+label (``va="center"``).
PANEL_A_ROW_Y = 0.90
TITLE_STRIP_H = 0.038


def get_paths(peak: int) -> dict[str, Path]:
    linear_base = (
        ARCADE_ROOT
        / f"ABC_SMC_RF_N512_combined_grid_linear_0.8_p{peak}_mean_only"
        / f"ABC_SMC_RF_N512_combined_grid_linear_0.8_p{peak}_mean_only"
    )
    dend_base = (
        ARCADE_ROOT
        / f"ABC_SMC_RF_N512_combined_grid_dendrogram_1.0_p{peak}_mean_only"
        / f"ABC_SMC_RF_N512_combined_grid_dendrogram_1.0_p{peak}_mean_only"
    )
    return {
        "lr_json": linear_base / f"lr_predictions_r0.8_p{peak}.json",
        "linear_posterior": linear_base / "iter_4" / "all_param_df.csv",
        "dend_json": dend_base / f"redundancy_analysis_threshold_1.0_p{peak}.json",
        "base_posterior": ARCADE_ROOT
        / "ABC_SMC_RF_N512_combined_grid_breast_only_mean"
        / "iter_4"
        / "all_param_df.csv",
    }


def build_figure(peak: int, font_size: int = FONT_LABEL, dpi: int = DPI) -> plt.Figure:
    apply_style()
    paths = get_paths(peak)

    models = load_lr_data(paths["lr_json"])
    linear_posterior = load_posterior(paths["linear_posterior"])
    peak_posterior = load_peak_posterior(paths["base_posterior"], peak=peak)

    fig = plt.figure(figsize=(WIDTH_1P5, 4.82), dpi=dpi)
    top_gs = gridspec.GridSpec(
        1,
        3,
        figure=fig,
        left=0.12,
        right=0.98,
        top=0.88,
        bottom=0.57,
        wspace=0.85,
    )
    bottom_gs = gridspec.GridSpec(
        1,
        1,
        figure=fig,
        left=0.33,
        right=0.98,
        top=0.49,
        bottom=0.12,
    )

    ax_s0 = fig.add_subplot(top_gs[0, 0])
    ax_s1 = fig.add_subplot(top_gs[0, 1])
    ax_s2 = fig.add_subplot(top_gs[0, 2])
    ax_dend = fig.add_subplot(bottom_gs[0, 0])

    draw_scatter_panels([ax_s0, ax_s1, ax_s2], models, linear_posterior, font_size=font_size)
    draw_dendrogram_panel(
        ax_dend,
        peak_posterior,
        ward_threshold=1.0,
        font_size=font_size,
    )

    panel_letter(fig, "A", 0.02, PANEL_A_ROW_Y, va="center")

    color_idx = (peak - 1 + _CLUSTER_COLOR_OFFSET) % len(peak_colors)
    cluster_marker = peak_markers[color_idx]
    cluster_color = peak_colors[color_idx]
    _ms = peak_marker_markersize(cluster_marker)
    # Strip vertically centered on ``PANEL_A_ROW_Y`` so marker/text match letter A.
    title_ax = fig.add_axes(
        (0.055, PANEL_A_ROW_Y - 0.5 * TITLE_STRIP_H, 0.38, TITLE_STRIP_H)
    )
    title_ax.set_axis_off()
    title_ax.plot(
        0.06,
        0.5,
        marker=cluster_marker,
        markersize=_ms,
        markerfacecolor=cluster_color,
        markeredgecolor="black",
        markeredgewidth=0.4,
        linestyle="none",
        transform=title_ax.transAxes,
        clip_on=False,
        zorder=3,
    )
    title_ax.text(
        0.22,
        0.5,
        f"Peak {peak}",
        ha="left",
        va="center",
        transform=title_ax.transAxes,
        fontsize=FONT_TITLE,
        fontweight="normal",
        color="#222222",
        clip_on=False,
    )

    panel_letter(fig, "B", 0.02, 0.51, va="top")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model simplification figure")
    parser.add_argument(
        "--peak",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Peak index 1-3. Default: 1.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Output path. If omitted, show the figure with the current matplotlib backend.",
    )
    parser.add_argument("--font-size", type=int, default=FONT_LABEL)
    parser.add_argument("--dpi", type=int, default=DPI)
    args = parser.parse_args()

    fig = build_figure(args.peak, font_size=args.font_size, dpi=args.dpi)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        savefig_both(fig, args.save)
        print(f"Saved to {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
