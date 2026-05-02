#!/usr/bin/env python3
"""Create feasible metric range panels and combine them into one figure.

This script mirrors ``combine_fit_exp_figure.py``: each panel is
generated as a standalone figure first, then the final figure redraws the same
panels on one canvas. Panel A shows the three-metric ABC run. Panels B-D show
single-metric fits with the fitted metric on the x-axis and the two non-fit
metrics as separate y-axis rows.
"""

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
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inverse_design.analyze.analyze_aggregated_results import (  # noqa: E402
    FEASIBLE_TARGET_METRICS,
    metric_to_colors,
    plot_histogram_comparison,
)
from inverse_design.plotting.style import (  # noqa: E402
    DPI,
    FONT_BASE,
    FONT_LABEL,
    FONT_PANEL,
    WIDTH_DOUBLE,
    apply_style,
    savefig_both,
)
from inverse_design.vis.vis_simulation_outcome import (  # noqa: E402
    DEFAULT_FIGURE_DIR,
    DEFAULT_METRICS_RANGES,
    plot_feasible_metric_relationship_grid,
)
from inverse_design.utils.utils import remove_outliers  # noqa: E402

ARCADE_OUTPUT_DIR = REPO_ROOT.parent / "ARCADE_OUTPUT"
DEFAULT_BREAST_MEAN_DATA_DIR = ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N512_combined_grid_breast_only_mean"
DEFAULT_OUTPUT_DIR = DEFAULT_FIGURE_DIR / "feasible"

PANEL_A_FILENAME = "A_feasible_metric_histograms.png"
SINGLE_METRIC_PANEL_RUNS = [
    {
        "panel": "B",
        "fit_metric": "doub_time",
        "title": "Fit DT=32",
        "data_dir": ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N512_combined_grid_doub_32",
        "filename": "B_fit_doub_time_relationship.png",
    },
    {
        "panel": "C",
        "fit_metric": "symmetry",
        "title": "Fit S=0.75",
        "data_dir": ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N512_combined_grid_sym_075",
        "filename": "C_fit_symmetry_relationship.png",
    },
    {
        "panel": "D",
        "fit_metric": "act_ratio",
        "title": "Fit A=0.70",
        "data_dir": ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N512_combined_grid_act_07",
        "filename": "D_fit_act_ratio_relationship.png",
    },
]

DEFAULT_HISTOGRAM_DATA_DIRS = {
    "doub_time": DEFAULT_BREAST_MEAN_DATA_DIR,
    "symmetry": DEFAULT_BREAST_MEAN_DATA_DIR,
    "act_ratio": DEFAULT_BREAST_MEAN_DATA_DIR,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate and combine feasible metric histogram and relationship panels."
    )
    parser.add_argument(
        "--hist-iterations",
        type=int,
        nargs="+",
        default=[0, 4],
        help="Iteration numbers to compare in panel A.",
    )
    parser.add_argument(
        "--panel-b-iteration",
        "--single-metric-iteration",
        dest="single_metric_iteration",
        type=int,
        default=2,
        help="Iteration to use from each single-metric ABC run for panels B-D.",
    )
    parser.add_argument(
        "--hist-data-dir",
        action="append",
        default=[],
        metavar="METRIC=PATH",
        help=(
            "Override panel A data directory for one metric. Can be passed multiple times, "
            "for example --hist-data-dir doub_time=/path/to/run."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["doub_time", "symmetry", "act_ratio"],
        help="Exactly three metrics to show in panel A and compare across panels B-D.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the combined figure is saved.",
    )
    parser.add_argument(
        "--output-name",
        default="feasible_metrics_ranges_combined.png",
        help="Combined figure filename.",
    )
    parser.add_argument("--target-doub-time", type=float, default=32)
    parser.add_argument("--target-symmetry", type=float, default=0.75)
    parser.add_argument("--target-act-ratio", type=float, default=0.7)
    parser.add_argument("--hist-y-max", type=float, default=125)
    parser.add_argument("--hist-bins", type=int, default=10)
    parser.add_argument(
        "--matrix-wspace",
        "--relationship-wspace",
        dest="matrix_wspace",
        type=float,
        default=0.05,
        help="Inner spacing for the B-D relationship panel marginal grids.",
    )
    parser.add_argument(
        "--matrix-hspace",
        "--relationship-hspace",
        dest="matrix_hspace",
        type=float,
        default=0.02,
        help="Inner spacing for the B-D relationship panel marginal grids.",
    )
    parser.add_argument("--width", type=float, default=7.1, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=6.7, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--font-size",
        type=int,
        default=FONT_LABEL,
        help="Legacy override; default follows shared figure style.",
    )
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--relationship-x-pad-ratio",
        type=float,
        default=1.08,
        help="Padding multiplier for fitted-metric x ranges in panels B-D.",
    )
    parser.add_argument(
        "--center-relationship-x-on-target",
        action="store_true",
        dest="center_relationship_x_on_target",
        default=True,
        help="Center B-D x-axis ranges on target values. This is the default.",
    )
    parser.add_argument(
        "--no-center-relationship-x-on-target",
        action="store_false",
        dest="center_relationship_x_on_target",
        help="Use default metric ranges for B-D x-axes instead of centering on target values.",
    )
    parser.add_argument(
        "--no-remove-outliers",
        action="store_true",
        help="Disable IQR outlier removal for both panels.",
    )
    return parser.parse_args(argv)


def _target_metrics_from_args(args):
    return {
        "doub_time": args.target_doub_time,
        "symmetry": args.target_symmetry,
        "act_ratio": args.target_act_ratio,
    }


def _histogram_data_dirs(args):
    data_dirs = dict(DEFAULT_HISTOGRAM_DATA_DIRS)
    for item in args.hist_data_dir:
        if "=" not in item:
            raise ValueError("--hist-data-dir entries must use METRIC=PATH syntax.")
        metric, path = item.split("=", 1)
        data_dirs[metric] = Path(path)
    return data_dirs


def _validate_metrics(args) -> None:
    if len(args.metrics) != 3:
        raise ValueError("--metrics must contain exactly three metrics for panels B-D.")

    missing_fit_metrics = [
        panel["fit_metric"]
        for panel in SINGLE_METRIC_PANEL_RUNS
        if panel["fit_metric"] not in args.metrics
    ]
    if missing_fit_metrics:
        raise ValueError(
            "Panels B-D use the default single-metric runs, so --metrics must include "
            + ", ".join(missing_fit_metrics)
            + "."
        )


def _sync_axis_text_sizes(ax, font_size: int) -> None:
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.title.set_size(font_size)
    ax.tick_params(axis="both", which="major", labelsize=FONT_BASE, width=0.5, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _draw_panel_label(
    fig, subplot_spec, label: str, *, x_offset: float = -0.045, y_offset: float = 0.012
) -> None:
    bbox = subplot_spec.get_position(fig)
    fig.text(
        bbox.x0 + x_offset,
        bbox.y1 + y_offset,
        label,
        ha="left",
        va="bottom",
        fontsize=FONT_PANEL,
        fontweight="bold",
    )


def _draw_panel_a(fig, subplot_spec, args, target_metrics):
    hist_data_dirs = _histogram_data_dirs(args)
    hist_data = [
        {
            metric: pd.read_csv(hist_data_dirs[metric] / f"iter_{iteration}" / "final_metrics.csv")
            for metric in args.metrics
        }
        for iteration in args.hist_iterations
    ]
    labels = [f"iter_{iteration}" for iteration in args.hist_iterations]
    panel_grid = gridspec.GridSpecFromSubplotSpec(
        len(hist_data),
        len(args.metrics),
        subplot_spec=subplot_spec,
        wspace=0.11,
        hspace=0.10,
    )

    y_lim = (0, args.hist_y_max)
    y_ticks = np.arange(0, args.hist_y_max + 1, 25)
    panel_axes = np.empty((len(hist_data), len(args.metrics)), dtype=object)
    for col_idx, metric in enumerate(args.metrics):
        axes = []
        for row_idx in range(len(hist_data)):
            sharex = axes[0] if axes else None
            ax = fig.add_subplot(panel_grid[row_idx, col_idx], sharex=sharex)
            axes.append(ax)
            panel_axes[row_idx, col_idx] = ax

        plot_histogram_comparison(
            [iteration_data[metric] for iteration_data in hist_data],
            labels,
            metric,
            color=metric_to_colors.get(metric, "k"),
            target_metric=target_metrics.get(metric, FEASIBLE_TARGET_METRICS.get(metric)),
            remove_outliers_flag=not args.no_remove_outliers,
            iqr_multiplier=args.iqr_multiplier,
            bins=args.hist_bins,
            y_lim=y_lim,
            y_ticks=y_ticks,
            fig=fig,
            axes=axes,
            manage_layout=False,
            return_fig=True,
        )
        for ax in axes:
            _sync_axis_text_sizes(ax, args.font_size)
        if col_idx == 0:
            for row_idx, ax in enumerate(axes):
                ax.text(
                    0.03,
                    0.82,
                    labels[row_idx].replace("iter_", "g"),
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=args.font_size,
                    fontweight="bold",
                )
        else:
            for ax in axes:
                ax.tick_params(axis="y", labelleft=False)

    bbox = subplot_spec.get_position(fig)
    fig.text(
        bbox.x0 - 0.068,
        (bbox.y0 + bbox.y1) / 2,
        "Number of samples",
        rotation=90,
        ha="center",
        va="center",
        fontsize=args.font_size,
    )
    return panel_axes


def _save_panel_a(args, target_metrics) -> Path:
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 1.76), dpi=args.dpi)
    fig.subplots_adjust(left=0.090, right=0.995, top=0.94, bottom=0.30)
    grid = fig.add_gridspec(1, 1)
    _draw_panel_a(fig, grid[0], args, target_metrics)
    _draw_panel_label(fig, grid[0], "A", x_offset=-0.075, y_offset=0.005)
    save_path = args.output_dir / PANEL_A_FILENAME
    savefig_both(fig, save_path)
    plt.close(fig)
    return save_path


def _single_metric_csv(panel: dict, args) -> Path:
    return Path(panel["data_dir"]) / f"iter_{args.single_metric_iteration}" / "final_metrics.csv"


def _unfit_metrics(args, fit_metric: str) -> tuple[str, str]:
    metrics = [metric for metric in args.metrics if metric != fit_metric]
    if len(metrics) != 2:
        raise ValueError(
            f"Expected exactly two non-fit metrics for {fit_metric}; got {', '.join(metrics)}."
        )
    return metrics[0], metrics[1]


def _relationship_metrics_ranges(
    metrics_df: pd.DataFrame,
    plot_metrics: tuple[str, ...],
    fit_metric: str,
    target_metrics: dict[str, float],
    args,
) -> dict[str, tuple[float, float]]:
    metrics_ranges = dict(DEFAULT_METRICS_RANGES)
    if fit_metric not in target_metrics:
        return metrics_ranges

    clean_df = metrics_df[list(plot_metrics)].replace([np.inf, -np.inf], np.nan).dropna()
    if not args.no_remove_outliers and len(clean_df) > 0:
        clean_df, _ = remove_outliers(clean_df, args.iqr_multiplier)

    values = clean_df[fit_metric].dropna()
    if len(values) == 0:
        return metrics_ranges

    target = target_metrics[fit_metric]
    half_width = max(abs(values.min() - target), abs(values.max() - target))
    half_width *= args.relationship_x_pad_ratio
    if half_width <= 0:
        half_width = 1.0
    metrics_ranges[fit_metric] = (target - half_width, target + half_width)
    return metrics_ranges


def _draw_single_metric_panel(fig, subplot_spec, args, panel: dict, target_metrics):
    fit_metric = panel["fit_metric"]
    y_metrics = _unfit_metrics(args, fit_metric)
    metrics_df = pd.read_csv(_single_metric_csv(panel, args))
    plot_metrics = (fit_metric, *y_metrics)
    metrics_ranges = _relationship_metrics_ranges(
        metrics_df, plot_metrics, fit_metric, target_metrics, args
    )
    _, axes, outlier_indices = plot_feasible_metric_relationship_grid(
        metrics_df,
        x_metrics=(fit_metric,),
        y_metric_grid=((y_metrics[0],), (y_metrics[1],)),
        metrics_ranges=metrics_ranges,
        target_metrics=target_metrics,
        target_line_orientation="vertical",
        remove_outliers_flag=not args.no_remove_outliers,
        iqr_multiplier=args.iqr_multiplier,
        fig=fig,
        subplot_spec=subplot_spec,
        wspace=args.matrix_wspace,
        hspace=args.matrix_hspace,
        show_all_axis_labels=False,
        show_all_y_labels=True,
        show_y_marginals="all",
        show_x_marginals="first_row",
        center_x_on_target=args.center_relationship_x_on_target,
    )
    for row_axes in axes:
        for ax_dict in row_axes:
            for ax in ax_dict.values():
                _sync_axis_text_sizes(ax, args.font_size)
        row_axes[0]["main"].yaxis.set_label_coords(-0.25, 0.5)
    return axes, outlier_indices


def _save_single_metric_panel(args, panel: dict, target_metrics) -> tuple[Path, int]:
    fig = plt.figure(figsize=(2.35, 4.05), dpi=args.dpi)
    fig.subplots_adjust(left=0.24, right=0.95, top=0.91, bottom=0.12)
    grid = fig.add_gridspec(1, 1)
    _, outlier_indices = _draw_single_metric_panel(fig, grid[0], args, panel, target_metrics)
    _draw_panel_label(fig, grid[0], panel["panel"], x_offset=-0.18, y_offset=0.025)
    save_path = args.output_dir / panel["filename"]
    savefig_both(fig, save_path)
    plt.close(fig)
    return save_path, len(outlier_indices)


def _save_combined_figure(args, target_metrics) -> tuple[Path, dict[str, int]]:
    fig = plt.figure(figsize=(min(args.width, WIDTH_DOUBLE), args.height), dpi=args.dpi)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.945, bottom=0.085)
    outer = fig.add_gridspec(
        2,
        3,
        height_ratios=(0.46, 1.0),
        hspace=0.22,
        wspace=0.30,
    )

    _draw_panel_a(fig, outer[0, :], args, target_metrics)
    outlier_counts = {}
    for col_idx, panel in enumerate(SINGLE_METRIC_PANEL_RUNS):
        _, outlier_indices = _draw_single_metric_panel(
            fig, outer[1, col_idx], args, panel, target_metrics
        )
        outlier_counts[panel["panel"]] = len(outlier_indices)

    _draw_panel_label(fig, outer[0, :], "A", x_offset=-0.070, y_offset=0.010)
    for col_idx, panel in enumerate(SINGLE_METRIC_PANEL_RUNS):
        _draw_panel_label(fig, outer[1, col_idx], panel["panel"], x_offset=-0.070, y_offset=-0.015)

    save_path = args.output_dir / args.output_name
    savefig_both(fig, save_path)
    plt.close(fig)
    return save_path, outlier_counts


def main(argv=None):
    args = parse_args(argv)
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _validate_metrics(args)

    target_metrics = _target_metrics_from_args(args)

    saved_paths = []
    print("Generating panel A histograms...")
    saved_paths.append(_save_panel_a(args, target_metrics))

    print("Generating panels B-D single-metric comparisons...")
    for panel in SINGLE_METRIC_PANEL_RUNS:
        save_path, _ = _save_single_metric_panel(args, panel, target_metrics)
        saved_paths.append(save_path)

    print("Combining generated panels...")
    combined_path, combined_outliers = _save_combined_figure(args, target_metrics)
    saved_paths.append(combined_path)

    for panel_label, count in combined_outliers.items():
        print(f"Panel {panel_label}: removed {count} outliers")
    for path in saved_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
