#!/usr/bin/env python3
"""Generate the fit-to-experiment figure panels as standalone files.

This script intentionally reuses the existing visualisation functions instead
of reimplementing their plots:

* panel A: ``analyze_aggregated_results.plot_histogram_comparison``
* panel B: ``vis_posterior_overview.plot_pca_with_peaks``
* panel C: direct bar panels using the same cluster markers/colors

Outputs are written to ``results/figures/fit_exp/`` by default.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inverse_design.analyze.analyze_aggregated_results import (  # noqa: E402
    FEASIBLE_TARGET_METRICS,
    metric_to_colors,
    plot_histogram_comparison,
)
from inverse_design.analyze.analyze_single_folder import analyze_single_folder  # noqa: E402
from inverse_design.config.parameter_config import PARAM_RANGES  # noqa: E402
from inverse_design.io.cluster_profiles import (  # noqa: E402
    load_cluster_profiles,
    load_targets,
    normalize_profiles,
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
from inverse_design.plotting.colormap import DEFAULT_METRIC_COLORS  # noqa: E402
from inverse_design.vis.utils import peak_colors, peak_markers  # noqa: E402
from inverse_design.vis.vis_posterior_overview import plot_pca_with_peaks  # noqa: E402

ARCADE_OUTPUT_DIR = REPO_ROOT.parent / "ARCADE_OUTPUT"
DEFAULT_BASE_DIR = ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N512_combined_grid_breast"
DEFAULT_PCA_BASE_DIR = ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2"
DEFAULT_CLUSTER_DIR = REPO_ROOT / "CLUSTER"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "fit_exp"
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "out" / "pca_cluster_arcade_inputs_breast_only_mean_2" / "summary.json"
)

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TOP_PAIR_PARAMS = 8
_CLUSTER_COLOR_OFFSET = 2

PANEL_A_METRICS = [
    "doub_time",
    "symmetry",
    "colony_growth",
    "doub_time_std",
    "symmetry_std",
]
PANEL_A_LABELS = {
    "doub_time": "Doubling time (hr)",
    "symmetry": "Symmetry (-)",
    "colony_growth": "Colony growth ($\\mu$m/day)",
    "doub_time_std": "Doubling time SD (hr)",
    "symmetry_std": "Symmetry SD (-)",
}
PANEL_C_METRICS = [
    "doub_time",
    "symmetry",
    "colony_growth",
    "doub_time_std",
    "symmetry_std",
]
PANEL_C_TITLES = {
    "doub_time": "Doubling time",
    "symmetry": "Symmetry",
    "colony_growth": "Colony growth\nrate",
    "doub_time_std": "Doubling time\nSD",
    "symmetry_std": "Symmetry\nSD",
}
CLUSTER_CASES = [
    ("mean", "Mean"),
    ("mode", "Mode"),
    ("cluster_1", "Peak 1"),
    ("cluster_2", "Peak 2"),
    ("cluster_3", "Peak 3"),
    ("cluster_4", "Peak 4"),
]
DROP_PARAM_COLUMNS = [
    "input_folder",
    "X_SPACING",
    "Y_SPACING",
    "DISTANCE_TO_CENTER",
]
TIMESTAMP_RE = re.compile(r"_(\d{6})\.CELLS\.json$")
DEFAULT_FIT_EXP_FONT_SIZE = FONT_LABEL + 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate standalone fit-to-experiment panels A, B, and C."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="ABC run directory used for panel A histograms.",
    )
    parser.add_argument(
        "--pca-base-dir",
        type=Path,
        default=DEFAULT_PCA_BASE_DIR,
        help="ABC run directory used for panel B PCA.",
    )
    parser.add_argument(
        "--cluster-dir",
        type=Path,
        default=DEFAULT_CLUSTER_DIR,
        help="Directory containing mean/mode/cluster_* ARCADE outputs for panel C.",
    )
    parser.add_argument("--prior-iteration", type=int, default=0)
    parser.add_argument("--posterior-iteration", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--panel-a-metrics",
        nargs="+",
        default=None,
        metavar="METRIC",
        help=(
            "Column order for panel A histograms (default: built-in fit_exp order). "
            "Example: doub_time doub_time_std symmetry symmetry_std colony_growth"
        ),
    )
    parser.add_argument(
        "--panel-a-only",
        action="store_true",
        help="Only generate panel A outputs (skip B, C, and combined figure).",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only generate fit_exp_combined outputs (skip standalone A, B, and C files).",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Path to summary.json for panel D equifinality profiles.",
    )
    parser.add_argument("--hist-bins", type=int, default=15)
    parser.add_argument("--hist-y-max", type=float, default=100)
    parser.add_argument(
        "--font-size",
        type=int,
        default=DEFAULT_FIT_EXP_FONT_SIZE,
        help="Legacy override; default follows shared figure style.",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--skip-cluster-cache",
        action="store_true",
        help="Recalculate panel C per-seed CSVs even when cached files exist.",
    )
    return parser.parse_args(argv)


def _available_panel_a_metrics(targets: dict[str, float], metric_order: list[str]) -> list[str]:
    return [metric for metric in metric_order if metric in targets]


def _sync_axis_text_sizes(ax, font_size: int) -> None:
    """Apply one visible text scale to axes created by different helpers."""
    tick_font_size = max(FONT_BASE + 1, font_size - 1)
    ax.xaxis.label.set_size(font_size)
    ax.yaxis.label.set_size(font_size)
    ax.title.set_size(font_size)
    ax.tick_params(axis="both", which="major", labelsize=tick_font_size, width=0.5, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _hist_y_ticks(hist_y_max: float) -> np.ndarray:
    if hist_y_max == 100:
        return np.array([0, 25, 50, 75, 100])
    if hist_y_max == 250:
        return np.array([0, 50, 100, 150, 200, 250])
    return np.linspace(0, hist_y_max, 4)


def _plot_panel_a_combined(args, targets: dict[str, float]) -> Path:
    """Create the five-metric histogram panel using the shared helper."""
    metrics = _available_panel_a_metrics(targets, args.panel_a_metrics)
    prior_df = pd.read_csv(args.base_dir / f"iter_{args.prior_iteration}" / "final_metrics.csv")
    posterior_df = pd.read_csv(
        args.base_dir / f"iter_{args.posterior_iteration}" / "final_metrics.csv"
    )
    y_lim = (0, args.hist_y_max)
    y_ticks = _hist_y_ticks(args.hist_y_max)

    fig, axes = plt.subplots(
        2,
        len(metrics),
        figsize=(2.0 * len(metrics), 1.9),
        sharey=True,
        dpi=args.dpi,
    )
    if len(metrics) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col_idx, metric in enumerate(metrics):
        plot_histogram_comparison(
            [prior_df, posterior_df],
            [f"iter_{args.prior_iteration}", f"iter_{args.posterior_iteration}"],
            metric,
            color=metric_to_colors.get(metric, "k"),
            target_metric=targets.get(metric, FEASIBLE_TARGET_METRICS.get(metric)),
            remove_outliers_flag=True,
            bins=args.hist_bins,
            y_lim=y_lim,
            y_ticks=y_ticks,
            xlabel=PANEL_A_LABELS.get(metric, metric),
            fig=fig,
            axes=[axes[0, col_idx], axes[1, col_idx]],
            manage_layout=False,
            return_fig=True,
        )
        for row_idx in range(2):
            _sync_axis_text_sizes(axes[row_idx, col_idx], args.font_size)
            axes[row_idx, col_idx].tick_params(axis="y", labelleft=col_idx == 0)

    axes[0, 0].text(
        0.03,
        0.84,
        "g0",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="center",
        fontsize=args.font_size,
        fontweight="bold",
    )
    axes[1, 0].text(
        0.03,
        0.84,
        "g4",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="center",
        fontsize=args.font_size,
        fontweight="bold",
    )
    fig.text(
        0.015,
        0.63,
        "Number of samples",
        rotation=90,
        ha="center",
        va="center",
        fontsize=args.font_size,
    )

    fig.set_size_inches(WIDTH_DOUBLE, 1.46, forward=True)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.96, bottom=0.30, wspace=0.08, hspace=0.12)
    save_path = args.output_dir / "A_histograms.png"
    savefig_both(fig, save_path)
    plt.close(fig)
    return save_path


def _plot_panel_a_individual(args, targets: dict[str, float]) -> list[Path]:
    """Also save the direct one-file-per-metric helper outputs for inspection."""
    metrics = _available_panel_a_metrics(targets, args.panel_a_metrics)
    posterior_metrics_files = [
        args.base_dir / f"iter_{iteration}" / "final_metrics.csv"
        for iteration in [args.prior_iteration, args.posterior_iteration]
    ]
    posterior_metrics_dfs = [pd.read_csv(path) for path in posterior_metrics_files]
    labels = [f"iter_{args.prior_iteration}", f"iter_{args.posterior_iteration}"]
    y_lim = (0, args.hist_y_max)
    y_ticks = _hist_y_ticks(args.hist_y_max)
    saved_paths = []

    for metric in metrics:
        save_path = args.output_dir / f"A_histogram_{metric}.png"
        fig, axes = plot_histogram_comparison(
            posterior_metrics_dfs,
            labels,
            metric,
            color=metric_to_colors.get(metric, "k"),
            target_metric=targets.get(metric, FEASIBLE_TARGET_METRICS.get(metric)),
            remove_outliers_flag=True,
            bins=args.hist_bins,
            y_lim=y_lim,
            y_ticks=y_ticks,
            xlabel=PANEL_A_LABELS.get(metric, metric),
            return_fig=True,
        )
        for ax in axes:
            _sync_axis_text_sizes(ax, args.font_size)
        savefig_both(fig, save_path)
        plt.close(fig)
        saved_paths.append(save_path)
    return saved_paths


def _load_posterior_parameters(args) -> pd.DataFrame:
    posterior_df = pd.read_csv(
        args.pca_base_dir / f"iter_{args.posterior_iteration}" / "all_param_df.csv"
    )
    posterior_df = posterior_df.drop(
        columns=[column for column in DROP_PARAM_COLUMNS if column in posterior_df.columns],
        errors="ignore",
    )
    return posterior_df.select_dtypes(include=[np.number]).dropna(axis=0, how="any")


def _plot_panel_b(args) -> Path:
    posterior_df = _load_posterior_parameters(args)
    save_path = args.output_dir / "B_pca_with_peaks.png"
    # Keep panel-B export compact so combined width matches panel A.
    fig, ax = plt.subplots(figsize=(1.9, 1.8), dpi=args.dpi)
    fig, ax = plot_pca_with_peaks(
        posterior_df,
        n_components=2,
        fig=fig,
        ax=ax,
        font_size=args.font_size,
        cluster_fill_alpha=0.0,
        point_size=7,
    )
    _zoom_axis_to_collections(ax)
    ax.set_box_aspect(1)
    _sync_axis_text_sizes(ax, args.font_size)
    savefig_both(fig, save_path)
    plt.close(fig)
    return save_path


def _zoom_axis_to_collections(ax, padding: float = 0.06) -> None:
    """Tighten axis limits around scatter collections with a small data padding."""
    offsets = []
    for collection in ax.collections:
        if hasattr(collection, "get_offsets"):
            xy = np.asarray(collection.get_offsets())
            if xy.size:
                offsets.append(xy)
    if not offsets:
        return
    points = np.vstack(offsets)
    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size == 0:
        return
    x_min, y_min = finite.min(axis=0)
    x_max, y_max = finite.max(axis=0)
    x_pad = max((x_max - x_min) * padding, 0.3)
    y_pad = max((y_max - y_min) * padding, 0.3)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)


def _cluster_timestamps(cluster_case_dir: Path) -> list[str]:
    timestamps = {
        match.group(1)
        for path in cluster_case_dir.glob("*.CELLS.json")
        if (match := TIMESTAMP_RE.search(path.name))
    }
    if not timestamps:
        raise FileNotFoundError(f"No ARCADE CELLS JSON files found in {cluster_case_dir}")
    return sorted(timestamps, key=int)


def _cluster_metrics_csvs(args) -> list[Path]:
    """Convert CLUSTER raw outputs into cached per-seed and aggregate CSVs."""
    cache_root = args.output_dir / "panel_c_metrics"
    csv_paths = []
    for case_dir_name, _ in CLUSTER_CASES:
        case_dir = args.cluster_dir / case_dir_name
        if not case_dir.exists():
            raise FileNotFoundError(f"Missing CLUSTER case directory: {case_dir}")

        save_dir = cache_root / case_dir_name
        save_path = save_dir / "final_metrics_seed.csv"
        aggregate_path = save_dir / "final_metrics.csv"
        if save_path.exists() and aggregate_path.exists() and not args.skip_cluster_cache:
            csv_paths.append(save_path)
            continue

        timestamps = _cluster_timestamps(case_dir)
        result = analyze_single_folder(case_dir, timestamps)
        save_dir.mkdir(parents=True, exist_ok=True)
        result["per_seed_df"].to_csv(save_path, index=False)
        pd.DataFrame([result["final_metrics"]]).to_csv(aggregate_path, index=False)
        csv_paths.append(save_path)

    return csv_paths


def _plot_panel_c(args, targets: dict[str, float], csv_paths: list[Path]) -> Path:
    target_df = pd.DataFrame({key: [value] for key, value in targets.items()})
    scenario_names = ["EXP", *[label for _, label in CLUSTER_CASES]]
    save_path = args.output_dir / "C_metric_comparison.png"
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 1.58), dpi=args.dpi)
    panel_grid = gridspec.GridSpec(1, len(PANEL_C_METRICS), figure=fig, wspace=0.32)
    _draw_panel_c(fig, panel_grid[0, :], target_df, csv_paths, scenario_names, args.font_size)
    savefig_both(fig, save_path)
    plt.close(fig)
    return save_path


def _metric_case_summary(
    csv_file: Path, metric: str
) -> tuple[float, tuple[float, float], np.ndarray | None]:
    """Return center, asymmetric y-error, and optional per-seed raw values."""
    if metric.endswith("_std"):
        df = pd.read_csv(Path(csv_file).with_name("final_metrics.csv"))
        if metric not in df:
            return np.nan, (0.0, 0.0), None
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        center = float(values.iloc[0]) if len(values) else np.nan
        return center, (0.0, 0.0), None

    df = pd.read_csv(csv_file)
    if metric not in df:
        return np.nan, (0.0, 0.0), None
    values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return np.nan, (0.0, 0.0), None
    q1, median, q3 = values.quantile([0.25, 0.5, 0.75])
    return float(median), (float(median - q1), float(q3 - median)), values.to_numpy()


def _draw_metric_bar(
    ax,
    metric: str,
    target_df: pd.DataFrame,
    csv_paths: list[Path],
    scenario_names: list[str],
    font_size: int,
) -> None:
    """Draw one metric bar panel, using aggregate-only values for std metrics."""
    is_std_metric = metric.endswith("_std")
    centers = [float(target_df[metric].iloc[0])]
    yerrs = [(0.0, 0.0)]
    raw_values: list[np.ndarray | None] = [None]

    if not is_std_metric and f"{metric}_std" in target_df:
        exp_std = float(target_df[f"{metric}_std"].iloc[0])
        yerrs[0] = (exp_std, exp_std)

    for csv_path in csv_paths:
        center, yerr, raw = _metric_case_summary(csv_path, metric)
        centers.append(center)
        yerrs.append(yerr)
        raw_values.append(raw)

    x_positions = np.arange(len(scenario_names))
    marker_to_size = {"o": 30, "s": 27, "^": 37, "D": 30, "*": 60, "p": 42}

    for idx, (x_pos, center, yerr) in enumerate(zip(x_positions, centers, yerrs)):
        if np.isnan(center):
            continue
        lower, upper = yerr
        show_errorbar = lower > 0 or upper > 0
        bar_color = DEFAULT_METRIC_COLORS.get(metric, "#cccccc") if idx == 0 else "white"
        ax.bar(
            x_pos,
            center,
            yerr=[[lower], [upper]] if show_errorbar else None,
            capsize=3 if show_errorbar else 0,
            alpha=0.75 if idx == 0 else 1.0,
            color=bar_color,
            edgecolor="black",
            linewidth=0.8,
            zorder=2,
        )
        if raw_values[idx] is not None:
            jitter = np.random.default_rng(idx).normal(0, 0.045, len(raw_values[idx]))
            ax.scatter(
                x_pos + jitter,
                raw_values[idx],
                color="black",
                alpha=0.55,
                s=9,
                linewidth=0,
                zorder=3,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["EXP"] + [""] * (len(scenario_names) - 1), rotation=45, ha="center")
    for idx, x_pos in enumerate(x_positions[1:], start=1):
        marker = peak_markers[(idx - 1) % len(peak_markers)]
        marker_kwargs = dict(
            marker=marker,
            s=marker_to_size.get(marker, 62),
            color=peak_colors[(idx - 1) % len(peak_colors)],
            edgecolor="black",
            linewidth=0.9,
            clip_on=False,
            zorder=10,
        )
        if idx < 3:
            marker_kwargs["facecolor"] = "none"
        ax.scatter(x_pos, -0.1, transform=ax.get_xaxis_transform(), **marker_kwargs)

    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25, axis="y", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="x", pad=1)
    _sync_axis_text_sizes(ax, font_size)


def _draw_panel_c(
    fig,
    subplot_spec,
    target_df: pd.DataFrame,
    csv_paths: list[Path],
    scenario_names: list[str],
    font_size: int,
) -> list:
    panel_grid = gridspec.GridSpecFromSubplotSpec(
        1,
        len(PANEL_C_METRICS),
        subplot_spec=subplot_spec,
        wspace=0.40,
    )
    axes = []
    for col_idx, metric in enumerate(PANEL_C_METRICS):
        ax = fig.add_subplot(panel_grid[0, col_idx])
        _draw_metric_bar(ax, metric, target_df, csv_paths, scenario_names, font_size)
        axes.append(ax)
    return axes


def _shape_legend_handles():
    handles = []
    for idx, (_, label) in enumerate(CLUSTER_CASES):
        marker = peak_markers[idx % len(peak_markers)]
        color = peak_colors[idx % len(peak_colors)]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="None",
                label=label,
                markerfacecolor="none" if idx < 2 else color,
                markeredgecolor="black",
                markeredgewidth=1.2,
                markersize={"o": 5.5, "s": 5.2, "^": 6.1, "D": 5.5, "*": 7.7, "p": 6.5}.get(
                    marker, 5.5
                ),
                color=color,
            )
        )
    return handles


def _draw_panel_label(
    fig, subplot_spec, label: str, *, x_offset: float = 0.0, y_offset: float = 0.012
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


def _draw_panel_a(fig, subplot_spec, args, targets: dict[str, float]) -> np.ndarray:
    metrics = _available_panel_a_metrics(targets, args.panel_a_metrics)
    prior_df = pd.read_csv(args.base_dir / f"iter_{args.prior_iteration}" / "final_metrics.csv")
    posterior_df = pd.read_csv(
        args.base_dir / f"iter_{args.posterior_iteration}" / "final_metrics.csv"
    )
    y_lim = (0, args.hist_y_max)
    y_ticks = _hist_y_ticks(args.hist_y_max)
    panel_grid = gridspec.GridSpecFromSubplotSpec(
        2,
        len(metrics),
        subplot_spec=subplot_spec,
        hspace=0.12,
        wspace=0.12,
    )
    axes = np.empty((2, len(metrics)), dtype=object)

    for col_idx, metric in enumerate(metrics):
        axes[0, col_idx] = fig.add_subplot(panel_grid[0, col_idx])
        axes[1, col_idx] = fig.add_subplot(panel_grid[1, col_idx], sharey=axes[0, col_idx])
        plot_histogram_comparison(
            [prior_df, posterior_df],
            [f"iter_{args.prior_iteration}", f"iter_{args.posterior_iteration}"],
            metric,
            color=metric_to_colors.get(metric, "k"),
            target_metric=targets.get(metric, FEASIBLE_TARGET_METRICS.get(metric)),
            remove_outliers_flag=True,
            bins=args.hist_bins,
            y_lim=y_lim,
            y_ticks=y_ticks,
            xlabel=PANEL_A_LABELS.get(metric, metric),
            fig=fig,
            axes=[axes[0, col_idx], axes[1, col_idx]],
            manage_layout=False,
            return_fig=True,
        )
        _sync_axis_text_sizes(axes[0, col_idx], args.font_size)
        _sync_axis_text_sizes(axes[1, col_idx], args.font_size)
        axes[0, col_idx].tick_params(axis="y", labelleft=col_idx == 0)
        axes[1, col_idx].tick_params(axis="y", labelleft=col_idx == 0)

    axes[0, 0].text(
        0.03,
        0.84,
        "g0",
        transform=axes[0, 0].transAxes,
        ha="left",
        va="center",
        fontsize=args.font_size,
        fontweight="bold",
    )
    axes[1, 0].text(
        0.03,
        0.84,
        "g4",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="center",
        fontsize=args.font_size,
        fontweight="bold",
    )
    bbox = subplot_spec.get_position(fig)
    fig.text(
        bbox.x0 - 0.052,
        (bbox.y0 + bbox.y1) / 2,
        "Number of samples",
        rotation=90,
        ha="center",
        va="center",
        fontsize=args.font_size,
    )
    return axes


def _draw_panel_b(fig, subplot_spec, args):
    posterior_df = _load_posterior_parameters(args)
    ax = fig.add_subplot(subplot_spec)
    plot_pca_with_peaks(
        posterior_df,
        n_components=2,
        fig=fig,
        ax=ax,
        font_size=args.font_size,
        cluster_fill_alpha=0.0,
        point_size=7,
    )
    _zoom_axis_to_collections(ax)
    ax.set_box_aspect(1)
    ax.set_anchor("W")
    _sync_axis_text_sizes(ax, args.font_size)
    return ax


def _pair_top_params(
    norm_profiles: dict[int, dict[str, float]],
    pair: tuple[int, int],
    n: int = TOP_PAIR_PARAMS,
) -> list[str]:
    idx_a, idx_b = pair
    common = [p for p in norm_profiles[idx_a] if p in norm_profiles[idx_b]]
    return sorted(common, key=lambda p: norm_profiles[idx_a][p], reverse=True)[:n]


def _param_display_name(name: str) -> str:
    display_name = name.replace("_MU", "").replace("_SIGMA", " σ").replace("_", " ").lower()
    return display_name[:1].upper() + display_name[1:]


def _draw_pair_panel(
    ax,
    pair: tuple[int, int],
    top_params: list[str],
    norm_profiles: dict[int, dict[str, float]],
    font_size: int,
    show_y_axis: bool = True,
) -> None:
    idx_a, idx_b = pair
    n_params = len(top_params)
    tick_font_size = max(FONT_BASE + 1, font_size - 1)
    color_a = peak_colors[(idx_a + _CLUSTER_COLOR_OFFSET) % len(peak_colors)]
    color_b = peak_colors[(idx_b + _CLUSTER_COLOR_OFFSET) % len(peak_colors)]
    marker_a = peak_markers[(idx_a + _CLUSTER_COLOR_OFFSET) % len(peak_markers)]
    marker_b = peak_markers[(idx_b + _CLUSTER_COLOR_OFFSET) % len(peak_markers)]

    ax.set_xlim(-0.5, n_params - 0.5)
    ax.set_ylim(-0.08, 1.2)
    ax.tick_params(axis="x", labelsize=tick_font_size, width=0.5, rotation=45)
    ax.tick_params(axis="y", labelsize=tick_font_size)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_yticks([0.0, 0.5, 1.0])
    if show_y_axis:
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=tick_font_size)
    else:
        ax.set_yticklabels([])

    ax.set_xticks(range(n_params))
    ax.set_xticklabels(
        [_param_display_name(p) for p in top_params], fontsize=tick_font_size, ha="right"
    )

    for col_idx, param in enumerate(top_params):
        x = col_idx
        y_a = norm_profiles[idx_a][param]
        y_b = norm_profiles[idx_b][param]
        _d_marker_size = {"o": 30, "s": 27, "^": 37, "D": 30, "*": 60, "p": 42}
        ax.axvline(x, color="#e8e8e8", lw=0.5, zorder=0)
        ax.plot([x, x], [y_a, y_b], color="#aaaaaa", lw=0.8, zorder=1)
        ax.scatter(
            x,
            y_a,
            marker=marker_a,
            color=color_a,
            s=_d_marker_size.get(marker_a, 30),
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax.scatter(
            x,
            y_b,
            marker=marker_b,
            color=color_b,
            s=_d_marker_size.get(marker_b, 30),
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.grid(True, axis="y", alpha=0.15, linewidth=0.4, zorder=0)


def _draw_panel_d(fig, subplot_spec, args) -> None:
    profiles = load_cluster_profiles(args.summary_json)
    norm = normalize_profiles(profiles, PARAM_RANGES)
    bbox = subplot_spec.get_position(fig)
    pair_grid = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=subplot_spec,
        hspace=2.35,
        wspace=0.18,
    )
    for pair_idx, pair in enumerate(PAIRS):
        row = pair_idx // 3
        col = pair_idx % 3
        top_params = _pair_top_params(norm, pair, n=TOP_PAIR_PARAMS)
        ax = fig.add_subplot(pair_grid[row, col])
        _draw_pair_panel(ax, pair, top_params, norm, args.font_size, show_y_axis=(col == 0))
    fig.text(
        bbox.x0 - 0.07,
        bbox.y1 - 0.05,
        "Normalized (prior)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=args.font_size,
    )
    fig.text(
        bbox.x0 - 0.07,
        bbox.y0 + 0.02,
        "Normalized (prior)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=args.font_size,
    )


def _plot_combined_figure(args, targets: dict[str, float], csv_paths: list[Path]) -> Path:
    """Redraw all panels into one canvas so raster panel scaling cannot distort text."""
    fig = plt.figure(figsize=(6.95, 5.92), dpi=args.dpi)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.06)
    outer = fig.add_gridspec(
        3,
        2,
        height_ratios=(0.55, 0.52, 0.85),
        width_ratios=(0.8, 5.2),
        hspace=0.38,
        wspace=0.04,
    )

    _draw_panel_a(fig, outer[0, :], args, targets)
    ax_b = _draw_panel_b(fig, outer[1, 0], args)
    _b_pos = outer[1, 0].get_position(fig)
    ax_b.set_position([_b_pos.x0 - 0.025, _b_pos.y0, _b_pos.width, _b_pos.height])
    target_df = pd.DataFrame({key: [value] for key, value in targets.items()})
    scenario_names = ["EXP", *[label for _, label in CLUSTER_CASES]]
    _draw_panel_c(fig, outer[1, 1], target_df, csv_paths, scenario_names, args.font_size)
    _draw_panel_d(fig, outer[2, :], args)

    _draw_panel_label(fig, outer[0, :], "A", x_offset=-0.05)
    _draw_panel_label(fig, outer[1, 0], "B", x_offset=-0.05, y_offset=0.005)
    _draw_panel_label(fig, outer[1, 1], "C", x_offset=-0.03, y_offset=0.005)
    _draw_panel_label(fig, outer[2, :], "D", x_offset=-0.05, y_offset=0.02)

    bbox_c = outer[1, 1].get_position(fig)
    fig.legend(
        handles=_shape_legend_handles(),
        loc="upper left",
        ncol=len(CLUSTER_CASES),
        frameon=False,
        fontsize=max(FONT_BASE + 1, args.font_size - 1),
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.0,
        labelspacing=0.2,
        bbox_to_anchor=(bbox_c.x0 + 0.03, bbox_c.y1 + 0.03),
    )
    save_path = args.output_dir / "fit_exp_combined.png"
    savefig_both(fig, save_path)
    plt.close(fig)
    return save_path


def main(argv=None) -> None:
    args = parse_args(argv)
    apply_style()
    if args.panel_a_metrics is None:
        args.panel_a_metrics = list(PANEL_A_METRICS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_targets(args.base_dir)

    if args.panel_a_only and args.combined_only:
        raise ValueError("--panel-a-only and --combined-only cannot be used together")

    saved = []
    if args.combined_only:
        print("Combining generated panels (A + B + C + D equifinality)...")
        csv_paths = _cluster_metrics_csvs(args)
        saved.append(_plot_combined_figure(args, targets, csv_paths))
        for path in saved:
            print(f"Saved {path}")
        return

    print("Generating panel A histograms...")
    panel_a_path = _plot_panel_a_combined(args, targets)
    saved.append(panel_a_path)
    saved.extend(_plot_panel_a_individual(args, targets))

    if args.panel_a_only:
        for path in saved:
            print(f"Saved {path}")
        return

    print("Generating panel B PCA overview...")
    panel_b_path = _plot_panel_b(args)
    saved.append(panel_b_path)

    print("Generating panel C metric comparison...")
    csv_paths = _cluster_metrics_csvs(args)
    panel_c_path = _plot_panel_c(args, targets, csv_paths)
    saved.append(panel_c_path)

    print("Combining generated panels (A + B + C + D equifinality)...")
    saved.append(_plot_combined_figure(args, targets, csv_paths))

    for path in saved:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
