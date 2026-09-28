#!/usr/bin/env python3
"""Simplification metrics figure: bar-chart grid across simplification levels."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TypeAlias

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.ticker import NullFormatter
from matplotlib.transforms import blended_transform_factory

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inverse_design.vis.utils import (  # noqa: E402
    peak_colors,
    peak_marker_markersize,
    peak_markers,
)
from inverse_design.plotting.style import (  # noqa: E402
    DPI,
    FONT_BASE,
    FONT_LABEL,
    FONT_PANEL,
    FONT_TITLE,
    WIDTH_DOUBLE,
    apply_style,
    savefig_both,
)
from inverse_design.analyze.errors import signed_percent_error  # noqa: E402

ARCADE_OUTPUT_DIR = REPO_ROOT.parent / "ARCADE_OUTPUT"
DEFAULT_CLUSTER_DIR = ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2"
DEFAULT_CLUSTER_METRICS_DIR = (
    REPO_ROOT / "out" / "pca_cluster_arcade_inputs_breast_only_mean_2" / "metrics"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "simplification"

EXP_TARGETS = {
    "doub_time": 45.5,
    "doub_time_std": 13.79,
    "symmetry": 0.806,
    "symmetry_std": 0.067,
    "colony_growth": 18.3,
}
_CLUSTER_COLOR_OFFSET = 2
TEXT = "#222222"
MUTED = "#666666"
LIGHT_EDGE = "#cfcfcf"
# transAxes x for panel letters A (MI) and B (metrics r-series); C uses 0.0.
PANEL_AB_LETTER_X = -0.1

# Panel B (r-series) vs C (t-series): title strip + subplot background (letters are black).
SECTION_KIND_STYLE: dict[str, dict[str, str]] = {
    "r": {
        "title_bg": "#e8eef8",
        "axes_bg": "#f7f9fd",
    },
    "t": {
        "title_bg": "#f6ebe3",
        "axes_bg": "#fdf9f5",
    },
}

# Mean metrics (cols 1–3) share ±50% axis; std metrics (cols 4–5) share ±100% axis.
SIMP_METRICS = [
    "doub_time",
    "symmetry",
    "colony_growth",
    "doub_time_std",
    "symmetry_std",
]
# Bar centers = i + shift (markers use axes-x via blended transform, not data x).
BAR_X_SHIFT = 0.88
BAR_WIDTH = 0.68
# x-position in axes coords (left margin); y in data (0 = reference line).
CLUSTER_MARKER_AXES_X = -0.40
SIMP_METRIC_LABELS = {
    "doub_time": "DT",
    "doub_time_std": "DT SD",
    "symmetry": "Sym",
    "symmetry_std": "Sym SD",
    "colony_growth": "CG",
}
SIMP_LEVELS_R = [
    ("orig", None, None),
    ("r09", "linear", "0.9"),
    ("r08", "linear", "0.8"),
    ("r07", "linear", "0.7"),
]
SIMP_LEVELS_T = [
    ("orig", None, None),
    ("t10", "dendrogram", "1.0"),
    ("t125", "dendrogram", "1.25"),
    ("t15", "dendrogram", "1.5"),
]
LEVEL_LABELS = {
    "orig": "Original",
    "r09": "r=0.9",
    "r08": "r=0.8",
    "r07": "r=0.7",
    "t10": "t=1.0",
    "t125": "t=1.25",
    "t15": "t=1.5",
}
# Std metrics use small targets, so % error can exceed ±50; use a wider axis than means.
_STD_METRICS = frozenset({"doub_time_std", "symmetry_std"})
# Nested column groups (means vs stds): tight within each group; modest gap between groups.
WSPACE_METRIC_TRIPLET = 0.10
WSPACE_METRIC_PAIR = 0.10
WSPACE_MEANS_STDS_GAP = 0.22
PeakMetricStats: TypeAlias = tuple[float, float, float] | None
PeakData: TypeAlias = dict[str, PeakMetricStats] | None


def _simp_run_dir(series: str, threshold: str, peak_k: int) -> Path | None:
    run_name = f"ABC_SMC_RF_N512_combined_grid_{series}_{threshold}_p{peak_k}_mean_only"
    run_dir = ARCADE_OUTPUT_DIR / run_name / run_name / "iter_4"
    return run_dir if run_dir.exists() else None


def _add_std_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-parameter-set within-seed std for base metrics if not present."""
    df = df.copy()
    for base in ("doub_time", "symmetry"):
        std_col = f"{base}_std"
        if std_col not in df.columns and base in df.columns:
            if "input_folder" in df.columns:
                df[std_col] = df.groupby("input_folder")[base].transform("std")
            else:
                df[std_col] = df[base].std()
    return df


def _peak_stats_from_csv(csv_path: Path) -> dict[str, PeakMetricStats]:
    """Load (median, iqr_lo, iqr_hi) per simplification metric."""
    df = _add_std_columns(pd.read_csv(csv_path))
    result: dict[str, PeakMetricStats] = {}
    for metric in SIMP_METRICS:
        if metric not in df.columns:
            result[metric] = None
            continue
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        values = _drop_iqr_outliers(values)
        if len(values) == 0:
            result[metric] = None
            continue
        q25 = float(values.quantile(0.25))
        median = float(values.median())
        q75 = float(values.quantile(0.75))
        result[metric] = (median, median - q25, q75 - median)
    return result


def _drop_iqr_outliers(values: pd.Series) -> pd.Series:
    """Drop finite values outside the 1.5 IQR fence before summary statistics."""
    if len(values) < 4:
        return values
    q25 = values.quantile(0.25)
    q75 = values.quantile(0.75)
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 0:
        return values
    lo = q25 - 1.5 * iqr
    hi = q75 + 1.5 * iqr
    return values[(values >= lo) & (values <= hi)]


def _load_peak_stats(run_dir: Path | None) -> PeakData:
    """None return means the whole run is missing."""
    if run_dir is None:
        return None
    csv_path = run_dir / "final_metrics_seed.csv"
    if not csv_path.exists():
        return None
    return _peak_stats_from_csv(csv_path)


def _simp_run_key(run_dir: Path) -> tuple[str, str, int] | None:
    run_name = run_dir.parent.name
    prefix = "ABC_SMC_RF_N512_combined_grid_"
    suffix = "_mean_only"
    if not run_name.startswith(prefix) or not run_name.endswith(suffix):
        return None
    try:
        series, threshold, peak_label = (
            run_name.removeprefix(prefix).removesuffix(suffix).rsplit("_", 2)
        )
    except ValueError:
        return None
    if not peak_label.startswith("p"):
        return None
    try:
        return series, threshold, int(peak_label[1:])
    except ValueError:
        return None


def _load_orig_peak_stats(cluster_dir: Path, peak_k: int) -> PeakData:
    csv_path = _orig_peak_csv_path(cluster_dir, peak_k)
    return _peak_stats_from_csv(csv_path) if csv_path.exists() else None


def _orig_peak_csv_path(cluster_dir: Path, peak_k: int) -> Path:
    cluster_name = f"cluster_{peak_k}"
    candidates = [
        cluster_dir / cluster_name / "final_metrics_seed.csv",
        cluster_dir / "metrics" / cluster_name / "final_metrics_seed.csv",
    ]
    if cluster_dir == DEFAULT_CLUSTER_DIR:
        candidates.append(DEFAULT_CLUSTER_METRICS_DIR / cluster_name / "final_metrics_seed.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _percent_axis_for_metric(metric: str) -> tuple[float, float, list[float]]:
    """Y limits and ticks for signed percent error; wider range for *_std metrics."""
    if metric in _STD_METRICS:
        return -100.0, 100.0, [-100, -50, 0, 50, 100]
    return -50.0, 50.0, [-50, -25, 0, 25, 50]


def _format_tick(value: float) -> str:
    if np.isclose(value, round(value)):
        return f"{value:.0f}"
    return f"{value:.1f}"


def _signed_percent_error(value: float, target: float) -> float:
    return signed_percent_error(value, target)


def _error_stats(metric: str, stats: tuple[float, float, float]) -> tuple[float, float, float]:
    """Return (signed_center, lo_spread, hi_spread) where spreads are IQR half-widths."""
    median, lo_err, hi_err = stats
    q25 = median - lo_err
    q75 = median + hi_err
    target = EXP_TARGETS[metric]
    center = _signed_percent_error(median, target)
    lo_spread = center - _signed_percent_error(q25, target)
    hi_spread = _signed_percent_error(q75, target) - center
    return center, max(0.0, lo_spread), max(0.0, hi_spread)


def _metric_error_ranges(
    cluster_dir: Path, pad_ratio: float = 1.02
) -> dict[str, tuple[float, float]]:
    """Data-driven y-ranges for percent-error metric panels."""
    values_by_metric: dict[str, list[float]] = {metric: [] for metric in SIMP_METRICS}
    levels = [SIMP_LEVELS_R[0], *SIMP_LEVELS_R[1:], *SIMP_LEVELS_T[1:]]
    for level_key, series, threshold in levels:
        for peak_k in range(1, 5):
            if level_key == "orig":
                peak_data = _load_orig_peak_stats(cluster_dir, peak_k)
            else:
                if series is None or threshold is None:
                    continue
                peak_data = _load_peak_stats(_simp_run_dir(series, threshold, peak_k))
            if peak_data is None:
                continue
            for metric in SIMP_METRICS:
                stats = peak_data.get(metric)
                if stats is None:
                    continue
                center, lo_err, hi_err = _error_stats(metric, stats)
                values_by_metric[metric].extend([center - lo_err, center, center + hi_err])

    ranges: dict[str, tuple[float, float]] = {}
    for metric, values in values_by_metric.items():
        finite = [float(value) for value in values if np.isfinite(value)]
        if not finite:
            ranges[metric] = (0.0, 1.0)
            continue
        lo = max(0.0, min(finite) / pad_ratio)
        hi = max(finite) * pad_ratio
        if hi <= lo:
            hi = lo + 1.0
        ranges[metric] = (lo, hi)
    return ranges


def _draw_bar_cell(
    ax,
    metric: str,
    level_data_list: list[PeakData],
    level_label_list: list[str],
    cluster_k: int,
    font_size: int,
    *,
    show_ytick_labels: bool,
    show_xtick_labels: bool,
) -> None:
    """Mini bar chart: x=simplification models, y=signed error (%). Orig bar colored."""
    n_levels = len(level_data_list)
    lo, hi = _percent_axis_for_metric(metric)[:2]
    span = hi - lo
    x_centers = [float(i) + BAR_X_SHIFT for i in range(n_levels)]
    half_w = BAR_WIDTH / 2.0
    x_right = (n_levels - 1) + BAR_X_SHIFT + half_w + 0.12

    first_bar_left = BAR_X_SHIFT - half_w
    ax.set_xlim(first_bar_left - 0.1, x_right)
    ax.set_ylim(lo, hi)
    ax.axhline(0, color="#000000", lw=0.8, zorder=1)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    tick_values = _percent_axis_for_metric(metric)[2]
    ax.set_yticks(tick_values)
    if show_ytick_labels:
        ax.set_yticklabels([str(v) for v in tick_values], fontsize=FONT_BASE)
    else:
        ax.yaxis.set_major_formatter(NullFormatter())

    ax.set_xticks(x_centers)
    ax.set_xticklabels(level_label_list, fontsize=FONT_BASE, rotation=45, ha="right")
    ax.tick_params(axis="x", labelbottom=show_xtick_labels)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["left"].set_color("#000000")
    ax.spines["left"].set_visible(True)
    _ypad = 2.0 if (show_ytick_labels and metric == "doub_time_std") else 3.5
    ax.tick_params(
        axis="y",
        length=2,
        width=0.7,
        color="#000000",
        labelcolor="#000000",
        labelsize=FONT_BASE,
        left=True,
        labelleft=show_ytick_labels,
        pad=_ypad,
    )

    color_idx = (cluster_k - 1 + _CLUSTER_COLOR_OFFSET) % len(peak_colors)
    cluster_color = peak_colors[color_idx]

    for i, peak_data in enumerate(level_data_list):
        is_orig = i == 0
        bar_color = cluster_color if is_orig else "#ffffff"
        x = x_centers[i]

        if peak_data is None or peak_data.get(metric) is None:
            ax.bar(
                x,
                span,
                bottom=lo,
                width=BAR_WIDTH * 0.95,
                fill=False,
                edgecolor="#aaaaaa",
                linewidth=0.5,
                linestyle="--",
                zorder=2,
            )
            ax.text(x, 0, "N/A", ha="center", va="center", fontsize=FONT_BASE, color="#aaaaaa")
        else:
            center, lo_spread, hi_spread = _error_stats(metric, peak_data[metric])
            value = float(np.clip(center, lo, hi))
            ax.bar(
                x,
                value,
                bottom=0,
                width=BAR_WIDTH,
                color=bar_color,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.82 if is_orig else 1.0,
                zorder=2,
            )
            lower_err = min(lo_spread, value - lo)
            upper_err = min(hi_spread, hi - value)
            if lower_err > 0 or upper_err > 0:
                ax.errorbar(
                    x,
                    value,
                    yerr=[[lower_err], [upper_err]],
                    color="#444444",
                    linewidth=1.2,
                    capsize=3,
                    capthick=1.0,
                    zorder=3,
                )


def _plot_section(
    fig,
    subplot_spec,
    section_levels: list[tuple[str, str | None, str | None]],
    section_title: str,
    args: argparse.Namespace,
    cluster_dir: Path,
    show_row_labels: bool = True,
    *,
    panel_label: str | None = None,
    section_kind: str = "r",
    panel_letter_x: float = 0.0,
    show_section_title: bool = True,
    show_row_markers: bool = True,
    title_strip_ratio: float = 0.075,
):
    peaks_to_plot = tuple(getattr(args, "peaks_to_plot", (1, 2, 3, 4)))
    n_levels = len(section_levels)
    n_clusters = len(peaks_to_plot)
    _style = SECTION_KIND_STYLE.get(section_kind, SECTION_KIND_STYLE["r"])
    outer = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=subplot_spec,
        height_ratios=[title_strip_ratio, 1.0],
        hspace=0.08,
    )
    title_ax = fig.add_subplot(outer[0])
    title_ax.axis("off")
    title_ax.set_facecolor(_style["title_bg"])
    # y=0.5 matches MI table panel letter (``combine_simplification_mi_table`` header).
    if panel_label:
        title_ax.text(
            panel_letter_x,
            0.5,
            panel_label,
            ha="left",
            va="center",
            fontsize=FONT_PANEL,
            fontweight="bold",
            color="#000000",
            transform=title_ax.transAxes,
        )
    if show_section_title:
        title_ax.text(
            0.5,
            0.5,
            section_title,
            ha="center",
            va="center",
            fontsize=FONT_TITLE,
            fontweight="normal",
            color=TEXT,
            transform=title_ax.transAxes,
        )

    # Two column groups: 3 means (DT, Sym, CG) + 2 stds (DT SD, Sym SD). width_ratios [3,2]
    # keeps each subplot column equal width; wspace differs within vs between groups.
    content = gridspec.GridSpecFromSubplotSpec(
        n_clusters,
        2,
        subplot_spec=outer[1],
        height_ratios=[1.0] * n_clusters,
        width_ratios=[3, 2],
        hspace=0.28,
        wspace=WSPACE_MEANS_STDS_GAP,
    )

    all_level_data: list[list[PeakData]] = []
    level_label_list: list[str] = []
    for level_key, series, threshold in section_levels:
        peak_stats = []
        for peak_k in peaks_to_plot:
            if level_key == "orig":
                stats = _load_orig_peak_stats(cluster_dir, peak_k)
            else:
                if series is None or threshold is None:
                    raise ValueError(f"Missing series or threshold for {level_key}")
                stats = _load_peak_stats(_simp_run_dir(series, threshold, peak_k))
            peak_stats.append(stats)
        all_level_data.append(peak_stats)
        level_label_list.append(LEVEL_LABELS[level_key])

    for cluster_idx in range(n_clusters):
        cluster_k = peaks_to_plot[cluster_idx]
        gs_means = gridspec.GridSpecFromSubplotSpec(
            1,
            3,
            subplot_spec=content[cluster_idx, 0],
            wspace=WSPACE_METRIC_TRIPLET,
        )
        gs_stds = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=content[cluster_idx, 1],
            wspace=WSPACE_METRIC_PAIR,
        )
        for col_idx in range(3):
            metric = SIMP_METRICS[col_idx]
            ax = fig.add_subplot(gs_means[0, col_idx])
            ax.set_facecolor(_style["axes_bg"])
            level_data_for_cluster = [
                all_level_data[level_idx][cluster_idx] for level_idx in range(n_levels)
            ]
            show_ytick_labels = col_idx in (0, 3)
            show_xtick_labels = cluster_idx == n_clusters - 1
            _draw_bar_cell(
                ax,
                metric,
                level_data_for_cluster,
                level_label_list,
                cluster_k,
                args.font_size,
                show_ytick_labels=show_ytick_labels,
                show_xtick_labels=show_xtick_labels,
            )
            if cluster_idx == 0:
                ax.set_title(
                    SIMP_METRIC_LABELS[metric], fontsize=FONT_TITLE, fontweight="normal", pad=3
                )
            if col_idx == 0 and show_row_labels and show_row_markers:
                color_idx = (cluster_k - 1 + _CLUSTER_COLOR_OFFSET) % len(peak_colors)
                _marker_tf = blended_transform_factory(ax.transAxes, ax.transData)
                ax.plot(
                    CLUSTER_MARKER_AXES_X,
                    0.0,
                    transform=_marker_tf,
                    marker=peak_markers[color_idx],
                    color=peak_colors[color_idx],
                    markersize=peak_marker_markersize(peak_markers[color_idx]),
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    clip_on=False,
                    zorder=4,
                )
        for j in range(2):
            col_idx = 3 + j
            metric = SIMP_METRICS[col_idx]
            ax = fig.add_subplot(gs_stds[0, j])
            ax.set_facecolor(_style["axes_bg"])
            level_data_for_cluster = [
                all_level_data[level_idx][cluster_idx] for level_idx in range(n_levels)
            ]
            show_ytick_labels = col_idx in (0, 3)
            show_xtick_labels = cluster_idx == n_clusters - 1
            _draw_bar_cell(
                ax,
                metric,
                level_data_for_cluster,
                level_label_list,
                cluster_k,
                args.font_size,
                show_ytick_labels=show_ytick_labels,
                show_xtick_labels=show_xtick_labels,
            )
            if cluster_idx == 0:
                ax.set_title(
                    SIMP_METRIC_LABELS[metric], fontsize=FONT_TITLE, fontweight="normal", pad=3
                )

    return title_ax


def plot_metrics_panel(
    fig,
    subplot_spec,
    args: argparse.Namespace,
    *,
    section_panel_labels: tuple[str | None, str | None] = (None, None),
):
    """Entry point for the combined wrapper. Returns ``(title_ax_r, title_ax_t)``."""
    triptych = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=subplot_spec,
        width_ratios=[1, 1],
        wspace=0.06,
    )

    lb, rb = section_panel_labels
    title_ax_r = _plot_section(
        fig,
        triptych[0, 0],
        SIMP_LEVELS_R,
        "Pairwise correlation threshold",
        args,
        args.cluster_dir,
        show_row_labels=True,
        panel_label=lb,
        section_kind="r",
        panel_letter_x=PANEL_AB_LETTER_X,
    )
    title_ax_t = _plot_section(
        fig,
        triptych[0, 1],
        SIMP_LEVELS_T,
        "Hierarchy threshold",
        args,
        args.cluster_dir,
        show_row_labels=False,
        panel_label=rb,
        section_kind="t",
        panel_letter_x=0.0,
    )
    return title_ax_r, title_ax_t


def plot_metric_section(
    fig,
    subplot_spec,
    args: argparse.Namespace,
    *,
    section: str,
    panel_label: str | None = None,
    show_section_title: bool = True,
    show_row_markers: bool = True,
    title_strip_ratio: float = 0.075,
):
    if section == "r":
        return _plot_section(
            fig,
            subplot_spec,
            SIMP_LEVELS_R,
            "Pairwise correlation threshold",
            args,
            args.cluster_dir,
            show_row_labels=True,
            panel_label=panel_label,
            section_kind="r",
            panel_letter_x=PANEL_AB_LETTER_X,
            show_section_title=show_section_title,
            show_row_markers=show_row_markers,
            title_strip_ratio=title_strip_ratio,
        )
    if section == "t":
        return _plot_section(
            fig,
            subplot_spec,
            SIMP_LEVELS_T,
            "Hierarchy threshold",
            args,
            args.cluster_dir,
            show_row_labels=False,
            panel_label=panel_label,
            section_kind="t",
            panel_letter_x=0.0,
            show_section_title=show_section_title,
            show_row_markers=show_row_markers,
            title_strip_ratio=title_strip_ratio,
        )
    raise ValueError(f"Unknown section: {section}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplification metrics figure: bar-chart grid.")
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--font-size", type=int, default=FONT_LABEL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.60), dpi=args.dpi)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.075)
    plot_metrics_panel(fig, fig.add_gridspec(1, 1)[0], args)
    save_path = args.output_dir / "simplified_metrics_grid.png"
    savefig_both(fig, save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
