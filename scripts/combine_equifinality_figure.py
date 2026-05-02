#!/usr/bin/env python3
"""Equifinality combined figure: convergence histograms | PCA clusters | metric validation."""

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
    WIDTH_DOUBLE,
    apply_style,
    savefig_both,
)
from inverse_design.plotting.colormap import DEFAULT_METRIC_COLORS, PEAK_COLORS  # noqa: E402
from inverse_design.io.cluster_profiles import (  # noqa: E402
    load_cluster_profiles,
    normalize_profiles,
)
from inverse_design.vis.utils import peak_colors, peak_markers  # noqa: E402
from inverse_design.vis.vis_parameter_comparison import plot_multi_boxcharts  # noqa: E402
from inverse_design.vis.vis_posterior_overview import plot_pca_with_peaks  # noqa: E402
from inverse_design.vis.vis_simulation_outcome import DEFAULT_FIGURE_DIR, METRIC_LABELS  # noqa: E402
from inverse_design.config.parameter_config import PARAM_RANGES  # noqa: E402

ARCADE_OUTPUT_DIR = REPO_ROOT.parent / "ARCADE_OUTPUT"
DEFAULT_BASE_DIR = ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2"

PANEL_B_METRICS = [
    "doub_time",
    "symmetry",
    "colony_growth",
    "doub_time_std",
    "symmetry_std",
]
PANEL_D_METRICS = [
    "doub_time",
    "symmetry",
    "colony_growth",
    "doub_time_std",
    "symmetry_std",
]
DEFAULT_SUMMARY_JSON = (
    REPO_ROOT / "out" / "pca_cluster_arcade_inputs_breast_only_mean_2" / "summary.json"
)
PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
TOP_PAIR_PARAMS = 8
_CLUSTER_COLOR_OFFSET = 2


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate the horizontal equifinality pipeline figure."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_BASE_DIR,
        help="Root ABC run directory containing iter_N subdirectories.",
    )
    parser.add_argument("--prior-iteration", type=int, default=0)
    parser.add_argument("--posterior-iteration", type=int, default=4)
    parser.add_argument(
        "--posterior-output-dirs",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Directories for posterior re-simulation outputs (mean, mode, peaks). "
            "If omitted, constructed automatically from --base-dir."
        ),
    )
    parser.add_argument(
        "--scenario-names",
        nargs="+",
        default=["EXP", "Mean", "Mode", "Peak 1", "Peak 2", "Peak 3", "Peak 4"],
    )
    parser.add_argument(
        "--targets-json",
        type=Path,
        default=None,
        help="Path to targets.json. Defaults to <base-dir>/targets.json.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_SUMMARY_JSON,
        help="Path to summary.json containing pca_peak_inverse cluster profiles.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results" / "figures" / "fit_exp"
    )
    parser.add_argument("--output-name", default="equifinality_combined.png")
    parser.add_argument("--width", type=float, default=WIDTH_DOUBLE)
    parser.add_argument("--height", type=float, default=4.25)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--hist-bins", type=int, default=10)
    parser.add_argument("--hist-y-max", type=float, default=320)
    parser.add_argument(
        "--font-size",
        type=int,
        default=FONT_LABEL,
        help="Legacy override; default follows shared figure style.",
    )
    parser.add_argument(
        "--spurious-scenario-index",
        type=int,
        default=-1,
        help=(
            "1-based index into scenario_names (EXP=0, then clusters 1..N) to render as "
            "spurious (gray crosshatch, grayed marker). -1 = none. "
            "Example: --spurious-scenario-index 3 grays out the third cluster."
        ),
    )
    return parser.parse_args(argv)


def _pair_top_params(
    norm_profiles: dict[int, dict[str, float]],
    pair: tuple[int, int],
    n: int = TOP_PAIR_PARAMS,
) -> list[str]:
    """Return top-n parameters by normalized value of the first cluster in the pair."""
    idx_a, idx_b = pair
    common = [p for p in norm_profiles[idx_a] if p in norm_profiles[idx_b]]
    return sorted(common, key=lambda p: norm_profiles[idx_a][p], reverse=True)[:n]


def _param_display_name(name: str) -> str:
    """Return shorter, human-friendly parameter labels."""
    return name.replace("_MU", "").replace("_SIGMA", " σ").replace("_", " ").title()


def _draw_pair_panel(
    ax,
    pair: tuple[int, int],
    top_params: list[str],
    norm_profiles: dict[int, dict[str, float]],
    font_size: int,
    show_y_labels: bool = True,
) -> None:
    """Draw one pairwise panel with two dots per parameter and a connector."""
    idx_a, idx_b = pair
    n_params = len(top_params)

    color_a = peak_colors[(idx_a + _CLUSTER_COLOR_OFFSET) % len(peak_colors)]
    color_b = peak_colors[(idx_b + _CLUSTER_COLOR_OFFSET) % len(peak_colors)]
    marker_a = peak_markers[idx_a % len(peak_markers)]
    marker_b = peak_markers[idx_b % len(peak_markers)]

    ax.set_xlim(-0.5, n_params - 0.5)
    ax.set_ylim(-0.08, 1.08)
    ax.tick_params(axis="x", labelsize=FONT_BASE, width=0.5, rotation=45)
    ax.tick_params(axis="y", labelsize=FONT_BASE, width=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_yticks([0.0, 0.5, 1.0])
    if show_y_labels:
        ax.set_yticklabels(["0", "0.5", "1"], fontsize=FONT_BASE)
        ax.set_ylabel("Normalized (prior)", fontsize=FONT_BASE, labelpad=2)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")

    x_labels = [_param_display_name(p) for p in top_params]
    ax.set_xticks(range(n_params))
    ax.set_xticklabels(x_labels, fontsize=FONT_BASE, ha="right")

    for col_idx, param in enumerate(top_params):
        x = col_idx
        y_a = norm_profiles[idx_a][param]
        y_b = norm_profiles[idx_b][param]
        ax.axvline(x, color="#e8e8e8", lw=0.5, zorder=0)
        ax.plot([x, x], [y_a, y_b], color="#aaaaaa", lw=0.8, zorder=1)
        ax.scatter(
            x,
            y_a,
            marker=marker_a,
            color=color_a,
            s=22,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax.scatter(
            x,
            y_b,
            marker=marker_b,
            color=color_b,
            s=22,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_title(f"P{idx_a + 1} vs P{idx_b + 1}", fontsize=font_size, pad=3)
    ax.grid(True, axis="y", alpha=0.15, linewidth=0.4, zorder=0)


def _posterior_output_dirs(args) -> list[Path]:
    if args.posterior_output_dirs:
        return list(args.posterior_output_dirs)
    base = args.base_dir / "POSTERIOR_OUTPUTS"
    planned_dirs = [
        base / "mean" / "iter_0",
        base / "mode" / "iter_0",
        *[base / f"peak_{i}" / "iter_0" for i in range(1, 5)],
    ]
    if all((directory / "final_metrics_seed.csv").exists() for directory in planned_dirs):
        return planned_dirs

    peak_dirs = []
    for i in range(1, 5):
        run_name = f"ABC_SMC_RF_N512_combined_grid_linear_0.7_p{i}_mean_only"
        peak_dirs.append(ARCADE_OUTPUT_DIR / run_name / run_name / f"iter_{args.posterior_iteration}")
    fallback_dirs = [
        args.base_dir / f"iter_{args.posterior_iteration}",
        args.base_dir / f"iter_{args.posterior_iteration}",
        *peak_dirs,
    ]
    return fallback_dirs


def _plot_panel_a(fig, subplot_spec, args) -> None:
    """Draw 6 pairwise cluster parameter profiles in a 2x3 subgrid."""
    profiles = load_cluster_profiles(args.summary_json)
    norm = normalize_profiles(profiles, PARAM_RANGES)
    pair_grid = gridspec.GridSpecFromSubplotSpec(
        2,
        3,
        subplot_spec=subplot_spec,
        hspace=1.40,
        wspace=0.18,
    )
    for pair_idx, pair in enumerate(PAIRS):
        row = pair_idx // 3
        col = pair_idx % 3
        top_params = _pair_top_params(norm, pair, n=TOP_PAIR_PARAMS)
        ax = fig.add_subplot(pair_grid[row, col])
        _draw_pair_panel(
            ax,
            pair,
            top_params,
            norm,
            args.font_size,
            show_y_labels=(col == 0),
        )


def _plot_panel_b(fig, subplot_spec, args, target_metrics):
    """Draw stacked prior/posterior convergence histograms."""
    panel_grid = gridspec.GridSpecFromSubplotSpec(
        len(PANEL_B_METRICS) * 2,
        1,
        subplot_spec=subplot_spec,
        hspace=0.25,
    )
    y_lim = (0, args.hist_y_max)
    y_ticks = np.arange(0, args.hist_y_max + 1, 100)

    df_prior = pd.read_csv(args.base_dir / f"iter_{args.prior_iteration}" / "final_metrics.csv")
    df_post = pd.read_csv(args.base_dir / f"iter_{args.posterior_iteration}" / "final_metrics.csv")
    for row_idx, metric in enumerate(PANEL_B_METRICS):
        axes = [
            fig.add_subplot(panel_grid[row_idx * 2, 0]),
            fig.add_subplot(panel_grid[row_idx * 2 + 1, 0]),
        ]
        plot_histogram_comparison(
            [df_prior, df_post],
            [f"iter_{args.prior_iteration}", f"iter_{args.posterior_iteration}"],
            metric,
            color=metric_to_colors.get(metric, "k"),
            target_metric=target_metrics.get(metric, FEASIBLE_TARGET_METRICS.get(metric)),
            remove_outliers_flag=True,
            bins=args.hist_bins,
            y_lim=y_lim,
            y_ticks=y_ticks,
            fig=fig,
            axes=axes,
            manage_layout=False,
            return_fig=True,
        )
        target_val = target_metrics.get(metric, FEASIBLE_TARGET_METRICS.get(metric))
        if target_val is not None:
            axes[0].axvline(
                target_val,
                color="red",
                linestyle="--",
                linewidth=0.8,
                alpha=0.65,
            )
        axes[0].set_ylabel("Count", fontsize=args.font_size)
        for ax in axes:
            ax.tick_params(labelsize=FONT_BASE, width=0.5)


def _plot_panel_c(fig, subplot_spec, args, posterior_df):
    """Draw PCA scatter with cluster-colored fills and convex hulls."""
    ax = fig.add_subplot(subplot_spec)
    plot_pca_with_peaks(
        posterior_df,
        n_components=2,
        fig=fig,
        ax=ax,
        font_size=args.font_size,
        cluster_fill_alpha=0.12,
    )


def _draw_metric_bar(
    ax,
    metric: str,
    target_df,
    csv_files: list,
    scenario_names: list[str],
    case_colors: list,
    font_size: int,
    spurious_idx: int = -1,
    tolerance_fraction: float = 0.15,
) -> None:
    """Draw one metric panel with tolerance band and optional spurious grayout."""
    is_std = metric.endswith("_std")
    suffix = "final_metrics.csv" if is_std else "final_metrics_seed.csv"
    all_means, all_stds, all_raw = [], [], []

    exp_mean = float(target_df[metric].iloc[0])
    all_means.append(exp_mean)
    all_stds.append(0.0)
    all_raw.append(None)

    for csv_file in csv_files:
        agg_path = Path(csv_file).parent / suffix
        if not agg_path.exists():
            all_means.append(np.nan)
            all_stds.append(0.0)
            all_raw.append(None)
            continue
        df = pd.read_csv(agg_path)
        if metric not in df:
            all_means.append(np.nan)
            all_stds.append(0.0)
            all_raw.append(None)
            continue
        values = df[metric].replace([np.inf, -np.inf], np.nan).dropna().values
        all_means.append(float(values.mean()) if len(values) else np.nan)
        all_stds.append(float(values.std()) if len(values) else 0.0)
        all_raw.append(values if (len(values) > 0 and not is_std) else None)

    ax.axhspan(
        exp_mean * (1 - tolerance_fraction),
        exp_mean * (1 + tolerance_fraction),
        color="#cccccc",
        alpha=0.28,
        zorder=0,
    )

    x_positions = np.arange(len(scenario_names))
    marker_to_size = {"o": 70, "s": 65, "^": 80, "D": 55, "*": 110, "p": 70}

    for j, (x_pos, mean_val, std_val) in enumerate(zip(x_positions, all_means, all_stds)):
        if np.isnan(mean_val):
            continue
        is_spurious = j == spurious_idx
        if j == 0:
            bar_color, hatch, alpha = DEFAULT_METRIC_COLORS.get(metric, "#cccccc"), None, 0.75
        elif is_spurious:
            bar_color, hatch, alpha = "#bbbbbb", "///", 0.5
        else:
            bar_color = case_colors[j - 1] if (j - 1) < len(case_colors) else "white"
            hatch, alpha = None, 0.6
        ax.bar(
            x_pos,
            mean_val,
            yerr=[[0], [std_val]],
            capsize=3,
            alpha=alpha,
            color=bar_color,
            edgecolor="black",
            linewidth=0.8,
            hatch=hatch,
            zorder=2,
        )
        if all_raw[j] is not None and not is_spurious:
            jitter = np.random.default_rng(j).normal(0, 0.06, len(all_raw[j]))
            ax.scatter(x_pos + jitter, all_raw[j], color="black", alpha=0.55, s=12, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["EXP"] + [""] * (len(scenario_names) - 1))
    y_min, y_max = ax.get_ylim()
    y_marker = y_min - 0.09 * (y_max - y_min)
    for j in range(1, len(scenario_names)):
        is_spurious = j == spurious_idx
        marker = peak_markers[(j - 1) % len(peak_markers)]
        ax.scatter(
            x_positions[j],
            y_marker,
            marker=marker,
            s=marker_to_size.get(marker, 70),
            color="#bbbbbb" if is_spurious else peak_colors[(j - 1) % len(peak_colors)],
            edgecolor="black",
            linewidth=1.0,
            clip_on=False,
            zorder=10,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=FONT_BASE, width=0.5, length=2.5)
    ax.grid(True, alpha=0.25, axis="y", linewidth=0.5)
    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=font_size, pad=4)


def _plot_panel_d(fig, subplot_spec, args, target_df, posterior_output_dirs):
    """Draw bar + dot metric comparisons with tolerance band and spurious option."""
    panel_grid = gridspec.GridSpecFromSubplotSpec(
        1,
        len(PANEL_D_METRICS),
        subplot_spec=subplot_spec,
        wspace=0.35,
    )
    csv_files = [directory / "final_metrics_seed.csv" for directory in posterior_output_dirs]
    n_cases = len(args.scenario_names) - 1
    case_colors = [PEAK_COLORS[i % len(PEAK_COLORS)] for i in range(n_cases)]

    for col_idx, metric in enumerate(PANEL_D_METRICS):
        ax = fig.add_subplot(panel_grid[0, col_idx])
        _draw_metric_bar(
            ax,
            metric,
            target_df,
            csv_files,
            args.scenario_names,
            case_colors,
            args.font_size,
            spurious_idx=args.spurious_scenario_index,
        )

    # Keep the public helper exercised with the same styling arguments, then
    # close its standalone figure; the composite panel is drawn directly above.
    preview_fig, _ = plot_multi_boxcharts(
        target_df,
        csv_files,
        args.scenario_names,
        target_metrics=PANEL_D_METRICS[:1],
        case_colors=case_colors,
        font_size=args.font_size,
    )
    plt.close(preview_fig)


def main(argv=None):
    args = parse_args(argv)
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(args.width, args.height), dpi=args.dpi)
    outer = gridspec.GridSpec(1, 1, figure=fig)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.22)
    _plot_panel_a(fig, outer[0], args)

    save_path = args.output_dir / args.output_name
    savefig_both(fig, save_path)
    plt.close(fig)
    print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
