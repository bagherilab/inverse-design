import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import seaborn as sns
from inverse_design.plotting.colormap import DEFAULT_METRIC_COLORS
from inverse_design.plotting.theme import apply_publication_style
from inverse_design.utils.utils import remove_outliers

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIGURE_DIR = REPO_ROOT / "results" / "figures"
DEFAULT_HISTOGRAM_DATA_DIR = (
    Path(__file__).resolve().parents[4]
    / "ARCADE_OUTPUT"
    / "ABC_SMC_RF_N512_combined_grid_breast_only_mean"
)
FEASIBLE_TARGET_METRICS = {
    "symmetry": 0.75,
    "doub_time": 32,
    "act_ratio": 0.7,
}

metric_to_colors = dict(DEFAULT_METRIC_COLORS)
metric_to_labels = {
    "symmetry": "Symmetry (-)",
    "doub_time": "Doubling time (h)",
    "colony_growth": "Colony growth",
    "act_ratio": "Activity (-)",
    "n_cells": "Cell count (cell)",
    "symmetry_std": "Symmetry SD",
    "doub_time_std": "Doubling time SD",
    "colony_growth_std": "Colony growth SD",
    "act_ratio_std": "Activity SD",
}
metric_to_xlim = {
    "symmetry": (0.46, 1.05),
    "doub_time": (13, 161),
    "colony_growth": (-16, 65),
    "act_ratio": (-0.02, 1.02),
    "n_cells": (0, 250),
    "symmetry_std": (-0.016, 0.172),
    "doub_time_std": (-4, 47),
}


def plot_histogram_comparison(
    dfs,
    labels,
    metric,
    color="k",
    save_path=None,
    target_metric=None,
    remove_outliers_flag=False,
    iqr_multiplier=1.5,
    bins=10,
    x_lim=None,
    y_lim=None,
    y_ticks=None,
    xlabel=None,
    show_target_on_first=False,
    neutral_first=True,
    show_distribution_labels=False,
    panel_height=1.0,
    fig_width=2.45,
    fig=None,
    axes=None,
    manage_layout=True,
    return_fig=False,
):
    """
    Create horizontal histogram plots for a single metric, with each subplot showing a different distribution.

    Args:
        dfs (list[pd.DataFrame]): List of DataFrames containing multiple samples
        labels (list[str]): Labels for each distribution
        metric (str): The metric to plot
        color (str): Base color for the histograms
        save_path (str, optional): Path to save the figure
        target_metric (float, optional): Target value to plot as vertical line
        remove_outliers_flag (bool): Whether to remove outliers using IQR method
        iqr_multiplier (float): Multiplier for IQR to determine outlier bounds (default: 1.5)
        bins (int): Number of histogram bins.
        x_lim (tuple[float, float], optional): Explicit x-axis limits. Defaults to metric_to_xlim
            when available, otherwise data bounds with padding.
        y_lim (tuple[float, float], optional): Explicit y-axis limits.
        y_ticks (sequence[float], optional): Explicit y-axis tick locations.
        xlabel (str, optional): Label for the bottom x-axis.
        show_target_on_first (bool): Draw the target line on the first distribution too.
        neutral_first (bool): Plot the first distribution in gray, useful for prior/posterior panels.
        show_distribution_labels (bool): Show each distribution label inside its panel.
        panel_height (float): Height of each stacked histogram panel in inches.
        fig_width (float): Figure width in inches.
        fig (matplotlib.figure.Figure, optional): Existing figure for composite layouts.
        axes (sequence[matplotlib.axes.Axes], optional): Existing axes to draw into.
        manage_layout (bool): Whether to call tight_layout/subplots_adjust.
        return_fig (bool): Return ``(fig, axes)`` instead of showing/closing.
    """

    apply_publication_style(
        font_size=8,
        axes_linewidth=0.8,
        tick_major_width=0.8,
        **{
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 300,
        },
    )

    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    n_distributions = len(dfs)
    if axes is None:
        fig, axes = plt.subplots(
            n_distributions,
            1,
            figsize=(fig_width, panel_height * n_distributions),
            sharex=True,
        )
        if n_distributions == 1:
            axes = [axes]
    else:
        axes = list(axes)
        if len(axes) != n_distributions:
            raise ValueError(
                f"Expected {n_distributions} axes for {n_distributions} distributions, "
                f"got {len(axes)}."
            )
        fig = fig or axes[0].figure

    valid_dfs = []
    for df in dfs:
        valid_df = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
        if remove_outliers_flag and len(valid_df) > 0:
            valid_df, _ = remove_outliers(valid_df, iqr_multiplier)
            valid_dfs.append(valid_df)
            print(f"Removed {len(df[metric]) - len(valid_df)} outliers for {metric} in df")
        else:
            valid_dfs.append(valid_df)

    if any(len(valid_df) == 0 for valid_df in valid_dfs):
        raise ValueError(f"No finite values available for metric '{metric}'.")

    if x_lim is None:
        x_lim = metric_to_xlim.get(metric)
    if x_lim is None:
        x_min = min(valid_df.min() for valid_df in valid_dfs)
        x_max = max(valid_df.max() for valid_df in valid_dfs)
        padding = 0.08 * (x_max - x_min) if x_max > x_min else 1.0
        x_lim = (x_min - padding, x_max + padding)

    # Calculate shared y-axis bounds using common bin edges for fair panel comparisons.
    y_max = 0
    for df in valid_dfs:
        hist_values, _ = np.histogram(df, bins=bins, range=x_lim)
        y_max = max(y_max, hist_values.max())

    if y_lim is None:
        y_lim = (0, np.ceil(y_max * 1.12 / 10) * 10)

    for i, (df, label) in enumerate(zip(valid_dfs, labels)):
        ax = axes[i]
        plot_color = "0.70" if neutral_first and i == 0 else metric_to_colors.get(metric, color)

        bin_edges = np.linspace(df.min(), df.max(), bins + 1)
        sns.histplot(
            data=df,
            ax=ax,
            color=plot_color,
            bins=bin_edges,
            edgecolor="black",
            linewidth=0.55,
            alpha=0.82,
        )

        # Clean, compact axes suitable for assembling into multi-panel figures.
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        if i == len(valid_dfs) - 1:
            ax.set_xlabel(xlabel or metric_to_labels.get(metric, metric), fontsize=8)
        else:
            ax.set_xlabel("")

        ax.set_ylabel("")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        else:
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

        # Remove x tick labels (but keep ticks)
        # ax.tick_params(axis='x', labelbottom=False)
        # Style grid
        ax.grid(axis="y", linestyle="--", linewidth=0.55, color="0.82")
        ax.set_axisbelow(True)

        if i != len(valid_dfs) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.tick_params(axis="x", labelbottom=True)
        ax.tick_params(axis="both", which="major", labelsize=8, width=0.8)

        # Set scientific notation for x-axis
        ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))

        if target_metric is not None and (show_target_on_first or i != 0):
            ax.axvline(
                x=target_metric,
                color="#d62728",
                linestyle=(0, (4, 3)),
                alpha=0.95,
                linewidth=1.35,
                zorder=5,
            )

        if show_distribution_labels and labels:
            ax.text(
                0.02,
                0.86,
                label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="0.25",
            )

    if manage_layout:
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.18)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        if not return_fig:
            plt.close()
    else:
        if return_fig:
            return fig, axes
        plt.show()

    if return_fig:
        return fig, axes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot feasible metric histogram panels for publication figures."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_HISTOGRAM_DATA_DIR,
        help="Folder containing iter_<n>/final_metrics.csv files.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=[0, 4],
        help="Iteration numbers to compare.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["doub_time", "symmetry", "act_ratio"],
        help="Metric columns to plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where figure files are saved.",
    )
    parser.add_argument(
        "--prefix",
        default="feasible_metric_histogram",
        help="Filename prefix for saved histograms.",
    )
    parser.add_argument("--target-doub-time", type=float, default=32)
    parser.add_argument("--target-symmetry", type=float, default=0.75)
    parser.add_argument("--target-act-ratio", type=float, default=0.7)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--y-max", type=float, default=320)
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--no-remove-outliers",
        action="store_true",
        help="Disable IQR outlier removal.",
    )
    parser.add_argument(
        "--show-distribution-labels",
        action="store_true",
        help="Show iteration labels inside each histogram panel.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    posterior_metrics_files = [
        args.data_dir / f"iter_{iteration}" / "final_metrics.csv" for iteration in args.iterations
    ]
    posterior_metrics_dfs = [pd.read_csv(path) for path in posterior_metrics_files]

    target_metrics = {
        "doub_time": args.target_doub_time,
        "symmetry": args.target_symmetry,
        "act_ratio": args.target_act_ratio,
    }
    labels = [f"iter_{iteration}" for iteration in args.iterations]
    y_lim = (0, args.y_max) if args.y_max is not None else None
    y_ticks = np.arange(0, args.y_max + 1, 100) if args.y_max is not None else None

    for metric_name in args.metrics:
        save_file = args.output_dir / f"{args.prefix}_{metric_name}.png"
        plot_histogram_comparison(
            posterior_metrics_dfs,
            labels,
            metric_name,
            color=metric_to_colors.get(metric_name, "k"),
            save_path=save_file,
            target_metric=target_metrics.get(metric_name, FEASIBLE_TARGET_METRICS.get(metric_name)),
            remove_outliers_flag=not args.no_remove_outliers,
            iqr_multiplier=args.iqr_multiplier,
            bins=args.bins,
            y_lim=y_lim,
            y_ticks=y_ticks,
            show_distribution_labels=args.show_distribution_labels,
        )
        print(f"Saved {save_file}")
