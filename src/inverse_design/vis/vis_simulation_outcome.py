import argparse
import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.preprocessing import MinMaxScaler
from inverse_design.utils.utils import remove_outliers
from matplotlib.widgets import RectangleSelector, Button
from inverse_design.plotting.colormap import metric_color
from inverse_design.plotting.theme import apply_publication_style


METRIC_LABELS = {
    "doub_time": "Doubling time (h)",
    "symmetry": "Symmetry (-)",
    "act_ratio": "Activity (-)",
    "n_cells": "Cell count (cell)",
    "colony_growth": "Colony growth",
}

TOP_MARGINAL_RATIO = 0.16
RIGHT_MARGINAL_RATIO = 0.10
INNER_MARGINAL_SPACE = 0.03

DEFAULT_METRICS_RANGES = {
    "doub_time": (25, 150),
    "symmetry": (0.52, 1.05),
    "act_ratio": (-0.01, 1.01),
    "n_cells": (0, 250),
}

DEFAULT_TARGET_METRICS = {
    "doub_time": 32,
    "symmetry": 0.75,
    "act_ratio": 0.7,
}

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIGURE_DIR = REPO_ROOT / "results" / "figures"
ARCADE_OUTPUT_DIR = REPO_ROOT.parent / "ARCADE_OUTPUT"
DEFAULT_OUTCOME_DATASET_DIRS = [
    ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N512_combined_grid_act_07",
    ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N512_combined_grid_sym_075",
    ARCADE_OUTPUT_DIR / "ABC_SMC_RF_N512_combined_grid_doub_32",
]
DEFAULT_OUTCOME_DATA_DIR = DEFAULT_OUTCOME_DATASET_DIRS[0]
DEFAULT_OUTCOME_METRICS_FILE = DEFAULT_OUTCOME_DATA_DIR / "iter_2" / "final_metrics.csv"
DEFAULT_OUTCOME_METRICS_FILES = [
    dataset_dir / "iter_2" / "final_metrics.csv" for dataset_dir in DEFAULT_OUTCOME_DATASET_DIRS
]


def load_and_prepare_data(param_file, metrics_file):
    # Load parameter and metrics data
    params_df = pd.read_csv(param_file)
    metrics_df = pd.read_csv(metrics_file)

    # Clean up parameter data
    params_df = params_df.drop("file_name", axis=1)
    metrics_list = ["doub_time", "act_t2", "colony_g_rate"]
    # Select std columns from metrics
    std_cols = [col for col in metrics_df.columns if "std" in col]
    std_cols = [col for col in metrics_df.columns if col in metrics_list]
    metrics_std_df = metrics_df[std_cols]

    # Normalize both dataframes to [0,1] range
    scaler = MinMaxScaler()
    params_normalized = pd.DataFrame(scaler.fit_transform(params_df), columns=params_df.columns)
    metrics_normalized = pd.DataFrame(
        scaler.fit_transform(metrics_std_df), columns=metrics_std_df.columns
    )

    return params_normalized, metrics_normalized


def plot_pairwise_relationships(metrics_df):
    # Calculate number of metrics
    n_metrics = len(metrics_df.columns)

    # Create a grid of subplots
    fig, axes = plt.subplots(n_metrics, n_metrics, figsize=(2 * n_metrics, 2 * n_metrics))

    # For each pair of metrics
    for i, metric1 in enumerate(metrics_df.columns):
        for j, metric2 in enumerate(metrics_df.columns):
            ax = axes[i, j]

            if i != j:
                # Calculate correlation and slope
                correlation = metrics_df[metric1].corr(metrics_df[metric2])
                slope, intercept = np.polyfit(metrics_df[metric1], metrics_df[metric2], 1)

                # Create scatter plot
                ax.scatter(metrics_df[metric1], metrics_df[metric2], alpha=0.5)

                # Add trend line
                x_range = np.array([metrics_df[metric1].min(), metrics_df[metric1].max()])
                ax.plot(
                    x_range,
                    slope * x_range + intercept,
                    "r--",
                    label=f"slope={slope:.2f}\nr={correlation:.2f}",
                )

                ax.legend(fontsize="small")
            else:
                # On diagonal, show density plot
                sns.kdeplot(data=metrics_df[metric1], ax=ax)
                ax.set_xlabel("")
                ax.set_ylabel("")

            # Only show labels on edge plots
            if i == n_metrics - 1:
                ax.set_xlabel(metric2)
            if j == 0:
                ax.set_ylabel(
                    metric1
                )  # Changed from metric2 to metric1 for correct y-axis labeling

            # Remove ticks for cleaner look
            ax.tick_params(labelsize="small")

    plt.tight_layout()
    plt.savefig("metric_pairwise_relationships.png")
    # plt.show()


def calculate_chaos_metric(metrics_df):
    # First normalize all std values globally to [0,1] range
    scaler = MinMaxScaler()
    normalized_stds = pd.DataFrame(
        scaler.fit_transform(metrics_df), columns=metrics_df.columns, index=metrics_df.index
    )

    # For each row (input), calculate Shannon entropy
    shannon_entropies = []
    for idx in normalized_stds.index:
        # Get distribution for this input
        if idx == 5:
            break
        p = normalized_stds.loc[idx]
        # Calculate Shannon entropy: -sum(p * log(p))
        # Adding small epsilon to avoid log(0)
        print(p)
        entropy = -np.sum(p * np.log(p + 1e-10))
        print(entropy)

        # Weight the entropy by the mean std value to account for absolute magnitude
        mean_std = metrics_df.loc[idx].mean()
        weighted_entropy = entropy * mean_std
        print(weighted_entropy)
        print("--------------------------------")
        shannon_entropies.append(weighted_entropy)
    # Create DataFrame with results
    chaos_df = pd.DataFrame(
        {
            "chaos_metric": shannon_entropies,
        },
        index=metrics_df.index,
    )

    # Plot distribution of chaos metric
    plt.figure(figsize=(10, 6))
    sns.histplot(data=chaos_df, x="chaos_metric", kde=True)
    plt.title("Distribution of Chaos Metric\n(Higher values indicate more chaotic behavior)")
    plt.xlabel("Chaos Metric (Weighted Shannon Entropy)")
    plt.ylabel("Count")
    # plt.show()

    return chaos_df


def plot_pairwise_scatter(
    df, metric_names, metrics_ranges, point_size=50, alpha=0.8, grid=True, save_path=None
):
    """
    Create three pairwise scatter plots to visualize relationships between three numerical features.
    Interactive selection is enabled - click and drag in any plot to select points.
    Selected points will be highlighted in all plots.

    Parameters:
    -----------
    df : pandas DataFrame
        The dataframe containing the features to plot
    metric_names : list or tuple of str
        Three column names from df to use as [feature_A, feature_B, feature_C]
    metrics_ranges : dict
        A dictionary mapping metric names to their ranges
    point_size : int, optional
        Size of the scatter points
    alpha : float, optional
        Transparency of points (0 to 1)
    grid : bool, optional
        Whether to show grid lines
    """
    # Validate inputs
    if len(metric_names) != 3:
        raise ValueError("metric_names must contain exactly three feature names")

    for name in metric_names:
        if name not in df.columns:
            raise ValueError(f"Feature '{name}' not found in DataFrame columns")
        if not pd.api.types.is_numeric_dtype(df[name]):
            raise ValueError(f"Feature '{name}' must be numeric")

    # Extract features
    feature_A, feature_B, feature_C = metric_names

    # Create the figure and axis
    fig, ax = plt.subplots(1, 4, figsize=(15, 4))
    ax = ax.flatten()

    # Store scatter plots for later reference
    scatters = []

    # Create three pairwise scatter plots
    # A vs B
    scatter1 = ax[0].scatter(
        df[feature_A], df[feature_B], s=point_size, alpha=alpha, edgecolors="k", picker=True
    )
    ax[0].set_xlabel(feature_A, fontsize=12)
    ax[0].set_ylabel(feature_B, fontsize=12)
    ax[0].set_title(f"{feature_A} vs {feature_B}", fontsize=14)
    ax[0].set_xlim(metrics_ranges[feature_A][0], metrics_ranges[feature_A][1])
    ax[0].set_ylim(metrics_ranges[feature_B][0], metrics_ranges[feature_B][1])
    if grid:
        ax[0].grid(True, linestyle="--", alpha=0.7)
    scatters.append(scatter1)

    # B vs C
    scatter2 = ax[1].scatter(
        df[feature_C], df[feature_B], s=point_size, alpha=alpha, edgecolors="k", picker=True
    )
    ax[1].set_xlabel(feature_C, fontsize=12)
    ax[1].set_ylabel(feature_B, fontsize=12)
    ax[1].set_title(f"{feature_C} vs {feature_B}", fontsize=14)
    ax[1].set_xlim(metrics_ranges[feature_C][0], metrics_ranges[feature_C][1])
    ax[1].set_ylim(metrics_ranges[feature_B][0], metrics_ranges[feature_B][1])
    if grid:
        ax[1].grid(True, linestyle="--", alpha=0.7)
    scatters.append(scatter2)

    # A vs C
    scatter3 = ax[2].scatter(
        df[feature_A], df[feature_C], s=point_size, alpha=alpha, edgecolors="k", picker=True
    )
    ax[2].set_xlabel(feature_A, fontsize=12)
    ax[2].set_ylabel(feature_C, fontsize=12)
    ax[2].set_title(f"{feature_A} vs {feature_C}", fontsize=14)
    ax[2].set_xlim(metrics_ranges[feature_A][0], metrics_ranges[feature_A][1])
    ax[2].set_ylim(metrics_ranges[feature_C][0], metrics_ranges[feature_C][1])

    if grid:
        ax[2].grid(True, linestyle="--", alpha=0.7)
    scatters.append(scatter3)

    # Create a heatmap of the correlation matrix
    corr_mask = np.triu(np.ones_like(df[metric_names].corr(), dtype=bool), k=1)
    sns.heatmap(df[metric_names].corr(), annot=True, cmap="Reds", ax=ax[3], mask=corr_mask)
    ax[3].set_title("Correlation Heatmap", fontsize=14)

    # Store the data for selection
    data = {"A": df[feature_A].values, "B": df[feature_B].values, "C": df[feature_C].values}

    # Selected indices (shared across all plots)
    selected_indices = set()

    # Function to reset selection
    def reset_selection(event):
        selected_indices.clear()

        # Reset colors for all scatter plots
        for scatter in scatters:
            scatter.set_facecolors(["#1f77b4"] * len(df))

        # Force redraw
        fig.canvas.draw_idle()
        print("Selection reset")

    # Function to handle selection
    def on_select(eclick, erelease):
        if eclick.inaxes != erelease.inaxes:
            return

        # Get the selection rectangle coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Determine which plot was selected
        ax_idx = ax.tolist().index(eclick.inaxes)
        if ax_idx >= 3:  # Skip if selection was in correlation plot
            return

        # Determine which features were plotted
        if ax_idx == 0:  # A vs B
            x_feat, y_feat = "A", "B"
        elif ax_idx == 1:  # B vs C
            x_feat, y_feat = "B", "C"
        else:  # C vs A
            x_feat, y_feat = "C", "A"

        # Find points within the selection rectangle
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Create mask for points in the selection
        mask = (
            (data[x_feat] >= x_min)
            & (data[x_feat] <= x_max)
            & (data[y_feat] >= y_min)
            & (data[y_feat] <= y_max)
        )

        # Get indices of selected points
        new_selected = set(np.where(mask)[0])

        # Toggle selection
        for idx in new_selected:
            if idx in selected_indices:
                selected_indices.remove(idx)
            else:
                selected_indices.add(idx)

        # Create updated mask
        updated_mask = np.zeros(len(df), dtype=bool)
        for idx in selected_indices:
            updated_mask[idx] = True

        # Update colors for all scatter plots
        for scatter in scatters:
            colors = np.array(["#1f77b4"] * len(df))  # Default blue color
            colors[updated_mask] = "#d62728"  # Red for selected points
            scatter.set_facecolors(colors)

        # Force redraw
        fig.canvas.draw_idle()

    # Create rectangle selectors with explicit button press/release handlers
    selectors = []
    for i in range(3):
        selector = RectangleSelector(
            ax[i],
            on_select,
            useblit=True,
            button=[1],  # Only use left mouse button
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )
        selectors.append(selector)

    # Keep a reference to the selectors to prevent garbage collection
    fig.selectors = selectors

    plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.12, wspace=0.2, hspace=0.15)

    fig.patch.set_visible(False)
    if save_path:
        plt.savefig(save_path)
    else:
        # Add a reset button
        reset_ax = plt.axes([0.90, 0.005, 0.08, 0.05])  # [x, y, width, height]
        reset_button = Button(reset_ax, "Reset", color="lightgoldenrodyellow", hovercolor="0.975")
        # Connect the reset button to the function
        reset_button.on_clicked(reset_selection)
        plt.show()
    return fig, ax, selectors


def _ordered_unique(items):
    return list(dict.fromkeys(items))


def _format_corr(value):
    if not np.isfinite(value):
        return "R = n/a"
    if abs(value) < 0.005:
        value = 0.0
    return f"R = {value:.2f}"


def _clean_metrics_df(df, metrics, remove_outliers_flag=True, iqr_multiplier=1.5):
    missing = [metric for metric in metrics if metric not in df.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    valid_df = df[metrics].replace([np.inf, -np.inf], np.nan).dropna()
    outlier_indices = []
    if remove_outliers_flag and len(valid_df) > 0:
        valid_df, outlier_indices = remove_outliers(valid_df, iqr_multiplier)
    return valid_df, outlier_indices


def _metric_range(df, metric, metrics_ranges):
    if metrics_ranges and metric in metrics_ranges:
        return metrics_ranges[metric]

    values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        raise ValueError(f"No finite values available for metric '{metric}'.")

    lower = values.min()
    upper = values.max()
    padding = 0.08 * (upper - lower) if upper > lower else 1.0
    return lower - padding, upper + padding


def _center_range_on_target(df, metric, x_lim, target_metrics, pad_fraction=0.05):
    if not target_metrics or metric not in target_metrics:
        return x_lim

    target = target_metrics[metric]
    values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    if len(values) == 0:
        return x_lim

    data_half_width = max(abs(values.min() - target), abs(values.max() - target))
    range_half_width = max(abs(x_lim[0] - target), abs(x_lim[1] - target))
    half_width = max(data_half_width, range_half_width) * (1 + pad_fraction)
    if half_width == 0:
        half_width = 1.0
    return target - half_width, target + half_width


def _style_marginal_axis(ax):
    ax.grid(False)
    ax.tick_params(
        left=False,
        right=False,
        bottom=False,
        top=False,
        labelleft=False,
        labelright=False,
        labelbottom=False,
        labeltop=False,
        width=0.7,
    )
    ax.yaxis.offsetText.set_visible(False)
    ax.xaxis.offsetText.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _style_main_axis(ax, show_xlabel, show_ylabel, inset_ylabel=False, ylabel_x=None):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", labelsize=8, width=0.8, length=3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(False)
    if not show_xlabel:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    if not show_ylabel or inset_ylabel:
        ax.set_ylabel("")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    elif not inset_ylabel:
        ax.yaxis.labelpad = 3
        ax.tick_params(axis="y", pad=2)
        if ylabel_x is not None:
            ax.yaxis.set_label_coords(ylabel_x, 0.5)


def _draw_target_lines(ax, x_metric, y_metric, target_metrics, orientation):
    if target_metrics is None:
        return

    draw_horizontal = orientation in {"horizontal", "both"} and y_metric in target_metrics
    draw_vertical = orientation in {"vertical", "both", "axis"} and x_metric in target_metrics

    if draw_horizontal:
        ax.axhline(
            target_metrics[y_metric],
            color=metric_color(y_metric),
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
        )
    if draw_vertical:
        ax.axvline(
            target_metrics[x_metric],
            color=metric_color(x_metric),
            linestyle=(0, (4, 3)),
            linewidth=1.2,
            alpha=0.9,
            zorder=4,
        )


def _muted_color(hex_color: str, alpha: float = 0.35) -> tuple:
    """Return RGBA tuple for hex_color at reduced alpha."""
    r = int(hex_color[1:3], 16) / 255
    g = int(hex_color[3:5], 16) / 255
    b = int(hex_color[5:7], 16) / 255
    return (r, g, b, alpha)


def plot_constraint_propagation_matrix(
    run_data,
    *,
    metrics=("doub_time", "symmetry", "act_ratio"),
    metrics_ranges=None,
    target_metrics=None,
    remove_outliers_flag=True,
    iqr_multiplier=1.5,
    bins=12,
    fig=None,
    subplot_spec=None,
    wspace=0.12,
    hspace=0.08,
    figsize=(5.35, 3.80),
    save_path=None,
):
    """4xN constraint propagation matrix of marginal histograms.

    Parameters
    ----------
    run_data : list of (label, fit_metric, df)
        Each tuple is one row of the matrix.
        ``label`` is the row label string.
        ``fit_metric`` is the column metric key that was constrained in this
        ABC run (e.g. ``"doub_time"``), or ``None`` for a failure row where
        no single metric was targeted.
        ``df`` is a DataFrame containing at least the columns in ``metrics``.
    metrics : sequence of str
        Column metrics, left to right. Must be present in every df.
    metrics_ranges : dict or None
        Fixed axis ranges per metric. Falls back to DEFAULT_METRICS_RANGES.
    target_metrics : dict or None
        Target values per metric for the red dashed reference lines.
        Falls back to DEFAULT_TARGET_METRICS.
    remove_outliers_flag : bool
        Whether to apply IQR outlier removal per row's dataframe.
    iqr_multiplier : float
        IQR multiplier used when remove_outliers_flag is True.
    bins : int
        Number of histogram bins per cell.
    fig : Figure or None
        Existing figure to draw into. A new figure is created if None.
    subplot_spec : SubplotSpec or None
        GridSpec slot to fill. Uses the full figure if None.
    wspace, hspace : float
        Grid spacing between cells.
    figsize : tuple
        (width, height) in inches, used only when fig is None.
    save_path : Path or str or None
        If given, save the figure here and close it.

    Returns
    -------
    fig : Figure
    axes_grid : list of list of Axes
        axes_grid[row_idx][col_idx]
    """
    apply_publication_style(
        font_size=8,
        axes_linewidth=0.8,
        tick_major_width=0.6,
        **{
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.dpi": 300,
        },
    )

    metrics_ranges = DEFAULT_METRICS_RANGES if metrics_ranges is None else metrics_ranges
    target_metrics = DEFAULT_TARGET_METRICS if target_metrics is None else target_metrics

    n_rows = len(run_data)
    n_cols = len(metrics)

    if fig is None:
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=0.18, right=0.99, top=0.93, bottom=0.08)

    if subplot_spec is None:
        outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=wspace, hspace=hspace)
    else:
        outer = gridspec.GridSpecFromSubplotSpec(
            n_rows,
            n_cols,
            subplot_spec=subplot_spec,
            wspace=wspace,
            hspace=hspace,
        )

    axes_grid = []

    for row_idx, (row_label, fit_metric, df) in enumerate(run_data):
        clean_df, _ = _clean_metrics_df(
            df,
            list(metrics),
            remove_outliers_flag=remove_outliers_flag,
            iqr_multiplier=iqr_multiplier,
        )

        y_max = 0
        for metric in metrics:
            x_lim = _metric_range(clean_df, metric, metrics_ranges)
            values = clean_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            counts, _ = np.histogram(values, bins=bins, range=x_lim)
            y_max = max(y_max, int(counts.max()))
        y_upper = max(1.0, y_max * 1.15)

        row_axes = []
        for col_idx, metric in enumerate(metrics):
            ax = fig.add_subplot(outer[row_idx, col_idx])
            is_diagonal = fit_metric == metric
            x_lim = _metric_range(clean_df, metric, metrics_ranges)
            values = clean_df[metric].replace([np.inf, -np.inf], np.nan).dropna()

            bar_color = (
                metric_color(metric)
                if is_diagonal
                else _muted_color(metric_color(metric), alpha=0.35)
            )
            ax.hist(
                values,
                bins=bins,
                range=x_lim,
                color=bar_color,
                edgecolor="none",
                linewidth=0,
                rasterized=True,
            )
            ax.set_xlim(x_lim)
            ax.set_ylim(0, y_upper)

            if target_metrics and metric in target_metrics:
                ax.axvline(
                    target_metrics[metric],
                    color="#dd2020",
                    linestyle=(0, (4, 3)),
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=4,
                )

            if is_diagonal:
                ax.set_facecolor("#eff6ff")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("#2563eb")
                    spine.set_linewidth(1.4)
            else:
                ax.set_facecolor("#fafafa")
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                ax.spines["left"].set_linewidth(0.5)
                ax.spines["bottom"].set_linewidth(0.5)

            ax.tick_params(axis="both", which="major", labelsize=7, length=2.5, width=0.6)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune="both"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))

            show_xlabel = row_idx == n_rows - 1
            show_ylabel = col_idx == 0

            if show_xlabel:
                ax.set_xlabel(METRIC_LABELS.get(metric, metric), fontsize=7.5)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis="x", labelbottom=False)

            if show_ylabel:
                ax.set_ylabel("Count", fontsize=7.5)
                label_color = metric_color(fit_metric) if fit_metric else "#666666"
                ax.text(
                    -0.27,
                    0.5,
                    row_label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=7.5,
                    color=label_color,
                    clip_on=False,
                    rotation=0,
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(axis="y", labelleft=False)

            row_axes.append(ax)
        axes_grid.append(row_axes)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)

    return fig, axes_grid


def plot_metric_joint_hexbin(
    df,
    x_metric,
    y_metric,
    *,
    metrics_ranges=None,
    target_metrics=None,
    target_line_orientation="axis",
    bins=18,
    gridsize=17,
    fig=None,
    subplot_spec=None,
    figsize=(2.45, 2.35),
    show_xlabel=True,
    show_ylabel=True,
    inset_ylabel=False,
    show_y_marginal=True,
    show_x_marginal=True,
    center_x_on_target=True,
    show_corr=True,
    save_path=None,
):
    """Plot one publication-style metric relationship panel with marginal histograms."""

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

    valid_df, _ = _clean_metrics_df(df, [x_metric, y_metric], remove_outliers_flag=False)
    if len(valid_df) == 0:
        raise ValueError(f"No finite paired values available for {x_metric} and {y_metric}.")

    x_lim = _metric_range(valid_df, x_metric, metrics_ranges)
    y_lim = _metric_range(valid_df, y_metric, metrics_ranges)
    if center_x_on_target:
        x_lim = _center_range_on_target(valid_df, x_metric, x_lim, target_metrics)

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if subplot_spec is None:
        subplot_spec = gridspec.GridSpec(1, 1, figure=fig)[0]

    inner = gridspec.GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=subplot_spec,
        width_ratios=(1.0, RIGHT_MARGINAL_RATIO),
        height_ratios=(TOP_MARGINAL_RATIO, 1.0),
        wspace=INNER_MARGINAL_SPACE,
        hspace=INNER_MARGINAL_SPACE,
    )
    ax_main = fig.add_subplot(inner[1, 0])
    ax_top = fig.add_subplot(inner[0, 0], sharex=ax_main)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_main)

    ax_main.hexbin(
        valid_df[x_metric],
        valid_df[y_metric],
        gridsize=gridsize,
        extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]],
        cmap="Greys",
        mincnt=1,
        linewidths=0,
        rasterized=True,
    )
    ax_top.hist(
        valid_df[x_metric],
        bins=bins,
        range=x_lim,
        color=metric_color(x_metric),
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
    )
    if not show_x_marginal:
        ax_top.set_visible(False)
    ax_right.hist(
        valid_df[y_metric],
        bins=bins,
        range=y_lim,
        orientation="horizontal",
        color=metric_color(y_metric),
        edgecolor="black",
        linewidth=0.45,
        alpha=0.9,
    )
    if not show_y_marginal:
        ax_right.set_visible(False)

    ax_main.set_xlim(x_lim)
    ax_main.set_ylim(y_lim)
    ax_main.set_box_aspect(1)
    ax_main.set_xlabel(METRIC_LABELS.get(x_metric, x_metric), fontsize=8)
    y_label = METRIC_LABELS.get(y_metric, y_metric)
    ax_main.set_ylabel(y_label if show_ylabel else "", fontsize=7.2)

    _draw_target_lines(
        ax_main,
        x_metric=x_metric,
        y_metric=y_metric,
        target_metrics=target_metrics,
        orientation=target_line_orientation,
    )

    if show_corr:
        corr = valid_df[x_metric].corr(valid_df[y_metric])
        ax_main.text(
            0.95,
            0.94,
            _format_corr(corr),
            transform=ax_main.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="black",
        )

    _style_main_axis(
        ax_main,
        show_xlabel=show_xlabel,
        show_ylabel=show_ylabel,
        inset_ylabel=inset_ylabel,
        ylabel_x=-0.38 if show_y_marginal else -0.28,
    )
    if show_ylabel and inset_ylabel:
        ax_main.text(
            0.03,
            0.05,
            y_label,
            transform=ax_main.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )
    _style_marginal_axis(ax_top)
    _style_marginal_axis(ax_right)
    ax_top.set_xlim(x_lim)
    ax_right.set_ylim(y_lim)

    axes = {"main": ax_main, "top": ax_top, "right": ax_right}
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)
    return fig, axes


def plot_feasible_metric_relationship_grid(
    metrics_df,
    *,
    x_metrics=("doub_time", "symmetry", "act_ratio"),
    y_metric_grid=(
        ("n_cells", "n_cells", "n_cells"),
        ("symmetry", "doub_time", "doub_time"),
        ("act_ratio", "act_ratio", "symmetry"),
    ),
    metrics_ranges=None,
    target_metrics=None,
    target_line_orientation="vertical",
    remove_outliers_flag=True,
    iqr_multiplier=1.5,
    figsize=(5.35, 4.85),
    fig=None,
    subplot_spec=None,
    wspace=0.18,
    hspace=0.10,
    show_all_axis_labels=False,
    show_all_y_labels=True,
    deduplicate_repeated_y_labels=False,
    show_y_marginals="first_col",
    show_x_marginals="first_row",
    center_x_on_target=True,
    save_path=None,
):
    """Create the feasible-metric relationship grid used for panel B."""

    metrics_ranges = DEFAULT_METRICS_RANGES if metrics_ranges is None else metrics_ranges
    target_metrics = DEFAULT_TARGET_METRICS if target_metrics is None else target_metrics

    n_rows = len(y_metric_grid)
    n_cols = len(x_metrics)
    if any(len(row) != n_cols for row in y_metric_grid):
        raise ValueError("Each y_metric_grid row must have the same length as x_metrics.")

    plot_metrics = _ordered_unique(
        list(x_metrics) + [metric for row in y_metric_grid for metric in row]
    )
    valid_df, outlier_indices = _clean_metrics_df(
        metrics_df,
        plot_metrics,
        remove_outliers_flag=remove_outliers_flag,
        iqr_multiplier=iqr_multiplier,
    )

    if fig is None:
        fig = plt.figure(figsize=figsize)
        fig.subplots_adjust(left=0.115, right=0.99, top=0.985, bottom=0.10)
    if subplot_spec is None:
        outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, wspace=wspace, hspace=hspace)
    else:
        outer = gridspec.GridSpecFromSubplotSpec(
            n_rows,
            n_cols,
            subplot_spec=subplot_spec,
            wspace=wspace,
            hspace=hspace,
        )
    axes = []

    for row_idx, y_metrics in enumerate(y_metric_grid):
        axis_row = []
        seen_y_ranges = set()
        for col_idx, (x_metric, y_metric) in enumerate(zip(x_metrics, y_metrics)):
            y_range = tuple(_metric_range(valid_df, y_metric, metrics_ranges))
            show_ylabel = show_all_y_labels or show_all_axis_labels or col_idx == 0
            if deduplicate_repeated_y_labels and y_range in seen_y_ranges:
                show_ylabel = False
            seen_y_ranges.add(y_range)
            if show_y_marginals == "all":
                show_y_marginal = True
            elif show_y_marginals == "none":
                show_y_marginal = False
            elif show_y_marginals == "row_last":
                show_y_marginal = col_idx == n_cols - 1
            elif show_y_marginals == "first_col":
                show_y_marginal = col_idx == 0
            else:
                raise ValueError(f"Unknown show_y_marginals mode: {show_y_marginals}")

            if show_x_marginals == "all":
                show_x_marginal = True
            elif show_x_marginals == "none":
                show_x_marginal = False
            elif show_x_marginals == "first_row":
                show_x_marginal = row_idx == 0
            else:
                raise ValueError(f"Unknown show_x_marginals mode: {show_x_marginals}")

            _, panel_axes = plot_metric_joint_hexbin(
                valid_df,
                x_metric,
                y_metric,
                metrics_ranges=metrics_ranges,
                target_metrics=target_metrics,
                target_line_orientation=target_line_orientation,
                fig=fig,
                subplot_spec=outer[row_idx, col_idx],
                show_xlabel=show_all_axis_labels or row_idx == n_rows - 1,
                show_ylabel=show_ylabel,
                inset_ylabel=False,
                show_y_marginal=show_y_marginal,
                show_x_marginal=show_x_marginal,
                center_x_on_target=center_x_on_target,
                show_corr=True,
            )
            axis_row.append(panel_axes)
        axes.append(axis_row)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close(fig)

    return fig, axes, outlier_indices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot the feasible metric relationship grid for publication figures."
    )
    parser.add_argument(
        "--metrics-files",
        nargs="+",
        type=Path,
        default=DEFAULT_OUTCOME_METRICS_FILES,
        help="CSV file(s) containing simulation outcome metrics. Multiple files are concatenated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="Directory where the figure file is saved.",
    )
    parser.add_argument(
        "--output-name",
        default="feasible_metric_relationships_grid.png",
        help="Output figure filename.",
    )
    parser.add_argument("--target-doub-time", type=float, default=32)
    parser.add_argument("--target-symmetry", type=float, default=0.75)
    parser.add_argument("--target-act-ratio", type=float, default=0.7)
    parser.add_argument(
        "--target-line-orientation",
        choices=["axis", "horizontal", "vertical", "both", "none"],
        default="vertical",
        help="Which target reference lines to draw.",
    )
    parser.add_argument("--iqr-multiplier", type=float, default=1.5)
    parser.add_argument(
        "--no-remove-outliers",
        action="store_true",
        help="Disable IQR outlier removal before plotting.",
    )
    parser.add_argument("--width", type=float, default=5.35, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=4.85, help="Figure height in inches.")
    parser.add_argument("--wspace", type=float, default=0.18)
    parser.add_argument("--hspace", type=float, default=0.10)
    parser.add_argument(
        "--y-marginals",
        choices=["all", "row_last", "first_col", "none"],
        default="first_col",
        help="Which right-side horizontal histograms to show.",
    )
    parser.add_argument(
        "--x-marginals",
        choices=["all", "first_row", "none"],
        default="first_row",
        help="Which top histograms to show.",
    )
    parser.add_argument(
        "--no-center-x-on-target",
        action="store_true",
        help="Disable x-axis centering around target values.",
    )
    parser.add_argument(
        "--edge-labels-only",
        action="store_true",
        help="Only show x labels on the bottom row and y labels on the left edge.",
    )
    parser.add_argument(
        "--deduplicate-y-labels",
        action="store_true",
        help="Within each row, hide repeated y-axis labels that share the same range.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.concat(
        [pd.read_csv(metrics_file) for metrics_file in args.metrics_files],
        ignore_index=True,
    )
    target_metrics = None
    if args.target_line_orientation != "none":
        target_metrics = {
            "doub_time": args.target_doub_time,
            "symmetry": args.target_symmetry,
            "act_ratio": args.target_act_ratio,
        }

    _, _, outlier_indices = plot_feasible_metric_relationship_grid(
        metrics_df,
        target_metrics=target_metrics,
        target_line_orientation=args.target_line_orientation,
        remove_outliers_flag=not args.no_remove_outliers,
        iqr_multiplier=args.iqr_multiplier,
        figsize=(args.width, args.height),
        wspace=args.wspace,
        hspace=args.hspace,
        show_all_axis_labels=False,
        show_all_y_labels=True,
        deduplicate_repeated_y_labels=args.deduplicate_y_labels,
        show_y_marginals=args.y_marginals,
        show_x_marginals=args.x_marginals,
        center_x_on_target=not args.no_center_x_on_target,
        save_path=args.output_dir / args.output_name,
    )
    print(f"Removed {len(outlier_indices)} outliers in df")
    print(f"Saved {args.output_dir / args.output_name}")


if __name__ == "__main__":
    main()
