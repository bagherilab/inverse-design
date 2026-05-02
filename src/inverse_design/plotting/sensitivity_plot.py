import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from inverse_design.analyze.sensitivity_analysis import align_dataframes, remove_nan_rows
from inverse_design.plotting.theme import apply_publication_style
import os


def get_varying_parameters(params_df, threshold=1e-10):
    """
    Identify parameters that vary in the dataset above a given threshold.

    Args:
        params_df (pd.DataFrame): DataFrame containing parameter values
        threshold (float): Minimum standard deviation to consider a parameter as varying

    Returns:
        list: List of column names for parameters that vary above the threshold
    """
    param_cols = [col for col in params_df.columns if col != "input_folder"]
    params_std = params_df[param_cols].std()
    return list(params_std[params_std.abs() > threshold].index)


def plot_sensitivity_bubble(param_df, metrics_df, target_metric="symmetry", save_file=None):
    """
    Create a bubble plot showing parameter relationships.

    Args:
        param_df (pd.DataFrame): DataFrame containing parameters
        metrics_df (pd.DataFrame): DataFrame containing metrics
        target_metric (str): Target metric to analyze (default: 'symmetry')
        save_file (str): Path to save the figure (optional)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get varying parameters
    varying_cols = get_varying_parameters(param_df)
    if len(varying_cols) != 2:
        raise ValueError(f"Expected 2 varying parameters, found {len(varying_cols)}")

    apply_publication_style(font_size=10, axes_linewidth=1.0)

    # Create figure - remove aspect ratio constraint to avoid compression
    fig, ax = plt.subplots(figsize=(8, 8))
    # Don't set aspect ratio here - we'll handle circle rendering differently

    # Get standard deviation column
    std_col = f"{target_metric}_std"
    if std_col not in metrics_df.columns:
        raise ValueError(f"Standard deviation column {std_col} not found in metrics file")

    # Normalize sizes between 50 and 500 based on standard deviation
    min_std = min(metrics_df[std_col])
    max_std = max(metrics_df[std_col])
    normalized_sizes = 20 + (metrics_df[std_col] - min_std) * (100 - 20) / (max_std - min_std)

    # Create custom colormap: skyblue for low, white for mean, lightcoral for high
    import matplotlib.colors as mcolors

    colors = ["white", "#af1b0a"]
    n_bins = 20
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)

    # Set color normalization with bounds at (min(sym) * 0.95, 1.0)
    min_metric = min(metrics_df[target_metric])
    max_metric = max(metrics_df[target_metric])
    # max_metric = 50

    # Use specified bounds
    color_min = min_metric * 0.95
    color_max = max_metric * 1.05
    norm = mcolors.Normalize(vmin=color_min, vmax=color_max)

    # Create scatter plot with color for symmetry and size for std
    # Use marker='o' to ensure circular bubbles regardless of aspect ratio
    scatter = ax.scatter(
        param_df[varying_cols[0]],
        param_df[varying_cols[1]],
        s=normalized_sizes,
        c=metrics_df[target_metric],
        cmap=cmap,
        norm=norm,
        alpha=0.7,
        edgecolors="black",
        linewidths=1.5,
        marker="o",  # Explicitly use circular markers
    )

    # Style the axes (consistent with previous figure)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Set labels with bold formatting
    ax.set_xlabel(varying_cols[0], fontsize=12, fontweight="bold")
    ax.set_ylabel(varying_cols[1], fontsize=12, fontweight="bold")

    ax.set_axisbelow(True)

    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Add colorbar for symmetry values
    from matplotlib.colorbar import ColorbarBase

    cbar_ax = fig.add_axes([0.85, 0.25, 0.02, 0.5])  # [left, bottom, width, height]
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="vertical")
    cbar.ax.tick_params(labelsize=12, width=1.5)
    # Set tick label weights separately
    for label in cbar.ax.get_yticklabels():
        label.set_weight("bold")
    cbar.set_label(target_metric.title(), rotation=270, labelpad=20, fontsize=12, weight="bold")

    legend_values = np.linspace(min_std, max_std, 5)
    legend_sizes = 20 + (legend_values - min_std) * (100 - 20) / (max_std - min_std)

    # Position for horizontal legend at the bottom
    legend_y = 0.05  # Y position for all circles
    legend_x_start = 0.18  # Starting X position
    scale_factor = 0.001  # Empirically determined scaling factor

    circle_radii = [np.sqrt(size / np.pi) * scale_factor for size in legend_sizes]

    # Calculate spacing to maintain equal distance between circumferences
    circle_positions = [legend_x_start]
    circumference_spacing = 0.02  # Desired spacing between circle edges

    for i in range(1, len(circle_radii)):
        new_pos = (
            circle_positions[i - 1] + circle_radii[i - 1] + circumference_spacing + circle_radii[i]
        )
        circle_positions.append(new_pos)

    # Create circles for the legend using the same aspect ratio considerations
    for i, (val, size, radius, x_pos) in enumerate(
        zip(legend_values, legend_sizes, circle_radii, circle_positions)
    ):
        # Create perfect circles by using the same coordinate system considerations
        circle = plt.Circle(
            (x_pos, legend_y),
            radius,
            facecolor="white",
            edgecolor="black",
            linewidth=1.75,
            transform=fig.transFigure,
        )
        fig.patches.append(circle)

    # Add labels for min and max values at the ends
    fig.text(
        circle_positions[0] - circle_radii[0] - 0.05,
        legend_y,
        f"{min_std:.3f}",
        fontsize=9,
        ha="center",
        va="center",
        transform=fig.transFigure,
    )
    fig.text(
        circle_positions[-1] + circle_radii[-1] + 0.05,
        legend_y,
        f"{max_std:.3f}",
        fontsize=9,
        ha="center",
        va="center",
        transform=fig.transFigure,
    )

    # Adjust layout to accommodate colorbar
    plt.subplots_adjust(right=0.85, bottom=0.15)
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close()
    else:
        plt.show()


def plot_partial_dependence(param_df, metrics_df, target_metric="symmetry", save_file=None):
    """
    Create partial dependence plots showing relationship between parameters and metric.
    For each subplot:
        - First subplot: Shows param_1 vs metric, with separate lines for each unique param_2 value
        - Second subplot: Shows param_2 vs metric, with separate lines for each unique param_1 value

    Args:
        param_df (pd.DataFrame): DataFrame containing parameters
        metrics_df (pd.DataFrame): DataFrame containing metrics
        target_metric (str): Target metric to analyze
        save_file (str): Path to save the figure (optional)
    """
    df = pd.concat([param_df, metrics_df], axis=1)
    varying_cols = get_varying_parameters(param_df)
    if len(varying_cols) != 2:
        raise ValueError(f"Expected 2 varying parameters, found {len(varying_cols)}")

    param_1, param_2 = varying_cols

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    markers = ["o", "x", "+", "s", "D", "P", "H", "X", "d", "p", "h", "v", "8"]

    for i, (main_param, other_param) in enumerate([(param_1, param_2), (param_2, param_1)]):
        unique_values = sorted(df[other_param].unique())
        palette = sns.color_palette(n_colors=len(unique_values))

        for j, value in enumerate(unique_values):
            mask = np.isclose(df[other_param], value, rtol=1e-10)
            subset = df[mask].copy()
            subset = subset.sort_values(by=main_param)
            axes[i].plot(
                subset[main_param],
                subset[target_metric],
                "-",
                marker=markers[j % len(markers)],
                color=palette[j],
                label=f"{other_param}={value:.3f}",
                markersize=4,
                alpha=0.6,
            )

        axes[i].set_xlabel(main_param)
        axes[i].set_ylabel(target_metric)
        axes[i].set_title(f"Relationship between {target_metric}\nand {main_param}")
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def main():
    target_metrics = ["colony_growth", "doub_time", "act_ratio", "symmetry"]
    parameter_base_folder = f"../../../ARCADE_OUTPUT/sensitivity_analysis"
    for i in range(len(target_metrics[1:2])):
        target_metric = target_metrics[i]
        param_df = pd.read_csv(f"{parameter_base_folder}/all_param_df.csv")
        metrics_df = pd.read_csv(f"{parameter_base_folder}/final_metrics.csv")
        param_df, metrics_df = align_dataframes(param_df, metrics_df)
        metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
        param_df, metrics_df = remove_nan_rows(param_df, metrics_df)
        # colors: #486b45 (symmetry), #545aab (colony growth rate), #bb883b (double time), #af1b0a (act_ratio)
        if 1:
            save_file = f"{parameter_base_folder}/sensitivity_bubble_{target_metric}.png"
            plot_sensitivity_bubble(
                param_df,
                metrics_df,
                target_metric=target_metric,
                save_file=save_file,
            )
        if 1:
            save_file = (
                f"{parameter_base_folder}/sensitivity_partial_dependence_{target_metric}.png"
            )
            plot_partial_dependence(
                param_df,
                metrics_df,
                target_metric=target_metric,
                save_file=save_file,
            )


if __name__ == "__main__":
    main()
