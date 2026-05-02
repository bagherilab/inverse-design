import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from inverse_design.plotting.colormap import metric_color
from inverse_design.plotting.theme import apply_publication_style


def vis_doub_time_distribution(df, save_path=None):

    # Clean column names (remove extra spaces)
    df.columns = df.columns.str.strip()

    # Calculate mean and std for each cancer type
    stats_data = []
    for cancer_type in df["Panel Name"].unique():
        subset = df[df["Panel Name"] == cancer_type]["Doubling Time"]
        stats_data.append(
            {
                "Cancer_Type": cancer_type,
                "Mean": subset.mean(),
                "Std": subset.std(),
                "Count": len(subset),
            }
        )

    # Add the glioblastoma data point
    stats_data.append(
        {"Cancer_Type": "Glioblastoma (U87MG)", "Mean": 30.8, "Std": 2.5, "Count": "N/A"}
    )
    stats_data.append(
        {"Cancer_Type": "Glioblastoma (GBP03)", "Mean": 25.4, "Std": 0.5, "Count": "N/A"}
    )

    # Convert to DataFrame for easier handling
    stats_df = pd.DataFrame(stats_data)

    apply_publication_style(font_size=12, axes_linewidth=1.5)

    # Create figure with academic styling
    fig, ax = plt.subplots(figsize=(int(len(stats_df) / 2), 4))

    # Create bar plot with upper error bars only (empty bars)
    bars = ax.bar(
        range(len(stats_df)),
        stats_df["Mean"],
        yerr=[np.zeros(len(stats_df)), stats_df["Std"]],  # Only upper error bars
        color=metric_color("doub_time"),
        alpha=0.7,
        capsize=5,
        error_kw={"elinewidth": 2, "capthick": 2, "color": "black"},
        edgecolor="black",  # Add border to each bar
        linewidth=2,
    )  # Border thickness

    # Add jittered swarm plots for each cancer type (except glioblastoma)
    cancer_types = df["Panel Name"].unique()
    for i, cancer_type in enumerate(cancer_types):
        subset = df[df["Panel Name"] == cancer_type]["Doubling Time"]
        # Add jitter to x-positions to avoid overlapping
        jitter = np.random.normal(0, 0.05, len(subset))  # Small random jitter
        x_positions = np.full(len(subset), i) + jitter
        ax.scatter(
            x_positions, subset, color="black", alpha=1.0, s=30, zorder=3
        )  # Ensure points are on top

    # Style the axes (consistent with previous figures)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)

    ax.set_ylabel("Doubling Time (hours)", fontsize=12, fontweight="bold")

    # Set x-axis labels
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df["Cancer_Type"], rotation=45, ha="right")

    # Set y-axis to log scale
    # ax.set_yscale('log')

    # Style grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Make tick labels bold and thicker (consistent with previous figures)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5)

    # Bold tick labels
    for label_tick in ax.get_xticklabels():
        label_tick.set_fontweight("bold")
    for label_tick in ax.get_yticklabels():
        label_tick.set_fontweight("bold")

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def create_bar_plot_from_manual_data(
    manual_data, y_label="Metric Value", color="k", save_path=None
):
    """
    Create a bar plot with academic styling from manually entered data.

    Parameters:
    manual_data: list of dictionaries with keys: 'Cancer_Type', 'Mean', 'Std', 'Count', 'Metric'
    y_label: string for y-axis label
    title: optional title for the plot
    figsize: tuple for figure size
    """

    # Convert manual data to DataFrame
    stats_df = pd.DataFrame(manual_data)

    apply_publication_style(font_size=12, axes_linewidth=1.5)

    # Create figure with academic styling
    if y_label == "Symmetry [-]":
        fig, ax = plt.subplots(figsize=(len(stats_df) + 0.1, 4))
    else:
        fig, ax = plt.subplots(figsize=(len(stats_df), 4))

    # Handle cases where Std might be NaN or None
    std_values = []
    for std in stats_df["Std"]:
        if pd.isna(std) or std is None or std == "N/A":
            std_values = None
        else:
            std_values.append(std)

    # Create bar plot with upper error bars only (empty bars)
    if std_values:
        bars = ax.bar(
            range(len(stats_df)),
            stats_df["Mean"],
            yerr=[np.zeros(len(stats_df)), std_values],  # Only upper error bars
            color=color,
            alpha=0.7,
            capsize=5,
            error_kw={"elinewidth": 2, "capthick": 2, "color": "black"},
            edgecolor="black",  # Add border to each bar
            linewidth=2,
        )  # Border thickness
    else:
        bars = ax.bar(
            range(len(stats_df)),
            stats_df["Mean"],
            color=color,
            alpha=0.7,
            capsize=5,
            error_kw={"elinewidth": 2, "capthick": 2, "color": "black"},
            edgecolor="black",  # Add border to each bar
            linewidth=2,
        )  # Border thickness
    # Style the axes (consistent with previous figures)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")

    # Set x-axis labels
    ax.set_xticks(range(len(stats_df)))
    ax.set_xticklabels(stats_df["Cancer_Type"], rotation=45, ha="right")

    # Style grid
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    # Make tick labels bold and thicker (consistent with previous figures)
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.5)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    if 1:
        df = pd.read_csv("../../data/NCI60.csv")
        vis_doub_time_distribution(df, save_path="../../data/NCI60_doub_time_exp.png")

    symmetry_data = [
        {"Cancer_Type": "Breast", "Mean": 0.806, "Std": 0.067, "Count": None, "Metric": "symmetry"},
        {
            "Cancer_Type": "Glioblastoma (U87MG)",
            "Mean": 0.91,
            "Std": 0.11,
            "Count": None,
            "Metric": "symmetry",
        },
    ]
    create_bar_plot_from_manual_data(
        symmetry_data,
        y_label="Symmetry [-]",
        color="#486b45",
        save_path="../../data/symmetry_exp.png",
    )
    colony_growth_data = [
        {
            "Cancer_Type": "Breast",
            "Mean": 18.3,
            "Std": None,
            "Count": None,
            "Metric": "colony_growth_rate",
        },
        {
            "Cancer_Type": "Glioblastoma (U87MG)",
            "Mean": 35.0,
            "Std": None,
            "Count": None,
            "Metric": "colony_growth_rate",
        },
        {
            "Cancer_Type": "Glioblastoma (GBP03)",
            "Mean": 56.8815,
            "Std": None,
            "Count": None,
            "Metric": "colony_growth_rate",
        },
    ]
    create_bar_plot_from_manual_data(
        colony_growth_data,
        y_label="Colony Growth (µm/day)",
        color="#545aab",
        save_path="../../data/colony_growth_rate_exp.png",
    )


if __name__ == "__main__":
    main()
