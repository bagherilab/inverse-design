import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.config.parameter_config import PARAMS_DEFAULTS, SOURCE_PARAMS_DEFAULTS
import json
import seaborn as sns
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.manifold import TSNE
from typing import List, Optional, Tuple
from inverse_design.analyze.abc_metrics_comparison import analyze_iterations
from inverse_design.io.scenarios import combined_grid_n512_breast
from inverse_design.io import ArcadeRunLayout
from inverse_design.plotting.colormap import PEAK_COLORS
from inverse_design.plotting.theme import apply_publication_style
from inverse_design.vis.vis_cache import load_cache_payload, select_simplified_dirs

TOTAL_PARAMS_DEFAULTS = {**PARAMS_DEFAULTS, **SOURCE_PARAMS_DEFAULTS}
apply_publication_style(font_size=12, axes_linewidth=1.0)
# Categorize points to nearest peaks and plot with different shapes
peak_colors = list(PEAK_COLORS)
peak_markers = [
    "o",
    "s",
    "^",
    "D",
    "*",
    "p",
]  # circle, square, triangle up, diamond, triangle down, pentagon
# peak_markers = ['o'] * 1 + ['s'] * 1
# peak_colors = ['#e41a1c'] * 1 + ['#377eb8'] * 1
marker_to_size = {"o": 18, "s": 18, "^": 37, "D": 18, "*": 70, "p": 35}


def plot_box_swarm_charts(
    original_df: pd.DataFrame,
    simplified_dfs: List[pd.DataFrame],
    scenario_names: List[str],
    target_peak_idx: int,
    figsize_per_subplot: tuple = (4, 3),
    max_rows: int = 5,
    swarm_size: float = 3,
    box_alpha: float = 0.7,
    save_path: str = None,
    dpi: int = 300,
):
    """
    Create box and swarm plots for each parameter across different scenarios.

    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataframe with all parameters as columns
    simplified_dfs : List[pd.DataFrame]
        List of simplified dataframes for each scenario
    scenario_names : List[str]
        Names for each scenario (including original)
    figsize_per_subplot : tuple, default (4, 3)
        Size of each individual subplot
    max_cols : int, default 5
        Maximum number of columns in the subplot grid
    swarm_size : float, default 3
        Size of points in swarm plot
    box_alpha : float, default 0.7
        Transparency of box plots
    save_path : str, optional
        Path to save the figure
    dpi : int, default 300
        Resolution for saved figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object containing all subplots
    axes : numpy.ndarray
        Array of subplot axes
    """

    # Validate inputs
    if len(simplified_dfs) + 1 != len(scenario_names):
        raise ValueError(
            "Number of scenario names should equal number of simplified_dfs + 1 (for original)"
        )

    # Get parameter names from original dataframe
    param_names = original_df.columns.tolist()
    n_params = len(param_names)

    # Find peak 1 data for the original df
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        original_df[param_names].values
    )
    point_assignments = []
    for point in pca_result[:, :2]:
        distances = [np.sqrt(np.sum((point - peak) ** 2)) for peak in peak_positions]
        nearest_peak = np.argmin(distances)
        point_assignments.append(nearest_peak)
    point_assignments = np.array(point_assignments)
    mask = point_assignments == target_peak_idx

    # Prepare data for plotting
    all_dfs = [original_df[mask]] + simplified_dfs

    # Create combined dataframe for easier plotting
    combined_data = []
    for i, (df, scenario) in enumerate(zip(all_dfs, scenario_names)):
        for param in param_names:
            if param in df.columns:
                param_data = df[param].dropna()
                for value in param_data:
                    combined_data.append({"Parameter": param, "Scenario": scenario, "Value": value})

    plot_df = pd.DataFrame(combined_data)

    # Calculate subplot grid dimensions
    n_rows = min(max_rows, n_params)
    n_cols = math.ceil(n_params / n_rows)

    # Calculate figure size
    fig_width = n_cols * figsize_per_subplot[0]
    fig_height = n_rows * figsize_per_subplot[1]

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Handle case where there's only one subplot
    if n_params == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten axes for easier iteration
    axes_flat = axes.flatten() if n_params > 1 else axes

    # Set color palette
    colors = sns.color_palette("Set2", len(scenario_names))

    # Create plots for each parameter
    for i, param in enumerate(param_names):
        print(f"Plotting {param}, {i+1}/{len(param_names)}")
        ax = axes_flat[i]

        # Filter data for current parameter
        param_data = plot_df[plot_df["Parameter"] == param]

        if len(param_data) == 0:
            ax.text(
                0.5, 0.5, f"No data for\n{param}", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(param)
            continue

        # Create box plot
        box_plot = sns.boxplot(data=param_data, x="Scenario", y="Value", ax=ax, palette=colors)

        # Create swarm plot overlay
        # swarm_plot = sns.swarmplot(data=param_data, x='Scenario', y='Value',
        #                          ax=ax, palette=colors, size=swarm_size, alpha=0.8)

        # Customize subplot
        ax.set_title(param, fontsize=10, fontweight="bold")
        ax.set_xlabel("Scenario", fontsize=9)
        ax.set_ylabel("Value", fontsize=9)

        # Rotate x-axis labels if they're long
        max_label_length = max(len(name) for name in scenario_names)
        if max_label_length > 8:
            ax.tick_params(axis="x", rotation=45)

        # Adjust tick label size
        ax.tick_params(axis="both", labelsize=8)

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig, axes


def plot_pca_with_simplified(
    posterior_data,
    simplified_posterior_dfs,
    n_components=2,
    simplified_labels=None,
    fig=None,
    ax=None,
    save_path=None,
    peak_positions=None,
):
    """
    Create a PCA plot with density contours and peaks.

    Args:
        posterior_data: DataFrame containing posterior samples
        param_subset: List of parameters to include in PCA (default: all parameters)
        n_components: Number of PCA components to plot (default: 2)
        fig: Optional figure object
        ax: Optional axis object

    Returns:
        tuple: (fig, ax) - Figure and axis objects
    """
    if simplified_labels is None:
        simplified_labels = [f"Scenario {i+1}" for i in range(len(simplified_posterior_dfs))]
    param_names = posterior_data.columns.tolist()
    # Prepare data
    data = posterior_data[param_names].values

    # Perform PCA and find peaks
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        data, n_components
    )
    scaler = StandardScaler()
    scaler.fit(data)

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3))

    if len(peak_positions) > 0:
        # Assign each point to the nearest peak
        point_assignments = []
        for point in pca_result[:, :2]:
            distances = [np.sqrt(np.sum((point - peak) ** 2)) for peak in peak_positions]
            nearest_peak = np.argmin(distances)
            point_assignments.append(nearest_peak)
        point_assignments = np.array(point_assignments)

        # Plot points grouped by their nearest peak
        for i in range(len(peak_positions)):
            mask = point_assignments == i

            if np.any(mask):
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    color=None,
                    alpha=0.75,
                    s=5,
                    marker="o",
                    facecolor="none",
                    edgecolors="black",
                    linewidths=0.5,
                )
    else:
        ax.scatter(
            pca_result[:, 0], pca_result[:, 1], c="#A0522D", alpha=0.75, s=30, edgecolors="none"
        )
    ax.scatter(
        [],
        [],
        c=None,
        alpha=0.75,
        s=15,
        facecolor="none",
        edgecolors="black",
        label=simplified_labels[0],
        linewidths=0.5,
    )

    contour_lines = ax.contour(X, Y, Z, levels=15, colors="black", alpha=0.4, linewidths=1.0)
    for i, simplified_posterior_df in enumerate(simplified_posterior_dfs):
        # if param_names is not in simplified_posterior_df.columns, add it with default values
        for param in param_names:
            if param not in simplified_posterior_df.columns:
                simplified_posterior_df[param] = TOTAL_PARAMS_DEFAULTS[param]
        # sort simplified_posterior columns to match param_names
        simplified_posterior_df = simplified_posterior_df[param_names]
        simplified_data = simplified_posterior_df[param_names].values
        simplified_in_pca = pca.transform(scaler.transform(simplified_data))
        color = peak_colors[i]
        ax.scatter(
            simplified_in_pca[:, 0],
            simplified_in_pca[:, 1],
            color=color,
            s=15,
            marker=peak_markers[i],
            facecolor="none",
            edgecolors=color,
            linewidths=0.5,
            zorder=10,
            label=simplified_labels[i + 1],
        )

    #
    ax.legend(loc="upper right", fontsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Set labels with bold formatting
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12, fontweight="bold"
    )
    ax.set_axisbelow(True)

    # Make tick labels bold and thicker (consistent with previous figures)
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig, ax


def cluster_emergent_behaviors(
    behavior_data: pd.DataFrame,
    point_assignments: np.ndarray,
    method: str = "kmeans",
    n_clusters: int = 3,
    dimensionality_reduction: str = "pca",
    n_components: int = 2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, object, object]:
    """
    Cluster samples based on their emergent behaviors.

    Args:
        behavior_data: DataFrame with emergent behaviors (rows=samples, cols=behaviors)
        method: Clustering method ('kmeans', 'hdbscan')
        n_clusters: Number of clusters (for kmeans)
        dimensionality_reduction: Method for dim reduction ('pca', 'tsne', 'umap')
        n_components: Number of components for dimensionality reduction
        random_state: Random state for reproducibility

    Returns:
        cluster_labels: Array of cluster assignments
        behavior_embedding: 2D embedding of behaviors
        clusterer: Fitted clustering object
        reducer: Fitted dimensionality reduction object
    """
    behavior_clean = behavior_data.dropna()
    valid_indices = behavior_clean.index
    print(f"valid_indices: {valid_indices}")
    if len(behavior_clean) == 0:
        raise ValueError("No valid behavior data after removing NaN values")

    print(
        f"Clustering {len(behavior_clean)} valid samples with {len(behavior_clean.columns)} behavior features"
    )

    # Standardize behavior data
    scaler = StandardScaler()
    behavior_scaled = scaler.fit_transform(behavior_clean)

    # Dimensionality reduction for visualization
    if dimensionality_reduction == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
    elif dimensionality_reduction == "tsne":
        reducer = TSNE(
            n_components=n_components,
            random_state=random_state,
            perplexity=min(30, len(behavior_clean) // 4),
        )
    else:
        raise ValueError("dimensionality_reduction must be 'pca', or 'tsne'")

    behavior_embedding = reducer.fit_transform(behavior_scaled)
    fig, ax = plt.subplots(figsize=(8, 6))
    # color code by point_assignments
    colors = [
        "red",
        "blue",
        "green",
        "yellow",
        "purple",
        "orange",
        "brown",
        "pink",
        "gray",
        "black",
    ]
    for i in range(len(np.unique(point_assignments))):
        ax.scatter(
            behavior_embedding[point_assignments == i, 0],
            behavior_embedding[point_assignments == i, 1],
            c=colors[i],
            s=10,
        )
    # ax.scatter(behavior_embedding[:, 0], behavior_embedding[:, 1], c='black', s=10)
    plt.show()
    # plt.savefig(f'{base_simplified_07_dir}/posterior_plots/behavior_embedding.png', dpi=300, bbox_inches='tight')
    # plt.close()
    # Clustering
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels_clean = clusterer.fit_predict(behavior_scaled)
    elif method == "hdbscan":
        clusterer = HDBSCAN(min_cluster_size=max(2, len(behavior_clean) // 10))
        cluster_labels_clean = clusterer.fit_predict(behavior_scaled)
    else:
        raise ValueError("method must be 'kmeans' or 'hdbscan'")

    # Expand cluster labels to include NaN samples
    cluster_labels = np.full(len(behavior_data), -1)  # -1 for invalid/NaN samples
    print(f"cluster_labels_clean: {cluster_labels_clean.shape}")
    print(f"valid_indices: {valid_indices.shape}")
    print(f"cluster_labels: {cluster_labels.shape}")
    cluster_labels[valid_indices] = cluster_labels_clean

    return cluster_labels, behavior_embedding, clusterer, reducer


def plot_behavior_colored_parameter_pca(
    posterior_data: pd.DataFrame,
    simplified_posterior_dfs: List[pd.DataFrame],
    final_metrics_df: pd.DataFrame,
    target_peak_idx: int = 0,
    clustering_method: str = "kmeans",
    n_behavior_clusters: int = 3,
    behavior_dim_reduction: str = "pca",
    metric_names: List[str] = None,
    n_components: int = 2,
    fig=None,
    ax=None,
    save_path: Optional[str] = None,
    show_behavior_plot: bool = True,
    simplified_labels: Optional[List[str]] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a parameter PCA plot colored by emergent behavior clusters.
    Uses existing perform_pca_and_find_peaks function to find peaks in parameter space,
    then applies the same peak assignments to behavior data.

    Args:
        posterior_data: DataFrame with parameter samples from original model
        simplified_posterior_dfs: List of DataFrames with simplified model samples
        final_metrics_df: DataFrame with pre-calculated emergent behaviors/metrics
        perform_pca_and_find_peaks: Function that performs PCA and finds peaks
        target_peak_idx: Index of peak to focus on for analysis
        clustering_method: Method for clustering behaviors ('kmeans' or 'hdbscan')
        n_behavior_clusters: Number of clusters for behavior clustering
        behavior_dim_reduction: Dimensionality reduction for behaviors ('pca', 'tsne', 'umap')
        n_components: Number of PCA components for parameter space
        fig, ax: Optional matplotlib figure and axis objects
        save_path: Optional path to save the figure
        show_behavior_plot: Whether to show the behavior space plot as well
        simplified_labels: Optional labels for simplified models

    Returns:
        Figure and axis objects, plus clustering results
    """

    print("Step 1: Finding peaks in parameter space...")

    # Ensure the indices match between posterior_data and final_metrics_df
    if len(posterior_data) != len(final_metrics_df):
        print(
            f"Warning: posterior_data has {len(posterior_data)} samples, "
            f"final_metrics_df has {len(final_metrics_df)} samples"
        )
        # Use intersection of indices
        common_indices = posterior_data.index.intersection(final_metrics_df.index)
        posterior_data = posterior_data.loc[common_indices]
        final_metrics_df = final_metrics_df.loc[common_indices]
        print(f"Using {len(common_indices)} common samples")

    # Get parameter names and prepare data for PCA
    param_names = posterior_data.columns.tolist()
    data = posterior_data[param_names].values

    # Perform PCA and find peaks using your existing function
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        data, n_components
    )

    # Assign each point to the nearest peak (same as your existing code)
    point_assignments = []
    if len(peak_positions) > 0:
        for point in pca_result[:, :2]:
            distances = [np.sqrt(np.sum((point - peak) ** 2)) for peak in peak_positions]
            nearest_peak = np.argmin(distances)
            point_assignments.append(nearest_peak)
        point_assignments = np.array(point_assignments)
    else:
        # No peaks found, assign all to peak 0
        point_assignments = np.zeros(len(pca_result))

    print(f"Found {len(peak_positions)} peaks in parameter space")
    print(f"Peak assignments: {np.bincount(point_assignments.astype(int))}")

    # Filter for target peak
    target_peak_mask = point_assignments == target_peak_idx
    print(f"Focusing on peak {target_peak_idx}: {target_peak_mask.sum()} samples")

    if target_peak_mask.sum() == 0:
        raise ValueError(f"No samples found for target peak {target_peak_idx}")

    # Filter both parameter and behavior data for target peak
    posterior_data_filtered = posterior_data  # [target_peak_mask]
    behavior_data_filtered = final_metrics_df  # [target_peak_mask]
    pca_result_filtered = pca_result  # [target_peak_mask]

    print("Step 2: Using pre-calculated emergent behaviors from final_metrics_df...")

    # Remove non-behavior columns (like sample_id, index, etc.)
    behavior_columns = [col for col in behavior_data_filtered.columns if col in metric_names]
    behavior_data = behavior_data_filtered[behavior_columns]

    print(f"Using {len(behavior_columns)} behavior features: {behavior_columns}")

    print("Step 3: Clustering based on emergent behaviors...")

    # Cluster based on emergent behaviors (only target peak data)
    cluster_labels, behavior_embedding, clusterer, behavior_reducer = cluster_emergent_behaviors(
        behavior_data,
        point_assignments,
        method=clustering_method,
        n_clusters=n_behavior_clusters,
        dimensionality_reduction=behavior_dim_reduction,
        n_components=2,
    )

    unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
    print(f"Found {len(unique_clusters)} behavior clusters")

    print("Step 4: Processing simplified data...")

    # Use the same scaler and PCA from original analysis
    scaler = StandardScaler()
    scaler.fit(data)

    # Transform simplified data to parameter PCA space
    simplified_param_embeddings = []
    for simplified_df in simplified_posterior_dfs:
        simplified_data = simplified_df[param_names].values
        simplified_in_pca = pca.transform(scaler.transform(simplified_data))
        simplified_param_embeddings.append(simplified_in_pca)

    print("Step 5: Creating visualization...")

    # Create figure
    if show_behavior_plot:
        if fig is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            axes = fig.get_axes()
            ax1, ax2 = axes[0], axes[1] if len(axes) > 1 else axes[0]
    else:
        if fig is None or ax is None:
            fig, ax1 = plt.subplots(figsize=(8, 6))
        else:
            ax1 = ax

    # Color palette for clusters
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(unique_clusters), 1)))
    cluster_colors = {cluster_id: colors[i] for i, cluster_id in enumerate(unique_clusters)}
    cluster_colors[-1] = "lightgray"  # Color for invalid/NaN samples

    # Plot parameter PCA colored by behavior clusters
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        if np.any(mask):
            ax1.scatter(
                pca_result_filtered[mask, 0],
                pca_result_filtered[mask, 1],
                c=[cluster_colors[cluster_id]],
                alpha=0.7,
                s=30,
                label=f"Behavior Cluster {cluster_id}",
                edgecolors="black",
                linewidths=0.5,
            )

    # Plot invalid samples if any
    invalid_mask = cluster_labels == -1
    if np.any(invalid_mask):
        ax1.scatter(
            pca_result_filtered[invalid_mask, 0],
            pca_result_filtered[invalid_mask, 1],
            c="lightgray",
            alpha=0.3,
            s=20,
            label="Invalid samples",
            edgecolors="black",
            linewidths=0.3,
        )

    # Add density contours from original analysis
    ax1.contour(X, Y, Z, levels=15, colors="black", alpha=0.4, linewidths=1.0)

    # Mark the target peak
    if len(peak_positions) > target_peak_idx:
        ax1.scatter(
            peak_positions[target_peak_idx, 0],
            peak_positions[target_peak_idx, 1],
            c="red",
            s=200,
            marker="*",
            label=f"Target Peak {target_peak_idx}",
            edgecolors="darkred",
            linewidths=2,
            zorder=15,
        )

    # Plot simplified models
    markers = ["s", "^", "D", "v", "<", ">", "p", "*", "h", "H"]
    for i, simplified_embedding in enumerate(simplified_param_embeddings):
        marker = markers[i % len(markers)]
        label = simplified_labels[i] if simplified_labels else f"Simplified Model {i+1}"
        ax1.scatter(
            simplified_embedding[:, 0],
            simplified_embedding[:, 1],
            c="orange",
            s=80,
            marker=marker,
            alpha=0.9,
            label=label,
            edgecolors="darkorange",
            linewidths=1.5,
            zorder=10,
        )

    # Style parameter PCA plot
    ax1.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12, fontweight="bold"
    )
    ax1.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12, fontweight="bold"
    )
    ax1.set_title(
        f"Parameter PCA - Peak {target_peak_idx}\n(Colored by Behavior Clusters)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Style axes
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.tick_params(axis="both", which="major", labelsize=10)

    # Plot behavior space if requested
    if show_behavior_plot:
        valid_mask = cluster_labels >= 0
        valid_indices = np.where(valid_mask)[0]

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            if np.any(cluster_mask):
                # Find valid behavior embedding indices for this cluster
                cluster_indices = np.where(cluster_mask)[0]
                # Map to behavior embedding indices (only valid samples were used)
                behavior_indices = [
                    i for i, orig_idx in enumerate(valid_indices) if orig_idx in cluster_indices
                ]

                if behavior_indices:
                    ax2.scatter(
                        behavior_embedding[behavior_indices, 0],
                        behavior_embedding[behavior_indices, 1],
                        c=[cluster_colors[cluster_id]],
                        alpha=0.7,
                        s=30,
                        label=f"Cluster {cluster_id}",
                        edgecolors="black",
                        linewidths=0.5,
                    )

        # Style behavior plot
        reducer_name = behavior_dim_reduction.upper()
        ax2.set_xlabel(f"{reducer_name}1", fontsize=12, fontweight="bold")
        ax2.set_ylabel(f"{reducer_name}2", fontsize=12, fontweight="bold")
        ax2.set_title(
            f"Behavior Space - Peak {target_peak_idx}\n(Clustering Basis)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Style axes
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    # Return results
    results = {
        "cluster_labels": cluster_labels,
        "behavior_embedding": behavior_embedding,
        "param_embedding": pca_result_filtered,
        "param_pca": pca,
        "behavior_reducer": behavior_reducer,
        "clusterer": clusterer,
        "unique_clusters": unique_clusters,
        "peak_positions": peak_positions,
        "point_assignments": point_assignments,
        "target_peak_mask": target_peak_mask,
    }

    return fig, ax1, results


def plot_box_swarm_charts_lr_analysis(
    original_df: pd.DataFrame,
    simplified_dfs: List[pd.DataFrame],
    scenario_names: List[str],
    best_fit_samples_idx_list: List[int],
    lr_prediction_summaries: List[dict],
    peak_idx: int,
    figsize_per_subplot: tuple = (2, 1),  # Changed to square
    max_rows: int = 5,
    swarm_size: float = 3,
    box_alpha: float = 0.7,
    save_path: str = None,
    dpi: int = 300,
):
    """
    Create bar plots comparing Original vs each simplified scenario separately.
    Creates 2 figures per scenario: individual parameters and derived parameters (from LR models).

    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataframe with all parameters as columns
    simplified_dfs : List[pd.DataFrame]
        List of simplified dataframes for each scenario
    scenario_names : List[str]
        Names for each scenario (including original)
    best_fit_samples_idx_list : List[int]
        List of indices for best fit samples for each scenario
    lr_prediction_summaries : List[dict]
        List of linear regression prediction summary dictionaries, one for each scenario (excluding original)
    peak_idx : int
        Index of the peak to use (maintained for compatibility)
    figsize_per_subplot : tuple, default (3, 3)
        Size of each individual subplot (now square)
    max_rows : int, default 5
        Maximum number of rows in the subplot grid
    swarm_size : float, default 3
        Size of points in swarm plot (unused in bar plots but kept for compatibility)
    box_alpha : float, default 0.7
        Transparency of box plots
    save_path : str, optional
        Base path to save the figures (will be modified for each comparison)
    dpi : int, default 300
        Resolution for saved figure

    Returns:
    --------
    all_figures : dict
        Dictionary containing all created figures organized by scenario and type
    """

    # Validate inputs
    if len(simplified_dfs) + 1 != len(scenario_names):
        raise ValueError(
            "Number of scenario names should equal number of simplified_dfs + 1 (for original)"
        )

    if len(best_fit_samples_idx_list) != len(simplified_dfs) + 1:
        raise ValueError("Number of best_fit_samples_idx_list should equal number of scenarios")

    if len(lr_prediction_summaries) != len(simplified_dfs):
        raise ValueError(
            "Number of lr_prediction_summaries should equal number of simplified scenarios"
        )

    # Get parameter names from original dataframe
    param_names = original_df.columns.tolist()

    # Find peak data for the original df
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        original_df[param_names].values
    )
    point_assignments = []
    for point in pca_result[:, :2]:
        distances = [np.sqrt(np.sum((point - peak) ** 2)) for peak in peak_positions]
        nearest_peak = np.argmin(distances)
        point_assignments.append(nearest_peak)
    point_assignments = np.array(point_assignments)
    mask = point_assignments == peak_idx

    # Prepare original data
    original_masked_df = original_df[mask].reset_index(drop=True)
    original_best_idx = best_fit_samples_idx_list[0]

    if original_best_idx >= len(original_masked_df):
        raise ValueError(f"Original best_idx {original_best_idx} out of range")

    original_selected = original_masked_df.iloc[original_best_idx : original_best_idx + 1]

    all_figures = {}

    # Create comparison plots for each simplified scenario
    for scenario_idx, (simplified_df, scenario_name, lr_summary) in enumerate(
        zip(simplified_dfs, scenario_names[1:], lr_prediction_summaries), 1
    ):

        print(f"\nProcessing comparison: Original vs {scenario_name}")

        # Get best fit sample for this scenario
        scenario_best_idx = best_fit_samples_idx_list[scenario_idx]
        if scenario_best_idx >= len(simplified_df):
            print(f"Warning: best_idx {scenario_best_idx} out of range for {scenario_name}")
            continue

        scenario_selected = simplified_df.iloc[scenario_best_idx : scenario_best_idx + 1]

        # Extract parameter categorization from LR summary
        param_categorization = lr_summary["parameter_categorization"]
        prediction_models = lr_summary["prediction_models"]["models"]

        # Get independent parameters
        independent_params = [
            p["parameter_name"]
            for p in param_categorization["independent_parameters"]["parameters"]
        ]

        # Get root parameters
        root_params = [
            p["parameter_name"] for p in param_categorization["root_parameters"]["parameters"]
        ]

        # Get derived parameters
        derived_params = [
            p["parameter_name"] for p in param_categorization["derived_parameters"]["parameters"]
        ]

        # Create prediction chains from the models
        # Build a more sophisticated chain structure that handles derived->derived predictions
        predictor_to_targets = {}  # Maps predictor -> list of targets
        target_to_predictor = {}  # Maps target -> its predictor
        param_to_r2 = {}

        for model in prediction_models:
            predictor_param = model["predictor_parameter"]["name"]
            target_param = model["target_parameter"]["name"]
            r2_value = model["prediction_quality"]["r_squared"]

            if predictor_param not in predictor_to_targets:
                predictor_to_targets[predictor_param] = []

            predictor_to_targets[predictor_param].append(target_param)
            target_to_predictor[target_param] = predictor_param
            param_to_r2[target_param] = r2_value

        # Build complete chains starting from each root parameter
        def build_chain_from_root(root_param, visited=None):
            """Recursively build a chain from a root parameter"""
            if visited is None:
                visited = set()

            if root_param in visited:
                return []  # Avoid cycles

            visited.add(root_param)
            chain = [root_param]

            # Find all parameters that this root/parameter predicts
            if root_param in predictor_to_targets:
                for target in predictor_to_targets[root_param]:
                    # Recursively build chains for each target
                    sub_chain = build_chain_from_root(target, visited.copy())
                    if sub_chain:
                        chain.extend(sub_chain)

            return chain

        # Build all chains starting from root parameters
        all_chains = []
        for root_param in root_params:
            chain = build_chain_from_root(root_param)
            if len(chain) > 1:  # Only include chains with at least one derived parameter
                all_chains.append(chain)

        # Convert to the expected format with r2_values
        chain_sequences = []
        for chain in all_chains:
            r2_values = []
            for param in chain[1:]:  # Skip root parameter
                if param in param_to_r2:
                    r2_values.append(param_to_r2[param])

            chain_sequences.append({"sequence": chain, "r2_values": r2_values})

        # Find any derived parameters that might not be in chains (orphaned)
        all_chained_params = set()
        for chain_seq in chain_sequences:
            all_chained_params.update(chain_seq["sequence"])

        orphaned_derived = [p for p in derived_params if p not in all_chained_params]

        if orphaned_derived:
            print(f"Warning: Found orphaned derived parameters: {orphaned_derived}")
            # Add them as individual chains
            for param in orphaned_derived:
                if param in target_to_predictor:
                    predictor = target_to_predictor[param]
                    r2_value = param_to_r2.get(param, 0.0)
                    chain_sequences.append(
                        {"sequence": [predictor, param], "r2_values": [r2_value]}
                    )

        print(f"Independent parameters ({len(independent_params)}): {independent_params}")
        print(f"Root parameters ({len(root_params)}): {root_params}")
        print(f"Derived parameters ({len(derived_params)}): {derived_params}")
        print(f"Individual plot: {len(independent_params)} subplots")
        print(f"Derived plot: {len(root_params) + len(derived_params)} subplots (in chains)")
        print(
            f"Total parameters to plot: {len(independent_params) + len(root_params) + len(derived_params)}"
        )
        print(f"Chain sequences: {len(chain_sequences)} chains")
        for i, chain_seq in enumerate(chain_sequences):
            print(f"  Chain {i+1}: {chain_seq['sequence']}")

        def create_comparison_data(param_list):
            combined_data = []
            comparison_scenarios = ["Original", scenario_name]
            comparison_data = [original_selected, scenario_selected]

            for scenario_data, comparison_scenario in zip(comparison_data, comparison_scenarios):
                for param in param_list:
                    if param in scenario_data.columns:
                        value = scenario_data[param].iloc[0]
                        if not pd.isna(value):
                            combined_data.append(
                                {
                                    "Parameter": param,
                                    "Scenario": comparison_scenario,
                                    "Value": value,
                                }
                            )
            return pd.DataFrame(combined_data)

        def create_individual_plot():
            """Create plot for independent parameters only"""
            if not independent_params:
                return None, None

            plot_df = create_comparison_data(independent_params)
            n_params = len(independent_params)

            # Calculate subplot grid dimensions
            n_rows = min(max_rows, math.ceil(n_params / math.ceil(n_params / max_rows)))
            n_cols = math.ceil(n_params / n_rows)

            # Calculate figure size (square subplots) with 1.5x height spacing
            spacing_factor = 1.5
            fig_width = n_cols * figsize_per_subplot[0]
            fig_height = n_rows * figsize_per_subplot[1] * spacing_factor

            # Create subplots with extra space for titles
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

            # Handle case where there's only one subplot
            if n_params == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)

            # Flatten axes for easier iteration
            axes_flat = axes.flatten() if n_params > 1 else axes

            # Create bar plots for each parameter
            for i, param in enumerate(independent_params):
                ax = axes_flat[i]

                # Filter data for current parameter
                param_data = plot_df[plot_df["Parameter"] == param]

                if len(param_data) == 0:
                    ax.text(
                        0.5,
                        0.3,  # Lower position to account for title space
                        f"No data for\n{param}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(param)
                    continue

                # Create bar plot - constrain to bottom 50% of the subplot
                scenarios = param_data["Scenario"].tolist()
                values = param_data["Value"].tolist()

                bars = ax.bar(
                    scenarios,
                    values,
                    alpha=box_alpha,
                    color="white",
                    edgecolor="black",
                    linewidth=1.5,
                    capsize=5,
                )
                # Add hatches to the bars other than the original
                for bar in bars[1:]:
                    bar.set_hatch("//")

                # Customize subplot
                # Format parameter names and split if too long
                title = param.replace("_MU", " ").replace("_", " ").capitalize()
                if len(title) > 14:
                    words = title.split()
                    mid = len(words) // 2
                    title = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.grid(True, alpha=0.3, axis="y")
                ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

                # Bold tick labels for both x and y axes
                for label_tick in ax.get_xticklabels():
                    label_tick.set_fontweight("bold")
                for label_tick in ax.get_yticklabels():
                    label_tick.set_fontweight("bold")

                # Set y-axis - keep bars in bottom portion of subplot
                if values:
                    y_min, y_max = min(values), max(values)
                    if y_min >= 0:
                        ax.set_ylim(0, y_max * 1.1)
                    else:
                        ax.set_ylim(y_min * 1.1, y_max * 1.1)

            # Hide empty subplots
            for i in range(n_params, len(axes_flat)):
                axes_flat[i].set_visible(False)

            # Adjust layout with increased vertical spacing (hspace increased to 2.25 = 1.5 * 1.5)
            plt.subplots_adjust(
                top=0.85, bottom=0.15, left=0.1, right=0.95, hspace=2.25, wspace=0.4
            )

            return fig, axes

        def create_derived_plot():
            """Create plot for root and derived parameters - ROOT at top, all derived at bottom with source legends"""
            if not chain_sequences:
                return None, None

            # First, let's build a complete graph structure
            all_params_in_chains = set()
            prediction_edges = []  # (predictor, target, r2_value)

            for model in prediction_models:
                predictor = model["predictor_parameter"]["name"]
                target = model["target_parameter"]["name"]
                r2_value = model["prediction_quality"]["r_squared"]
                prediction_edges.append((predictor, target, r2_value))
                all_params_in_chains.add(predictor)
                all_params_in_chains.add(target)

            # Simplified 2-level hierarchy:
            # Level 0: ROOT parameters (top row)
            # Level 1: ALL derived parameters (bottom row)

            params_with_predictors = {target for _, target, _ in prediction_edges}

            # Find root parameters (those that predict but aren't predicted)
            roots_in_chains = []
            derived_in_chains = []

            for param in all_params_in_chains:
                if param not in params_with_predictors:
                    roots_in_chains.append(param)
                else:
                    derived_in_chains.append(param)

            # Create predictor lookup for derived parameters
            param_predictors = {}
            for predictor, target, r2_value in prediction_edges:
                param_predictors[target] = (predictor, r2_value)

            # Calculate layout dimensions
            n_levels = 2  # Always 2 levels: ROOT and derived
            max_params_per_level_root = len(roots_in_chains)
            max_params_per_level_derived = min(
                5, len(derived_in_chains)
            )  # Max 5 columns for derived

            # For derived parameters, we might need multiple rows if more than 5 parameters
            n_derived_rows = math.ceil(len(derived_in_chains) / 5) if derived_in_chains else 1

            # Calculate figure dimensions (square subplots)
            max_cols = max(max_params_per_level_root, max_params_per_level_derived)
            fig_width = max_cols * figsize_per_subplot[0]
            fig_height = (1 + n_derived_rows) * figsize_per_subplot[
                1
            ]  # 1 row for ROOT + n rows for derived

            fig = plt.figure(figsize=(fig_width, fig_height))

            # Level 0: ROOT parameters
            for i, param in enumerate(roots_in_chains):
                x_pos = (i + 0.5) / max_cols if max_params_per_level_root > 0 else 0.5
                y_pos = (
                    1 - (0 + 0.5) / (1 + n_derived_rows) * 1.8
                )  # ROOT row, add *0.7 to compress vertically

                # Create subplot - square dimensions
                subplot_width = 0.8 / max_cols if max_cols > 0 else 0.8
                subplot_height = 0.8 / (1 + n_derived_rows)

                left = x_pos - subplot_width / 2
                bottom = y_pos - subplot_height / 2

                ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

                # Get data for this parameter
                plot_df = create_comparison_data([param])

                if len(plot_df) == 0:
                    ax.text(
                        0.5,
                        0.3,  # Lower position for title space
                        f"No data for\n{param}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(param)
                    continue

                # Create bar plot
                scenarios = plot_df["Scenario"].tolist()
                values = plot_df["Value"].tolist()

                bars = ax.bar(
                    scenarios,
                    values,
                    alpha=box_alpha,
                    color="white",
                    edgecolor="black",
                    linewidth=1.5,
                    capsize=5,
                )
                # Add hatches to the bars other than the original
                for bar in bars[1:]:
                    bar.set_hatch("//")

                # Bold tick labels for both x and y axes
                for label_tick in ax.get_xticklabels():
                    label_tick.set_fontweight("bold")
                for label_tick in ax.get_yticklabels():
                    label_tick.set_fontweight("bold")
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

                # Create title (just parameter name, no ROOT indication)
                param_display = param.replace("_MU", " ").replace("_", " ").capitalize()
                title = param_display

                # Customize subplot
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.tick_params(axis="both", labelsize=12, width=1)
                ax.grid(True, alpha=0.3, axis="y")
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

                # Set y-axis
                if values:
                    y_min, y_max = min(values), max(values)
                    if y_min >= 0:
                        ax.set_ylim(0, y_max * 1.1)
                    else:
                        ax.set_ylim(y_min * 1.1, y_max * 1.1)

                # Add ROOT legend (no color)
                ax.text(
                    0.02,
                    0.98,
                    "ROOT",
                    transform=ax.transAxes,
                    fontsize=6,
                    fontweight="bold",
                    verticalalignment="top",
                    horizontalalignment="left",
                    bbox=dict(
                        boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9
                    ),
                )

            # Level 1: ALL derived parameters (with max 5 columns per row)
            for i, param in enumerate(derived_in_chains):
                row = i // 5  # Which row (0, 1, 2, ...)
                col = i % 5  # Which column in that row (0, 1, 2, 3, 4)

                x_pos = (col + 0.5) / max_cols if max_params_per_level_root > 0 else 0.5
                y_pos = 1 - (1 + row + 0.5) / (1 + n_derived_rows) * 1.8  # Add *0.7 here too
                # Create subplot - square dimensions
                subplot_width = 0.8 / max_cols if max_cols > 0 else 0.8
                subplot_height = 0.8 / (1 + n_derived_rows)

                left = x_pos - subplot_width / 2
                bottom = y_pos - subplot_height / 2

                ax = fig.add_axes([left, bottom, subplot_width, subplot_height])

                # Get data for this parameter
                plot_df = create_comparison_data([param])

                if len(plot_df) == 0:
                    ax.text(
                        0.5,
                        0.3,  # Lower position for title space
                        f"No data for\n{param}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(param)
                    continue

                # Create bar plot
                scenarios = plot_df["Scenario"].tolist()
                values = plot_df["Value"].tolist()

                bars = ax.bar(
                    scenarios,
                    values,
                    alpha=box_alpha,
                    color="white",
                    edgecolor="black",
                    linewidth=1.5,
                    capsize=5,
                )
                # Add hatches to the bars other than the original
                for bar in bars[1:]:
                    bar.set_hatch("//")

                # Bold tick labels for both x and y axes
                for label_tick in ax.get_xticklabels():
                    label_tick.set_fontweight("bold")
                for label_tick in ax.get_yticklabels():
                    label_tick.set_fontweight("bold")
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

                # Create title (just parameter name with R² value)
                param_display = param.replace("_MU", " ").replace("_", " ").capitalize()
                if param in param_to_r2:
                    r2_value = param_to_r2[param]
                    title = f"{param_display}\n(R² = {r2_value:.3f})"
                else:
                    title = param_display

                # Customize subplot
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.tick_params(axis="both", labelsize=12, width=1)
                ax.grid(True, alpha=0.3, axis="y")
                ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

                # Set y-axis
                if values:
                    y_min, y_max = min(values), max(values)
                    if y_min >= 0:
                        ax.set_ylim(0, y_max * 1.1)
                    else:
                        ax.set_ylim(y_min * 1.1, y_max * 1.1)

                # Add source legend for derived parameters (no "from:" prefix, no color)
                if param in param_predictors:
                    predictor, r2_value = param_predictors[param]
                    predictor_display = predictor.replace("_MU", "").replace("_", " ")

                    # Add white background text box showing just the source name
                    ax.text(
                        0.02,
                        0.98,
                        predictor_display,
                        transform=ax.transAxes,
                        fontsize=6,
                        fontweight="bold",
                        verticalalignment="top",
                        horizontalalignment="left",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            facecolor="white",
                            edgecolor="black",
                            alpha=0.9,
                        ),
                    )

            # Add much more space between subplots to prevent title overlap
            return fig, None

        # Create the two types of plots
        fig_individual, axes_individual = create_individual_plot()
        fig_derived, axes_derived = create_derived_plot()

        # Store figures in results dictionary
        scenario_key = scenario_name.replace(" ", "_").replace("=", "").replace(".", "")
        all_figures[scenario_key] = {
            "individual": (fig_individual, axes_individual),
            "derived": (fig_derived, axes_derived),
        }

        # Save figures if path provided
        if save_path:
            base_path = save_path.rsplit(".", 1)[0]  # Remove extension
            extension = save_path.rsplit(".", 1)[1] if "." in save_path else "png"

            if fig_individual:
                individual_path = f"{base_path}_{scenario_key}_individual.{extension}"
                if not os.path.exists(os.path.dirname(individual_path)):
                    os.makedirs(os.path.dirname(individual_path))
                fig_individual.savefig(
                    individual_path, dpi=dpi, bbox_inches="tight", transparent=True
                )
                print(f"Individual parameters figure saved to: {individual_path}")

            if fig_derived:
                derived_path = f"{base_path}_{scenario_key}_derived.{extension}"
                if not os.path.exists(os.path.dirname(derived_path)):
                    os.makedirs(os.path.dirname(derived_path))
                fig_derived.savefig(derived_path, dpi=dpi, bbox_inches="tight", transparent=True)
                print(f"Derived parameters figure saved to: {derived_path}")

    return all_figures


def _create_parameter_comparison_plot(
    params,
    original_selected,
    scenario_selected,
    scenarios,
    figsize_per_subplot,
    param_type,
    scenario_name,
    save_path,
):
    """Helper function to create parameter comparison plots."""
    if not params:
        return

    n_params = len(params)
    n_rows = 2
    n_cols = math.ceil(n_params / n_rows)

    fig_width = n_cols * figsize_per_subplot[0]
    fig_height = n_rows * figsize_per_subplot[1]

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # Handle single subplot case
    if n_params == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()
    for i, param in enumerate(params):
        ax = axes_flat[i]
        original_value = original_selected[param].iloc[0]
        scenario_value = scenario_selected[param].iloc[0]
        values = [original_value, scenario_value]
        bars = ax.bar(
            scenarios, values, alpha=0.7, color="white", edgecolor="black", linewidth=1.5, capsize=5
        )

        for bar in bars[1:]:
            bar.set_hatch("//")

        # Format parameter names and split if too long
        title = param.replace("_MU", " ").replace("_", " ").capitalize()
        if len(title) > 14:
            words = title.split()
            mid = len(words) // 2
            title = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        # Bold tick labels for both x and y axes
        for label_tick in ax.get_xticklabels():
            label_tick.set_fontweight("bold")
        for label_tick in ax.get_yticklabels():
            label_tick.set_fontweight("bold")
        # Set y-axis
        if values:
            y_min, y_max = min(values), max(values)
            if y_min >= 0:
                ax.set_ylim(0, y_max * 1.1)
            else:
                ax.set_ylim(y_min * 1.1, y_max * 1.1)

    # Hide empty subplots
    for i in range(n_params, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(
            f"{save_path}_{scenario_name}_{param_type}.png",
            dpi=300,
            bbox_inches="tight",
            transparent=True,
        )
    plt.close(fig)


def plot_dendrogram_posterior(
    original_df,
    simplified_dfs,
    best_fit_samples_idx_list,
    den_analysis_list,
    scenario_names,
    peak_idx,
    save_path,
    figsize_per_subplot=(2, 2),
):
    # Get parameter names from original dataframe
    param_names = original_df.columns.tolist()

    # Find peak data for the original df
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        original_df[param_names].values
    )
    point_assignments = []
    for point in pca_result[:, :2]:
        distances = [np.sqrt(np.sum((point - peak) ** 2)) for peak in peak_positions]
        nearest_peak = np.argmin(distances)
        point_assignments.append(nearest_peak)
    point_assignments = np.array(point_assignments)
    mask = point_assignments == peak_idx
    # Prepare original data
    original_masked_df = original_df[mask].reset_index(drop=True)
    original_best_idx = best_fit_samples_idx_list[0]

    if original_best_idx >= len(original_masked_df):
        raise ValueError(f"Original best_idx {original_best_idx} out of range")

    original_selected = original_masked_df.iloc[original_best_idx : original_best_idx + 1]
    # Create comparison plots for each simplified scenario
    for scenario_idx, (simplified_df, scenario_name, den_analysis) in enumerate(
        zip(simplified_dfs, scenario_names[1:], den_analysis_list), 1
    ):

        print(f"\nProcessing comparison: Original vs {scenario_name}")

        # Get best fit sample for this scenario
        scenario_best_idx = best_fit_samples_idx_list[scenario_idx]
        scenario_selected = simplified_df.iloc[scenario_best_idx : scenario_best_idx + 1]

        rep_params = den_analysis["representative_parameters"]
        individual_params = den_analysis["individual_params"]

        scenarios = ["Original", scenario_name]

        # Create figures using helper function
        _create_parameter_comparison_plot(
            rep_params,
            original_selected,
            scenario_selected,
            scenarios,
            figsize_per_subplot,
            "representative",
            scenario_name,
            save_path,
        )

        _create_parameter_comparison_plot(
            individual_params,
            original_selected,
            scenario_selected,
            scenarios,
            figsize_per_subplot,
            "individual",
            scenario_name,
            save_path,
        )


# ---------------------------------------------------------------------------
# Figure 3 - Model Simplification (Option 1: streamlined two-row layout)
# ---------------------------------------------------------------------------


def _draw_importance_bars(
    ax: "plt.Axes",
    param_names: list,
    importance: list,
    cluster_color: str,
    cluster_label: str,
) -> None:
    """Horizontal importance bars with highest-importance params at the top."""
    cmap = plt.get_cmap("RdYlBu_r")
    n_params = len(param_names)
    colors = [cmap(float(value)) for value in importance]
    y_positions = np.arange(n_params)

    ax.barh(y_positions, importance, color=colors, edgecolor="none", height=0.7)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(
        [param.replace("_MU", "").replace("_", " ").capitalize() for param in param_names],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Normalized importance", fontsize=8)
    ax.set_title(cluster_label, color=cluster_color, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)


def _draw_simplification_grid(
    ax: "plt.Axes",
    param_names: list,
    dropped_dict: dict,
    threshold_labels: list,
    cluster_color: str,
) -> None:
    """Grid of params x thresholds; dropped cells shown with cluster-colored hatch."""
    from matplotlib.patches import Rectangle

    n_params = len(param_names)
    n_thresh = len(threshold_labels)

    for row_idx, param in enumerate(param_names):
        for col_idx, threshold_label in enumerate(threshold_labels):
            is_dropped = param in dropped_dict.get(threshold_label, set())
            rect = Rectangle(
                (col_idx - 0.5, row_idx - 0.5),
                1,
                1,
                facecolor=cluster_color if is_dropped else "white",
                edgecolor="lightgray",
                linewidth=0.5,
                alpha=0.20 if is_dropped else 1.0,
                zorder=1,
            )
            ax.add_patch(rect)
            if is_dropped:
                ax.add_patch(
                    Rectangle(
                        (col_idx - 0.5, row_idx - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor=cluster_color,
                        linewidth=0.5,
                        hatch="///",
                        zorder=2,
                    )
                )

    ax.set_xlim(-0.5, n_thresh - 0.5)
    ax.set_ylim(-0.5, n_params - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(n_thresh))
    ax.set_xticklabels(threshold_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(
        [param.replace("_MU", "").replace("_", " ").capitalize() for param in param_names],
        fontsize=7,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0, labelsize=7)


def _draw_accuracy_plot(
    ax: "plt.Axes",
    cluster_panels: list,
    threshold_labels: list,
) -> None:
    """Line plot of mean relative error vs simplification threshold."""
    for panel in cluster_panels:
        ax.plot(
            range(len(threshold_labels)),
            panel["errors"],
            marker="o",
            color=panel["color"],
            linewidth=1.8,
            markersize=4,
            label=panel["label"],
        )
        ax.axhline(
            panel["original_error"],
            color=panel["color"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.5,
        )

    ax.set_xticks(range(len(threshold_labels)))
    ax.set_xticklabels(threshold_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean relative error", fontsize=8)
    ax.set_xlabel("Simplification threshold", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.legend(frameon=False, fontsize=7)


def plot_simplification_figure(
    cluster_panels: list,
    threshold_labels: list,
    figsize: tuple = (7.5, 5.2),
    save_path: str | None = None,
):
    """Create the two-row model simplification summary figure.

    Row 0 shows horizontal parameter-importance bars per cluster. Row 1 shows
    dropped-parameter grids per cluster plus a central accuracy line plot.
    """
    n_clusters = len(cluster_panels)
    if n_clusters not in (1, 2):
        raise ValueError(f"n_clusters must be 1 or 2; got {n_clusters}")

    fig = plt.figure(figsize=figsize)

    gs_top = GridSpec(
        1,
        n_clusters,
        figure=fig,
        left=0.08,
        right=0.97,
        top=0.95,
        bottom=0.54,
        wspace=0.30,
    )
    axes_importance = [fig.add_subplot(gs_top[0, i]) for i in range(n_clusters)]

    if n_clusters == 2:
        gs_bottom = GridSpec(
            1,
            3,
            figure=fig,
            left=0.08,
            right=0.97,
            top=0.46,
            bottom=0.08,
            wspace=0.25,
            width_ratios=[3, 2, 3],
        )
        axes_grid = [fig.add_subplot(gs_bottom[0, 0]), fig.add_subplot(gs_bottom[0, 2])]
        ax_accuracy = fig.add_subplot(gs_bottom[0, 1])
    else:
        gs_bottom = GridSpec(
            1,
            2,
            figure=fig,
            left=0.08,
            right=0.97,
            top=0.46,
            bottom=0.08,
            wspace=0.25,
            width_ratios=[3, 2],
        )
        axes_grid = [fig.add_subplot(gs_bottom[0, 0])]
        ax_accuracy = fig.add_subplot(gs_bottom[0, 1])

    axes_dict = {}

    for idx, (panel, ax) in enumerate(zip(cluster_panels, axes_importance)):
        _draw_importance_bars(
            ax,
            panel["param_names"],
            panel["importance"],
            panel["color"],
            panel["label"],
        )
        axes_dict[f"imp_c{idx}"] = ax

    for idx, (panel, ax) in enumerate(zip(cluster_panels, axes_grid)):
        _draw_simplification_grid(
            ax,
            panel["param_names"],
            panel["dropped"],
            threshold_labels,
            panel["color"],
        )
        axes_dict[f"grid_c{idx}"] = ax

    _draw_accuracy_plot(ax_accuracy, cluster_panels, threshold_labels)
    axes_dict["accuracy"] = ax_accuracy

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes_dict


def main():
    # Create all plots
    warnings.filterwarnings("ignore")

    # Load path bundles from analyze/scenarios/registry.yaml (same as parameter_importance __main__)
    _paths = combined_grid_n512_breast()
    base_original_dir = _paths.base_original_dir

    save_path_folder = "../../../ARCADE_OUTPUT/simplified_models_vis"

    base_layout = ArcadeRunLayout.from_root(base_original_dir)
    posterior_df = pd.read_csv(base_layout.all_param_df_csv("iter_4"))
    final_metrics_df = pd.read_csv(base_layout.final_metrics_csv("iter_4"))
    with open(base_layout.targets_json(), encoding="utf-8") as target_file:
        target_metrics = json.load(target_file)

    drop_cols = [
        "input_folder",
        "X_SPACING",
        "Y_SPACING",
        "DISTANCE_TO_CENTER",
    ]  # , "CAPILLARY_DENSITY", "AUTOPHAGY_RATE_SIGMA"]
    posterior_df.drop(columns=drop_cols, inplace=True)
    metric_names = list(target_metrics.keys())
    simplified_methods = ["linear", "threshold"]
    simplified_method = simplified_methods[0]
    scenario_names = (
        ["Original", "r=0.9", "r=0.8", "r=0.7", "r=0.01"]
        if simplified_method == "linear"
        else ["Original", "t=1.0", "t=1.25", "t=1.5", "t=5.0"]
    )
    peak_idx = 2

    simplified_posterior_dirs = select_simplified_dirs(_paths, simplified_method, peak_idx)
    simplified_posterior_dfs = []
    for dir in simplified_posterior_dirs:
        simplified_layout = ArcadeRunLayout.from_root(dir)
        simplified_posterior_dfs.append(pd.read_csv(simplified_layout.all_param_df_csv("iter_4")))
    print("Generating visualization plots...")
    if 1:
        cache_root = Path(os.getenv("VIS_CACHE_ROOT", "results/vis_cache"))
        use_cache = os.getenv("VIS_USE_CACHE", "1").lower() not in {"0", "false", "no"}
        best_fit_samples_idx_list = None
        if use_cache:
            try:
                cache_payload = load_cache_payload(cache_root, simplified_method, peak_idx)
                best_fit_samples_idx_list = cache_payload["best_fit_samples_idx_list"]
                print(f"Loaded visualization cache: {cache_root}")
            except FileNotFoundError:
                print(
                    f"Cache payload not found for method={simplified_method}, peak={peak_idx + 1}; "
                    "falling back to analyze_iterations."
                )
            except KeyError as exc:
                print(f"Cache payload missing key {exc!s}; falling back to analyze_iterations.")

        if best_fit_samples_idx_list is None:
            all_simulation_metrics_list = []
            best_fit_samples_idx_list = []
            for idx, posterior_dir in enumerate([base_original_dir] + simplified_posterior_dirs):
                (
                    iterations,
                    mean_errors,
                    best_errors,
                    all_simulation_metrics,
                    all_distance_results,
                    best_fit_samples_idx,
                ) = analyze_iterations(
                    posterior_dir,
                    metric_names,
                    target_metrics,
                    n_iterations=5,
                    verbose=False,
                    find_peaks=idx == 0,
                    target_peak=peak_idx,
                )
                all_simulation_metrics_list.append(all_simulation_metrics)
                best_fit_samples_idx_list.append(
                    best_fit_samples_idx[-1]
                )  # best sample idx for the last iteration
        summary_metrics_paths = [
            ArcadeRunLayout.from_root(simplified_posterior_dir).iteration_subdir("iter_4")
            / "final_metrics_seed.csv"
            for simplified_posterior_dir in simplified_posterior_dirs
        ]
        seed_metrics_dfs = [
            pd.read_csv(summary_metrics_path) for summary_metrics_path in summary_metrics_paths
        ]
        if simplified_method == "linear":
            r_thresholds = [0.9, 0.8, 0.7, 0.01]

            chain_analysis_list = [
                json.load(
                    open(
                        f"{simplified_posterior_dir}/lr_predictions_r{r_thresholds[idx]}_p{peak_idx+1}.json"
                    )
                )
                for idx, simplified_posterior_dir in enumerate(simplified_posterior_dirs)
            ]
            plot_box_swarm_charts_lr_analysis(
                posterior_df,
                simplified_posterior_dfs,
                scenario_names,
                best_fit_samples_idx_list=best_fit_samples_idx_list,
                lr_prediction_summaries=chain_analysis_list,
                peak_idx=peak_idx,
                save_path=f"{save_path_folder}/bar_plot_{simplified_method}_p{peak_idx+1}_chain_analysis.png",
            )
        else:
            thresholds = [1.0, 1.25, 1.5, 5.0]
            den_analysis_list = [
                json.load(
                    open(
                        f"{simplified_posterior_dir}/redundancy_analysis_threshold_{thresholds[idx]}_p{peak_idx+1}.json"
                    )
                )
                for idx, simplified_posterior_dir in enumerate(simplified_posterior_dirs)
            ]
            plot_dendrogram_posterior(
                posterior_df,
                simplified_posterior_dfs,
                best_fit_samples_idx_list,
                den_analysis_list,
                scenario_names,
                peak_idx,
                save_path=f"{save_path_folder}/dendrogram_posterior_{simplified_method}_p{peak_idx+1}.png",
            )

    param_subset = posterior_df.columns.tolist()
    param_subset = [param for param in param_subset if not param.endswith("_SIGMA")]
    print("All visualizations have been created and saved!")


if __name__ == "__main__":
    main()
