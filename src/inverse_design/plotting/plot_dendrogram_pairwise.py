import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform

from inverse_design.plotting.theme import apply_publication_style


def colorize_ward_pairs(data, param_names, save_path=None, figsize=(8, 8), verbose=True):
    """
    Performs Ward hierarchical clustering and finds pairs of parameters
    with the lowest Ward distances. Colors these pairs in the dendrogram.

    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data for clustering
    param_names : array-like
        Names of the parameters corresponding to each sample
    save_path : str, optional
        Path to save the plot
    figsize : tuple, default=(8, 8)
        Figure size for the dendrogram plot
    verbose : bool, default=True
        Whether to print summary information

    Returns:
    --------
    pairs : list of tuples
        List of parameter pairs with their Ward distances, sorted by distance
    pair_indices : list of tuples
        List of sample index pairs corresponding to the parameter pairs
    """

    apply_publication_style(font_size=10, axes_linewidth=1.0)

    # Convert to numpy array if needed
    data = np.array(data)
    param_names = np.array(param_names)

    # Calculate pairwise Ward distances
    distances = pdist(data, metric="euclidean")
    # Square the distances for Ward linkage (Ward distance is squared Euclidean)
    ward_distances = distances**2

    # Convert to square matrix for easier indexing
    distance_matrix = squareform(ward_distances)

    # Find all pairs and their distances
    n_samples = len(data)
    pairs_with_distances = []

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            pairs_with_distances.append(
                ((param_names[i], param_names[j]), distance_matrix[i, j], (i, j))
            )

    # Sort pairs by Ward distance (lowest first)
    pairs_with_distances.sort(key=lambda x: x[1])

    # Extract sorted information
    pairs = [(pair[0], pair[1]) for pair in pairs_with_distances]
    pair_indices = [pair[2] for pair in pairs_with_distances]

    # Perform Ward linkage for dendrogram
    linkage_matrix = linkage(data, method="ward")

    # Create colors for the closest pairs
    peak_colors = ["#f6b192", "#a51429", "#67001f", "#d25849", "mediumorchid", "darkorange"]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    n_samples = len(data)

    # Create mapping from nodes to samples
    node_to_samples = create_node_to_samples_mapping(linkage_matrix, n_samples)

    # Get the top pairs to highlight (let's highlight top 3 pairs)
    top_pairs = pairs[: min(3, len(pairs))]
    top_pair_indices = pair_indices[: min(3, len(pairs))]

    # Create custom color function for dendrogram
    def link_color_func(k):
        """
        Color links that connect the closest pairs
        """
        if k in node_to_samples:
            samples_in_node = node_to_samples[k]

            # Check if this node represents one of our top pairs
            for pair_idx, (idx1, idx2) in enumerate(top_pair_indices):
                if len(samples_in_node) == 2 and set(samples_in_node) == {idx1, idx2}:
                    color_name = peak_colors[pair_idx % len(peak_colors)]
                    return matplotlib.colors.to_hex(color_name)

        return "gray"

    # Plot dendrogram
    dend = dendrogram(
        linkage_matrix,
        link_color_func=link_color_func,
        orientation="right",
        no_plot=False,
        labels=param_names,
        ax=ax,
    )

    # Style the axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Color the leaf labels for parameters in top pairs
    ylbls = ax.get_ymajorticklabels()
    highlighted_params = set()

    for pair_idx, (param1, param2) in enumerate(top_pairs):
        highlighted_params.add(param1)
        highlighted_params.add(param2)

    for lbl in ylbls:
        param = lbl.get_text()
        if param in highlighted_params:
            # Find which pair this parameter belongs to
            for pair_idx, (param1, param2) in enumerate(top_pairs):
                if param in [param1, param2]:
                    color = peak_colors[pair_idx % len(peak_colors)]
                    lbl.set_color(color)
                    lbl.set_fontweight("bold")
                    break
        else:
            lbl.set_color("gray")

    # Set labels
    ax.set_xlabel("Ward Distance", fontsize=12, fontweight="bold")
    ax.set_axisbelow(True)

    # Style tick labels
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
    else:
        plt.show()

    # Print summary information
    if verbose:
        print("Parameter pairs ranked by Ward distance (closest first):")
        print("-" * 60)
        for i, ((param1, param2), distance) in enumerate(pairs[:20]):  # Show top 20
            print(f"{i+1:2d}. {param1} <-> {param2}: {distance:.4f}")

        if len(pairs) > 20:
            print(f"... and {len(pairs) - 20} more pairs")

        print(f"\nHighlighted pairs (top 3 closest):")
        for i, ((param1, param2), distance) in enumerate(top_pairs):
            print(f"  Pair {i+1}: {param1} <-> {param2} (distance: {distance:.4f})")

    return pairs, pair_indices


def create_node_to_samples_mapping(linkage_matrix, n_samples):
    """
    Create a mapping from internal node IDs to the samples they contain.
    """
    node_to_samples = {}

    # Initialize leaf nodes (original samples)
    for i in range(n_samples):
        node_to_samples[i] = [i]

    # Process internal nodes
    for i, (left, right, distance, size) in enumerate(linkage_matrix):
        new_node_id = n_samples + i
        left_samples = node_to_samples[int(left)]
        right_samples = node_to_samples[int(right)]
        node_to_samples[new_node_id] = left_samples + right_samples

    return node_to_samples


# Example usage and test function
def test_ward_clustering():
    # Load the data
    base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast"
    posterior_df = pd.read_csv(f"{base_dir}/iter_4/all_param_df.csv")

    drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
    posterior_df.drop(columns=drop_cols, inplace=True)
    param_names = np.array(posterior_df.columns.tolist())
    posterior_df = posterior_df.to_numpy()
    samples_stand = (posterior_df - np.mean(posterior_df, axis=0)) / np.std(posterior_df, axis=0)
    correlation_matrix = np.corrcoef(samples_stand, rowvar=False)
    # Test the function
    pairs, pair_indices = colorize_ward_pairs(
        data=correlation_matrix,
        param_names=param_names,
        figsize=(6, 6),
        save_path=f"{base_dir}/redundancy_analysis_pairwise.png",
    )

    return pairs, pair_indices


if __name__ == "__main__":
    # Uncomment the line below to run the test
    test_ward_clustering()
