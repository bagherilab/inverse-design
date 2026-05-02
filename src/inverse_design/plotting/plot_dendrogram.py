import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import pandas as pd
from collections import Counter
from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.plotting.theme import apply_publication_style

apply_publication_style(font_size=12, axes_linewidth=1.0)


def create_node_to_samples_mapping(linkage_matrix, n_samples):
    """
    Create mapping from internal node IDs to the samples they contain.
    """
    n_internal = len(linkage_matrix)
    node_to_samples = {}

    # Initialize leaf nodes (original samples)
    for i in range(n_samples):
        node_to_samples[i] = [i]

    # Build internal nodes
    for i, (left, right, dist, count) in enumerate(linkage_matrix):
        internal_node_id = n_samples + i
        left_samples = node_to_samples[int(left)]
        right_samples = node_to_samples[int(right)]
        node_to_samples[internal_node_id] = left_samples + right_samples

    return node_to_samples


def colorize_ward_clusters(
    data, param_names, n_group, n_min_sample, save_path=None, figsize=(8, 8), verbose=True
):
    """
    Performs Ward hierarchical clustering and colorizes only n_group clusters
    with lowest Ward distances that contain at least n_min_sample samples.
    Automatically highlights the sample with the lowest Ward distance within each valid cluster.

    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data for clustering
    n_group : int
        Number of groups/clusters to colorize
    n_min_sample : int
        Minimum number of samples required in each cluster
    save_path : str, optional
        Path to save the plot
    figsize : tuple, default=(12, 8)
        Figure size for the dendrogram plot

    Returns:
    --------
    threshold_distance : float
        The Ward distance threshold used for clustering
    cluster_labels : array
        Cluster labels for each sample
    valid_clusters : list
        List of cluster IDs that meet the criteria
    highlight_samples : list
        List of sample indices that are highlighted (sample with lowest Ward distance in each cluster)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from collections import Counter

    apply_publication_style(font_size=10, axes_linewidth=1.0)

    # Convert to numpy array if needed
    data = np.array(data)

    # Perform Ward linkage
    linkage_matrix = linkage(data, method="ward")

    # Get all possible thresholds (Ward distances from linkage matrix)
    ward_distances = linkage_matrix[:, 2]

    # Sort distances to try from lowest to highest
    sorted_distances = np.sort(ward_distances)

    valid_clusters = []
    threshold_distance = None
    cluster_labels = None

    # Try different thresholds to find n_group valid clusters
    for distance in sorted_distances:
        # Get cluster labels at this threshold
        temp_labels = fcluster(linkage_matrix, distance, criterion="distance")
        # Count samples in each cluster
        cluster_counts = Counter(temp_labels)

        # Find clusters that meet minimum sample requirement
        valid_cluster_ids = [
            cluster_id for cluster_id, count in cluster_counts.items() if count >= n_min_sample
        ]

        # If we have enough valid clusters, use this threshold
        if len(valid_cluster_ids) >= n_group:
            # Select the n_group clusters with most samples (or first n_group)
            valid_cluster_ids = sorted(
                valid_cluster_ids, key=lambda x: cluster_counts[x], reverse=True
            )[:n_group]

            threshold_distance = distance
            cluster_labels = temp_labels
            valid_clusters = valid_cluster_ids
            break
    if threshold_distance is None:
        raise ValueError(
            f"Could not find {n_group} clusters with at least {n_min_sample} samples each"
        )

    # Select one representative sample from each valid cluster
    highlight_samples = []
    rep_params = []
    redundant_params = {}
    for cluster_id in valid_clusters:
        # Get all samples in this cluster
        cluster_samples = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

        if len(cluster_samples) == 1:
            # If only one sample in cluster, select it
            highlight_samples.append(cluster_samples[0])
        else:
            # Calculate Ward distance for each sample within the cluster
            cluster_data = data[cluster_samples]

            # Calculate cluster centroid
            cluster_centroid = np.mean(cluster_data, axis=0)

            # Calculate Ward distance for each sample to the centroid
            ward_distances = []
            for sample_idx in cluster_samples:
                sample_data = data[sample_idx]
                # Ward distance is the squared Euclidean distance
                ward_dist = np.sum((sample_data - cluster_centroid) ** 2)
                ward_distances.append(ward_dist)
            # Select the sample with minimum Ward distance
            min_distance_idx = np.argmin(ward_distances)
            highlight_samples.append(cluster_samples[min_distance_idx])
            rep_params.append(str(param_names[cluster_samples[min_distance_idx]]))
            redundant_params[rep_params[-1]] = [
                str(param) for param in param_names[cluster_samples] if str(param) not in rep_params
            ]

    # Use consistent colors with previous figures
    peak_colors = ["lightcoral", "skyblue", "gold", "lightgreen", "plum", "orange"]
    peak_colors = ["crimson", "dodgerblue", "orange", "forestgreen", "mediumorchid", "darkorange"]
    # peak_colors = ['#bb883b', '#486b45', '#545aab', '#af1b0a', 'mediumorchid', 'darkorange']
    peak_colors = ["#67001f", "#a51429", "#f6b192", "#d25849"]
    peak_colors = ["#f6b192", "#a51429", "#67001f", "#d25849"]
    # Create figure with consistent styling
    fig, ax = plt.subplots(figsize=figsize)
    n_samples = len(cluster_labels)
    node_to_samples = create_node_to_samples_mapping(linkage_matrix, n_samples)

    # Create custom color function for dendrogram leaves
    def link_color_func(k):
        """
        k is an internal node ID. We need to check what clusters
        the samples in this internal node belong to.
        """
        if k in node_to_samples:
            samples_in_node = node_to_samples[k]

            # Get cluster labels for samples in this node
            node_clusters = [cluster_labels[sample_idx] for sample_idx in samples_in_node]

            # If all samples in this node belong to the same valid cluster, color it
            if len(set(node_clusters)) == 1:  # All samples have same cluster
                cluster_id = node_clusters[0]
                if cluster_id in valid_clusters:
                    cluster_idx = valid_clusters.index(cluster_id)
                    color_name = peak_colors[cluster_idx % len(peak_colors)]
                    return matplotlib.colors.to_hex(color_name)

        return "gray"

    # Plot dendrogram without automatic coloring first
    dend = dendrogram(
        linkage_matrix,
        link_color_func=link_color_func,
        orientation="right",  # Make it horizontal
        no_plot=False,
        labels=param_names,
        ax=ax,
    )

    # Style the axes (consistent with previous figures)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Color the leaf labels based on cluster membership
    ylbls = ax.get_ymajorticklabels()
    for lbl in ylbls:
        param = lbl.get_text()
        if param in param_names:
            sample_idx = np.where(param_names == param)[0][0]
            cluster_id = cluster_labels[sample_idx]
            if cluster_id in valid_clusters:
                cluster_idx = valid_clusters.index(cluster_id)
                color = peak_colors[cluster_idx % len(peak_colors)]
                lbl.set_color(color)
            else:
                lbl.set_color("gray")

            if sample_idx in highlight_samples:
                lbl.set_fontweight("bold")

    # Add vertical threshold line with consistent styling
    """
    ax.axvline(
        x=threshold_distance,
        color="black",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label=f"Threshold: {threshold_distance:.2f}",
    )
    """
    # Set labels with bold formatting
    ax.set_xlabel("Ward Distance", fontsize=12, fontweight="bold")
    ax.set_axisbelow(True)
    # Make tick labels bold and thicker (consistent with previous figures)
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)

    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    # for label in ax.get_yticklabels():
    #    label.set_fontweight('bold')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
    else:
        plt.show()

    # Print summary information
    if verbose:
        print(f"Threshold Ward distance: {threshold_distance:.4f}")
        print(f"Number of valid clusters found: {len(valid_clusters)}")
        print(f"Valid cluster IDs: {valid_clusters}")
        print(
            f"Highlight sample indices (samples with lowest Ward distance in each cluster): {highlight_samples}"
        )
        print(f"Representative params: {param_names[highlight_samples]}")
        print("Redundant params:")
        for rep_param, redundant_param in redundant_params.items():
            print(f"  {rep_param}: {redundant_param}")
        for cluster_id in valid_clusters:
            count = sum(1 for label in cluster_labels if label == cluster_id)
            cluster_samples = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            highlight_marker = (
                " (has highlighted sample)" if cluster_samples[0] in highlight_samples else ""
            )

    return (
        threshold_distance,
        cluster_labels,
        valid_clusters,
        rep_params,
        redundant_params,
        [lbl.get_text() for lbl in ylbls],
    )


def colorize_ward_clusters_threshold(
    data,
    param_names,
    ward_threshold,
    save_path=None,
    figsize=(5, 4),
    verbose=True,
    ax=None,
):
    """
    Performs Ward hierarchical clustering and colorizes all clusters below a specified Ward distance threshold.

    Parameters:
    -----------
    data : array-like, shape (n_samples, n_features)
        Input data for clustering
    param_names : array-like
        Names/labels for each sample
    ward_threshold : float
        Ward distance threshold - clusters with distance below this will be colorized
    save_path : str, optional
        Path to save the plot
    figsize : tuple, default=(8, 8)
        Figure size for the dendrogram plot
    verbose : bool, default=True
        Whether to print clustering information

    Returns:
    --------
    threshold_distance : float
        The Ward distance threshold used for clustering (same as input)
    cluster_labels : array
        Cluster labels for each sample
    clusters : list
        List of all cluster IDs found at the threshold
    rep_params : list
        Representative parameter names (one per cluster)
    redundant_params : dict
        Dictionary mapping representative params to their redundant counterparts
    leaf_labels : list
        List of leaf labels from the dendrogram
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
    from collections import Counter

    # Convert to numpy array if needed
    data = np.array(data)
    param_names = np.array(param_names)

    # Perform Ward linkage
    linkage_matrix = linkage(data, method="ward")

    # Get cluster labels at the specified threshold
    cluster_labels = fcluster(linkage_matrix, ward_threshold, criterion="distance")

    # Count samples in each cluster
    cluster_counts = Counter(cluster_labels)

    # Separate clusters (>1 sample) from individual parameters (1 sample)
    clusters = [cluster_id for cluster_id, count in cluster_counts.items() if count > 1]
    individual_params = [cluster_id for cluster_id, count in cluster_counts.items() if count == 1]

    clusters = sorted(clusters)
    individual_params = sorted(individual_params)

    # Select one representative sample from each cluster (only for multi-sample clusters)
    highlight_samples = []
    rep_params = []
    redundant_params = {}

    for cluster_id in clusters:
        # Get all samples in this cluster
        cluster_samples = [i for i, label in enumerate(cluster_labels) if label == cluster_id]

        # Calculate Ward distance for each sample within the cluster
        cluster_data = data[cluster_samples]

        # Calculate cluster centroid
        cluster_centroid = np.mean(cluster_data, axis=0)

        # Calculate Ward distance for each sample to the centroid
        ward_distances = []
        for sample_idx in cluster_samples:
            sample_data = data[sample_idx]
            # Ward distance is the squared Euclidean distance
            ward_dist = np.sum((sample_data - cluster_centroid) ** 2)
            ward_distances.append(ward_dist)

        # Select the sample with minimum Ward distance
        min_distance_idx = np.argmin(ward_distances)
        representative_sample = cluster_samples[min_distance_idx]
        highlight_samples.append(representative_sample)
        rep_params.append(str(param_names[representative_sample]))

        # Store redundant parameters
        redundant_params[rep_params[-1]] = [
            str(param_names[sample_idx])
            for sample_idx in cluster_samples
            if sample_idx != representative_sample
        ]

    # Define colors for clusters
    peak_colors = [
        "mediumorchid",
        "darkorange",
        "lightcoral",
        "skyblue",
        "gold",
        "lightgreen",
        "plum",
        "crimson",
        "dodgerblue",
        "orange",
        "forestgreen",
    ]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    n_samples = len(cluster_labels)
    node_to_samples = create_node_to_samples_mapping(linkage_matrix, n_samples)

    # Create custom color function for dendrogram links
    def link_color_func(k):
        """
        k is an internal node ID. Color the link if all samples in this node
        belong to the same valid cluster.
        """
        if k in node_to_samples:
            samples_in_node = node_to_samples[k]

            # Get cluster labels for samples in this node
            node_clusters = [cluster_labels[sample_idx] for sample_idx in samples_in_node]

            # If all samples in this node belong to the same cluster, color it
            if len(set(node_clusters)) == 1:  # All samples have same cluster
                cluster_id = node_clusters[0]
                if cluster_id in clusters:
                    cluster_idx = clusters.index(cluster_id)
                    color_name = peak_colors[cluster_idx % len(peak_colors)]
                    return matplotlib.colors.to_hex(color_name)

        return "gray"

    param_names_formatted = np.array(
        [param.replace("_MU", " ").replace("_", " ").capitalize() for param in param_names]
    )
    # Plot dendrogram
    dend = dendrogram(
        linkage_matrix,
        link_color_func=link_color_func,
        orientation="right",  # Horizontal orientation
        no_plot=False,
        labels=param_names_formatted,
        ax=ax,
    )

    # Style the axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Color the leaf labels based on cluster membership
    ylbls = ax.get_ymajorticklabels()
    for lbl in ylbls:
        param = lbl.get_text()
        if param in param_names_formatted:
            sample_idx = np.where(param_names_formatted == param)[0][0]
            cluster_id = cluster_labels[sample_idx]

            if cluster_id in clusters:  # Multi-sample cluster
                cluster_idx = clusters.index(cluster_id)
                color = peak_colors[cluster_idx % len(peak_colors)]
                lbl.set_color(color)
            else:  # Individual parameter
                lbl.set_color("gray")

            # Make representative samples bold
            if sample_idx in highlight_samples:
                lbl.set_fontweight("bold")

    # Add vertical threshold line
    ax.axvline(
        x=ward_threshold,
        color="black",
        linestyle="--",
        alpha=0.8,
        linewidth=2,
        label=f"Threshold: {ward_threshold:.2f}",
    )

    # Set labels
    ax.set_xlabel("Ward Distance", fontsize=12, fontweight="bold")
    ax.set_axisbelow(True)
    # Format tick labels
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")

    if created_fig:
        plt.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        else:
            plt.show()

    # Print summary information
    if verbose:
        print(f"Ward distance threshold: {ward_threshold:.4f}")
        print(f"Parameter clusters (>1 param): {len(clusters)}")
        print(f"Individual parameters (1 param): {len(individual_params)}")
        print(f"Cluster IDs: {clusters}")
        print(f"Representative params: {[param_names[i] for i in highlight_samples]}")

        print("\nCluster details:")
        for cluster_id in clusters:
            count = sum(1 for label in cluster_labels if label == cluster_id)
            cluster_samples = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
            rep_sample = [i for i in cluster_samples if i in highlight_samples][0]
            print(f"  Cluster {cluster_id}: {count} parameters (rep: {param_names[rep_sample]})")

        print(f"\nIndividual parameters:")
        individual_sample_indices = [
            i for i, label in enumerate(cluster_labels) if label in individual_params
        ]
        for idx in individual_sample_indices:
            print(f"  {param_names[idx]}")

        if redundant_params:
            print("\nRedundant parameters:")
            for rep_param, redundant_list in redundant_params.items():
                if redundant_list:
                    print(f"  {rep_param}: {redundant_list}")

    # Assert check: individual + rep + redundant = total number of parameters
    total_individual = len(individual_params)
    total_rep = len(rep_params)
    total_redundant = sum(len(redundant_list) for redundant_list in redundant_params.values())
    total_params = len(param_names)

    assert (
        total_individual + total_rep + total_redundant == total_params
    ), f"Parameter count mismatch: individual({total_individual}) + rep({total_rep}) + redundant({total_redundant}) = {total_individual + total_rep + total_redundant} != total({total_params})"

    if verbose:
        print(
            f"\nParameter count verification: {total_individual} individual + {total_rep} representative + {total_redundant} redundant = {total_params} total ✓"
        )

    return (
        ward_threshold,
        cluster_labels,
        clusters,
        rep_params,
        redundant_params,
        [lbl.get_text() for lbl in ylbls],
    )


# Example usage and test function
def test_ward_clustering():
    # Load the data
    base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2"
    base_dir = "../../data/N1024_breast_mean_only/"
    posterior_df = pd.read_csv(f"{base_dir}/iter_4/all_param_df.csv")

    drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
    peak_idx = 1
    posterior_df.drop(columns=drop_cols, inplace=True)
    # Perform PCA and find peaks using your existing function
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        posterior_df, n_components=2, random_state=0
    )
    distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[peak_idx]) ** 2, axis=1))
    closest_indices = np.argsort(distances)[:50]
    peak_data = posterior_df.iloc[closest_indices]
    posterior_df = peak_data
    param_names = np.array(posterior_df.columns.tolist())
    correlation_matrix = np.corrcoef(posterior_df.values, rowvar=False)
    if 1:
        for threshold in [1.0]:  # , 1.25, 1.5]:
            ward_threshold, labels, valid_clusters, rep_params, redundant_params, ylbls = (
                colorize_ward_clusters_threshold(
                    data=correlation_matrix,
                    param_names=param_names,
                    ward_threshold=threshold,
                    save_path=f"{base_dir}/redundancy_analysis_threshold_{threshold}_p{peak_idx+1}.png",
                )
            )
    return threshold, labels, valid_clusters, rep_params, redundant_params, ylbls


if __name__ == "__main__":
    # Uncomment the line below to run the test
    test_ward_clustering()
