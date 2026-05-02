import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import pandas as pd
from typing import Tuple, List, Dict, Optional
import json


class SVDRedundancyDetector:
    """
    Detect redundant parameters using SVD analysis of ABC posterior distributions.

    This class identifies parameter combinations that control similar emergent behaviors
    by analyzing the singular value decomposition of the parameter correlation structure.
    """

    def __init__(
        self,
        parameter_samples: pd.DataFrame,
        parameter_names: List[str] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Initialize the detector with ABC posterior samples.

        Args:
            parameter_samples: DataFrame of shape (n_samples, n_parameters)
            parameter_names: List of parameter names for interpretability
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.samples = parameter_samples.to_numpy()
        self.n_samples, self.n_params = parameter_samples.shape
        self.param_names = parameter_names or [f"param_{i}" for i in range(self.n_params)]

        # Split data into train and test sets
        from sklearn.model_selection import train_test_split

        self.train_idx, self.test_idx = train_test_split(
            np.arange(self.n_samples), test_size=test_size, random_state=random_state
        )
        self.train_samples = self.samples[self.train_idx]
        self.test_samples = self.samples[self.test_idx]

        # Storage for analysis results
        self.correlation_matrix = None
        self.svd_results = None
        self.redundancy_groups = None
        self.scalers = [StandardScaler() for _ in range(self.n_params)]

    def standardize_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Standardize parameters to zero mean and unit variance using only training data statistics.

        Returns:
            Tuple of (standardized_train_samples, standardized_test_samples)
        """
        self.samples_std_train = np.zeros_like(self.train_samples)
        self.samples_std_test = np.zeros_like(self.test_samples)

        for i in range(self.n_params):
            # Fit scaler on training data only
            self.scalers[i].fit(self.train_samples[:, i].reshape(-1, 1))
            # Transform both train and test data
            self.samples_std_train[:, i] = (
                self.scalers[i].transform(self.train_samples[:, i].reshape(-1, 1)).ravel()
            )
            self.samples_std_test[:, i] = (
                self.scalers[i].transform(self.test_samples[:, i].reshape(-1, 1)).ravel()
            )

        return self.samples_std_train, self.samples_std_test

    def compute_svd_analysis(self, method: str = "samples") -> np.ndarray:
        """
        Perform SVD analysis using different approaches.

        Args:
            method: 'samples' (SVD on raw samples) or 'correlation' (SVD on correlation matrix)

        Returns:
            Correlation matrix of standardized parameters
        """
        if not hasattr(self, "samples_std_train"):
            self.standardize_parameters()

        # Always compute correlation matrix for comparison using training data only
        self.correlation_matrix = np.corrcoef(self.samples_std_train.T)

        if method == "samples":
            # SVD on standardized training samples directly (preferred approach)
            # This preserves sample-wise relationships and variance structure
            U, s, Vt = np.linalg.svd(self.samples_std_train, full_matrices=False)

            # Convert to parameter space interpretation
            # V (Vt.T) contains parameter loadings for each component
            parameter_loadings = Vt.T  # Shape: (n_params, n_components)
            self.svd_results = {
                "U_train": U,  # Training sample loadings (n_train, n_components)
                "singular_values": s,  # Singular values
                "Vt": Vt,  # Parameter directions (n_components, n_params)
                "parameter_loadings": parameter_loadings,  # Parameter loadings (n_params, n_components)
                "explained_variance_ratio": (s**2 / (len(self.train_idx) - 1))
                / np.sum(s**2 / (len(self.train_idx) - 1)),
                "method": "samples",
            }

            # Project test data onto the same SVD space
            self.svd_results["U_test"] = self.samples_std_test @ Vt.T

        elif method == "correlation":
            # SVD on correlation matrix (analyzes parameter-parameter relationships)
            # This focuses on linear correlations but loses sample variance information
            U, s, Vt = np.linalg.svd(self.correlation_matrix)

            self.svd_results = {
                "U": U,  # Parameter eigenvectors
                "singular_values": s,  # Singular values of correlation matrix
                "Vt": Vt,  # Same as U for symmetric matrix
                "parameter_loadings": U,  # Parameter loadings
                "explained_variance_ratio": s / np.sum(s),
                "method": "correlation",
            }
        else:
            raise ValueError("Method must be 'samples' or 'correlation'")

        return self.correlation_matrix

    def identify_redundant_groups(
        self,
        threshold: float = 0.7,
        method: str = "loading_similarity",
        svd_method: str = "samples",
        max_n_components: int = 10,
    ) -> Dict:
        """
        Identify groups of redundant parameters using SVD analysis.

        Args:
            threshold: Similarity threshold for grouping parameters
            method: Method for identifying redundancy ('loading_similarity' or 'correlation_clustering')
            svd_method: SVD approach ('samples' or 'correlation')

        Returns:
            Dictionary containing redundancy group information
        """
        if self.svd_results is None:
            self.compute_svd_analysis(method=svd_method)

        if method == "loading_similarity":
            groups = self._group_by_loading_similarity(threshold, max_n_components)
        elif method == "correlation_clustering":
            groups = self._group_by_correlation_clustering(threshold)
        else:
            raise ValueError("Method must be 'loading_similarity' or 'correlation_clustering'")

        self.redundancy_groups = groups
        return groups

    def _group_by_loading_similarity(self, threshold: float, max_n_components) -> Dict:
        """Group parameters based on similar loadings in dominant SVD components."""
        # Get parameter loadings (works for both SVD methods)
        if self.svd_results["method"] == "samples":
            parameter_loadings = self.svd_results["parameter_loadings"]
            singular_values = self.svd_results["singular_values"]
        else:  # correlation method
            parameter_loadings = self.svd_results["U"]
            singular_values = self.svd_results["singular_values"]

        # Focus on components that explain significant variance
        explained_var = self.svd_results["explained_variance_ratio"]
        cumulative_variance = np.cumsum(explained_var)
        n_components = np.argmax(cumulative_variance >= 0.8) + 1  # 80% of variance
        n_components = min(
            n_components, min(max_n_components, parameter_loadings.shape[1])
        )  # Limit components
        self.n_components = n_components
        # Weight loadings by importance (singular values or explained variance)
        if self.svd_results["method"] == "samples":
            # For samples SVD, weight by explained variance
            weights = np.sqrt(explained_var[:n_components])
        else:
            # For correlation SVD, weight by singular values
            weights = singular_values[:n_components] / np.sum(singular_values[:n_components])

        weighted_loadings = parameter_loadings[:, :n_components] * weights
        # Compute pairwise similarities between parameter loading patterns
        similarity_matrix = np.abs(np.corrcoef(weighted_loadings))

        # Ensure symmetry and handle numerical precision issues
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        np.fill_diagonal(similarity_matrix, 1.0)  # Ensure diagonal is exactly 1

        # Create distance matrix and ensure it's valid
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)  # Ensure diagonal is exactly 0

        # Convert to condensed form for linkage (only upper triangle)
        distance_condensed = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(distance_condensed, method="ward")

        # Extract clusters
        from scipy.cluster.hierarchy import fcluster

        cluster_threshold = 1 - threshold
        cluster_labels = fcluster(linkage_matrix, cluster_threshold, criterion="distance")

        # Organize results
        groups = {}
        for i, label in enumerate(cluster_labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(i)

        # Filter out singleton groups and format results
        redundant_groups = {}
        group_id = 1
        for cluster_id, param_indices in groups.items():
            if len(param_indices) > 1:  # Only keep groups with multiple parameters
                redundant_groups[f"Group_{group_id}"] = {
                    "parameters": [self.param_names[i] for i in param_indices],
                    "indices": param_indices,
                    "similarity_scores": {
                        (self.param_names[i], self.param_names[j]): similarity_matrix[i, j]
                        for i in param_indices
                        for j in param_indices
                        if i < j
                    },
                    "svd_method": self.svd_results["method"],
                }
                group_id += 1

        return redundant_groups

    def _group_by_correlation_clustering(self, threshold: float) -> Dict:
        """Group parameters based on direct correlation clustering."""
        correlation_abs = np.abs(self.correlation_matrix)

        # Create adjacency matrix based on correlation threshold
        adjacency = (correlation_abs >= threshold).astype(int)
        np.fill_diagonal(adjacency, 0)  # Remove self-connections

        # Find connected components (groups of highly correlated parameters)
        from scipy.sparse.csgraph import connected_components

        n_components, labels = connected_components(adjacency, directed=False)

        # Organize results
        groups = {}
        for i, label in enumerate(labels):
            if label not in groups:
                groups[label] = []
            groups[label].append(i)

        # Filter and format results
        redundant_groups = {}
        group_id = 1
        for cluster_id, param_indices in groups.items():
            if len(param_indices) > 1:
                # Calculate average correlation within group
                group_correlations = []
                for i in range(len(param_indices)):
                    for j in range(i + 1, len(param_indices)):
                        idx1, idx2 = param_indices[i], param_indices[j]
                        group_correlations.append(correlation_abs[idx1, idx2])

                redundant_groups[f"Group_{group_id}"] = {
                    "parameters": [self.param_names[i] for i in param_indices],
                    "indices": param_indices,
                    "avg_correlation": np.mean(group_correlations),
                    "correlations": group_correlations,
                    "similarity_scores": {
                        (self.param_names[i], self.param_names[j]): correlation_abs[i, j]
                        for i in param_indices
                        for j in param_indices
                        if i < j
                    },
                    "svd_method": self.svd_results["method"],
                }
                group_id += 1

        return redundant_groups

    def analyze_parameter_importance(self, n_components: int = None) -> pd.DataFrame:
        """
        Analyze parameter importance across SVD components.

        Args:
            n_components: Number of components to analyze (default: all)

        Returns:
            DataFrame with parameter importance scores
        """
        if self.svd_results is None:
            self.compute_svd_analysis()

        # Get appropriate loadings based on SVD method
        if self.svd_results["method"] == "samples":
            loadings_matrix = self.svd_results["parameter_loadings"]
            singular_values = self.svd_results["singular_values"]
        else:
            loadings_matrix = self.svd_results["U"]
            singular_values = self.svd_results["singular_values"]

        if self.n_components is None:
            n_components = loadings_matrix.shape[1]
        n_components = min(n_components, loadings_matrix.shape[1])

        # Create importance matrix
        importance_data = []
        for i, param_name in enumerate(self.param_names):
            row = {"parameter": param_name}
            for j in range(n_components):
                row[f"component_{j+1}"] = loadings_matrix[i, j]
                if self.svd_results["method"] == "samples":
                    # Weight by explained variance for samples SVD
                    weight = np.sqrt(self.svd_results["explained_variance_ratio"][j])
                else:
                    # Weight by singular values for correlation SVD
                    weight = singular_values[j]
                row[f"weighted_component_{j+1}"] = loadings_matrix[i, j] * weight

            # Overall importance
            if self.svd_results["method"] == "samples":
                weights = np.sqrt(self.svd_results["explained_variance_ratio"][:n_components])
            else:
                weights = singular_values[:n_components]
            row["overall_importance"] = np.sum((loadings_matrix[i, :n_components] * weights) ** 2)
            importance_data.append(row)

        return pd.DataFrame(importance_data)

    def plot_redundancy_analysis(
        self, method: str = "loading_similarity", figsize: Tuple[int, int] = (15, 5)
    ):
        """Create comprehensive visualization of redundancy analysis."""
        if self.svd_results is None:
            self.compute_svd_analysis()
        if method == "loading_similarity":
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            # 1. Explained variance
            ax1 = axes[0]
            explained_var = self.svd_results["explained_variance_ratio"]
            cumulative_var = np.cumsum(explained_var)
            ax1.bar(range(1, len(explained_var) + 1), explained_var, alpha=0.7, label="Individual")
            ax1.plot(range(1, len(cumulative_var) + 1), cumulative_var, "ro-", label="Cumulative")
            ax1.set_xlabel("Component")
            ax1.set_ylabel("Explained Variance Ratio")
            ax1.set_title("Explained Variance by Component")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Parameter clustering dendrogram
            ax3 = axes[2]
            ax3.set_title("Parameter Clustering Dendrogram")
            parameter_loadings = self.svd_results["parameter_loadings"]
            weights = np.sqrt(self.svd_results["explained_variance_ratio"])
            weighted_loadings = parameter_loadings * weights
            weighted_loadings = weighted_loadings[:, : self.n_components]
            correlation_abs = np.abs(np.corrcoef(weighted_loadings))
            # Ensure symmetry for distance matrix
            correlation_abs = (correlation_abs + correlation_abs.T) / 2
            np.fill_diagonal(correlation_abs, 1.0)

            distance_matrix = 1 - correlation_abs
            np.fill_diagonal(distance_matrix, 0.0)

            # Use condensed form for linkage
            distance_condensed = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(distance_condensed, method="ward")
            ddata = dendrogram(
                linkage_matrix,
                ax=ax3,
                orientation="top",
                no_labels=True,
                distance_sort=True,
                above_threshold_color="k",
                color_threshold=0.5,
            )

            leaf_order = ddata["leaves"]
            correct_labels = [self.param_names[i] for i in leaf_order]
            n_leaves = len(leaf_order)
            xticks = np.arange(5, 10 * n_leaves, 10)
            ax3.set_ylabel("Variance increase")
            ax3.set_xticks(xticks)
            ax3.set_xticklabels(correct_labels, rotation=90, ha="center", fontsize=8)
            ax3.set_xlim(0, 10 * n_leaves)

            # 3. PC components vs. parameter loadings
            ax2 = axes[1]
            if self.svd_results["method"] == "samples":
                loadings_matrix = self.svd_results["parameter_loadings"]
            else:
                loadings_matrix = self.svd_results["U"]

            n_comp_show = min(5, loadings_matrix.shape[1])
            # Reorder loadings matrix according to dendrogram leaf order
            loadings_matrix = loadings_matrix[leaf_order, :]
            loadings_df = pd.DataFrame(
                loadings_matrix[:, :n_comp_show],
                index=correct_labels,
                columns=[f"PC{i+1}" for i in range(n_comp_show)],
            )
            sns.heatmap(
                loadings_df.T, cmap="RdBu_r", center=0, ax=ax2, cbar_kws={"label": "Loading"}
            )
            ax2.set_title(f"Parameter Loadings")
            # Force all ticks to be shown
            ax2.set_xticks([i + 0.5 for i in range(len(correct_labels))])
            ax2.set_xticklabels(correct_labels, rotation=90, ha="center")
            ax2.set_yticks([i + 0.5 for i in range(n_comp_show)])
            ax2.set_yticklabels([f"PC{i+1}" for i in range(n_comp_show)])

            ax2.tick_params(axis="both", labelsize=8)
        else:
            # plot only the dendrogram using the correlation matrix
            fig, ax = plt.subplots(figsize=(8, 5))
            correlation_abs = np.abs(np.corrcoef(self.correlation_matrix))
            # Ensure symmetry for distance matrix
            correlation_abs = (correlation_abs + correlation_abs.T) / 2
            np.fill_diagonal(correlation_abs, 1.0)

            distance_matrix = 1 - correlation_abs
            np.fill_diagonal(distance_matrix, 0.0)

            # Use condensed form for linkage
            print(f"distance_matrix shape = {distance_matrix.shape}")
            print(f"distance_matrix = {distance_matrix}")
            # removed legacy debug hook
            distance_condensed = squareform(distance_matrix, checks=False)
            linkage_matrix = linkage(distance_condensed, method="ward")
            ddata = dendrogram(
                linkage_matrix,
                ax=ax,
                orientation="top",
                no_labels=True,
                distance_sort=True,
                above_threshold_color="k",
                color_threshold=0.5,
            )

            leaf_order = ddata["leaves"]
            correct_labels = [self.param_names[i] for i in leaf_order]
            n_leaves = len(leaf_order)
            xticks = np.arange(5, 10 * n_leaves, 10)
            ax.set_ylabel("Correlation")
            ax.set_xticks(xticks)
            ax.set_xticklabels(correct_labels, rotation=90, ha="center", fontsize=6)
            ax.set_xlim(0, 10 * n_leaves)
        plt.tight_layout()
        return fig

    def get_redundancy_summary(self) -> str:
        """Generate a text summary of redundancy analysis results."""
        if self.redundancy_groups is None:
            self.identify_redundant_groups()

        summary = "SVD-Based Parameter Redundancy Analysis Summary\n"
        summary += "=" * 50 + "\n\n"

        # Overall statistics
        summary += f"Total parameters analyzed: {self.n_params}\n"
        summary += f"Number of samples: {self.n_samples}\n\n"

        # SVD results
        if self.svd_results:
            n_significant = np.sum(self.svd_results["explained_variance_ratio"] > 0.05)
            summary += f"Significant SVD components (>5% variance): {n_significant}\n"
            summary += f"Variance explained by first 3 components: {np.sum(self.svd_results['explained_variance_ratio'][:3]):.2%}\n\n"

        # Redundancy groups
        if self.redundancy_groups:
            summary += f"Redundant parameter groups identified: {len(self.redundancy_groups)}\n\n"

            for group_name, group_info in self.redundancy_groups.items():
                summary += f"{group_name}:\n"
                summary += f"  Parameters: {', '.join(group_info['parameters'])}\n"
                if "avg_correlation" in group_info:
                    summary += f"  Average correlation: {group_info['avg_correlation']:.3f}\n"
        else:
            summary += "No significant redundant parameter groups found.\n"

        return summary

    def identify_representative_parameters(self) -> List[str]:
        """Identify representative parameters from the redundancy groups.
        Representative parameters are the parameters that have the highest mean similarity
        with other parameters in their group.
        """
        if self.redundancy_groups is None:
            self.identify_redundant_groups()
        self.representative_parameters = {}
        for group_name, group_info in self.redundancy_groups.items():
            parameters = group_info["parameters"]
            similarity_scores = group_info["similarity_scores"]

            # Calculate mean similarity score for each parameter
            param_mean_scores = {}
            for param in parameters:
                # Get all similarities involving this parameter
                param_scores = [
                    score for (p1, p2), score in similarity_scores.items() if param in (p1, p2)
                ]
                param_mean_scores[param] = np.mean(param_scores)

            # Find parameter with highest mean similarity
            best_param = max(param_mean_scores.items(), key=lambda x: x[1])[0]
            self.representative_parameters[group_name] = {
                "parameters": parameters,
                "best_param": best_param,
                "mean_similarity": param_mean_scores[best_param],
            }
        return self.representative_parameters

    def get_pairwise_similarities(self) -> Dict:
        """
        Get pairwise similarity scores between redundant parameters.

        Returns:
            Dictionary containing pairwise similarities for each redundant group
        """
        if self.redundancy_groups is None:
            self.identify_redundant_groups()

        pairwise_similarities = {}
        for group_name, group_info in self.redundancy_groups.items():
            pairwise_similarities[group_name] = {
                "parameters": group_info["parameters"],
                "similarities": group_info["similarity_scores"],
            }

        return pairwise_similarities

    def print_pairwise_similarities(self):
        """Print pairwise similarities between redundant parameters in a readable format."""
        similarities = self.get_pairwise_similarities()

        for group_name, group_info in similarities.items():
            print(f"\n{group_name}:")
            print("Parameters:", ", ".join(group_info["parameters"]))
            print("\nPairwise Similarities:")
            for (param1, param2), score in group_info["similarities"].items():
                print(f"{param1} - {param2}: {score:.3f}")
            print("-" * 50)

    def predict_redundant_parameters(self, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Predict redundant parameters using representative parameters via MLR.
        Works with samples in SVD space and predicts between parameters.

        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing prediction results and model performance metrics
        """
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        import numpy as np

        if self.redundancy_groups is None:
            self.identify_redundant_groups()

        # Get representative parameters
        rep_params = self.representative_parameters

        # Initialize results dictionary
        results = {}
        self.n_components = 32
        # Project all samples to SVD space
        U = self.svd_results["U_train"]  # Shape: (n_train, n_components)
        s = self.svd_results["singular_values"]
        Vt = self.svd_results["Vt"]  # Shape: (n_components, n_params)
        # Use only the first n_components
        U_reduced = U[:, : self.n_components]  # Shape: (n_train, n_components)
        s_reduced = s[: self.n_components]  # Shape: (n_components,)
        Vt_reduced = Vt[: self.n_components, :]  # Shape: (n_components, n_params)

        # Project data to reduced SVD space
        # Shape: (n_train, n_params, n_components)
        projected_data = np.einsum("sk,kp->spk", U_reduced, Vt_reduced)
        X_approx = (U_reduced * s_reduced) @ Vt_reduced  # shape: (n_train, n_params)
        print(f"projected_data.shape: {projected_data.shape}")
        print(f"X_approx.shape: {X_approx.shape}")
        # print(f"X_approx: {X_approx[:1]}")
        # print(f"samples: {self.samples_std_train[:1]}")

        # Create train/test indices once
        indices = np.arange(len(self.train_idx))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state
        )

        # For each redundant group
        for group_name, group_info in self.redundancy_groups.items():
            print("=" * 50)
            print(f"Group: {group_name}")

            parameters = group_info["parameters"]
            rep_param = rep_params[group_name]["best_param"]
            redundant_params = [p for p in parameters if p != rep_param]

            # Get indices
            rep_idx = self.param_names.index(rep_param)
            redundant_indices = [self.param_names.index(p) for p in redundant_params]

            # Get projected data
            X = projected_data[:, rep_idx, :]  # Shape: (n_train, n_components)
            y = projected_data[
                :, redundant_indices, :
            ]  # Shape: (n_train, n_redundant, n_components)

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Store results for this group
            group_results = {"models": [], "predictions": [], "metrics": {}}

            # Train individual models for each redundant parameter
            for i, param in enumerate(redundant_params):
                print(f"Training MLR for {param} using {rep_param}")

                # Train MLR for this parameter's SVD components
                mlr = LinearRegression()
                mlr.fit(
                    X_train, y_train[:, i, :]
                )  # Input: (n_train, n_components) -> Output: (n_train, n_components)

                # Make predictions
                y_pred_svd = mlr.predict(X_test)  # Shape: (n_test, n_components)

                # Store model and predictions
                group_results["models"].append(mlr)
                group_results["predictions"].append(y_pred_svd)

                # Evaluate in SVD space
                r2_svd = r2_score(y_test[:, i, :], y_pred_svd)
                rmse_svd = np.sqrt(mean_squared_error(y_test[:, i, :], y_pred_svd))

                # Transform back to original space for evaluation
                param_idx = redundant_indices[i]

                # Reconstruct original values from SVD components
                y_test_orig = (y_test[:, i, :] @ Vt_reduced[:, param_idx]).reshape(-1, 1)
                y_pred_orig = (y_pred_svd @ Vt_reduced[:, param_idx]).reshape(-1, 1)

                # Apply inverse scaling
                y_test_orig = self.scalers[param_idx].inverse_transform(y_test_orig)
                y_pred_orig = self.scalers[param_idx].inverse_transform(y_pred_orig)

                # Calculate metrics in original space
                r2_orig = r2_score(y_test_orig, y_pred_orig)
                rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
                relative_error = rmse_orig / np.std(y_test_orig)

                # Store metrics
                group_results["metrics"][param] = {
                    "r2_svd": r2_svd,
                    "rmse_svd": rmse_svd,
                    "r2_orig": r2_orig,
                    "rmse_orig": rmse_orig,
                    "relative_error": relative_error,
                }

                print(f"  R² (original): {r2_orig:.4f}, RMSE (original): {rmse_orig:.4f}")
            # Store results for this group
            results[group_name] = group_results
        return results

    def print_prediction_results(self, results: Dict):
        """
        Print prediction results in a readable format.

        Args:
            results: Dictionary containing prediction results from predict_redundant_parameters
        """
        print("\nPrediction Results Summary")
        print("=" * 50)

        for group_name in results["mlr"].keys():
            print(f"\nGroup: {group_name}")
            print(
                f"Representative Parameter: {results['mlr'][group_name]['representative_parameter']}"
            )
            print(
                f"Redundant Parameters: {', '.join(results['mlr'][group_name]['redundant_parameters'])}"
            )

            print("\nMLR Results:")
            print("-" * 30)
            for metric in results["mlr"][group_name]["metrics"]:
                print(f"\nParameter: {metric['parameter']}")
                print(f"R² Score: {float(metric['r2_score']):.3f}")
                print(f"RMSE (projected): {float(metric['rmse_projected']):.3f}")
                print(f"RMSE (original): {float(metric['rmse_original']):.3f}")
                print(f"Relative Error: {float(metric['relative_error']):.3f}")
                # print(f"Coefficient: {metric['coefficient'].tolist()}")
                # print(f"Intercept: {metric['coefficient'].tolist()}")

    def plot_regression_results(
        self, results: Dict, figsize: Tuple[int, int] = (10, 5), save_path: str = None
    ):
        """
        Plot regression results for each group showing relationships between representative
        and redundant parameters.

        Args:
            results: Dictionary containing prediction results from predict_redundant_parameters
            figsize: Figure size for each plot
        """
        for group_name, group_info in results["mlr"].items():
            rep_param = group_info["representative_parameter"]
            redundant_params = group_info["redundant_parameters"]
            train_idx = group_info["train_indices"]
            test_idx = group_info["test_indices"]

            # Get data
            rep_idx = self.param_names.index(rep_param)
            redundant_indices = [self.param_names.index(p) for p in redundant_params]

            X = self.samples_std_train[:, rep_idx]
            y = self.samples_std_train[:, redundant_indices]

            # Create subplot grid
            n_cols = len(redundant_params)
            fig, axes = plt.subplots(1, n_cols, figsize=figsize)
            if n_cols == 1:
                axes = [axes]

            # Plot each redundant parameter
            for i, (ax, param) in enumerate(zip(axes, redundant_params)):
                # Plot train data
                ax.scatter(X[train_idx], y[train_idx, i], alpha=0.5, label="Train", color="blue")

                # Plot test data
                ax.scatter(X[test_idx], y[test_idx, i], alpha=0.5, label="Test", color="red")

                # Plot regression line
                x_line = np.linspace(X.min(), X.max(), 100)
                y_line = group_info["models"][i].predict(x_line.reshape(-1, 1))
                ax.plot(x_line, y_line, "k-", label="Regression")

                # Add labels and title
                ax.set_xlabel(f"Representative Parameter\n({rep_param})")
                ax.set_ylabel(f"Redundant Parameter\n({param})")
                ax.set_title(f'R² = {group_info["metrics"][i]["r2_score"]:.2f}')
                ax.legend()
                ax.grid(True, alpha=0.7)
            plt.tight_layout()
            if save_path:
                plt.savefig(f"{save_path}/{group_name.lower()}.png")
            else:
                plt.show()

    def calculate_svd_reconstruction_error(self, n_components: int = 5) -> Dict:
        """
        Calculate reconstruction error using first n SVD components.

        Args:
            n_components: Number of SVD components to use for reconstruction

        Returns:
            Dictionary containing reconstruction error metrics
        """
        if self.svd_results is None:
            self.compute_svd_analysis()

        # Get SVD components
        U = self.svd_results["U_train"]
        s = self.svd_results["singular_values"]
        Vt = self.svd_results["Vt"]

        # Reconstruct data using first n components
        U_reduced = U[:, :n_components]
        s_reduced = s[:n_components]
        Vt_reduced = Vt[:n_components, :]

        # Reconstruct data in standardized space
        reconstructed_data_std = U_reduced @ np.diag(s_reduced) @ Vt_reduced

        # Transform back to original space
        reconstructed_data = self.scalers[0].inverse_transform(reconstructed_data_std)

        # Calculate reconstruction error in original space
        mse = np.mean((self.samples_std_train - reconstructed_data) ** 2, axis=0)
        rmse = np.sqrt(mse)
        relative_error = rmse / np.std(self.samples_std_train, axis=0)

        # Calculate explained variance ratio (in standardized space)
        total_var = np.sum(s**2)
        explained_var = np.sum(s_reduced**2)
        explained_var_ratio = explained_var / total_var

        results = {
            "reconstruction_error": {
                "mse": mse,
                "rmse": rmse,
                "relative_error": relative_error,
                "original_data": self.samples_std_train,
                "reconstructed_data": reconstructed_data,
            },
            "explained_variance": {
                "total": total_var,
                "explained": explained_var,
                "ratio": explained_var_ratio,
            },
            "n_components": n_components,
        }

        return results

    def plot_reconstruction_error(
        self, results: Dict, figsize: Tuple[int, int] = (15, 5), save_path: str = None
    ):
        """
        Plot reconstruction error analysis.

        Args:
            results: Dictionary from calculate_svd_reconstruction_error
            figsize: Figure size
            save_path: Path to save the figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot relative reconstruction error for each parameter
        relative_error = results["reconstruction_error"]["relative_error"]
        ax1.bar(range(len(relative_error)), relative_error)
        ax1.set_xlabel("Parameter Index")
        ax1.set_ylabel("Relative Reconstruction Error")
        ax1.set_title(f'Reconstruction Error (n={results["n_components"]} components)')
        ax1.grid(True, alpha=0.3)

        # Plot explained variance ratio
        explained_var = results["explained_variance"]["ratio"]
        ax2.bar(["Explained", "Unexplained"], [explained_var, 1 - explained_var])
        ax2.set_ylabel("Variance Ratio")
        ax2.set_title("Explained Variance Ratio")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        # Print summary statistics
        print("\nSVD Reconstruction Analysis")
        print("=" * 50)
        print(f"Number of components used: {results['n_components']}")
        print(f"Explained variance ratio: {results['explained_variance']['ratio']:.3f}")

        # Get original and reconstructed data
        original_data = results["reconstruction_error"]["original_data"]
        reconstructed_data = results["reconstruction_error"]["reconstructed_data"]

        # Calculate mean absolute error for each parameter
        mae = np.mean(np.abs(original_data - reconstructed_data), axis=0)
        # Create detailed comparison for each parameter
        print("\nDetailed Parameter Reconstruction Analysis")
        print("=" * 50)

        # Sort parameters by relative error
        error_params = list(
            zip(self.param_names, results["reconstruction_error"]["relative_error"])
        )

        for param_name, rel_error in error_params[:5]:
            param_idx = self.param_names.index(param_name)

            # Get statistics for this parameter
            orig_mean = np.mean(original_data[:, param_idx])
            orig_std = np.std(original_data[:, param_idx])
            recon_mean = np.mean(reconstructed_data[:, param_idx])
            recon_std = np.std(reconstructed_data[:, param_idx])
            abs_error = mae[param_idx]
            rmse = results["reconstruction_error"]["rmse"][param_idx]
            print(f"\nParameter: {param_name}")
            print("-" * 30)
            print(f"Original values - Mean: {orig_mean:.3f}, Std: {orig_std:.3f}")
            print(f"Reconstructed values - Mean: {recon_mean:.3f}, Std: {recon_std:.3f}")
            print(f"Absolute Error (MAE): {abs_error:.3f}")
            print(f"Root Mean Square Error (RMSE): {rmse:.3f}")
            print(f"Relative Error: {rel_error:.3f}")

            # Create scatter plot for this parameter
            plt.figure(figsize=(8, 6))
            plt.scatter(
                original_data[:, param_idx],
                reconstructed_data[:, param_idx],
                alpha=0.5,
                label="Data points",
            )

            # Add perfect reconstruction line
            min_val = min(original_data[:, param_idx].min(), reconstructed_data[:, param_idx].min())
            max_val = max(original_data[:, param_idx].max(), reconstructed_data[:, param_idx].max())
            plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect reconstruction")

            plt.xlabel("Original Values")
            plt.ylabel("Reconstructed Values")
            plt.title(
                f"Reconstruction Analysis for {param_name}\nRMSE: {rmse:.3f}, Relative Error: {rel_error:.3f}"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)

            if save_path:
                path_wo_png = save_path.split(".png")[0]
                plt.savefig(f"{path_wo_png}_{param_name}_reconstruction.png")
            else:
                plt.show()
            plt.close()


# Example usage function
def analyze_abc_posterior_redundancy(
    parameter_samples: np.ndarray,
    parameter_names: List[str] = None,
    correlation_threshold: float = 0.7,
    method: str = "loading_similarity",
    svd_method: str = "correlation",
    max_n_components: int = 10,
    verbose: bool = False,
    save_path: str = None,
) -> SVDRedundancyDetector:
    """
    Convenient function to perform complete redundancy analysis.

    Args:
        parameter_samples: ABC posterior samples (n_samples, n_parameters)
        parameter_names: List of parameter names
        correlation_threshold: Threshold for identifying redundant groups
        svd_method: 'samples' or 'correlation'
        plot: Whether to create visualization plots

    Returns:
        Fitted SVDRedundancyDetector instance
    """
    detector = SVDRedundancyDetector(parameter_samples, parameter_names)

    # Perform analysis
    detector.standardize_parameters()
    detector.compute_svd_analysis(method=svd_method)
    detector.identify_redundant_groups(
        threshold=correlation_threshold,
        method=method,
        svd_method=svd_method,
        max_n_components=max_n_components,
    )

    # Generate results
    if save_path:
        detector.plot_redundancy_analysis(method=method).savefig(save_path)
    else:
        detector.plot_redundancy_analysis(method=method)
        plt.show()
    if verbose:
        print(detector.get_redundancy_summary())

    # Show parameter importance
    # importance_df = detector.analyze_parameter_importance(n_components=5)
    # print("Parameter Importance Analysis:")
    # print(importance_df.round(3))

    return detector


def main():

    # Load the data
    base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast"
    prior_df = pd.read_csv(f"{base_dir}/iter_0/all_param_df.csv")
    posterior_df = pd.read_csv(f"{base_dir}/iter_4/all_param_df.csv")
    final_metrics_df = pd.read_csv(f"{base_dir}/iter_4/final_metrics.csv")

    drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
    prior_df.drop(columns=drop_cols, inplace=True)
    posterior_df.drop(columns=drop_cols, inplace=True)
    param_names = prior_df.columns.tolist()
    max_n_components = 6
    correlation_threshold = 0.7
    if not os.path.exists(f"{base_dir}/simplified_model/"):
        os.makedirs(f"{base_dir}/simplified_model/")
    detector = analyze_abc_posterior_redundancy(
        posterior_df,
        param_names,
        correlation_threshold=correlation_threshold,
        method="correlation_clustering",
        svd_method="correlation",
        max_n_components=max_n_components,
        save_path=f"{base_dir}/simplified_model/redundancy_analysis_{correlation_threshold}.png",
    )
    parameters_groups = detector.identify_representative_parameters()
    total_parameters = []
    for group in parameters_groups.values():
        total_parameters.extend(group["parameters"])

    representative_parameters = [group["best_param"] for group in parameters_groups.values()]

    # Redundant parameters are the parameters that are not in the same group as the representative parameters
    redundant_params = {}
    for group in parameters_groups.values():
        redundant_params[group["best_param"]] = [
            param for param in group["parameters"] if param not in representative_parameters
        ]
    print(f"Total parameters in groups: {total_parameters}, shape: {len(total_parameters)}")
    print("-" * 50)
    print(
        f"Representative parameters: {representative_parameters}, shape: {len(representative_parameters)}"
    )
    print("-" * 50)
    n_redundant_params = sum([len(params) for params in redundant_params.values()])

    print(f"Redundant parameters: {redundant_params}, total = {n_redundant_params}")
    assert n_redundant_params + len(representative_parameters) == len(total_parameters)

    # Save the representative parameters to a json file
    with open(
        f"{base_dir}/simplified_model/representative_parameters_{correlation_threshold}.json", "w"
    ) as f:
        json.dump(representative_parameters, f)
    # Save the redundant parameters to a json file
    with open(
        f"{base_dir}/simplified_model/redundant_parameters_{correlation_threshold}.json", "w"
    ) as f:
        json.dump(redundant_params, f)

    if 0:
        print("\nRepresentative parameters:")
        print(representative_parameters)

    if 0:
        # Perform prediction analysis
        print("\nPerforming prediction analysis...")
        prediction_results = detector.predict_redundant_parameters(test_size=0.2, random_state=42)
        detector.print_prediction_results(prediction_results)
    if 0:
        # Plot regression results
        save_path = f"{base_dir}/regression_results"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        detector.plot_regression_results(prediction_results, save_path=save_path)

    if 0:
        # Calculate and plot SVD reconstruction error
        reconstruction_results = detector.calculate_svd_reconstruction_error(n_components=5)
        detector.plot_reconstruction_error(
            reconstruction_results, save_path=f"{base_dir}/svd_reconstruction_error.png"
        )


if __name__ == "__main__":
    main()
