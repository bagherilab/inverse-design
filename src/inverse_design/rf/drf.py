import numpy as np
from typing import Optional, List, Dict, Union, Tuple, Literal
from scipy.spatial.distance import pdist, squareform
from .rf_base import BaseRF
from inverse_design.common.enum import Target


class DRF(BaseRF):
    """
    Implementation of ABC Distributional Random Forest for multivariate parameter inference.

    This class implements Algorithms 3 and 4 from the paper, suitable for
    multivariate parameter inference.
    """

    def __init__(
        self,
        n_trees: int = 500,
        min_samples_leaf: int = 5,
        n_try: Optional[int] = None,
        random_state: Optional[int] = None,
        criterion: Literal["CART", "MMD"] = "CART",
        subsample_ratio: float = 0.5,
        n_fourier_features: int = 50,
    ):
        """
        Initialize the ABC-DRF model.

        Parameters:
        -----------
        n_trees : int, default=500
            Number of trees in the forest.
        min_samples_leaf : int, default=5
            Minimum number of samples required to be at a leaf node.
        n_try : int, optional
            Number of statistics to consider when looking for the best split.
            Default is a sample from Poisson(|S|/3).
        random_state : int, optional
            Controls the randomization of the subsampling and feature sampling.
        criterion : {'CART', 'MMD'}, default='CART'
            The function to measure the quality of a split.
            - 'CART': uses the CART criterion (L2 loss)
            - 'MMD': uses the Maximum Mean Discrepancy criterion
        subsample_ratio : float, default=0.5
            The ratio of samples to use for building each tree.
        n_fourier_features : int, default=50
            Number of Fourier features to use when criterion='MMD'.
        """
        super().__init__(n_trees, min_samples_leaf, n_try, random_state)
        self.criterion = criterion
        self.subsample_ratio = subsample_ratio
        self.n_fourier_features = n_fourier_features

    def _build_forest(self) -> None:
        """
        Build the random forest following Algorithm 3 from the paper.
        """
        parameters, statistics = self.reference_table
        n_samples, n_statistics = statistics.shape

        # Default n_try if not provided
        if self.n_try is None:
            self.n_try = max(1, n_statistics // 3)

        self.trees = []

        # Calculate subsample sizes
        nsample_tree = int(n_samples * self.subsample_ratio)
        nsample_leaf = n_samples - nsample_tree  # Samples for predictions

        for t in range(self.n_trees):
            # Subsample for tree building (not bootstrap)
            if t % 10 == 0:
                print(f"Building tree {t+1} of {self.n_trees}")
            all_indices = np.arange(n_samples)
            self.rng.shuffle(all_indices)

            tree_indices = all_indices[:nsample_tree]
            leaf_indices = all_indices[nsample_tree:n_samples]

            # Build tree
            tree = self._build_tree(
                parameters[tree_indices], statistics[tree_indices], tree_indices
            )

            # Assign leaf node IDs to leaf samples
            tree_leaf_assignments = self._assign_samples_to_leaves(tree, statistics[leaf_indices])

            # Store leaf sample indices and their assignments
            tree["leaf_samples"] = leaf_indices
            tree["leaf_assignments"] = tree_leaf_assignments

            self.trees.append(tree)

    def _build_tree(
        self, parameters: np.ndarray, statistics: np.ndarray, indices: np.ndarray
    ) -> Dict:
        """
        Recursively build a decision tree.

        Parameters:
        -----------
        parameters : np.ndarray
            Parameter values for samples in the tree.
        statistics : np.ndarray
            Summary statistics for samples in the tree.
        indices : np.ndarray
            Original indices of the samples in the reference table.

        Returns:
        --------
        tree : Dict
            A decision tree represented as a dictionary.
        """
        tree = {"nodes": [], "next_id": 1}

        # Queue for nodes to process
        queue = [(0, np.arange(len(parameters)))]
        while queue:
            node_id, node_samples = queue.pop(0)

            # Create a node
            node = {
                "id": node_id,
                "samples": indices[node_samples],
            }

            # Check if this should be a leaf node
            if len(node_samples) <= self.min_samples_leaf:
                node["leaf"] = True
                tree["nodes"].append(node)
                continue

            # Check if all statistics are the same
            if np.all(np.std(statistics[node_samples], axis=0) == 0):
                node["leaf"] = True
                tree["nodes"].append(node)
                continue

            # Sample number of statistics to try (Poisson distributed)
            n_try_poisson = self.rng.poisson(self.n_try)
            n_try_actual = max(1, min(n_try_poisson, statistics.shape[1]))

            # Select statistics to consider for splitting
            stat_candidates = self.rng.choice(statistics.shape[1], size=n_try_actual, replace=False)

            # Find the best split based on the chosen criterion
            if self.criterion == "CART":
                best_stat_idx, best_threshold, best_score = self._find_best_split_cart(
                    parameters[node_samples], statistics[node_samples], stat_candidates
                )
            else:  # MMD criterion
                best_stat_idx, best_threshold, best_score = self._find_best_split_mmd(
                    parameters[node_samples], statistics[node_samples], stat_candidates
                )

            # Check if a valid split was found
            if best_stat_idx == -1:
                node["leaf"] = True
                tree["nodes"].append(node)
                continue

            # Split the node
            left_samples = node_samples[statistics[node_samples, best_stat_idx] <= best_threshold]
            right_samples = node_samples[statistics[node_samples, best_stat_idx] > best_threshold]

            assert len(left_samples) + len(right_samples) == len(
                node_samples
            ), f"Split samples ({len(left_samples)} + {len(right_samples)}) do not sum to original samples ({len(node_samples)})"
            # Check if split is valid
            if (
                len(left_samples) < self.min_samples_leaf
                or len(right_samples) < self.min_samples_leaf
            ):
                node["leaf"] = True
                tree["nodes"].append(node)
                continue

            # Configure the current node as an internal node
            node["leaf"] = False
            node["split_stat_idx"] = best_stat_idx
            node["threshold"] = best_threshold
            node["left_child"] = tree["next_id"]
            node["right_child"] = tree["next_id"] + 1
            tree["next_id"] += 2  # Increment counter for next nodes

            tree["nodes"].append(node)

            # Add children to the queue
            queue.append((node["left_child"], left_samples))
            queue.append((node["right_child"], right_samples))

        return tree

    def _find_best_split_cart(
        self, parameters: np.ndarray, statistics: np.ndarray, statistic_indices: List[int]
    ) -> Tuple[int, float, float]:
        """
        Find the best split using the CART criterion (L2 loss).

        This is similar to the base class method but optimized for multivariate parameters.

        Parameters:
        -----------
        parameters : np.ndarray of shape (n_samples, n_parameters)
            Parameter values.
        statistics : np.ndarray of shape (n_samples, n_statistics)
            Summary statistics.
        statistic_indices : List[int]
            Indices of statistics to consider for splitting.

        Returns:
        --------
        best_stat_idx : int
            Index of the statistic used for the best split.
        best_threshold : float
            Threshold value for the best split.
        best_score : float
            CART criterion value for the best split (higher is better).
        """
        n_samples = parameters.shape[0]
        best_score = -float("inf")
        best_stat_idx = -1
        best_threshold = 0.0

        for stat_idx in statistic_indices:
            # Sort samples based on the current statistic
            sort_idx = np.argsort(statistics[:, stat_idx])
            sorted_params = parameters[sort_idx]
            sorted_stats = statistics[sort_idx]

            # Try different split points
            for i in range(1, n_samples):
                # Only consider unique values as split points
                if sorted_stats[i - 1, stat_idx] == sorted_stats[i, stat_idx]:
                    continue

                # Split the data
                left_params = sorted_params[:i]
                right_params = sorted_params[i:]

                # Skip if either side is too small
                if (
                    len(left_params) < self.min_samples_leaf
                    or len(right_params) < self.min_samples_leaf
                ):
                    continue

                # Filter out non-float elements and convert to NumPy arrays
                left_params_non_str = np.array(
                    [[p for p in params if isinstance(p, float)] for params in left_params]
                )
                right_params_non_str = np.array(
                    [[p for p in params if isinstance(p, float)] for params in right_params]
                )

                # Compute means for each side
                left_mean = np.mean(left_params_non_str, axis=0)
                right_mean = np.mean(right_params_non_str, axis=0)

                # Compute the CART criterion (equation 4 in the paper)
                left_size = len(left_params_non_str)
                right_size = len(right_params_non_str)
                cart_score = (left_size * right_size / (n_samples**2)) * np.sum(
                    (right_mean - left_mean) ** 2
                )

                if cart_score > best_score:
                    best_score = cart_score
                    best_stat_idx = stat_idx
                    best_threshold = (
                        sorted_stats[i - 1, stat_idx] + sorted_stats[i, stat_idx]
                    ) / 2.0

        return best_stat_idx, best_threshold, best_score

    def _find_best_split_mmd(
        self, parameters: np.ndarray, statistics: np.ndarray, statistic_indices: List[int]
    ) -> Tuple[int, float, float]:
        """
        Find the best split using the Maximum Mean Discrepancy (MMD) criterion.

        Parameters:
        -----------
        parameters : np.ndarray of shape (n_samples, n_parameters)
            Parameter values.
        statistics : np.ndarray of shape (n_samples, n_statistics)
            Summary statistics.
        statistic_indices : List[int]
            Indices of statistics to consider for splitting.

        Returns:
        --------
        best_stat_idx : int
            Index of the statistic used for the best split.
        best_threshold : float
            Threshold value for the best split.
        best_score : float
            MMD criterion value for the best split (higher is better).
        """
        n_samples = parameters.shape[0]

        best_score = -float("inf")
        best_stat_idx = -1
        best_threshold = 0.0

        # Filter out non-float elements and convert to NumPy arrays
        parameters_non_str = np.array(
            [[p for p in params if isinstance(p, float)] for params in parameters]
        )
        n_params = parameters_non_str.shape[1]

        # Compute median pairwise distance between parameters (for kernel bandwidth)
        pairwise_dist = pdist(parameters_non_str, metric="euclidean")
        sigma = np.median(pairwise_dist) if len(pairwise_dist) > 0 else 1.0

        # Generate random Fourier features
        random_omega = self.rng.normal(0, 1.0 / sigma, size=(self.n_fourier_features, n_params))

        # Precompute Fourier features for all samples
        fourier_features = np.zeros((n_samples, self.n_fourier_features))
        for l in range(self.n_fourier_features):
            fourier_features[:, l] = np.cos(np.dot(parameters_non_str, random_omega[l]))

        for stat_idx in statistic_indices:
            # Sort samples based on the current statistic
            sort_idx = np.argsort(statistics[:, stat_idx])
            sorted_fourier = fourier_features[sort_idx]
            sorted_stats = statistics[sort_idx]

            # Try different split points
            for i in range(1, n_samples):
                # Only consider unique values as split points
                if sorted_stats[i - 1, stat_idx] == sorted_stats[i, stat_idx]:
                    continue

                # Split the data
                left_size = i
                right_size = n_samples - i

                # Skip if either side is too small
                if left_size < self.min_samples_leaf or right_size < self.min_samples_leaf:
                    continue

                # Calculate MMD using Fourier features
                mmd_score = 0.0
                for l in range(self.n_fourier_features):
                    left_mean = np.mean(sorted_fourier[:left_size, l])
                    right_mean = np.mean(sorted_fourier[left_size:, l])
                    mmd_score += ((left_size * right_size) / (n_samples**2)) * (
                        (left_mean - right_mean) ** 2
                    )

                if mmd_score > best_score:
                    best_score = mmd_score
                    best_stat_idx = stat_idx
                    best_threshold = (
                        sorted_stats[i - 1, stat_idx] + sorted_stats[i, stat_idx]
                    ) / 2.0

        return best_stat_idx, best_threshold, best_score

    def _assign_samples_to_leaves(self, tree: Dict, statistics: np.ndarray) -> np.ndarray:
        """
        Assign each sample to a leaf node in the tree.

        Parameters:
        -----------
        tree : Dict
            The decision tree.
        statistics : np.ndarray
            Summary statistics for samples to assign.

        Returns:
        --------
        leaf_assignments : np.ndarray
            Leaf node ID for each sample.
        """
        n_samples = statistics.shape[0]
        leaf_assignments = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            leaf_assignments[i] = self._find_leaf_id(tree, statistics[i])

        return leaf_assignments

    def _find_leaf_id(self, tree: Dict, statistics: np.ndarray) -> int:
        """
        Find the ID of the leaf node that the given statistics would fall into.

        Parameters:
        -----------
        tree : Dict
            A decision tree.
        statistics : np.ndarray of shape (n_statistics,)
            Summary statistics.

        Returns:
        --------
        leaf_id : int
            The ID of the leaf node.
        """
        node_id = 0
        while True:
            node = tree["nodes"][node_id]

            if node["leaf"]:
                return node_id

            if statistics[node["split_stat_idx"]] <= node["threshold"]:
                node_id = node["left_child"]
            else:
                node_id = node["right_child"]

    def predict_weights(self, observed_statistics: List[Target]) -> np.ndarray:
        """
        Compute weights for particles following Algorithm 4 from the paper.

        Parameters:
        -----------
        observed_statistics : List[Target]
            Observed summary statistics.

        Returns:
        --------
        weights : np.ndarray of shape (n_samples,)
            Weights for each particle in the reference table.
        """
        if not self.trees:
            raise ValueError("The forest has not been built yet. Call fit() first.")

        parameters, _ = self.reference_table
        n_samples = parameters.shape[0]

        # Initialize weights
        weights = np.zeros(n_samples)

        for tree in self.trees:
            # Find the leaf node ID for the observed statistics
            obs_leaf_id = self._find_leaf_id(tree, observed_statistics)

            # Get all leaf samples that fall into the same leaf
            leaf_samples = tree["leaf_samples"]
            leaf_assignments = tree["leaf_assignments"]

            # Count samples in the observed leaf
            leaf_count = np.sum(leaf_assignments == obs_leaf_id)

            if leaf_count > 0:
                # Update weights for samples in this leaf
                for i, (sample_idx, leaf_id) in enumerate(zip(leaf_samples, leaf_assignments)):
                    if leaf_id == obs_leaf_id:
                        weights[sample_idx] += 1.0 / leaf_count

        # Normalize weights across trees
        weights /= self.n_trees

        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights /= np.sum(weights)

        return weights
