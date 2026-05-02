import numpy as np
from typing import Optional, List, Dict, Union, Tuple
from collections import defaultdict
from .rf_base import BaseRF


class RF(BaseRF):
    """
    Implementation of ABC Random Forest for single parameter inference.

    This class implements Algorithms 1 and 2 from the paper, suitable for
    one-dimensional parameter inference.
    """

    def __init__(
        self,
        n_trees: int = 500,
        min_samples_leaf: int = 5,
        n_try: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the ABC-RF model.

        Parameters:
        -----------
        n_trees : int, default=500
            Number of trees in the forest.
        min_samples_leaf : int, default=5
            Minimum number of samples required to be at a leaf node.
        n_try : int, optional
            Number of statistics to consider when looking for the best split.
            Default is |S|/3.
        random_state : int, optional
            Controls the randomization of the bootstrapping and feature sampling.
        """
        super().__init__(n_trees, min_samples_leaf, n_try, random_state)

    def _build_forest(self) -> None:
        """
        Build the random forest following Algorithm 1 from the paper.
        """
        parameters, statistics = self.reference_table
        n_samples, n_statistics = statistics.shape

        # Default n_try if not provided
        if self.n_try is None:
            self.n_try = max(1, n_statistics // 3)

        self.trees = []

        for t in range(self.n_trees):
            # Bootstrap samples
            bootstrap_indices = self.rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_counts = np.bincount(bootstrap_indices, minlength=n_samples)

            # Build tree
            tree = self._build_tree(
                parameters[bootstrap_indices],
                statistics[bootstrap_indices],
                bootstrap_indices,
                bootstrap_counts,
            )

            self.trees.append(tree)

    def _build_tree(
        self,
        parameters: np.ndarray,
        statistics: np.ndarray,
        indices: np.ndarray,
        bootstrap_counts: np.ndarray,
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
        bootstrap_counts : np.ndarray
            Number of times each sample from the reference table appears in the bootstrap.

        Returns:
        --------
        tree : Dict
            A decision tree represented as a dictionary.
        """
        tree = {"nodes": []}

        # Queue for nodes to process
        queue = [(0, np.arange(len(parameters)))]

        while queue:
            node_id, node_samples = queue.pop(0)

            # Create a node
            node = {
                "id": node_id,
                "samples": indices[node_samples],
                "counts": bootstrap_counts[indices[node_samples]],
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

            # Select statistics to consider for splitting
            stat_candidates = self.rng.choice(
                statistics.shape[1], size=min(self.n_try, statistics.shape[1]), replace=False
            )

            # Find the best split
            best_stat_idx, best_threshold, best_loss = self._find_best_split(
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
            node["left_child"] = len(tree["nodes"]) + 1
            node["right_child"] = len(tree["nodes"]) + 2

            tree["nodes"].append(node)

            # Add children to the queue
            queue.append((node["left_child"], left_samples))
            queue.append((node["right_child"], right_samples))

        return tree

    def predict_weights(self, observed_statistics: np.ndarray) -> np.ndarray:
        """
        Compute weights for particles following Algorithm 2 from the paper.

        Parameters:
        -----------
        observed_statistics : np.ndarray of shape (n_statistics,)
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
            # Find the leaf node for the observed statistics
            leaf_node = self._find_leaf(tree, observed_statistics)

            # Update weights for particles in this leaf
            for sample_idx, count in zip(leaf_node["samples"], leaf_node["counts"]):
                if count > 0:  # Only consider samples that were in the bootstrap
                    weights[sample_idx] += count / np.sum(leaf_node["counts"])

        # Normalize weights across trees
        weights /= self.n_trees

        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights /= np.sum(weights)

        return weights

    def _find_leaf(self, tree: Dict, statistics: np.ndarray) -> Dict:
        """
        Find the leaf node that the given statistics would fall into.

        Parameters:
        -----------
        tree : Dict
            A decision tree.
        statistics : np.ndarray of shape (n_statistics,)
            Summary statistics.

        Returns:
        --------
        leaf : Dict
            The leaf node.
        """
        node_id = 0
        while True:
            node = tree["nodes"][node_id]

            if node["leaf"]:
                return node

            if statistics[node["split_stat_idx"]] <= node["threshold"]:
                node_id = node["left_child"]
            else:
                node_id = node["right_child"]
