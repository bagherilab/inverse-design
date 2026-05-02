import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Union, Optional
import logging


class BaseRF(ABC):
    """
    Base class for ABC Random Forest implementations.
    This class defines the common interface and functionality for all RF variants.
    """

    def __init__(
        self,
        n_trees: int = 500,
        min_samples_leaf: int = 5,
        n_try: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """
        Initialize the RF model.

        Parameters:
        -----------
        n_trees : int, default=500
            Number of trees in the forest.
        min_samples_leaf : int, default=5
            Minimum number of samples required to be at a leaf node.
        n_try : int, optional
            Number of statistics to consider when looking for the best split.
            Default is |S|/3 for ABC-RF and Poisson(|S|/3) for ABC-DRF.
        random_state : int, optional
            Controls both the randomization of the bootstrapping and the
            sampling of the features to consider when looking for the best split.
        """
        self.n_trees = n_trees
        self.min_samples_leaf = min_samples_leaf
        self.n_try = n_try
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        self.trees = []
        self.reference_table = None

    def fit(self, parameters: np.ndarray, statistics: np.ndarray) -> None:
        """
        Build the random forest from parameters and their corresponding statistics.

        Parameters:
        -----------
        parameters : np.ndarray of shape (n_samples, n_parameters)
            Parameter values.
        statistics : np.ndarray of shape (n_samples, n_statistics)
            Summary statistics corresponding to simulated data from parameters.
        """
        self.reference_table = (parameters, statistics)
        self._check_input(parameters, statistics)
        self._build_forest()
        return self

    @abstractmethod
    def _build_forest(self) -> None:
        """
        Build the random forest. Implementation depends on the specific RF variant.
        """
        pass

    @abstractmethod
    def predict_weights(self, observed_statistics: np.ndarray) -> np.ndarray:
        """
        Compute weights for particles in the reference table given observed statistics.

        Parameters:
        -----------
        observed_statistics : np.ndarray of shape (n_statistics,)
            Observed summary statistics.

        Returns:
        --------
        weights : np.ndarray of shape (n_samples,)
            Weights for each particle in the reference table.
        """
        pass

    def posterior_sample(
        self, observed_statistics: np.ndarray, n_samples: int = 1000
    ) -> np.ndarray:
        """
        Generate samples from the posterior distribution.

        Parameters:
        -----------
        observed_statistics : np.ndarray of shape (n_statistics,)
            Observed summary statistics.
        n_samples : int, default=1000
            Number of samples to generate.

        Returns:
        --------
        samples : np.ndarray of shape (n_samples, n_parameters)
            Samples from the posterior distribution.
        """
        weights = self.predict_weights(observed_statistics)
        parameters = self.reference_table[0]
        idx = self.rng.choice(len(parameters), size=n_samples, p=weights)
        return parameters[idx]

    def _check_input(self, parameters: np.ndarray, statistics: np.ndarray) -> None:
        """
        Check the validity of input data.

        Parameters:
        -----------
        parameters : np.ndarray of shape (n_samples, n_parameters)
            Parameter values.
        statistics : np.ndarray of shape (n_samples, n_statistics)
            Summary statistics.
        """
        if parameters.shape[0] != statistics.shape[0]:
            raise ValueError(
                f"Number of parameter sets ({parameters.shape[0]}) does not match "
                f"number of statistics sets ({statistics.shape[0]})"
            )

        if parameters.ndim == 1:
            # For backward compatibility, reshape to have explicit parameter dimension
            self.reference_table = (parameters.reshape(-1, 1), statistics)

    def _compute_l2_loss(self, parameters: np.ndarray) -> float:
        """
        Compute the L2 loss for a set of parameters.

        Parameters:
        -----------
        parameters : np.ndarray of shape (n_samples, n_parameters)
            Parameter values.

        Returns:
        --------
        loss : float
            The L2 loss value.
        """
        mean = np.mean(parameters, axis=0)
        return np.sum((parameters - mean) ** 2)

    def _find_best_split(
        self, parameters: np.ndarray, statistics: np.ndarray, statistic_indices: List[int]
    ) -> Tuple[int, float, float]:
        """
        Find the best split that minimizes the L2 loss.

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
        best_loss : float
            L2 loss value for the best split.
        """
        n_samples = parameters.shape[0]
        best_loss = float("inf")
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

                # Compute the loss after splitting
                left_loss = self._compute_l2_loss(left_params)
                right_loss = self._compute_l2_loss(right_params)
                total_loss = left_loss + right_loss

                if total_loss < best_loss:
                    best_loss = total_loss
                    best_stat_idx = stat_idx
                    best_threshold = (
                        sorted_stats[i - 1, stat_idx] + sorted_stats[i, stat_idx]
                    ) / 2.0

        return best_stat_idx, best_threshold, best_loss

    def variable_importance(self) -> np.ndarray:
        """
        Compute the importance of each summary statistic.

        Returns:
        --------
        importances : np.ndarray of shape (n_statistics,)
            Importance of each summary statistic.
        """
        if not self.trees:
            raise ValueError("The forest has not been built yet. Call fit() first.")

        n_statistics = self.reference_table[1].shape[1]
        importances = np.zeros(n_statistics)

        # Calculate importance based on how often each statistic is used for splitting
        for tree in self.trees:
            for node in tree["nodes"]:
                if "split_stat_idx" in node:
                    importances[node["split_stat_idx"]] += 1

        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)

        return importances


class TreeNode:
    """
    A node in a decision tree.
    """

    def __init__(self, samples: np.ndarray):
        """
        Initialize a node with a set of sample indices.

        Parameters:
        -----------
        samples : np.ndarray
            Indices of samples in this node.
        """
        self.samples = samples
        self.left = None
        self.right = None
        self.split_stat_idx = None
        self.threshold = None
        self.leaf = False

    def set_split(self, stat_idx: int, threshold: float):
        """
        Set the splitting criterion for this node.

        Parameters:
        -----------
        stat_idx : int
            Index of the statistic used for splitting.
        threshold : float
            Threshold value for splitting.
        """
        self.split_stat_idx = stat_idx
        self.threshold = threshold

    def set_leaf(self):
        """Mark the node as a leaf."""
        self.leaf = True
