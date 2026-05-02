import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import json
from numpy.linalg import slogdet
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.special import digamma
from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.io import ArcadeRunLayout
from inverse_design.plotting.theme import apply_publication_style
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

apply_publication_style(font_size=12, axes_linewidth=1.0)


class ABCParameterImportance:
    """
    Calculate parameter importance from ABC posterior samples using multiple methods.
    """

    def __init__(
        self,
        posterior_params: np.ndarray,
        target_metrics: np.ndarray,
        param_names: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
    ):
        """
        Initialize with posterior parameters and corresponding metrics.

        Parameters:
        -----------
        posterior_params : np.ndarray, shape (n_samples, n_params)
            Posterior parameter samples from ABC
        target_metrics : np.ndarray, shape (n_samples, n_metrics)
            Corresponding target metrics for each parameter sample
        param_names : List[str], optional
            Names of parameters
        metric_names : List[str], optional
            Names of target metrics
        """
        self.posterior_params = np.array(posterior_params)
        self.target_metrics = np.array(target_metrics)

        if self.target_metrics.ndim == 1:
            self.target_metrics = self.target_metrics.reshape(-1, 1)

        self.n_samples, self.n_params = self.posterior_params.shape
        self.n_metrics = self.target_metrics.shape[1]

        self.param_names = param_names
        self.metric_names = metric_names

        # Storage for results
        self.importance_results = {}

    def correlation_importance(self) -> Dict[str, np.ndarray]:
        """
        Calculate parameter importance using correlation coefficients.

        Returns:
        --------
        Dict with 'pearson' and 'spearman' correlation matrices
        """
        pearson_corr = np.zeros((self.n_params, self.n_metrics))
        spearman_corr = np.zeros((self.n_params, self.n_metrics))

        for i in range(self.n_params):
            for j in range(self.n_metrics):
                # Pearson correlation
                pearson_corr[i, j], _ = pearsonr(
                    self.posterior_params[:, i], self.target_metrics[:, j]
                )
                # Spearman correlation
                spearman_corr[i, j], _ = spearmanr(
                    self.posterior_params[:, i], self.target_metrics[:, j]
                )

        self.importance_results["correlation"] = {
            "pearson": pearson_corr,
            "spearman": spearman_corr,
        }

        return self.importance_results["correlation"]

    def random_forest_importance(self) -> Dict[str, np.ndarray]:
        """
        Calculate parameter importance using Random Forest feature importance.

        Returns:
        --------
        Dict with feature importance for each metric
        """
        rf_importance = np.zeros((self.n_params, self.n_metrics))

        for j in range(self.n_metrics):
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(self.posterior_params, self.target_metrics[:, j])
            rf_importance[:, j] = rf.feature_importances_

        self.importance_results["random_forest"] = rf_importance
        return rf_importance

    def linear_regression_importance(self) -> Dict[str, np.ndarray]:
        """
        Calculate parameter importance using standardized regression coefficients.

        Returns:
        --------
        Dict with standardized coefficients and R² scores
        """
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.posterior_params)

        coefficients = np.zeros((self.n_params, self.n_metrics))
        r2_scores = np.zeros(self.n_metrics)

        for j in range(self.n_metrics):
            lr = LinearRegression()
            lr.fit(X_scaled, self.target_metrics[:, j])
            coefficients[:, j] = np.abs(lr.coef_)  # Use absolute values for importance

            y_pred = lr.predict(X_scaled)
            r2_scores[j] = r2_score(self.target_metrics[:, j], y_pred)

        self.importance_results["linear_regression"] = {
            "coefficients": coefficients,
            "r2_scores": r2_scores,
        }

        return self.importance_results["linear_regression"]

    def variance_based_importance(self) -> Dict[str, np.ndarray]:
        """
        Calculate parameter importance based on variance contribution.
        Uses the concept that parameters with higher variance contribute more to output variance.
        """
        # Normalize parameters to [0, 1]
        param_normalized = (self.posterior_params - self.posterior_params.min(axis=0)) / (
            self.posterior_params.max(axis=0) - self.posterior_params.min(axis=0)
        )

        # Calculate variance contribution for each parameter-metric pair
        variance_importance = np.zeros((self.n_params, self.n_metrics))

        for i in range(self.n_params):
            for j in range(self.n_metrics):
                # Bin parameter values and calculate variance of metrics within bins
                n_bins = min(10, len(np.unique(param_normalized[:, i])))
                if n_bins > 1:
                    bins = np.linspace(0, 1, n_bins + 1)
                    bin_indices = np.digitize(param_normalized[:, i], bins) - 1
                    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

                    # Calculate between-bin variance vs within-bin variance
                    total_var = np.var(self.target_metrics[:, j])
                    within_bin_var = 0

                    for bin_idx in range(n_bins):
                        mask = bin_indices == bin_idx
                        if np.sum(mask) > 1:
                            within_bin_var += np.sum(mask) * np.var(self.target_metrics[mask, j])

                    within_bin_var /= self.n_samples
                    between_bin_var = total_var - within_bin_var

                    # Importance is the fraction of variance explained by this parameter
                    variance_importance[i, j] = between_bin_var / total_var if total_var > 0 else 0

        self.importance_results["variance_based"] = variance_importance
        return variance_importance

    def calculate_all_importance(self) -> Dict[str, Union[np.ndarray, Dict]]:
        """
        Calculate parameter importance using all methods.

        Returns:
        --------
        Dict containing results from all importance calculation methods
        """
        print("Calculating correlation-based importance...")
        self.correlation_importance()

        print("Calculating Random Forest importance...")
        self.random_forest_importance()

        print("Calculating linear regression importance...")
        self.linear_regression_importance()

        print("Calculating variance-based importance...")
        self.variance_based_importance()

        return self.importance_results

    def analyze_parameter_relationships(
        self,
        param_subset_A: List[str],
        param_subset_B: List[str],
        range_percentile: Tuple[float, float] = (25, 75),
    ) -> Dict:
        """
        Analyze relationships between parameter subsets (e.g., cancellation effects, collinearity).

        Parameters:
        -----------
        param_subset_A : List[str]
            Names of parameters in subset A (e.g., ['param_A', 'param_B', 'param_C'])
        param_subset_B : List[str]
            Names of parameters in subset B (e.g., ['param_D', 'param_E', 'param_F'])
        range_percentile : Tuple[float, float]
            Percentile range to define "certain range" for subset A

        Returns:
        --------
        Dict containing analysis results
        """
        # Get parameter indices
        idx_A = [
            self.param_names.index(name) for name in param_subset_A if name in self.param_names
        ]
        idx_B = [
            self.param_names.index(name) for name in param_subset_B if name in self.param_names
        ]

        if not idx_A or not idx_B:
            raise ValueError("Some parameter names not found in param_names")

        # Extract parameter values
        params_A = self.posterior_params[:, idx_A]
        params_B = self.posterior_params[:, idx_B]

        # Define range for subset A based on percentiles
        A_lower = np.percentile(params_A, range_percentile[0], axis=0)
        A_upper = np.percentile(params_A, range_percentile[1], axis=0)

        # Find samples where all parameters in A are within the specified range
        in_range_mask = np.all((params_A >= A_lower) & (params_A <= A_upper), axis=1)

        if np.sum(in_range_mask) < 10:
            print(f"Warning: Only {np.sum(in_range_mask)} samples in specified range")

        # Analyze subset B when A is constrained
        params_B_constrained = params_B[in_range_mask]
        params_B_full = params_B

        results = {
            "n_samples_in_range": np.sum(in_range_mask),
            "range_definition": {
                "lower_bounds": dict(zip(param_subset_A, A_lower)),
                "upper_bounds": dict(zip(param_subset_A, A_upper)),
            },
        }

        # 1. Collinearity analysis within subset B
        if len(idx_B) > 1:
            # Calculate correlation matrix for B parameters when A is constrained
            corr_matrix_constrained = np.corrcoef(params_B_constrained.T)
            corr_matrix_full = np.corrcoef(params_B_full.T)

            results["collinearity_analysis"] = {
                "correlation_matrix_constrained": corr_matrix_constrained,
                "correlation_matrix_full": corr_matrix_full,
                "correlation_difference": corr_matrix_constrained - corr_matrix_full,
                "high_correlation_pairs": [],
            }

            # Find high correlation pairs
            for i in range(len(idx_B)):
                for j in range(i + 1, len(idx_B)):
                    corr_constrained = corr_matrix_constrained[i, j]
                    if abs(corr_constrained) > 0.7:
                        results["collinearity_analysis"]["high_correlation_pairs"].append(
                            {
                                "param1": param_subset_B[i],
                                "param2": param_subset_B[j],
                                "correlation_constrained": corr_constrained,
                                "correlation_full": corr_matrix_full[i, j],
                            }
                        )

        # 2. Cancellation effect analysis
        # Calculate effective parameter combinations
        results["cancellation_analysis"] = {}

        for i, param_name in enumerate(param_subset_B):
            param_values_constrained = params_B_constrained[:, i]
            param_values_full = params_B_full[:, i]

            # Compare variance when A is constrained vs unconstrained
            var_constrained = np.var(param_values_constrained)
            var_full = np.var(param_values_full)

            # Check if parameter becomes more constrained (potential cancellation)
            variance_reduction = 1 - (var_constrained / var_full) if var_full > 0 else 0

            results["cancellation_analysis"][param_name] = {
                "variance_full": var_full,
                "variance_constrained": var_constrained,
                "variance_reduction_ratio": variance_reduction,
                "mean_full": np.mean(param_values_full),
                "mean_constrained": np.mean(param_values_constrained),
                "std_full": np.std(param_values_full),
                "std_constrained": np.std(param_values_constrained),
            }

        # 3. Linear combinations that show cancellation
        if len(idx_B) > 1:
            # Try different linear combinations of B parameters
            combinations_to_test = [
                ("sum", np.ones(len(idx_B))),
                ("difference", np.array([1, -1] + [0] * (len(idx_B) - 2))),
                ("weighted_sum", np.random.randn(len(idx_B))),
            ]

            results["linear_combinations"] = {}

            for combo_name, weights in combinations_to_test:
                if len(weights) != len(idx_B):
                    continue

                combo_full = params_B_full @ weights
                combo_constrained = params_B_constrained @ weights

                results["linear_combinations"][combo_name] = {
                    "weights": dict(zip(param_subset_B, weights)),
                    "variance_full": np.var(combo_full),
                    "variance_constrained": np.var(combo_constrained),
                    "variance_reduction_ratio": (
                        1 - (np.var(combo_constrained) / np.var(combo_full))
                        if np.var(combo_full) > 0
                        else 0
                    ),
                }

        return results

    def plot_importance_comparison(self, figsize=(15, 10), save_path=None):
        """
        Plot comparison of different importance methods.
        """
        if not self.importance_results:
            self.calculate_all_importance()

        n_methods = len([k for k in self.importance_results.keys() if k != "linear_regression"]) + 2
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        plot_idx = 0

        # Plot correlation methods
        if "correlation" in self.importance_results:
            for corr_type in ["pearson", "spearman"]:
                ax = axes[plot_idx]
                corr_data = self.importance_results["correlation"][corr_type]

                im = ax.imshow(np.abs(corr_data), cmap="viridis", aspect="auto")
                ax.set_title(f"{corr_type.capitalize()} Correlation")
                ax.set_xlabel("Metrics")
                ax.set_ylabel("Parameters")
                ax.set_xticks(range(self.n_metrics))
                ax.set_xticklabels(self.metric_names, rotation=45)
                ax.set_yticks(range(self.n_params))
                ax.set_yticklabels(self.param_names)
                plt.colorbar(im, ax=ax)
                plot_idx += 1

        # Plot Random Forest importance
        if "random_forest" in self.importance_results:
            ax = axes[plot_idx]
            rf_data = self.importance_results["random_forest"]

            im = ax.imshow(rf_data, cmap="viridis", aspect="auto")
            ax.set_title("Random Forest Importance")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Parameters")
            ax.set_xticks(range(self.n_metrics))
            ax.set_xticklabels(self.metric_names, rotation=45)
            ax.set_yticks(range(self.n_params))
            ax.set_yticklabels(self.param_names)
            plt.colorbar(im, ax=ax)
            plot_idx += 1

        # Plot Linear Regression importance
        if "linear_regression" in self.importance_results:
            ax = axes[plot_idx]
            lr_data = self.importance_results["linear_regression"]["coefficients"]

            im = ax.imshow(lr_data, cmap="viridis", aspect="auto")
            ax.set_title("Linear Regression Coefficients")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Parameters")
            ax.set_xticks(range(self.n_metrics))
            ax.set_xticklabels(self.metric_names, rotation=45)
            ax.set_yticks(range(self.n_params))
            ax.set_yticklabels(self.param_names)
            plt.colorbar(im, ax=ax)
            plot_idx += 1

        # Plot Variance-based importance
        if "variance_based" in self.importance_results:
            ax = axes[plot_idx]
            var_data = self.importance_results["variance_based"]

            im = ax.imshow(var_data, cmap="viridis", aspect="auto")
            ax.set_title("Variance-based Importance")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Parameters")
            ax.set_xticks(range(self.n_metrics))
            ax.set_xticklabels(self.metric_names, rotation=45)
            ax.set_yticks(range(self.n_params))
            ax.set_yticklabels(self.param_names)
            plt.colorbar(im, ax=ax)
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes)):
            axes[i].set_visible(False)

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def plot_overall_importance_comparison(self, figsize=(15, 8), save_path=None):
        """
        Plot comparison of different overall importance methods.
        """
        if "overall" not in self.importance_results:
            self.overall_importance_ranking()

        overall_results = self.importance_results["overall"]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Bar chart comparing different overall methods
        methods_to_plot = [
            "ensemble_average",
            "borda_count",
            "pca_based",
            "multivariate_regression",
        ]
        available_methods = [m for m in methods_to_plot if m in overall_results]

        x = np.arange(len(self.param_names))
        width = 0.8 / len(available_methods)

        colors = plt.cm.Set3(np.linspace(0, 1, len(available_methods)))

        for i, method in enumerate(available_methods):
            scores = overall_results[method]
            ax1.bar(
                x + i * width,
                scores,
                width,
                label=method.replace("_", " ").title(),
                color=colors[i],
                alpha=0.8,
            )

        ax1.set_xlabel("Parameters")
        ax1.set_ylabel("Importance Score")
        ax1.set_title("Overall Parameter Importance Comparison")
        ax1.set_xticks(x + width * (len(available_methods) - 1) / 2)
        ax1.set_xticklabels(self.param_names)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Heatmap of rankings
        ranking_data = []
        ranking_labels = []

        for method in available_methods:
            scores = overall_results[method]
            rankings = stats.rankdata(-scores, method="min")  # Higher score = lower rank
            ranking_data.append(rankings)
            ranking_labels.append(method.replace("_", " ").title())

        if ranking_data:
            ranking_matrix = np.array(ranking_data)
            im = ax2.imshow(ranking_matrix, cmap="RdYlBu_r", aspect="auto")

            # Add text annotations
            for i in range(len(ranking_labels)):
                for j in range(len(self.param_names)):
                    text = ax2.text(
                        j,
                        i,
                        f"{ranking_matrix[i, j]:.0f}",
                        ha="center",
                        va="center",
                        color="black",
                        fontweight="bold",
                    )

            ax2.set_title("Parameter Importance Rankings\n(1 = Most Important)")
            ax2.set_xlabel("Parameters")
            ax2.set_ylabel("Methods")
            ax2.set_xticks(range(len(self.param_names)))
            ax2.set_xticklabels(self.param_names)
            ax2.set_yticks(range(len(ranking_labels)))
            ax2.set_yticklabels(ranking_labels)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax2)
            cbar.set_label("Rank")

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        return fig

    def multivariate_importance(self, method="pca_based") -> np.ndarray:
        """
        Calculate overall parameter importance considering all metrics simultaneously.

        Parameters:
        -----------
        method : str
            Method for multivariate importance calculation:
            - 'pca_based': Use PCA to find directions of maximum variance
            - 'canonical_correlation': Use canonical correlation analysis
            - 'multivariate_regression': Use multivariate linear regression
            - 'distance_based': Use distance correlation

        Returns:
        --------
        np.ndarray: Overall importance scores for each parameter
        """
        from sklearn.decomposition import PCA
        from sklearn.cross_decomposition import CCA

        if method == "pca_based":
            # Use PCA on metrics to find principal directions
            pca = PCA(n_components=min(self.n_metrics, 3))  # Use top 3 PCs
            pca.fit(self.target_metrics)

            # Project metrics onto principal components
            metrics_pca = pca.transform(self.target_metrics)

            # Calculate importance for each PC and weight by explained variance
            importance_weighted = np.zeros(self.n_params)

            for pc_idx in range(pca.n_components_):
                pc_metric = metrics_pca[:, pc_idx]
                pc_importance = np.zeros(self.n_params)
                # Calculate correlation-based importance for this PC
                for i in range(self.n_params):
                    pc_importance[i] = abs(
                        np.corrcoef(self.posterior_params[:, i], pc_metric)[0, 1]
                    )

                # Weight by explained variance ratio
                weight = pca.explained_variance_ratio_[pc_idx]
                importance_weighted += weight * pc_importance

            self.importance_results["multivariate_pca"] = importance_weighted
            return importance_weighted

        elif method == "canonical_correlation":
            # Use Canonical Correlation Analysis
            if self.n_metrics > 1:
                cca = CCA(n_components=min(self.n_params, self.n_metrics))
                X_c, Y_c = cca.fit_transform(self.posterior_params, self.target_metrics)

                # Calculate importance based on canonical loadings
                importance = np.mean(np.abs(cca.x_weights_), axis=1)
                self.importance_results["multivariate_cca"] = importance
                return importance
            else:
                # Fall back to simple correlation for single metric
                return np.abs(np.corrcoef(self.posterior_params.T, self.target_metrics.T)[:-1, -1])

        elif method == "multivariate_regression":
            # Use multivariate linear regression (all metrics as dependent variables)
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.linear_model import LinearRegression

            # Standardize inputs
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.posterior_params)

            # Fit multivariate regression
            multi_reg = MultiOutputRegressor(LinearRegression())
            multi_reg.fit(X_scaled, self.target_metrics)

            # Calculate importance as average absolute coefficient across all outputs
            coefficients = np.array([estimator.coef_ for estimator in multi_reg.estimators_])
            importance = np.mean(np.abs(coefficients), axis=0)

            self.importance_results["multivariate_regression"] = importance
            return importance

        elif method == "distance_based":
            # Use distance correlation (requires dcor package, fall back to simple method)
            try:
                import dcor

                importance = np.zeros(self.n_params)
                for i in range(self.n_params):
                    # Calculate distance correlation between parameter and all metrics
                    importance[i] = dcor.distance_correlation(
                        self.posterior_params[:, i].reshape(-1, 1), self.target_metrics
                    )
                self.importance_results["distance_correlation"] = importance
                return importance
            except ImportError:
                print("dcor package not available, falling back to PCA-based method")
                return self.multivariate_importance(method="pca_based")

        else:
            raise ValueError(f"Unknown method: {method}")

    def overall_importance_ranking(self, metric_weights=None, methods_to_include=None) -> Dict:
        """
        Calculate overall parameter importance using multiple aggregation strategies.

        Parameters:
        -----------
        metric_weights : np.ndarray, optional
            Weights for each metric when combining individual importances
        methods_to_include : List[str], optional
            Which importance methods to include in the overall ranking

        Returns:
        --------
        Dict containing different overall importance rankings
        """
        if not self.importance_results:
            self.calculate_all_importance()

        if metric_weights is None:
            metric_weights = np.ones(self.n_metrics) / self.n_metrics
        else:
            metric_weights = np.array(metric_weights)
            metric_weights = metric_weights / np.sum(metric_weights)  # Normalize

        overall_results = {}

        # 1. Weighted average across metrics for each method
        if methods_to_include is None:
            methods_to_include = [
                "correlation",
                "random_forest",
                "linear_regression",
                "variance_based",
            ]

        for method in methods_to_include:
            if method == "correlation":
                if "correlation" in self.importance_results:
                    # Use Pearson correlation
                    pearson_importance = np.abs(self.importance_results["correlation"]["pearson"])
                    overall_results["pearson_weighted"] = pearson_importance @ metric_weights

                    # Use Spearman correlation
                    spearman_importance = np.abs(self.importance_results["correlation"]["spearman"])
                    overall_results["spearman_weighted"] = spearman_importance @ metric_weights

            elif method in self.importance_results:
                if method == "linear_regression":
                    importance_matrix = self.importance_results[method]["coefficients"]
                else:
                    importance_matrix = self.importance_results[method]

                overall_results[f"{method}_weighted"] = importance_matrix @ metric_weights

        # 2. Multivariate importance methods
        overall_results["pca_based"] = self.multivariate_importance("pca_based")
        overall_results["canonical_correlation"] = self.multivariate_importance(
            "canonical_correlation"
        )
        overall_results["multivariate_regression"] = self.multivariate_importance(
            "multivariate_regression"
        )

        # 3. Ensemble importance (average across all methods)
        method_scores = []
        method_names = []

        for method_name, scores in overall_results.items():
            if len(scores) == self.n_params:  # Ensure valid scores
                # Normalize scores to [0, 1] for fair averaging
                normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
                method_scores.append(normalized_scores)
                method_names.append(method_name)

        if method_scores:
            ensemble_importance = np.mean(method_scores, axis=0)
            overall_results["ensemble_average"] = ensemble_importance

        # 4. Rank-based ensemble (Borda count)
        if method_scores:
            rank_scores = np.zeros(self.n_params)
            for scores in method_scores:
                # Convert to ranks (higher importance = lower rank number)
                ranks = stats.rankdata(-scores)  # Negative for descending order
                rank_scores += ranks

            # Convert back to importance (lower total rank = higher importance)
            borda_importance = len(method_scores) * self.n_params - rank_scores + len(method_scores)
            borda_importance = borda_importance / np.sum(borda_importance)  # Normalize
            overall_results["borda_count"] = borda_importance

        self.importance_results["overall"] = overall_results
        return overall_results

    def get_importance_summary(self, include_overall=True) -> pd.DataFrame:
        """
        Get a summary table of parameter importance across all methods.

        Parameters:
        -----------
        include_overall : bool
            Whether to include overall importance measures
        """
        if not self.importance_results:
            self.calculate_all_importance()

        # Calculate overall importance if requested
        if include_overall:
            self.overall_importance_ranking()

        summary_data = []

        for i, param_name in enumerate(self.param_names):
            row = {"Parameter": param_name}

            # Individual metric importance (average across metrics)
            if "correlation" in self.importance_results:
                pearson_avg = np.mean(
                    np.abs(self.importance_results["correlation"]["pearson"][i, :])
                )
                spearman_avg = np.mean(
                    np.abs(self.importance_results["correlation"]["spearman"][i, :])
                )
                row["Pearson_Avg"] = pearson_avg
                row["Spearman_Avg"] = spearman_avg

            if "random_forest" in self.importance_results:
                rf_avg = np.mean(self.importance_results["random_forest"][i, :])
                row["RandomForest_Avg"] = rf_avg

            if "linear_regression" in self.importance_results:
                lr_avg = np.mean(self.importance_results["linear_regression"]["coefficients"][i, :])
                row["LinearReg_Avg"] = lr_avg

            if "variance_based" in self.importance_results:
                var_avg = np.mean(self.importance_results["variance_based"][i, :])
                row["Variance_Avg"] = var_avg

            # Overall importance measures
            if include_overall and "overall" in self.importance_results:
                overall_results = self.importance_results["overall"]

                if "ensemble_average" in overall_results:
                    row["Overall_Ensemble"] = overall_results["ensemble_average"][i]

                if "borda_count" in overall_results:
                    row["Overall_Borda"] = overall_results["borda_count"][i]

                if "pca_based" in overall_results:
                    row["Overall_PCA"] = overall_results["pca_based"][i]

                if "multivariate_regression" in overall_results:
                    row["Overall_MultiReg"] = overall_results["multivariate_regression"][i]

            summary_data.append(row)

        df = pd.DataFrame(summary_data)

        # Add ranking columns for overall importance
        if include_overall and "overall" in self.importance_results:
            for col in df.columns:
                if col.startswith("Overall_"):
                    rank_col = col.replace("Overall_", "Rank_")
                    df[rank_col] = df[col].rank(ascending=False, method="min").astype(int)

        return df


def calculate_ksg_mi(X, Y, k=3):
    """
    Calculate mutual information using the Kraskov-Stögbauer-Grassberger (KSG) estimator.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features) or (n_samples,)
        Input features/parameters
    Y : array-like, shape (n_samples,) or (n_samples, n_targets)
        Target variable(s)/metrics
    k : int, default=3
        Number of nearest neighbors for KSG estimator

    Returns:
    --------
    mi_values : array-like
        Mutual information values for each feature-target pair
    """
    X = np.array(X)
    Y = np.array(Y)

    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Ensure Y is 2D for consistent processing
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_features = X.shape[1]
    n_targets = Y.shape[1]
    mi_matrix = np.zeros((n_features, n_targets))

    # Find nan values indices

    nan_indices = np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1)
    X = X[~nan_indices]
    Y = Y[~nan_indices]
    # Calculate MI for each feature-target pair
    for i in range(n_features):
        for j in range(n_targets):
            # Use sklearn's implementation which is based on KSG
            mi_matrix[i, j] = mutual_info_regression(
                X[:, i].reshape(-1, 1), Y[:, j], n_neighbors=k, random_state=42
            )[0]

    return mi_matrix


def compute_gaussian_mi_matrix(X, Y, param_names=None, metric_names=None):
    """Backward-compatible name for :func:`calculate_ksg_mi` (used by ``mi_suggestions``)."""
    _ = param_names, metric_names
    return calculate_ksg_mi(X, Y)


def compute_ksg_mi_matrix(X, Y, param_names=None, metric_names=None):
    """Backward-compatible alias for :func:`calculate_ksg_mi` (used by ``mi_suggestions``)."""
    _ = param_names, metric_names
    return calculate_ksg_mi(X, Y)


def plot_gaussian_mi_results(mi_matrix, parameter_names=None, metric_names=None, save_path=None):
    """Plot MI matrix from :func:`compute_gaussian_mi_matrix`."""
    return plot_mi_importance(mi_matrix, parameter_names, metric_names, save_path=save_path)


def plot_ksg_mi_results(mi_matrix, parameter_names=None, metric_names=None, save_path=None):
    """Plot MI matrix from :func:`compute_ksg_mi_matrix`."""
    return plot_mi_importance(mi_matrix, parameter_names, metric_names, save_path=save_path)


def _create_horizontal_bar_plot(sorted_params, sorted_means, sorted_stds):
    # Create horizontal bar plot
    fig, ax = plt.subplots(figsize=(6, 5))
    y_pos = np.arange(len(sorted_params))
    bars = ax.barh(
        y_pos,
        sorted_means,
        xerr=[np.zeros(len(sorted_means)), sorted_stds],
        capsize=5,
        alpha=0.7,
        color=None,
        facecolor="none",
        edgecolor="black",
        linewidth=1.5,
    )

    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_params, fontsize=12)  # , fontweight='bold')
    # ax.set_xlabel('Mean Mutual Information Across Metrics', fontsize=12, fontweight='bold')
    # ax.set_ylabel('Parameters', fontsize=12, fontweight='bold')
    # ax.set_xlim(0, 0.5)
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    # ax.grid(True, alpha=0.3, axis='x')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    return fig, ax


def plot_mi_importance(
    mi_matrix, parameter_names=None, metric_names=None, save_path=None, title=None
):
    """
    Plot parameter importance (mean MI across metrics) with standard deviation.

    Parameters:
    -----------
    mi_matrix : array-like, shape (n_parameters, n_metrics)
        Mutual information matrix where each row is a parameter and each column is a metric
    parameter_names : list, optional
        Names of parameters/features
    metric_names : list, optional
        Names of metrics/targets (used for reference only)
    title : str
        Plot title
    """

    mi_matrix = np.array(mi_matrix)
    n_parameters, n_metrics = mi_matrix.shape
    # normalize mi_matrix across parameters
    mi_matrix = mi_matrix / np.max(mi_matrix, axis=0, keepdims=True)

    # Calculate mean and std across metrics for each parameter
    mean_importance = np.mean(mi_matrix, axis=1)  # Mean across metrics
    std_importance = np.std(mi_matrix, axis=1)  # Std across metrics

    # Default parameter names if not provided
    if parameter_names is None:
        parameter_names = [f"Parameter_{i+1}" for i in range(n_parameters)]

    # Sort by mean importance (descending)
    sorted_idx = np.argsort(mean_importance)
    sorted_params = [parameter_names[i] for i in sorted_idx]
    sorted_means = mean_importance[sorted_idx]
    sorted_stds = std_importance[sorted_idx]
    fig, ax = _create_horizontal_bar_plot(sorted_params, sorted_means, sorted_stds)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    return {
        "parameter_names": sorted_params,
        "mean_importance": sorted_means,
        "std_importance": sorted_stds,
        "ranking": sorted_idx,
    }


def run_complete_mi_analysis(X, Y, parameter_names=None, metric_names=None, save_path=None):
    """
    Run complete MI analysis: calculate MI for each parameter-metric pair,
    then show parameter importance across all metrics.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_parameters)
        Parameter data
    Y : array-like, shape (n_samples, n_metrics)
        Metric data
    parameter_names : list, optional
        Names of parameters
    metric_names : list, optional
        Names of metrics

    Returns:
    --------
    dict : Complete analysis results
    """

    # Calculate MI matrix
    mi_matrix = calculate_ksg_mi(X, Y)
    # Default names
    if parameter_names is None:
        parameter_names = [f"Param_{i+1}" for i in range(mi_matrix.shape[0])]
    if metric_names is None:
        metric_names = [f"Metric_{i+1}" for i in range(mi_matrix.shape[1])]

    # Plot the results
    results = plot_mi_importance(mi_matrix, parameter_names, metric_names, save_path=save_path)
    return {
        #'mi_matrix': mi_matrix,
        "parameter_names": results["parameter_names"],
        "importance": results["mean_importance"],
        "std_error": results["std_importance"],
        #'importance_ranking': results
    }


def analyze_parameter_rankings(data_dict, figsize=(10, 8), save_path=None):
    """
    Analyze and visualize how parameter rankings change across different thresholds.
    Shows only the ranking heatmap.

    Parameters:
    data_dict: Dictionary where keys are threshold names and values are lists of tuples
               [(parameter_name, mutual_info_value, std_error), ...]
    figsize: Tuple for figure size
    save_path: Path to save the figure

    Returns:
    fig, ax: matplotlib figure and axis objects
    """

    # Convert data to DataFrame for easier manipulation
    all_data = []
    for threshold, params in data_dict.items():
        for param_name, mi_value, std_error in params:
            all_data.append(
                {
                    "threshold": threshold,
                    "parameter": param_name,
                    "mutual_info": mi_value,
                    "std_error": std_error,
                }
            )

    df = pd.DataFrame(all_data)

    # Calculate rankings for each threshold
    df["rank"] = df.groupby("threshold")["mutual_info"].rank(method="dense", ascending=False)

    sorted_params = df[df["threshold"] == "Original"]["parameter"].unique()[::-1]

    # Get all unique parameters
    all_params = df["parameter"].unique()
    sort_order = [np.where(all_params == param)[0][0] for param in sorted_params]
    thresholds = list(data_dict.keys())

    # Create MI matrix
    mi_matrix = np.full((len(all_params), len(thresholds)), np.nan)
    param_to_idx = {param: i for i, param in enumerate(all_params)}
    threshold_to_idx = {thresh: i for i, thresh in enumerate(thresholds)}

    for _, row in df.iterrows():
        param_idx = param_to_idx[row["parameter"]]
        thresh_idx = threshold_to_idx[row["threshold"]]
        mi_matrix[param_idx, thresh_idx] = row["mutual_info"]

    mi_matrix_sorted = mi_matrix[sort_order, :]
    sorted_params = [all_params[i] for i in sort_order]

    # Create single heatmap visualization
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Create discrete color mapping for MI categories
    # Low MI: < 0.3 (light blue), Medium MI: 0.3-0.7 (yellow), High MI: > 0.7 (red)
    from matplotlib.colors import ListedColormap, BoundaryNorm

    # Define colors for each category
    colors = ["lightblue", "yellow", "red"]  # Low, Medium, High
    cmap = ListedColormap(colors)

    # Create MI category matrix (0=low, 1=medium, 2=high)
    mi_category_matrix = np.full_like(mi_matrix_sorted, np.nan)
    mi_category_matrix[mi_matrix_sorted < 0.3] = 0  # Low
    mi_category_matrix[(mi_matrix_sorted >= 0.3) & (mi_matrix_sorted < 0.7)] = 1  # Medium
    mi_category_matrix[mi_matrix_sorted >= 0.7] = 2  # High

    # Create heatmap with discrete color scheme
    im = ax.imshow(mi_category_matrix, cmap=cmap, aspect="auto", vmin=-0.5, vmax=2.5)

    # Set ticks and labels
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels(thresholds, rotation=45, ha="right", fontsize=12)  # , fontweight='bold')
    ax.set_yticks(range(len(sorted_params)))
    ax.set_yticklabels(
        [param.replace("_", " ").replace(" MU", "").capitalize() for param in sorted_params],
        fontsize=12,
    )  # , fontweight='bold')

    if 1:
        # Add MI values to heatmap
        for i in range(len(sorted_params)):
            for j in range(len(thresholds)):
                if not np.isnan(mi_matrix_sorted[i, j]):
                    mi_val = mi_matrix_sorted[i, j]
                    category = mi_category_matrix[i, j]
                    # Choose text color based on background color for better visibility
                    # Light blue (low): black text, Yellow (medium): black text, Red (high): white text
                    text_color = "white" if category == 2 else "black"
                    ax.text(
                        j,
                        i,
                        f"{mi_val:.2f}",
                        ha="center",
                        va="center",
                        # fontweight='bold',
                        fontsize=9,
                        color=text_color,
                    )

    # Add grid for better readability
    ax.set_xticks(np.arange(-0.5, len(thresholds), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(sorted_params), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
    # Hide minor tick marks but keep the grid lines
    ax.tick_params(axis="both", which="minor", length=0)
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    # for label in ax.get_xticklabels():
    #    label.set_fontweight('bold')
    # for label in ax.get_yticklabels():
    #    label.set_fontweight('bold')
    # ax.grid(True, alpha=0.3, axis='x')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_linewidth(1.0)
    # ax.spines['bottom'].set_linewidth(1.0)

    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Ranking heatmap saved to: {save_path}")

    return fig, ax


def extract_mi_data_from_results(results, scenario_name):
    """
    Extract mutual information data from run_complete_mi_analysis results.

    Parameters:
    -----------
    results : dict
        Results from run_complete_mi_analysis
    scenario_name : str
        Name of the scenario/threshold

    Returns:
    --------
    list : List of tuples (parameter_name, mean_mi_across_metrics, std_mi_across_metrics)
    """
    mi_matrix = results["mi_matrix"]
    parameter_names = results["parameter_names"]

    # Calculate mean and std MI across all metrics for each parameter
    mean_mi_per_param = np.mean(mi_matrix, axis=1)
    std_mi_per_param = np.std(mi_matrix, axis=1)

    # Create list of tuples
    param_data = []
    for i, param_name in enumerate(parameter_names):
        param_data.append((param_name, mean_mi_per_param[i], std_mi_per_param[i]))

    return param_data


def build_data_dict_from_scenario_results(scenario_results_dict):
    """
    Build data_dict for ranking analysis from multiple scenario results.

    Parameters:
    -----------
    scenario_results_dict : dict
        Dictionary where keys are scenario names and values are results from run_complete_mi_analysis

    Returns:
    --------
    dict : data_dict formatted for analyze_parameter_rankings
    """
    data_dict = {}

    for scenario_name, results in scenario_results_dict.items():
        data_dict[scenario_name] = extract_mi_data_from_results(results, scenario_name)

    return data_dict


def ksg_mutual_information(parameters, metrics, k=3, standardize=True):
    """
    Calculate KSG mutual information between each parameter and the metrics.

    Parameters:
    -----------
    parameters : np.ndarray
        Shape (n_samples, n_parameters) - input parameters
    metrics : np.ndarray
        Shape (n_samples, n_metrics) - output metrics
    k : int
        Number of nearest neighbors for KSG estimation (default: 3)
    standardize : bool
        Whether to standardize inputs (recommended, default: True)

    Returns:
    --------
    mi_values : np.ndarray
        Shape (n_parameters,) - mutual information for each parameter
    """
    n_samples, n_parameters = parameters.shape
    n_metrics = metrics.shape[1]

    if parameters.shape[0] != metrics.shape[0]:
        raise ValueError("Number of samples must match between parameters and metrics")

    # Standardize inputs if requested
    if standardize:
        # Standardize parameters (each column independently)
        parameters = (parameters - np.mean(parameters, axis=0)) / np.std(parameters, axis=0)
        # Standardize metrics (each column independently)
        metrics = (metrics - np.mean(metrics, axis=0)) / np.std(metrics, axis=0)

    mi_values = np.zeros(n_parameters)

    for param_idx in range(n_parameters):
        # Get current parameter (X) and all metrics (Y)
        X = parameters[:, param_idx : param_idx + 1]  # Shape: (n_samples, 1)
        Y = metrics  # Shape: (n_samples, n_metrics)

        # Create joint space Z = [X, Y]
        Z = np.hstack([X, Y])  # Shape: (n_samples, 1 + n_metrics)

        # Calculate distances in joint space Z
        Z_distances = cdist(Z, Z, metric="euclidean")
        np.fill_diagonal(Z_distances, np.inf)  # Exclude self-distances

        # Find k-th nearest neighbor distances in joint space
        kth_distances = np.partition(Z_distances, k - 1, axis=1)[:, k - 1]

        # Count neighbors within k-th distance in marginal spaces
        nx_counts = np.zeros(n_samples)
        ny_counts = np.zeros(n_samples)

        for i in range(n_samples):
            eps = kth_distances[i]

            # Count neighbors in X space (parameter space)
            X_distances = np.abs(X - X[i]).flatten()
            nx_counts[i] = np.sum(X_distances < eps)

            # Count neighbors in Y space (metrics space)
            Y_distances = cdist(Y[i : i + 1], Y, metric="euclidean").flatten()
            ny_counts[i] = np.sum(Y_distances < eps)

        # KSG estimator formula
        # MI = digamma(k) - mean(digamma(nx + 1)) - mean(digamma(ny + 1)) + digamma(n)
        mi_estimate = (
            digamma(k)
            - np.mean(digamma(nx_counts + 1))
            - np.mean(digamma(ny_counts + 1))
            + digamma(n_samples)
        )

        mi_values[param_idx] = max(0, mi_estimate)  # Ensure non-negative MI

    return mi_values


# Example usage and demonstration
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Load the data

    from inverse_design.io.scenarios import combined_grid_n512_breast

    _paths = combined_grid_n512_breast()
    base_original_dir = _paths.base_original_dir
    linear_model_dirs = list(_paths.linear_model_dirs)
    dendrogram_threshold_dirs = list(_paths.dendrogram_threshold_dirs)
    ext_linear_dirs = list(_paths.ext_linear_dirs)
    ext_dendrogram_threshold_dirs = list(_paths.ext_dendrogram_threshold_dirs)
    save_path_folder = "../../../ARCADE_OUTPUT/simplified_models_vis"

    base_layout = ArcadeRunLayout.from_root(base_original_dir)
    posterior_df = pd.read_csv(base_layout.all_param_df_csv("iter_4"))
    final_metrics_df = pd.read_csv(base_layout.final_metrics_csv("iter_4"))
    with open(base_layout.targets_json(), encoding="utf-8") as target_file:
        target_metrics = json.load(target_file)
    target_metrics = {k: v for k, v in target_metrics.items() if not k.endswith("std")}
    final_metrics_df = final_metrics_df[target_metrics.keys()]

    drop_cols = [
        "input_folder",
        "X_SPACING",
        "Y_SPACING",
        "DISTANCE_TO_CENTER",
    ]  # , "CAPILLARY_DENSITY", "AUTOPHAGY_RATE_SIGMA"]
    posterior_df.drop(columns=drop_cols, inplace=True)
    param_names = posterior_df.columns
    metrics_names = list(target_metrics.keys())
    simplified_methods = ["linear", "threshold"]
    simplified_method = simplified_methods[1]
    scenario_names = (
        ["Original", "r=0.9", "r=0.8", "r=0.7", "r=0.01"]
        if simplified_method == "linear"
        else ["Original", "t=1.0", "t=1.25", "t=1.5", "t=5.0"]
    )
    peak_idx = 3
    simplified_posterior_dirs = (
        linear_model_dirs if simplified_method == "linear" else dendrogram_threshold_dirs
    )
    if peak_idx == 3 and simplified_method == "linear":
        scenario_names = scenario_names[0:1] + scenario_names[3:]  # for peak 3
        simplified_posterior_dirs = simplified_posterior_dirs[peak_idx * 3 + 2 : (peak_idx + 1) * 3]
    else:
        simplified_posterior_dirs = simplified_posterior_dirs[peak_idx * 3 : (peak_idx + 1) * 3]
    simplified_posterior_dirs += (
        ext_linear_dirs[peak_idx : peak_idx + 1]
        if simplified_method == "linear"
        else ext_dendrogram_threshold_dirs[peak_idx : peak_idx + 1]
    )
    simplified_posterior_dfs = []
    simplified_final_metrics_dfs = []
    simplified_analysis_results = []
    for idx, dir in enumerate(simplified_posterior_dirs):
        simplified_layout = ArcadeRunLayout.from_root(dir)
        simplified_posterior_dfs.append(pd.read_csv(simplified_layout.all_param_df_csv("iter_4")))
        simplified_final_metrics_dfs.append(
            pd.read_csv(simplified_layout.final_metrics_csv("iter_4"))
        )
        r = scenario_names[idx + 1].split("=")[1]
        if simplified_method == "linear":
            simplified_analysis_results.append(
                json.load(open(f"{dir}/lr_predictions_r{r}_p{peak_idx+1}.json"))
            )
    if simplified_method == "threshold":
        simplified_analysis_results = [None] * (len(simplified_posterior_dirs) + 1)
    else:
        simplified_analysis_results = [None] + simplified_analysis_results
    print("\nGenerating plots...")
    # Initialize analyzer
    if 0:
        posterior_df_filtered = posterior_df.drop(
            columns=[col for col in posterior_df.columns if col.endswith("_SIGMA")]
        )
        param_names_filtered = [col for col in param_names if not col.endswith("_SIGMA")]
        analyzer = ABCParameterImportance(
            posterior_df_filtered, final_metrics_df, param_names_filtered, metrics_names
        )
        # Calculate all importance measures
        results = analyzer.calculate_all_importance()

        # Get summary
        summary = analyzer.get_importance_summary()
        print("Parameter Importance Summary:")
        print(summary)
        analyzer.plot_importance_comparison(
            save_path=f"{save_path_folder}/importance_comparison.png"
        )
        # Plot overall importance comparison
        analyzer.plot_overall_importance_comparison(
            save_path=f"{save_path_folder}/overall_importance_comparison.png"
        )
    importance_dfs = []
    if 1:
        importance_dfs_dict = {}
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()
        # Perform PCA and find peaks using your existing function
        pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
            posterior_df, n_components=2
        )
        # Assign each point to the nearest peak (same as your existing code)
        point_assignments = []
        if len(peak_positions) > 0:
            for point in pca_result[:, :2]:
                distances = [np.sqrt(np.sum((point - peak) ** 2)) for peak in peak_positions]
                nearest_peak = np.argmin(distances)
                point_assignments.append(nearest_peak)
            point_assignments = np.array(point_assignments)
        target_peak_mask = point_assignments == peak_idx
        posterior_df_filtered = posterior_df[target_peak_mask]
        final_metrics_df_filtered = final_metrics_df[target_peak_mask]
        # drop if param ends with _SIGMA
        posterior_df_filtered = posterior_df_filtered.drop(
            columns=[col for col in posterior_df_filtered.columns if col.endswith("_SIGMA")]
        )
        param_names_filtered = [col for col in param_names if not col.endswith("_SIGMA")]
        process_posterior_dfs = [posterior_df_filtered] + simplified_posterior_dfs
        process_final_metrics_dfs = [final_metrics_df_filtered] + simplified_final_metrics_dfs
        for idx, (posterior_df, final_metrics_df, simplified_analysis_result) in enumerate(
            zip(process_posterior_dfs, process_final_metrics_dfs, simplified_analysis_results)
        ):
            sampled_parameter_names = param_names_filtered
            if idx != 0:
                # drop_cols should be the intersection of drop_cols and posterior_df.columns
                drop_cols = list(set(drop_cols) & set(posterior_df.columns))
                posterior_df.drop(columns=drop_cols, inplace=True)
                posterior_df.drop(
                    columns=[col for col in posterior_df.columns if col.endswith("_SIGMA")],
                    inplace=True,
                )
                final_metrics_df = final_metrics_df[metrics_names]
                if simplified_method == "linear":
                    sampled_parameter_names = simplified_analysis_result["sampling_instructions"][
                        "parameters_to_sample"
                    ]["independent"]
                    sampled_parameter_names += simplified_analysis_result["sampling_instructions"][
                        "parameters_to_sample"
                    ]["root"]
                    sampled_parameter_names = [
                        param for param in sampled_parameter_names if param in param_names_filtered
                    ]
                else:
                    sampled_parameter_names = posterior_df.columns
            save_path = f"{save_path_folder}/ksg_mi_all_{scenario_names[idx]}.png"
            # Find indices of rows that are not nan in both posterior_df and final_metrics_df
            valid_indices = (
                ~np.isnan(posterior_df).any(axis=1)
                & ~np.isnan(final_metrics_df).any(axis=1)
                & ~np.isinf(final_metrics_df).any(axis=1)
                & ~np.isinf(posterior_df).any(axis=1)
            )
            posterior_df = posterior_df[sampled_parameter_names]
            if 0:
                if len(posterior_df.columns) <= 2:
                    importance_df = pd.DataFrame(
                        {
                            "parameter": posterior_df.columns,
                            "mutual_info": np.zeros(len(posterior_df.columns)),
                            "std_error": np.zeros(len(posterior_df.columns)),
                        }
                    ).sort_values(by="mutual_info", ascending=True)
                    importance_dfs.append(importance_df)
                    continue

                X = scaler_X.fit_transform(posterior_df[valid_indices])
                Y = scaler_Y.fit_transform(final_metrics_df[valid_indices])

                pca = PCA(n_components=2)
                pca.fit(X)
                X_pca = pca.transform(X)
                loadings = pca.components_.T  # shape: (n_parameters, n_components)

                # Variance explained by each PC
                explained_var = pca.explained_variance_ratio_
                importance = np.sum(np.abs(loadings) * explained_var, axis=1)

                # Put into a dataframe for readability
                importance_df = pd.DataFrame(
                    {
                        "parameter": posterior_df.columns,
                        "mutual_info": importance,
                        "std_error": np.zeros(len(posterior_df.columns)),
                    }
                ).sort_values(by="mutual_info", ascending=True)
                importance_dfs.append(importance_df)
                continue
            else:
                X = scaler_X.fit_transform(posterior_df[valid_indices])
                Y = scaler_Y.fit_transform(final_metrics_df[valid_indices])
                results = run_complete_mi_analysis(
                    X,
                    Y,
                    posterior_df.columns,
                    metrics_names,
                    save_path=f"{save_path_folder}/ksg_mi_p{peak_idx+1}_{scenario_names[idx]}_{simplified_method}.png",
                )
                # Store results for ranking analysis
                importance_array = np.array(
                    [
                        (param, float(results["importance"][i]), float(results["std_error"][i]))
                        for i, param in enumerate(results["parameter_names"])
                    ]
                )
                importance_df = pd.DataFrame(
                    importance_array, columns=["parameter", "mutual_info", "std_error"]
                ).sort_values(by="mutual_info", ascending=True)
                importance_dfs.append(importance_df)
                importance_dfs_dict[scenario_names[idx]] = np.array(importance_dfs[-1])
                # missing parameters
                missing_params = [
                    param
                    for param in param_names_filtered
                    if param not in results["parameter_names"]
                ]
                print(f"missing parameters: {missing_params}")
        # importance_dfs_dict = {scenario_names[idx]: np.array(importance_df) for idx, importance_df in enumerate(importance_dfs)}
        for key, value in importance_dfs_dict.items():
            print(f"key: {key}")
            print(f"value: {value}")
        # removed legacy debug hook
        print(f"importance_dfs_dict: {importance_dfs_dict['Original']}")
        fig, axes = analyze_parameter_rankings(
            importance_dfs_dict,
            figsize=(4, 5),
            save_path=f"{save_path_folder}/parameter_rankings_pca_p{peak_idx+1}_{simplified_method}_mi_analysis.png",
        )
