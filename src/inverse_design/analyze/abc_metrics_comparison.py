"""ABC posterior metrics comparison and plotting helpers."""

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler
from inverse_design.plotting.colormap import DEFAULT_METRIC_COLORS
from inverse_design.plotting.theme import apply_publication_style
from inverse_design.utils.utils import remove_outliers

metric_colors = dict(DEFAULT_METRIC_COLORS)
apply_publication_style(font_size=12, axes_linewidth=1.0)


class ABCMetricsComparison:
    """
    Class to quantify differences between target metrics and simulation metrics
    from ABC posterior distribution sampling.
    """

    def __init__(self, target_metrics, simulation_metrics, metric_names=None):
        """
        Initialize with target and simulation metrics.

        Parameters:
        -----------
        target_metrics : dict or array-like
            Target metric values. If dict, keys should be metric names.
            If array-like, should be [doubling_time, std_doubling, symmetry, std_symmetry]
        simulation_metrics : array-like of shape (n_samples, n_metrics)
            Simulation metrics from posterior sampling
        metric_names : list, optional
            Names of metrics. Default: ['doubling_time', 'std_doubling', 'symmetry', 'std_symmetry']
        """

        if metric_names is None:
            self.metric_names = ["doubling_time", "std_doubling", "symmetry", "std_symmetry"]
        else:
            self.metric_names = metric_names

        # Convert target_metrics to array if it's a dict
        if isinstance(target_metrics, dict):
            self.target_metrics = np.array([target_metrics[name] for name in self.metric_names])
        else:
            self.target_metrics = np.array(target_metrics)

        self.simulation_metrics = np.array(simulation_metrics)

        # Ensure shapes are compatible
        if len(self.target_metrics) != self.simulation_metrics.shape[1]:
            raise ValueError(
                "Number of target metrics must match number of simulation metric columns"
            )

    def compute_distance_metrics(self):
        """
        Compute various distance metrics between target and simulation metrics.
        """
        n_samples = self.simulation_metrics.shape[0]
        results = {}

        # 1. Euclidean distance (raw)
        euclidean_distances = np.sqrt(
            np.sum((self.simulation_metrics - self.target_metrics) ** 2, axis=1)
        )
        results["euclidean_raw"] = {
            "distances": euclidean_distances,
            "mean": np.mean(euclidean_distances),
            "std": np.std(euclidean_distances),
            "median": np.median(euclidean_distances),
        }

        # 2. Normalized Euclidean distance (by target values)
        target_normalized = self.target_metrics.copy()
        target_normalized[target_normalized == 0] = 1e-10  # Avoid division by zero
        normalized_diff = (self.simulation_metrics - self.target_metrics) / np.abs(
            target_normalized
        )
        euclidean_normalized = np.sqrt(np.sum(normalized_diff**2, axis=1))
        results["euclidean_normalized"] = {
            "distances": euclidean_normalized,
            "mean": np.mean(euclidean_normalized),
            "std": np.std(euclidean_normalized),
            "median": np.median(euclidean_normalized),
        }

        # 3. Standardized distance (z-score based)
        scaler = StandardScaler()
        # Fit on combined data to get proper scaling
        combined_data = np.vstack([self.target_metrics.reshape(1, -1), self.simulation_metrics])
        combined_scaled = scaler.fit_transform(combined_data)
        target_scaled = combined_scaled[0]
        sim_scaled = combined_scaled[1:]

        standardized_distances = np.sqrt(np.sum((sim_scaled - target_scaled) ** 2, axis=1))
        results["euclidean_standardized"] = {
            "distances": standardized_distances,
            "mean": np.mean(standardized_distances),
            "std": np.std(standardized_distances),
            "median": np.median(standardized_distances),
        }

        # 4. Manhattan distance (L1 norm)
        manhattan_distances = np.sum(np.abs(self.simulation_metrics - self.target_metrics), axis=1)
        results["manhattan_raw"] = {
            "distances": manhattan_distances,
            "mean": np.mean(manhattan_distances),
            "std": np.std(manhattan_distances),
            "median": np.median(manhattan_distances),
        }

        # 5. Relative percentage error for each metric
        relative_errors = (
            (self.simulation_metrics - self.target_metrics)
            / np.maximum(np.abs(self.target_metrics), 1e-10)
            * 100
        )
        results["relative_errors"] = {
            "errors": relative_errors,
            "mean_per_metric": np.mean(relative_errors, axis=0),
            "std_per_metric": np.std(relative_errors, axis=0),
            "overall_mean": np.mean(relative_errors),
        }

        # 6. Mahalanobis-like distance (using simulation covariance)
        sim_cov = np.cov(self.simulation_metrics.T)
        try:
            sim_cov_inv = np.linalg.inv(sim_cov)
            mahal_distances = []
            for sim in self.simulation_metrics:
                diff = sim - self.target_metrics
                mahal_dist = np.sqrt(diff.T @ sim_cov_inv @ diff)
                mahal_distances.append(mahal_dist)
            mahal_distances = np.array(mahal_distances)
            results["mahalanobis"] = {
                "distances": mahal_distances,
                "mean": np.mean(mahal_distances),
                "std": np.std(mahal_distances),
                "median": np.median(mahal_distances),
            }
        except np.linalg.LinAlgError:
            results["mahalanobis"] = None
            print("Warning: Could not compute Mahalanobis distance (singular covariance matrix)")

        return results

    def compute_metric_specific_statistics(self):
        """
        Compute statistics for each individual metric.
        """
        stats_per_metric = {}

        for i, metric_name in enumerate(self.metric_names):
            target_val = self.target_metrics[i]
            sim_vals = self.simulation_metrics[:, i]
            stats_per_metric[metric_name] = {
                "target_value": target_val,
                "simulation_mean": np.mean(sim_vals),
                "simulation_std": np.std(sim_vals),
                "simulation_median": np.median(sim_vals),
                "bias": np.mean(sim_vals) - target_val,
                "mse": mean_squared_error([target_val] * len(sim_vals), sim_vals),
                "mae": mean_absolute_error([target_val] * len(sim_vals), sim_vals),
                "relative_bias_percent": (
                    (np.mean(sim_vals) - target_val) / max(abs(target_val), 1e-10)
                )
                * 100,
                "coverage_95": np.percentile(sim_vals, [2.5, 97.5]),
                "target_in_95_interval": (
                    np.percentile(sim_vals, 2.5) <= target_val <= np.percentile(sim_vals, 97.5)
                ),
            }

        return stats_per_metric

    def plot_comparison(self, figsize=(15, 5), save_path=None):
        """
        Create comprehensive comparison plots.
        """
        fig, axes = plt.subplots(1, len(self.metric_names), figsize=figsize)
        # Individual metric comparisons
        for i, metric_name in enumerate(self.metric_names):
            ax = axes[i]

            if ax is not None:
                target_val = self.target_metrics[i]
                sim_vals = self.simulation_metrics[:, i]

                # Histogram of simulation values
                ax.hist(
                    sim_vals,
                    bins=30,
                    alpha=0.7,
                    density=True,
                    color="lightblue",
                    edgecolor="black",
                    linewidth=0.5,
                )

                # Target value line
                ax.axvline(target_val, color="red", linestyle="--", linewidth=2, label=f"Target")

                # Simulation mean line
                sim_mean = np.mean(sim_vals)
                ax.axvline(sim_mean, color="blue", linestyle="-", linewidth=2, label=f"Sim mean")
                ax.set_title(f'{metric_name.replace("_", " ").title()}')
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                if i == len(self.metric_names) - 1:
                    ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        return fig

    def generate_summary_report(self, verbose=False):
        """
        Generate a comprehensive summary report.
        """
        distance_results = self.compute_distance_metrics()
        metric_stats = self.compute_metric_specific_statistics()

        if verbose:
            print("=" * 80)
            print("ABC POSTERIOR METRICS COMPARISON REPORT")
            print("=" * 80)

            print("\n1. OVERALL DISTANCE METRICS:")
            print("-" * 40)
            for dist_name, dist_data in distance_results.items():
                if dist_data is not None:
                    if "mean" in dist_data:
                        print(
                            f"{dist_name.replace('_', ' ').title():.<25} "
                            f"Mean: {dist_data['mean']:.4f}, "
                            f"Std: {dist_data['std']:.4f}, "
                            f"Median: {dist_data['median']:.4f}"
                        )

            print("\n2. INDIVIDUAL METRIC ANALYSIS:")
            print("-" * 40)
            for metric_name, stats in metric_stats.items():
                print(f"\n{metric_name.replace('_', ' ').title()}:")
                print(f"  Target Value:           {stats['target_value']:.4f}")
                print(f"  Simulation Mean:        {stats['simulation_mean']:.4f}")
                print(f"  Simulation Std:         {stats['simulation_std']:.4f}")
                print(f"  Bias:                   {stats['bias']:.4f}")
                print(f"  Relative Bias (%):      {stats['relative_bias_percent']:.2f}%")
                print(f"  MSE:                    {stats['mse']:.6f}")
                print(f"  MAE:                    {stats['mae']:.4f}")
                print(
                    f"  95% Coverage:           [{stats['coverage_95'][0]:.4f}, {stats['coverage_95'][1]:.4f}]"
                )
                print(f"  Target in 95% Interval: {stats['target_in_95_interval']}")

            print("\n3. RELATIVE ERROR SUMMARY:")
            print("-" * 40)
            rel_errors = distance_results["relative_errors"]
            for i, metric_name in enumerate(self.metric_names):
                print(
                    f"{metric_name.replace('_', ' ').title():.<25} "
                    f"Mean Error: {rel_errors['mean_per_metric'][i]:.2f}%, "
                    f"Std Error: {rel_errors['std_per_metric'][i]:.2f}%"
                )
            print(f"{'Overall Mean Relative Error':.<25} {rel_errors['overall_mean']:.2f}%")

        return distance_results, metric_stats


def analyze_iteration(
    parameter_base_folder,
    iteration,
    metric_names,
    target_metrics,
    target_peak=0,
    verbose=False,
    find_peaks=False,
    save_plots=True,
):
    """
    Analyze metrics for a single iteration.

    Parameters:
    -----------
    parameter_base_folder : str
        Base folder path containing iteration results
    iteration : int
        Iteration number to analyze
    metric_names : list
        List of metric names to analyze
    target_metrics : array-like
        Target metric values

    Returns:
    --------
    tuple
        (comparator, distance_results, metric_stats, best_fit_sample, best_fit_distance)
    """
    from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks

    posterior_metrics_file = f"{parameter_base_folder}/iter_{iteration}/final_metrics.csv"
    posterior_metrics_df = pd.read_csv(posterior_metrics_file)
    posterior_params_df = pd.read_csv(f"{parameter_base_folder}/iter_{iteration}/all_param_df.csv")
    drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
    drop_cols = [col for col in drop_cols if col in posterior_params_df.columns]
    try:
        posterior_params_df.drop(columns=drop_cols, inplace=True)
    except:
        drop_cols = ["input_folder", "DISTANCE_TO_CENTER"]
        posterior_params_df.drop(columns=drop_cols, inplace=True)
    if find_peaks:
        pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
            posterior_params_df, n_components=2
        )
        # Get samples around this peak
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[target_peak]) ** 2, axis=1))
        closest_indices = np.argsort(distances)[:50]
        posterior_metrics_df = posterior_metrics_df.iloc[closest_indices]
        posterior_params_df = posterior_params_df.iloc[closest_indices]
    n_samples = posterior_metrics_df.shape[0]
    posterior_metrics_df = posterior_metrics_df[metric_names]
    posterior_metrics_df = posterior_metrics_df.replace([np.inf, -np.inf], np.nan).dropna()
    posterior_metrics_df, _ = remove_outliers(posterior_metrics_df, 1.5)
    posterior_metrics_df = posterior_metrics_df.dropna()
    print(f"Removed {n_samples - posterior_metrics_df.shape[0]} outliers for {metric_names} in df")

    n_samples = posterior_metrics_df.shape[0]
    simulation_metrics = np.array(
        [
            posterior_metrics_df[metric_names[0]],  # symmetry
            posterior_metrics_df[metric_names[1]],  # symmetry_std
            posterior_metrics_df[metric_names[2]],  # doub_time
            posterior_metrics_df[metric_names[3]],  # doub_time_std
            posterior_metrics_df[metric_names[4]],  # colony_growth
        ]
    ).T

    # Create comparison object
    comparator = ABCMetricsComparison(target_metrics, simulation_metrics, metric_names)

    # Generate report
    distance_results, metric_stats = comparator.generate_summary_report(verbose=verbose)

    if save_plots:
        save_path = f"{parameter_base_folder}/iter_{iteration}/posterior_comparison.png"
        comparator.plot_comparison(save_path=save_path)

    # Get best fit sample
    best_fit_idx = np.argmin(distance_results["euclidean_normalized"]["distances"])
    best_fit_sample = simulation_metrics[best_fit_idx]
    best_fit_distance = distance_results["euclidean_normalized"]["distances"][best_fit_idx]

    return (
        comparator,
        distance_results,
        metric_stats,
        best_fit_sample,
        best_fit_distance,
        best_fit_idx,
    )


def plot_error_trends(
    iterations, metric_names, all_simulation_metrics, target_metrics, cmap="viridis", save_path=None
):
    """
    Plot error trends across iterations using stacked bars for metric-wise contributions.

    Parameters:
    -----------
    iterations : list
        List of iteration numbers
    mean_errors : list
        List of mean errors for each iteration
    best_errors : list
        List of best errors for each iteration
    metric_names : list
        List of metric names
    all_simulation_metrics : list
        List of simulation metrics arrays for each iteration
    target_metrics : array-like
        Target metric values
    save_path : str, optional
        Path to save the plot
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Calculate metric-wise contributions for mean errors
    mean_metric_errors = []
    for sim_metrics in all_simulation_metrics:
        # Get the normalized differences for each metric
        metric_errors = np.abs(
            (sim_metrics - target_metrics) / np.maximum(np.abs(target_metrics), 1e-10)
        )
        mean_metric_errors.append(np.mean(metric_errors, axis=0))

    mean_metric_errors = np.array(mean_metric_errors)

    # Plot stacked bars for mean errors
    bottom = np.zeros(len(iterations))
    for i, metric in enumerate(metric_names):
        ax1.bar(
            iterations,
            mean_metric_errors[:, i],
            bottom=bottom,
            label=metric,
            color=cmap((i + 1) / len(metric_names)),
            alpha=0.9,
            # edgecolor='black',
            # linewidth=0.75
        )
        bottom += mean_metric_errors[:, i]

    ax1.set_title("Mean Error Contributions by Metric")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Normalized Error")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Calculate metric-wise contributions for best errors
    best_metric_errors = []
    for sim_metrics in all_simulation_metrics:
        # Get the best sample's metric-wise errors
        metric_errors = np.abs(
            (sim_metrics - target_metrics) / np.maximum(np.abs(target_metrics), 1e-10)
        )
        best_idx = np.argmin(np.sum(metric_errors, axis=1))
        best_metric_errors.append(metric_errors[best_idx])

    best_metric_errors = np.array(best_metric_errors)

    # Plot stacked bars for best errors
    bottom = np.zeros(len(iterations))
    for i, metric in enumerate(metric_names):
        ax2.bar(
            iterations,
            best_metric_errors[:, i],
            bottom=bottom,
            label=metric,
            color=cmap((i + 1) / len(metric_names)),
            alpha=0.9,
            # edgecolor='black',
            # linewidth=0.75
        )
        bottom += best_metric_errors[:, i]

    ax2.set_title("Best Sample Error Contributions by Metric")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Normalized Error")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_individual_metric_errors(
    iterations, metric_names, all_simulation_metrics, target_metrics, save_path=None
):
    """
    Plot error trends and metric values for each individual metric across iterations.

    Parameters:
    -----------
    iterations : list
        List of iteration numbers
    metric_names : list
        List of metric names
    all_simulation_metrics : list
        List of simulation metrics arrays for each iteration
    target_metrics : array-like
        Target metric values
    save_path : str, optional
        Path to save the plot
    """
    n_metrics = len(metric_names)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))

    for i, (metric_name, ax) in enumerate(zip(metric_names, axes)):
        # Create secondary y-axis for metric values
        ax2 = ax.twinx()

        # Calculate errors for this metric across iterations
        metric_errors = []
        metric_values = []
        for sim_metrics in all_simulation_metrics:
            # Calculate normalized error for this metric
            errors = np.abs(
                (sim_metrics[:, i] - target_metrics[i])
                / np.maximum(np.abs(target_metrics[i]), 1e-10)
            )
            metric_errors.append(
                {
                    "mean": np.mean(errors),
                    "std": np.std(errors),
                    "min": np.min(errors),
                    "max": np.max(errors),
                }
            )
            # Store metric values
            metric_values.append(
                {
                    "mean": np.mean(sim_metrics[:, i]),
                    "std": np.std(sim_metrics[:, i]),
                    "min": np.min(sim_metrics[:, i]),
                    "max": np.max(sim_metrics[:, i]),
                }
            )

        # Plot error statistics
        means = [err["mean"] for err in metric_errors]
        stds = [err["std"] for err in metric_errors]
        mins = [err["min"] for err in metric_errors]
        maxs = [err["max"] for err in metric_errors]

        # Plot mean error line
        ax.plot(iterations, means, "b-o", label="Mean Error", linewidth=2)

        # Plot std deviation as shaded area
        ax.fill_between(
            iterations,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            alpha=0.2,
            color="blue",
            label="±1 Std Dev",
        )

        # Plot metric values
        metric_means = [val["mean"] for val in metric_values]
        metric_stds = [val["std"] for val in metric_values]
        metric_mins = [val["min"] for val in metric_values]
        metric_maxs = [val["max"] for val in metric_values]

        # Plot mean metric value
        ax2.plot(iterations, metric_means, "k-o", label="Mean Value", linewidth=2)

        # Plot std deviation as shaded area
        ax2.fill_between(
            iterations,
            np.array(metric_means) - np.array(metric_stds),
            np.array(metric_means) + np.array(metric_stds),
            alpha=0.2,
            color="gray",
            label="±1 Std Dev",
        )

        # Plot target value as horizontal line
        ax2.axhline(y=target_metrics[i], color="r", linestyle="-", label="Target Value", alpha=0.5)

        # Customize plot
        ax.set_title(f'{metric_name.replace("_", " ").title()} Over Iterations')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Normalized Error", color="b")
        ax.tick_params(axis="y", labelcolor="b")
        ax.grid(True, alpha=0.3)

        # Customize secondary y-axis
        ax2.set_ylabel("Metric Value", color="k")
        ax2.tick_params(axis="y", labelcolor="k")

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, bbox_to_anchor=(1.15, 1), loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def lighten_color(color, amount=0.5):
    """Lightens the given color by blending it with white."""
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    white = np.array([1, 1, 1])
    return tuple((1 - amount) * np.array(c) + amount * white)


def plot_enhanced_metric_analysis(
    iterations,
    metric_names,
    all_simulation_metrics,
    target_metrics,
    save_path=None,
    plot_style="combined",
):
    """
    Enhanced visualization of metric errors and values across iterations.

    Parameters:
    -----------
    iterations : list
        List of iteration numbers
    metric_names : list
        List of metric names
    all_simulation_metrics : list
        List of simulation metrics arrays for each iteration
    target_metrics : array-like
        Target metric values
    save_path : str, optional
        Path to save the plot
    plot_style : str, optional
        'combined', 'separate', or 'dashboard' layout style
    """

    n_metrics = len(metric_names)

    if plot_style == "dashboard":
        return _plot_dashboard_style(
            iterations, metric_names, all_simulation_metrics, target_metrics, save_path
        )
    elif plot_style == "separate":
        return _plot_separate_style(
            iterations, metric_names, all_simulation_metrics, target_metrics, save_path
        )
    else:
        return _plot_combined_style(
            iterations, metric_names, all_simulation_metrics, target_metrics, save_path
        )


def _calculate_metric_statistics(all_simulation_metrics, target_metrics):
    """Calculate comprehensive statistics for metrics."""
    n_metrics = len(target_metrics)
    metric_stats = []
    error_stats = []

    for sim_metrics in all_simulation_metrics:
        iteration_metrics = []
        iteration_errors = []

        for i in range(n_metrics):
            # Metric values
            values = sim_metrics[:, i]
            iteration_metrics.append(
                {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75),
                    "min": np.min(values),
                    "max": np.max(values),
                }
            )

            # Error percentage
            errors = (values - target_metrics[i]) / target_metrics[i] * 100
            iteration_errors.append(
                {
                    "mean": np.mean(errors),
                    "std": np.std(errors),
                    "q25": np.percentile(errors, 25),
                    "q75": np.percentile(errors, 75),
                    "min": np.min(errors),
                    "max": np.max(errors),
                }
            )

        metric_stats.append(iteration_metrics)
        error_stats.append(iteration_errors)

    return metric_stats, error_stats


def _plot_combined_style(
    iterations, metric_names, all_simulation_metrics, target_metrics, save_path
):
    """Plot metrics and errors in a clean combined style."""
    n_metrics = len(metric_names)
    metric_stats, error_stats = _calculate_metric_statistics(all_simulation_metrics, target_metrics)

    # Create figure with better spacing
    fig = plt.figure(figsize=(15, 3 * n_metrics + 2))

    for i, metric_name in enumerate(metric_names):
        # Create subplot with better positioning
        ax = plt.subplot(n_metrics, 2, 2 * i + 1)
        ax2 = plt.subplot(n_metrics, 2, 2 * i + 2)

        # Extract data for this metric
        metric_means = [stats[i]["mean"] for stats in metric_stats]
        metric_q25 = [stats[i]["q25"] for stats in metric_stats]
        metric_q75 = [stats[i]["q75"] for stats in metric_stats]

        error_means = [stats[i]["mean"] for stats in error_stats]
        error_q25 = [stats[i]["q25"] for stats in error_stats]
        error_q75 = [stats[i]["q75"] for stats in error_stats]

        # Plot metric values with confidence intervals
        ax.plot(
            iterations,
            metric_means,
            "o-",
            linewidth=2.5,
            markersize=6,
            color="#2E86AB",
            label="Mean Value",
        )
        ax.fill_between(
            iterations, metric_q25, metric_q75, alpha=0.3, color="#2E86AB", label="IQR (25%-75%)"
        )
        ax.axhline(
            y=target_metrics[i],
            color="#F24236",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="Target",
        )

        # Plot error trends
        ax2.plot(
            iterations,
            error_means,
            "o-",
            linewidth=2.5,
            markersize=6,
            color="#A23B72",
            label="Mean Error",
        )
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=2, alpha=0.8)
        ax2.fill_between(
            iterations, error_q25, error_q75, alpha=0.3, color="#A23B72", label="IQR (25%-75%)"
        )

        # Customize metric plot
        ax.set_title(
            f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight="bold", pad=10
        )
        ax.set_ylabel("Metric Value", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        if i == 0:
            ax.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

        # Customize error plot
        ax2.set_title(f"Error percentage", fontsize=12, fontweight="bold", pad=10)
        ax2.set_ylabel("Error percentage (%)", fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax2.legend(loc="upper right", frameon=True, fancybox=True, shadow=True)

        # Set x-axis labels only for bottom plots
        if i == n_metrics - 1:
            ax.set_xlabel("Iteration", fontsize=10)
            ax2.set_xlabel("Iteration", fontsize=10)

    plt.tight_layout(pad=2.0)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    else:
        plt.show()


def _plot_dashboard_style(
    iterations, metric_names, all_simulation_metrics, target_metrics, save_path
):
    """Create a comprehensive dashboard-style visualization."""
    n_metrics = len(metric_names)
    metric_stats, error_stats = _calculate_metric_statistics(all_simulation_metrics, target_metrics)

    # Create dashboard layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.25)

    # Color palette
    colors = sns.color_palette("husl", n_metrics)

    # 1. Overall convergence plot (top left)
    ax1 = fig.add_subplot(gs[0, :2])
    for i, (metric_name, color) in enumerate(zip(metric_names, colors)):
        error_means = [stats[i]["mean"] for stats in error_stats]
        ax1.plot(
            iterations,
            error_means,
            "o-",
            linewidth=2,
            color=color,
            label=metric_name.replace("_", " ").title(),
        )
    ax1.set_title("Overall Convergence Trends", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Normalized Error")
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")  # Log scale for better error visualization

    # 2. Current vs Target comparison (top right)
    ax2 = fig.add_subplot(gs[0, 2:])
    current_values = [metric_stats[-1][i]["mean"] for i in range(n_metrics)]
    x_pos = np.arange(n_metrics)

    bars1 = ax2.bar(
        x_pos - 0.2,
        current_values,
        0.4,
        label="Current",
        color=[colors[i] for i in range(n_metrics)],
        alpha=0.7,
    )
    bars2 = ax2.bar(x_pos + 0.2, target_metrics, 0.4, label="Target", color="gray", alpha=0.7)

    ax2.set_title("Current vs Target Values", fontsize=14, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([name.replace("_", "\n") for name in metric_names], rotation=0)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    # 3. Individual metric plots (bottom section)
    for i, (metric_name, color) in enumerate(zip(metric_names, colors)):
        row = 1 + i // 2
        col = (i % 2) * 2
        ax = fig.add_subplot(gs[row, col : col + 2])

        # Extract data
        metric_means = [stats[i]["mean"] for stats in metric_stats]
        metric_stds = [stats[i]["std"] for stats in metric_stats]
        error_means = [stats[i]["mean"] for stats in error_stats]

        # Create twin axes
        ax_twin = ax.twinx()

        # Plot metric values
        line1 = ax.plot(
            iterations,
            metric_means,
            "o-",
            linewidth=2.5,
            color=color,
            label="Metric Value",
            markersize=6,
        )
        ax.fill_between(
            iterations,
            np.array(metric_means) - np.array(metric_stds),
            np.array(metric_means) + np.array(metric_stds),
            alpha=0.2,
            color=color,
        )
        ax.axhline(y=target_metrics[i], color="red", linestyle="--", alpha=0.8, linewidth=2)

        # Plot error
        line2 = ax_twin.plot(
            iterations,
            error_means,
            "s-",
            linewidth=2,
            color="darkred",
            alpha=0.7,
            label="Error",
            markersize=4,
        )

        # Styling
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight="bold")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Metric Value", color=color)
        ax_twin.set_ylabel("Normalized Error", color="darkred")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="y", labelcolor=color)
        ax_twin.tick_params(axis="y", labelcolor="darkred")

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right")

    plt.suptitle("Metric Analysis Dashboard", fontsize=16, fontweight="bold", y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def _plot_separate_style(
    iterations, metric_names, all_simulation_metrics, target_metrics, save_path
):
    """Create separate detailed plots for each metric."""
    n_metrics = len(metric_names)
    metric_stats, error_stats = _calculate_metric_statistics(all_simulation_metrics, target_metrics)

    # Create separate figure for each metric
    for i, metric_name in enumerate(metric_names):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            f'{metric_name.replace("_", " ").title()} - Detailed Analysis',
            fontsize=16,
            fontweight="bold",
        )

        # Extract data for this metric
        metric_means = [stats[i]["mean"] for stats in metric_stats]
        metric_stds = [stats[i]["std"] for stats in metric_stats]
        metric_mins = [stats[i]["min"] for stats in metric_stats]
        metric_maxs = [stats[i]["max"] for stats in metric_stats]

        error_means = [stats[i]["mean"] for stats in error_stats]
        error_stds = [stats[i]["std"] for stats in error_stats]

        # 1. Metric evolution with uncertainty
        ax1.plot(iterations, metric_means, "o-", linewidth=3, markersize=8, color="#2E86AB")
        ax1.fill_between(
            iterations,
            np.array(metric_means) - np.array(metric_stds),
            np.array(metric_means) + np.array(metric_stds),
            alpha=0.3,
            color="#2E86AB",
            label="±1 Std Dev",
        )
        ax1.fill_between(
            iterations, metric_mins, metric_maxs, alpha=0.1, color="#2E86AB", label="Min-Max Range"
        )
        ax1.axhline(
            y=target_metrics[i],
            color="#F24236",
            linestyle="--",
            linewidth=3,
            alpha=0.8,
            label="Target",
        )
        ax1.set_title("Metric Value Evolution")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Metric Value")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Error evolution
        ax2.plot(iterations, error_means, "o-", linewidth=3, markersize=8, color="#A23B72")
        ax2.fill_between(
            iterations,
            np.array(error_means) - np.array(error_stds),
            np.array(error_means) + np.array(error_stds),
            alpha=0.3,
            color="#A23B72",
        )
        ax2.set_title("Error Evolution")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Normalized Error")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # 3. Box plot of metric distributions
        metric_distributions = []
        for sim_metrics in all_simulation_metrics:
            metric_distributions.append(sim_metrics[:, i])

        box_plot = ax3.boxplot(
            metric_distributions, positions=iterations, widths=0.8, patch_artist=True
        )
        for patch in box_plot["boxes"]:
            patch.set_facecolor("#2E86AB")
            patch.set_alpha(0.7)
        ax3.axhline(y=target_metrics[i], color="#F24236", linestyle="--", linewidth=2, alpha=0.8)
        ax3.set_title("Distribution by Iteration")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Metric Value")
        ax3.grid(True, alpha=0.3)

        # 4. Convergence rate
        if len(iterations) > 1:
            convergence_rate = np.diff(error_means) / np.diff(iterations)
            ax4.plot(
                iterations[1:], convergence_rate, "o-", linewidth=2, markersize=6, color="#F18F01"
            )
            ax4.axhline(y=0, color="black", linestyle="-", alpha=0.3)
            ax4.set_title("Error Change Rate")
            ax4.set_xlabel("Iteration")
            ax4.set_ylabel("Error Change per Iteration")
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            metric_save_path = save_path.replace(".png", f"_{metric_name}.png")
            plt.savefig(metric_save_path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            plt.show()


def analyze_iterations(
    parameter_base_folder,
    metric_names,
    target_metrics,
    n_iterations=10,
    target_peak=0,
    verbose=False,
    find_peaks=False,
    save_plots=True,
):
    """
    Analyze metrics across multiple iterations for a single parameter folder.

    Parameters:
    -----------
    parameter_base_folder : str
        Base folder path containing iteration results
    metric_names : list
        List of metric names to analyze
    target_metrics : array-like
        Target metric values
    n_iterations : int, optional
        Number of iterations to analyze. Default is 10.
    verbose : bool, optional
        Whether to print detailed information. Default is False.

    Returns:
    --------
    tuple
        (iterations, mean_errors, best_errors, all_simulation_metrics, all_distance_results)
    """
    iterations = []
    mean_errors = []
    best_errors = []
    all_simulation_metrics = []
    all_distance_results = []
    best_fit_samples_idx = []
    for i in range(n_iterations):
        print(f"\nAnalyzing iteration {i}")
        print("=" * 40)

        (
            comparator,
            distance_results,
            metric_stats,
            best_fit_sample,
            best_fit_distance,
            best_fit_idx,
        ) = analyze_iteration(
            parameter_base_folder,
            i,
            metric_names,
            target_metrics,
            target_peak=target_peak,
            verbose=verbose,
            find_peaks=find_peaks,
            save_plots=save_plots,
        )

        # Store errors and metrics for plotting
        iterations.append(i)
        mean_errors.append(distance_results["euclidean_normalized"]["mean"])
        best_errors.append(best_fit_distance)
        all_simulation_metrics.append(comparator.simulation_metrics)
        all_distance_results.append(distance_results)
        best_fit_samples_idx.append(best_fit_idx)
        # Print best fit sample details
        print(f"\nBest fit sample for iteration {i}:")
        print(f"Best fit distance (normalized): {best_fit_distance:.4f}")
        if verbose:
            for metric_name, metric_value in zip(metric_names, best_fit_sample):
                print(f"{metric_name}: {metric_value:.4f}")
    return (
        iterations,
        mean_errors,
        best_errors,
        all_simulation_metrics,
        all_distance_results,
        best_fit_samples_idx,
    )


def plot_size_comparison(results_dict, metric_names, save_path=None):
    """
    Plot comparison of error percentages across different sizes for each metric.

    Parameters:
    -----------
    results_dict : dict
        Dictionary containing results for each size, with keys being sizes and values
        containing the analysis results
    metric_names : list
        List of metric names
    save_path : str, optional
        Path to save the plots
    """
    sizes = sorted(results_dict.keys())
    n_metrics = len(metric_names)

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 2 * n_metrics))
    if n_metrics == 1:
        axes = [axes]

    # Plot error percentage for each metric
    for i, (metric_name, ax) in enumerate(zip(metric_names, axes)):
        # Get error percentages for this metric across sizes
        error_percentages = []
        for size in sizes:
            # Get the last iteration's error percentage for this metric
            last_iter_errors = results_dict[size]["all_distance_results"][-1]["relative_errors"][
                "errors"
            ]
            metric_error = last_iter_errors[:, i]  # Get absolute error percentage for this metric
            error_percentages.append(np.mean(metric_error))  # Take mean across samples

        # Plot error percentage vs size
        ax.plot(sizes, error_percentages, "o-", linewidth=2, markersize=8, color="#2E86AB")
        ax.axhline(y=0, color="black", linestyle="--", linewidth=2)
        ax.set_title(metric_name.replace("_", " ").title())
        if i == len(metric_names) - 1:
            ax.set_xlabel("Size")
        ax.set_ylabel("Error Percentage (%)")
        ax.grid(True, alpha=0.7)
        ax.set_xscale("log", base=2)  # Log scale for sizes

        # Add value labels on points
        for x, y in zip(sizes, error_percentages):
            ax.annotate(
                f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center"
            )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def plot_scenario_comparison_boxswarm(
    all_simulation_metrics_list,
    scenario_names,
    metric_names,
    target_metrics,
    save_path=None,
    figsize=(12, 6),
):
    """
    Create box plots with swarm plots to compare error percentages across scenarios.

    Parameters:
    -----------
    all_simulation_metrics_list : list
        List containing simulation metrics arrays for each scenario
    scenario_names : list
        List of scenario names corresponding to the metrics
    metric_names : list
        List of metric names
    target_metrics : array-like
        Target metric values
    save_path : str, optional
        Path to save the plot
    figsize : tuple, optional
        Figure size (width, height)
    """

    n_scenarios = len(scenario_names)

    # Create figure with subplots: 2x3 grid for 5 metrics only
    fig, axes_2d = plt.subplots(2, 3, figsize=figsize)

    # Keep both 2D and flattened versions for easy access
    axes = axes_2d.flatten()

    # Prepare all data at once
    all_data = []

    for i, metric_name in enumerate(metric_names):
        for j, (sim_metrics_list, scenario_name) in enumerate(
            zip(all_simulation_metrics_list, scenario_names)
        ):
            # Get the last iteration's metrics for this scenario
            last_iteration_metrics = sim_metrics_list[-1]
            metric_values = last_iteration_metrics[:, i]

            # Calculate error percentages relative to target
            target_value = target_metrics[i]
            error_percentages = ((metric_values - target_value) / target_value) * 100

            # Add to combined dataset
            for sample_id, error_pct in enumerate(error_percentages):
                all_data.append(
                    {
                        "Scenario": scenario_name,
                        "Metric": metric_name,
                        "Error_Percentage": error_pct,
                        "Absolute_Error": np.abs(error_pct),
                        "sample_id": sample_id,
                    }
                )

    df = pd.DataFrame(all_data)

    def add_significance_bars(ax, data_groups, scenario_names):
        """Add significance bars and annotations above box plots"""
        # Perform pairwise statistical tests
        comparisons = list(combinations(range(len(scenario_names)), 2))
        p_values = []

        for i, j in comparisons:
            # Use Mann-Whitney U test (non-parametric)
            stat, p_val = stats.mannwhitneyu(
                data_groups[i], data_groups[j], alternative="two-sided"
            )
            p_values.append((i, j, p_val))

        # Get the maximum y-value for positioning bars
        max_y = max([max(group) for group in data_groups])
        y_offset = max_y * 0.1
        bar_height = max_y * 0.1
        y_axis_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        # Add significance bars
        significance_count = 0
        for idx, (i, j, p_val) in enumerate(p_values):
            # Skip if not significant
            if p_val >= 0.05 / len(comparisons):
                continue

            # Position bars at different heights
            y_pos = max_y + y_offset + (significance_count * bar_height * 2)

            # Draw horizontal line
            ax.plot([i, j], [y_pos, y_pos], color="black", linewidth=1.5)
            # Draw vertical lines
            ax.plot(
                [i, i],
                [y_pos - bar_height / 2, y_pos + bar_height / 2],
                color="black",
                linewidth=1.5,
            )
            ax.plot(
                [j, j],
                [y_pos - bar_height / 2, y_pos + bar_height / 2],
                color="black",
                linewidth=1.5,
            )

            # Add significance annotation
            if p_val < 0.001 / len(comparisons):
                sig_text = "***"
            elif p_val < 0.01 / len(comparisons):
                sig_text = "**"
            else:
                sig_text = "*"

            ax.text(
                (i + j) / 2,
                y_pos - y_axis_range * 0.05,
                sig_text,
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

            significance_count += 1

        # ax.set_ylim(-100, 180)

    def create_boxswarm_plot(ax, data, title, ylabel, metric_name, use_absolute=False):
        """Helper function to create consistent box+swarm plots"""
        error_col = "Absolute_Error" if use_absolute else "Error_Percentage"

        # Get the color for this metric
        metric_color = metric_colors.get(metric_name.lower(), "gray")

        box_data = [data[data["Scenario"] == scenario][error_col] for scenario in scenario_names]

        # Create box plot
        box_plot = ax.boxplot(
            box_data, positions=range(n_scenarios), patch_artist=True, showfliers=False, widths=0.6
        )

        # Color the boxes with lighter colors for higher group numbers
        for j, patch in enumerate(box_plot["boxes"]):
            lighten_amt = min(0.7, j * 0.25)  # Increase lightness with group number
            patch.set_facecolor(lighten_color(metric_color, lighten_amt))
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.5)

        # Style box plot elements
        for element in ["whiskers", "caps", "medians"]:
            for item in box_plot[element]:
                item.set_color("black")
                item.set_linewidth(1.5)

        # Add swarm plot with lighter color
        for j, scenario in enumerate(scenario_names):
            scenario_data = data[data["Scenario"] == scenario][error_col]
            if len(scenario_data) > 0:
                lighten_amt = min(0.7, j * 0.25)
                x_positions = np.random.normal(j, 0.08, len(scenario_data))
                ax.scatter(
                    x_positions,
                    scenario_data,
                    color=lighten_color(metric_color, lighten_amt),
                    alpha=1.0,
                    s=15,
                    edgecolor="black",
                    linewidth=0.5,
                )

        # Add zero reference line for non-absolute errors (dashed with higher transparency)
        if not use_absolute:
            ax.axhline(y=0, color="black", linestyle="--", linewidth=1.5, alpha=0.4)

        # Style the axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        # Customize plot
        # ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
        # ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
        ax.set_xticks(range(n_scenarios))
        ax.set_xticklabels(scenario_names, fontsize=12, fontweight="bold")

        # Add grid
        ax.grid(False)
        ax.set_axisbelow(True)

        # Make tick labels bold
        ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

        # Add significance testing
        # add_significance_bars(ax, box_data, scenario_names)

    # Plot all metrics (up to 5 total)
    plot_count = 0
    for i, metric_name in enumerate(metric_names):
        if plot_count >= 5:  # Limit to 5 individual metrics
            break
        ax = axes[plot_count]
        metric_data = df[df["Metric"] == metric_name]
        ylabel = "Error Percentage (%)" if plot_count % 3 == 0 else None
        create_boxswarm_plot(
            ax=ax,
            data=metric_data,
            title=metric_name.replace("_", " ").title(),
            ylabel=ylabel,
            metric_name=metric_name,
            use_absolute=False,
        )
        plot_count += 1

    # Remove unused subplots (if any)
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])

    # Create legend - now with both metric colors and scenario colors
    legend_elements = []

    # Add metric color legend
    for metric_name in metric_names[:5]:  # Only show first 5 metrics
        if metric_name.lower() in metric_colors:
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker="s",
                    color="w",
                    markerfacecolor=metric_colors[metric_name.lower()],
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    label=f'{metric_name.replace("_", " ").title()}',
                    alpha=0.7,
                )
            )

    legend_elements.append(
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            linewidth=1.5,
            alpha=0.4,
            label="Target Reference (0%)",
        )
    )
    if 0:
        # Place legend
        legend = fig.legend(
            handles=legend_elements,
            loc="lower left",
            bbox_to_anchor=(0.05, 0.02),
            ncol=3,
            fontsize=9,
            frameon=False,
        )

        # Make legend text bold
        for text in legend.get_texts():
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, hspace=0.5, wspace=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
    else:
        plt.show()

    return fig


def create_bar_plots_seed(
    target_metrics,
    best_fit_samples_idx_list,
    seed_metrics_dfs,
    metrics_names,
    series_labels=None,
    colors=None,
    figsize=(8, 3),
    title_prefix="Metric Comparison",
    save_path=None,
    show_std=False,
):
    """
    Create bar plots for multiple data series across different metrics.

    Parameters:
    -----------
    data : list of arrays
        List containing arrays of metric values for each series
    metrics_names : list
        Names of the metrics (each will get its own subplot)
    series_labels : list, optional
        Labels for each data series (scenarios) (default: "Series 1", "Series 2", etc.)
    colors : list, optional
        Colors for each series (default: predefined color palette)
    figsize : tuple, optional
        Figure size (default: (12, 4))
    title_prefix : str, optional
        Prefix for subplot titles (default: "Metric Comparison")
    save_path : str, optional
        Path to save the figure
    show_std : bool, optional
        Whether to show error bars (requires data to be 2D with std values)

    Returns:
    --------
    fig, axes : matplotlib figure and axes objects
    """
    # Get best fit samples from seed metrics dfs
    best_fit_samples = []
    for i in range(len(best_fit_samples_idx_list)):
        best_sample = seed_metrics_dfs[i][
            seed_metrics_dfs[i]["input_folder"].str.split("_").str[-1].values
            == str(best_fit_samples_idx_list[i] + 1)
        ]
        best_fit_samples.append(best_sample[metrics_names])

    # Convert data to numpy array
    data_array = np.array(best_fit_samples)
    metric_medians = np.median(data_array, axis=1)
    metric_stds = np.std(data_array, axis=1)
    # Set default series labels if not provided
    if series_labels is None:
        series_labels = [f"Series {i+1}" for i in range(len(data_array))]
    metric_medians = np.vstack([target_metrics[:-2], metric_medians])
    target_stds = np.append(target_metrics[-2:], 0)
    metric_stds = np.vstack([target_stds, metric_stds])
    # Create subplots - one for each metric
    fig, axes = plt.subplots(1, len(metrics_names), figsize=figsize, sharey=False)
    if len(metrics_names) == 1:
        axes = [axes]

    # Plot each metric
    for i, metric_name in enumerate(metrics_names):
        ax = axes[i]
        # Create bar plot
        x_positions = np.arange(len(series_labels))
        # Create bars with individual colors
        bar_color = "white"  # Default white for all bars
        bar_alpha = 0.7

        # Only color the EXP bar (first bar in each subplot)
        bars = ax.bar(
            x_positions,
            metric_medians[:, i],
            yerr=np.vstack([[np.zeros(len(metric_stds[:, i]))], [metric_stds[:, i]]]),
            capsize=5,
            alpha=bar_alpha,
            color=[
                metric_colors[metric_name.lower()] if j == 0 else bar_color
                for j in range(len(x_positions))
            ],
            edgecolor="black",
            linewidth=1.5,
        )
        # Add swarm plot
        # Add swarm plots for cases with raw data (skip first case)
        for j, case_raw in enumerate(best_fit_samples):
            x_jitter = np.random.normal(j + 1, 0.02, len(case_raw))
            ax.scatter(x_jitter, case_raw[metric_name], color="black", alpha=0.6, s=20, zorder=3)
        # Set x-axis labels (scenario names)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(series_labels, rotation=45, ha="right")

        # Set y-axis to start from 0
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(0, y_max * 1.1)

        # Add grid
        ax.grid(True, alpha=0.3, axis="y")

        # Bold tick labels for both x and y axes (from template)
        for label_tick in ax.get_xticklabels():
            label_tick.set_fontweight("bold")
        for label_tick in ax.get_yticklabels():
            label_tick.set_fontweight("bold")

    # Adjust layout
    plt.tight_layout()

    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, axes


if __name__ == "__main__":
    np.random.seed(42)
    verbose = False
    target_metrics = {
        "doub_time": 45.5,
        "symmetry": 0.806,
        "colony_growth": 18.3,
        "doub_time_std": 13.79,
        "symmetry_std": 0.067,
    }
    metric_names = list(target_metrics.keys())
    target_metrics = np.array([target_metrics[name] for name in metric_names])

    from inverse_design.io.scenarios import combined_grid_n512_breast

    _paths = combined_grid_n512_breast()
    base_folder_pattern = _paths.base_original_dir

    if 1:
        linear_model_dirs = list(_paths.linear_model_dirs)
        dendrogram_threshold_dirs = list(_paths.dendrogram_threshold_dirs)
        ext_linear_dirs = list(_paths.ext_linear_dirs)
        ext_dendrogram_threshold_dirs = list(_paths.ext_dendrogram_threshold_dirs)
        peak_idx = 0
        simplified_methods = ["linear", "threshold"]
        simplified_method = simplified_methods[1]
        scenario_names = (
            ["Original", "r=0.9", "r=0.8", "r=0.7", "r=0.01"]
            if simplified_method == "linear"
            else ["Original", "t=1.0", "t=1.25", "t=1.5", "t=5.0"]
        )
        if simplified_method == "linear":
            processed_dirs = (
                [base_folder_pattern]
                + linear_model_dirs[peak_idx * 3 : (peak_idx + 1) * 3]
                + ext_linear_dirs[peak_idx : peak_idx + 1]
            )
        else:
            processed_dirs = (
                [base_folder_pattern]
                + dendrogram_threshold_dirs[peak_idx * 3 : (peak_idx + 1) * 3]
                + ext_dendrogram_threshold_dirs[peak_idx : peak_idx + 1]
            )
        if peak_idx == 3 and simplified_method == "linear":
            scenario_names = scenario_names[0:1] + scenario_names[3:]  # for peak 3
            processed_dirs = (
                processed_dirs[0:1] + processed_dirs[3:4] + ext_linear_dirs[peak_idx : peak_idx + 1]
            )
        all_simulation_metrics_list = []
        best_fit_samples_idx_list = []
        for idx, processed_dir in enumerate(processed_dirs):
            folder_path = processed_dir
            (
                iterations,
                mean_errors,
                best_errors,
                all_simulation_metrics,
                all_distance_results,
                best_fit_samples_idx,
            ) = analyze_iterations(
                folder_path,
                metric_names,
                target_metrics,
                n_iterations=5,
                verbose=verbose,
                find_peaks=idx == 0,
                target_peak=peak_idx,
            )
            all_simulation_metrics_list.append(all_simulation_metrics)
            best_fit_samples_idx_list.append(
                best_fit_samples_idx[-1]
            )  # best sample idx for the last iteration
        save_path_folder = "../../../ARCADE_OUTPUT/simplified_models_vis"
        summary_metrics_paths = [
            f"{processed_dir}/iter_4/final_metrics_seed.csv" for processed_dir in processed_dirs
        ]
        seed_metrics_dfs = [
            pd.read_csv(summary_metrics_path) for summary_metrics_path in summary_metrics_paths
        ]
        create_bar_plots_seed(
            target_metrics,
            best_fit_samples_idx_list,
            seed_metrics_dfs,
            [metric_name for metric_name in metric_names if "std" not in metric_name],
            series_labels=["EXP"] + scenario_names,
            colors=None,
            figsize=(6, 3),
            save_path=f"{save_path_folder}/bar_plot_{simplified_method}_p{peak_idx+1}_mean_only.png",
        )
        # Generate box + swarm plots for scenario comparison
        if 1:
            plot_scenario_comparison_boxswarm(
                all_simulation_metrics_list,
                scenario_names,
                metric_names,
                target_metrics,
                save_path=f"{save_path_folder}/scenario_comparison_boxswarm_{simplified_method}_p{peak_idx+1}.png",
                figsize=(11, 4),
            )
