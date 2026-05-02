import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from inverse_design.analyze.abc_metrics_comparison import analyze_iterations
from inverse_design.analyze.utils.analyze_utils import calculate_percentage_error
from inverse_design.plotting.colormap import merged_metric_colors
from inverse_design.plotting.theme import (
    apply_journal_style_nature_baseline,
    apply_publication_style,
)

apply_journal_style_nature_baseline()

# Nature Communications color palette (professional, colorblind-friendly)
nat_colors = merged_metric_colors()
nat_colors["peak_I"] = nat_colors["peak_i"]
nat_colors["area_I"] = nat_colors["area_i"]
nat_colors["final_R"] = nat_colors["final_r"]
apply_publication_style(font_size=10, axes_linewidth=1.0)
# Generate sample data with realistic trends
np.random.seed(42)
iterations = np.arange(0, 10)
n_iter = len(iterations)

# Define target values for percentage error calculation
targets = {
    "symmetry": 0.806,
    "symmetry_std": 0.067,
    "doub_time": 45.5,
    "doub_time_std": 13.79,
    "colony_growth": 18.3,
}
targets2 = {
    "peak_I": 0.8371166865790522,
    "time_to_peak": 33.37865296171455,
    "final_R": 0.832337714058144,
    "area_I": 0.4488309996162221,
    "growth_rate": 0.0068769230738393,
}
np.random.seed(42)
verbose = False
metric_names = list(targets.keys())
target_metrics = np.array([targets[name] for name in metric_names])

# Base folder pattern
base_folder_pattern = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N{size}_combined_grid_breast"


def plot_metrics_and_time(
    iterations, metrics_data, total_error, total_error_ci, comp_time, save_path=None, figsize=(5, 3)
):
    """
    Plot metrics errors and computational time.

    Parameters:
    -----------
    iterations : array-like
        Array of iteration numbers
    metrics_data : list of tuples
        List of (error_vals, error_ci, label, color, marker) for each metric
    total_error : array-like
        Array of total error values
    total_error_ci : array-like
        Array of total error confidence intervals
    comp_time : array-like
        Array of computational time values
    save_path : str, optional
        Path to save the plot
    figsize : tuple, optional
        Figure size (width, height)
    """

    # Create the plot
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot individual metrics with confidence intervals
    line_objects = []

    for i, (error_vals, error_ci, label, color, marker) in enumerate(metrics_data):
        # Plot main line with consistent styling
        line = ax1.plot(
            iterations,
            error_vals,
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            label=label,
            linestyle="-",
            alpha=0.8,
        )[0]
        line_objects.append(line)

    # Plot total error with thicker line and different style
    ax1.fill_between(
        iterations,
        np.array(total_error) - np.array(total_error_ci),
        np.array(total_error) + np.array(total_error_ci),
        color="black",
        alpha=0.2,
        linewidth=0,
    )
    total_line = ax1.plot(
        iterations,
        total_error,
        color="black",
        linewidth=2.5,
        marker="s",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        markeredgewidth=2,
        label="Total Error",
        linestyle="-",
        alpha=1.0,
    )[0]
    line_objects.append(total_line)

    # Style the primary y-axis (consistent with previous figures)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_linewidth(1.0)
    ax1.spines["bottom"].set_linewidth(1.0)

    # Customize primary y-axis with bold formatting
    ax1.set_xlabel("Iteration", fontsize=12)
    ax1.set_ylabel("Error (%)", fontsize=12)
    ax1.set_xlim(-0.2, len(iterations) + 0.5)
    ax1.set_xticks(np.arange(0, len(iterations), 1))

    # Add subtle grid (consistent with previous figures)
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax1.set_axisbelow(True)

    # Make tick labels bold and thicker (consistent with previous figures)
    ax1.tick_params(axis="both", which="major", labelsize=14, width=1.5)

    ax1.set_ylim(-20, 320)

    # Error metrics legend elements
    error_legend_elements = []
    for line_obj in line_objects:
        error_legend_elements.append(line_obj)

    # Clear all plots
    # ax1.clear()
    # Add legend
    legend_to_names = {
        "Peak I": "Peak I",
        "Time To Peak": "Time$_{peak,I}$",
        "Final R": "R$_{final}$",
        "Area I": "Infection burden",
        "Growth Rate": "Infection rate",
        "Total Error": "Total Error",
    }

    # ax1.legend(error_legend_elements, [legend_to_names[line.get_label()] for line in error_legend_elements], loc='upper right', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()

    return fig


def plot_metrics_by_size(
    sizes, results_dict, target_metrics, comp_time=None, save_path=None, figsize=(4.5, 3)
):
    """
    Plot metrics errors against sample size for final iteration.

    Parameters:
    -----------
    sizes : array-like
        Array of sample sizes
    results_dict : dict
        Dictionary containing results for each size
    target_metrics : dict
        Dictionary of target values for each metric (e.g., {'peak_I': 0.837, 'time_to_peak': 33.38, ...})
    comp_time : array-like, optional
        Array of computational time values (kept for compatibility but not plotted)
    save_path : str, optional
        Path to save the plot
    figsize : tuple, optional
        Figure size (width, height)
    """

    # Create the plot (single axis, no computational time)
    fig, ax = plt.subplots(figsize=figsize)

    # Get metric names and count
    metric_names = list(target_metrics.keys())
    num_metrics = len(metric_names)

    # Extract metrics data for each size (final iteration only)
    metrics_data = []
    for size in sizes:
        results = results_dict[size]
        # Get the final iteration's metrics
        final_iter_metrics = results["all_simulation_metrics"][-1]

        # Calculate mean values and errors for each metric
        metric_errors = {}
        metric_errors_ci = {}

        for i, metric_name in enumerate(metric_names):
            # Calculate mean value for this metric
            metric_mean = np.mean(final_iter_metrics[:, i])

            # Calculate standard deviation for confidence intervals
            metric_std_ci = np.std(final_iter_metrics[:, i])

            # Calculate percentage error
            metric_error = calculate_percentage_error(metric_mean, target_metrics[metric_name])

            # Calculate percentage error confidence interval
            metric_error_ci = (metric_std_ci / target_metrics[metric_name]) * 100

            metric_errors[metric_name] = metric_error
            metric_errors_ci[metric_name] = metric_error_ci

        # Calculate total error
        weights = [1 / num_metrics] * num_metrics
        total_error = sum(
            weights[i] * metric_errors[metric_name] for i, metric_name in enumerate(metric_names)
        )

        # Calculate total error confidence interval (propagated uncertainty)
        total_error_ci = np.sqrt(
            sum(
                (weights[i] * metric_errors_ci[metric_name]) ** 2
                for i, metric_name in enumerate(metric_names)
            )
        )

        # Store all metric data
        size_data = metric_errors.copy()
        size_data["total"] = total_error
        size_data["total_ci"] = total_error_ci
        metrics_data.append(size_data)

    # Convert metrics data to arrays
    metric_error_arrays = {}
    for metric_name in metric_names:
        metric_error_arrays[metric_name] = np.array([data[metric_name] for data in metrics_data])

    total_errors = np.array([data["total"] for data in metrics_data])
    total_errors_ci = np.array([data["total_ci"] for data in metrics_data])

    # Define colors and markers for plotting
    default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    default_markers = ["o", "s", "*", "D", "v"]

    # Create metrics plot data dynamically
    line_objects = []
    metrics_plot_data = []

    for i, metric_name in enumerate(metric_names):
        color = default_colors[i % len(default_colors)]
        marker = default_markers[i % len(default_markers)]

        # Try to use nat_colors if available, otherwise use defaults
        try:
            color = nat_colors[metric_name]
        except (NameError, KeyError):
            pass

        # Format label for display
        label = metric_name.replace("_", " ").title()

        metrics_plot_data.append((metric_error_arrays[metric_name], label, color, marker))

    # Plot individual metrics (no confidence intervals)
    for error_vals, label, color, marker in metrics_plot_data:
        line = ax.plot(
            sizes,
            error_vals,
            color=color,
            linewidth=2.5,
            marker=marker,
            markersize=6,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.5,
            label=label,
            linestyle="-",
            alpha=0.8,
        )[0]
        line_objects.append(line)

    # Plot total error with confidence interval
    total_color = "black"
    try:
        total_color = nat_colors["black"]
    except (NameError, KeyError):
        pass

    ax.fill_between(
        sizes,
        total_errors - total_errors_ci,
        total_errors + total_errors_ci,
        color=total_color,
        alpha=0.2,
        linewidth=0,
    )
    total_line = ax.plot(
        sizes,
        total_errors,
        color=total_color,
        linewidth=2.5,
        marker="s",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=total_color,
        markeredgewidth=2,
        label="Total Error",
        linestyle="-",
        alpha=1.0,
    )[0]
    line_objects.append(total_line)

    # Style the axes (consistent with previous figures)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    # Customize axis with bold formatting
    ax.set_xlabel("Sample Size", fontsize=12)
    # ax.set_ylabel('Error (%)', fontsize=12)
    ax.set_xscale("log", base=2)  # Log scale for sample sizes

    # Add subtle grid (consistent with previous figures)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Make tick labels bold and thicker (consistent with previous figures)
    ax.tick_params(axis="both", which="major", labelsize=14, width=1.5)

    # Set y-axis range
    ax.set_ylim(-20, 100)

    # Improve layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, transparent=True)
        plt.close()
    else:
        plt.show()

    return fig


# Example usage:
if __name__ == "__main__":
    # Base folder pattern
    base_folder_pattern = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N{size}_combined_grid_breast"
    # base_folder_pattern = "../../../SIR_OUTPUT/n3_t365_l30/n{size}"
    if 0:
        size = 512
        folder_path = base_folder_pattern.format(size=size)
        (
            iterations,
            mean_errors,
            best_errors,
            all_simulation_metrics,
            all_distance_results,
            best_fit_samples_idx,
        ) = analyze_iterations(
            folder_path, metric_names, target_metrics, n_iterations=10, verbose=verbose
        )
        results_dict = {
            size: {
                "iterations": iterations,
                "mean_errors": mean_errors,
                "best_errors": best_errors,
                "all_simulation_metrics": all_simulation_metrics,
                "all_distance_results": all_distance_results,
            }
        }

    if 0:
        size = 512
        results = results_dict[size]
        iterations = np.array(results["iterations"])
        n_iter = len(iterations)

        # Get metric names and count
        metric_names = list(targets.keys())
        num_metrics = len(metric_names)

        # Extract mean values and standard deviations for each metric dynamically
        metric_means = {}
        metric_std_cis = {}

        for i, metric_name in enumerate(metric_names):
            # Extract mean values for this metric across all iterations
            metric_means[metric_name] = np.array(
                [np.mean(metrics[:, i]) for metrics in results["all_simulation_metrics"]]
            )

            # Calculate standard deviations for confidence intervals
            metric_std_cis[metric_name] = np.array(
                [np.std(metrics[:, i]) for metrics in results["all_simulation_metrics"]]
            )

        # Calculate percentage errors for each metric
        metric_errors = {}
        metric_error_cis = {}

        for metric_name in metric_names:
            # Calculate percentage errors
            metric_errors[metric_name] = calculate_percentage_error(
                metric_means[metric_name], targets[metric_name]
            )

            # Calculate percentage error confidence intervals
            metric_error_cis[metric_name] = (
                metric_std_cis[metric_name] / targets[metric_name]
            ) * 100

        # Calculate total error (weighted average)
        weights = [1 / num_metrics] * num_metrics  # Equal weights
        total_error = sum(
            weights[i] * metric_errors[metric_name] for i, metric_name in enumerate(metric_names)
        )

        # Total error confidence interval (propagated uncertainty)
        total_error_ci = np.sqrt(
            sum(
                (weights[i] * metric_error_cis[metric_name]) ** 2
                for i, metric_name in enumerate(metric_names)
            )
        )

        comp_time = np.array([510, 807, 1099, 1486, 1869, 2233, 2597, 2961, 3330, 3698])
        # comp_time = np.array([501.5, 1038, 1914, 4123])
        comp_time = np.maximum(comp_time, 0.1)  # Ensure positive values

        # Define colors and markers for plotting
        default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        default_markers = ["o", "s", "*", "D", "v"]

        # Prepare metrics data for plotting dynamically
        metrics_data = []
        for i, metric_name in enumerate(metric_names):
            color = default_colors[i % len(default_colors)]
            marker = default_markers[i % len(default_markers)]

            # Try to use nat_colors if available, otherwise use defaults
            try:
                color = nat_colors[metric_name]
            except (NameError, KeyError):
                pass

            # Format label for display
            label = metric_name.replace("_", " ").title()

            metrics_data.append(
                (metric_errors[metric_name], metric_error_cis[metric_name], label, color, marker)
            )

    if 0:
        # Create the plot
        fig = plot_metrics_and_time(
            iterations=iterations,
            metrics_data=metrics_data,
            total_error=total_error,
            total_error_ci=total_error_ci,
            comp_time=comp_time,
            save_path="../../../SIR_OUTPUT/n3_t365_l30/iteration_metrics.png",
            # save_path='../../../ARCADE_OUTPUT/iteration_metrics.png'
        )

    if 1:  # Multiple sizes analysis
        sizes = [2**i for i in range(7, 11)]
        results_dict = {}
        for size in sizes:
            folder_path = base_folder_pattern.format(size=size)
            (
                iterations,
                mean_errors,
                best_errors,
                all_simulation_metrics,
                all_distance_results,
                best_fit_samples_idx,
            ) = analyze_iterations(
                folder_path, metric_names, target_metrics, n_iterations=5, verbose=verbose
            )
            results_dict[size] = {
                "iterations": iterations,
                "mean_errors": mean_errors,
                "best_errors": best_errors,
                "all_simulation_metrics": all_simulation_metrics,
                "all_distance_results": all_distance_results,
            }

        # Create size-based plot
        comp_time = np.array([501.5, 1038, 1914, 4123])  # Computational time for each size
        fig = plot_metrics_by_size(
            sizes=sizes,
            results_dict=results_dict,
            comp_time=comp_time,
            target_metrics=targets,
            # save_path='../../../SIR_OUTPUT/n3_t365_l30/size_metrics.png'
            save_path="../../../ARCADE_OUTPUT/size_metrics.png",
        )
        # plt.show()
