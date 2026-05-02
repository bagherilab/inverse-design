import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from inverse_design.config.metrics_config import DEFAULT_METRICS
from inverse_design.analyze.utils.analyze_utils import calculate_metrics_statistics


def calculate_normalized_error(metric_values, target_value, loss_function="absolute", scaler=None):
    """
    Calculate normalized error percentages for a set of metric values.

    Args:
        metric_values (np.array): Array of metric values to evaluate
        target_value (float): Target value for the metric
        scaler (MinMaxScaler, optional): Pre-fitted scaler. If None, creates new one.

    Returns:
        tuple: (normalized_error_percent, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(metric_values.reshape(-1, 1))

    normalized_values = scaler.transform(metric_values.reshape(-1, 1))
    normalized_target = scaler.transform(np.array([target_value]).reshape(-1, 1))[0][0]
    if loss_function == "absolute":
        error = calculate_absolute_error(normalized_values, normalized_target)
    elif loss_function == "squared":
        error = calculate_squared_error(normalized_values, normalized_target)

    return error, scaler


def calculate_squared_error(values, target_value):
    """
    Calculate squared error (distance) between values and target value.

    Args:
        values (np.array): Array of metric values
        target_value (float): Target value for the metric

    Returns:
        np.array: Squared error values
    """
    return (values - target_value) ** 2


def calculate_absolute_error(values, target_value):
    """
    Calculate absolute error (distance) between values and target value.

    Args:
        values (np.array): Array of metric values
        target_value (float): Target value for the metric

    Returns:
        np.array: Absolute error values
    """
    return np.abs(values - target_value)


def calculate_metric_errors(
    posterior_df_dict,
    prior_df,
    default_metrics,
    metric,
    target,
    sample_sizes,
    loss_function="absolute",
):
    """
    Calculate error percentages for a metric across different sample sizes and reference values.

    Args:
        posterior_df_dict (dict): Dictionary of posterior dataframes for different sample sizes
        prior_df (dict): Prior distribution metrics
        default_metrics (dict): Default metrics values
        metric (str): Name of the metric to analyze
        target (float): Target value for the metric
        sample_sizes (list): List of sample sizes to analyze
        loss_function (str): Loss function to use

    Returns:
        dict: Dictionary containing errors and their standard deviations
    """
    # Fit scaler on full dataset
    metric_values = posterior_df_dict[sample_sizes[-1]][metric]["metric"]
    _, scaler = calculate_normalized_error(metric_values, target, loss_function)

    # Calculate errors for each sample size
    posterior_errors = []
    posterior_errors_std = []
    for n in sample_sizes:
        # Get mode and std for this sample size
        posterior_values = posterior_df_dict[n][metric]["mode"]
        posterior_std = posterior_df_dict[n][metric]["std"]

        # Calculate normalized error
        posterior_error, _ = calculate_normalized_error(
            np.array([posterior_values]), target, loss_function, scaler
        )
        posterior_errors.append(posterior_error[0][0])

        # Calculate normalized std by transforming (mode ± std) and taking half the difference
        upper_bound = posterior_values + posterior_std
        lower_bound = posterior_values - posterior_std
        normalized_upper = scaler.transform(np.array([upper_bound]).reshape(-1, 1))[0][0]
        normalized_lower = scaler.transform(np.array([lower_bound]).reshape(-1, 1))[0][0]
        normalized_std = (normalized_upper - normalized_lower) / 2
        posterior_errors_std.append(normalized_std)

    # Calculate reference error percentages
    prior_error, _ = calculate_normalized_error(
        np.array([prior_df[metric]["mode"]]), target, loss_function, scaler
    )
    prior_error = prior_error[0][0]

    # Calculate normalized prior std
    prior_std = prior_df[metric]["std"]
    prior_upper = prior_df[metric]["mode"] + prior_std
    prior_lower = prior_df[metric]["mode"] - prior_std
    normalized_prior_upper = scaler.transform(np.array([prior_upper]).reshape(-1, 1))[0][0]
    normalized_prior_lower = scaler.transform(np.array([prior_lower]).reshape(-1, 1))[0][0]
    prior_error_std = (normalized_prior_upper - normalized_prior_lower) / 2

    # Calculate default error and std
    default_error, _ = calculate_normalized_error(
        np.array([default_metrics[metric]["mean"]]), target, loss_function, scaler
    )
    default_error = default_error[0][0]

    # Calculate normalized default std
    default_std = default_metrics[metric]["std"]
    default_upper = default_metrics[metric]["mean"] + default_std
    default_lower = default_metrics[metric]["mean"] - default_std
    normalized_default_upper = scaler.transform(np.array([default_upper]).reshape(-1, 1))[0][0]
    normalized_default_lower = scaler.transform(np.array([default_lower]).reshape(-1, 1))[0][0]
    default_error_std = (normalized_default_upper - normalized_default_lower) / 2

    error_dict = {
        "posterior_errors": posterior_errors,
        "posterior_errors_std": posterior_errors_std,
        "prior_error": prior_error,
        "prior_error_std": prior_error_std,
        "default_error": default_error,
        "default_error_std": default_error_std,
    }

    return error_dict


def _plot_error_with_bands(
    ax,
    plot_x,
    errors,
    errors_std=None,
    prior_errors=None,
    prior_errors_std=None,
    default_error=None,
    default_error_std=None,
    use_log2_scale=False,
):
    """Helper function to plot errors with error bands.

    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        plot_x (array-like): X-axis values
        errors (array-like): Error values to plot
        errors_std (array-like): Standard deviation of errors
        prior_errors (float or array-like, optional): Prior error value(s)
        prior_errors_std (float or array-like, optional): Prior error standard deviation(s)
        default_error (float, optional): Default error value
        default_error_std (float, optional): Default error standard deviation
        use_log2_scale (bool): Whether to use log2 scale for x-axis
    """
    # Plot posterior with error band
    best_absolute_error = np.min(errors)
    ax.plot(plot_x, errors, "-o", label=f"Posterior (lowest = {best_absolute_error:.2f})")
    if errors_std is not None:
        ax.fill_between(
            plot_x,
            np.array(errors) - np.array(errors_std),
            np.array(errors) + np.array(errors_std),
            alpha=0.2,
        )

    # Plot prior with error band
    if prior_errors is not None:
        if isinstance(prior_errors, (float, int)):
            ax.axhline(
                y=prior_errors,
                color="gray",
                linestyle="--",
                label=f"Prior error: {prior_errors:.2f}",
            )
            if prior_errors_std is not None:
                ax.axhspan(
                    prior_errors - prior_errors_std,
                    prior_errors + prior_errors_std,
                    color="gray",
                    alpha=0.2,
                )
        else:
            best_prior_error = np.min(prior_errors)
            ax.plot(
                plot_x,
                prior_errors,
                "--",
                color="gray",
                label=f"Prior errors (lowest = {best_prior_error:.2f})",
            )
            if prior_errors_std is not None:
                ax.fill_between(
                    plot_x,
                    np.array(prior_errors) - np.array(prior_errors_std),
                    np.array(prior_errors) + np.array(prior_errors_std),
                    color="gray",
                    alpha=0.2,
                )

    # Plot default with error band
    if default_error is not None:
        ax.axhline(
            y=default_error,
            color="green",
            linestyle="--",
            label=f"Default error: {default_error:.2f}",
        )
        if default_error_std is not None:
            ax.axhspan(
                default_error - default_error_std,
                default_error + default_error_std,
                color="green",
                alpha=0.2,
            )

    if use_log2_scale:
        ax.set_xticks(plot_x)
        ax.set_xticklabels([f"2^{int(x)}" for x in plot_x])

    ax.grid(True)
    ax.legend()


def plot_error_analysis(
    posterior_df_dict,
    prior_df,
    target_metrics,
    default_metrics,
    metrics_list,
    x_values,
    x_label,
    loss_function="absolute",
    use_log2_scale=False,
    save_path=None,
):
    """
    Generic plotting function for error analysis vs a given parameter (samples or acceptance percentage).

    Args:
        posterior_df_dict (dict): Dictionary of posterior dataframes
        prior_df (dict): Prior distribution metrics
        target_metrics (dict): Target values for each metric
        default_metrics (dict): Default metrics values
        metrics_list (list): List of metrics to analyze
        x_values (list): Values for x-axis
        x_label (str): Label for x-axis
        loss_function (str): Loss function to use
        use_log2_scale (bool): Whether to use log2 scale for x-axis
        save_path (str, optional): Path to save the plot
    """
    n_plots = len(metrics_list) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    plot_x = np.log2(x_values) if use_log2_scale else x_values
    all_errors = {"posterior_errors": [], "prior_errors": [], "default_errors": []}

    # Plot for each individual metric
    for idx, metric in enumerate(metrics_list):
        target = target_metrics[metric]
        error_dict = calculate_metric_errors(
            posterior_df_dict,
            prior_df,
            default_metrics,
            metric,
            target,
            x_values,
            loss_function,
        )
        all_errors["posterior_errors"].append(error_dict["posterior_errors"])
        all_errors["prior_errors"].append(error_dict["prior_error"])
        all_errors["default_errors"].append(error_dict["default_error"])

        _plot_error_with_bands(
            axes[idx],
            plot_x,
            error_dict["posterior_errors"],
            error_dict["posterior_errors_std"],
            error_dict["prior_error"],
            error_dict["prior_error_std"],
            error_dict["default_error"],
            error_dict["default_error_std"],
            use_log2_scale,
        )

        axes[idx].set_xlabel(x_label)
        axes[idx].set_ylabel("Error")
        axes[idx].set_title(f"{loss_function} Error vs {x_label} for {metric} (target = {target})")

    # Plot total errors using the helper
    total_errors = np.sum(all_errors["posterior_errors"], axis=0)
    total_prior_error = np.sum(all_errors["prior_errors"])
    total_default_error = np.sum(all_errors["default_errors"])

    _plot_error_with_bands(
        axes[-1],
        plot_x,
        total_errors,
        None,
        total_prior_error,
        None,
        total_default_error,
        None,
        use_log2_scale,
    )

    axes[-1].set_xlabel(x_label)
    axes[-1].set_ylabel("Total Error")
    axes[-1].set_title(f"Total {loss_function} Error vs {x_label} (All Metrics)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def plot_error_vs_samples(
    posterior_df_dict,
    prior_df,
    target_metrics,
    default_metrics,
    metrics_list,
    loss_function="absolute",
    save_path=None,
):
    """Wrapper function for plotting absolute error vs number of samples."""
    sample_sizes = sorted(posterior_df_dict.keys())
    plot_error_analysis(
        posterior_df_dict,
        prior_df,
        target_metrics,
        default_metrics,
        metrics_list,
        sample_sizes,
        "Number of Samples",
        loss_function,
        use_log2_scale=True,
        save_path=save_path,
    )


def plot_error_vs_acceptance(
    posterior_df_dict,
    prior_df,
    target_metrics,
    default_metrics,
    metrics_list,
    loss_function="absolute",
    save_path=None,
):
    """Wrapper function for plotting absolute error vs acceptance percentage."""
    acceptance_percentages = sorted(posterior_df_dict.keys())
    plot_error_analysis(
        posterior_df_dict,
        prior_df,
        target_metrics,
        default_metrics,
        metrics_list,
        acceptance_percentages,
        "Acceptance Percentage",
        loss_function,
        use_log2_scale=False,
        save_path=save_path,
    )


def plot_error_vs_n_parameters(
    posterior_df_dict,
    prior_metrics_dict,
    target_metrics,
    default_metrics,
    metrics_list,
    loss_function="absolute",
    use_log2_scale=True,
    save_path=None,
):
    """
    Plot error vs number of parameters for each metric and mean error.

    Args:
        posterior_df_dict (dict): Dictionary of posterior dataframes for different parameter counts
        prior_metrics_dict (dict): Dictionary of prior metrics for different parameter counts
        target_metrics (dict): Target values for each metric
        default_metrics (dict): Default metrics values
        metrics_list (list): List of metrics to analyze
        loss_function (str): Loss function to use
        save_path (str, optional): Path to save the plot
    """
    n_plots = len(metrics_list) + 1
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    n_parameters = sorted(posterior_df_dict.keys())
    plot_x = np.log2(n_parameters)  # Use log2 scale for parameters
    all_errors = {"posterior_errors": [], "prior_errors": [], "default_errors": []}

    # Plot for each individual metric
    for idx, metric in enumerate(metrics_list):
        target = target_metrics[metric]
        posterior_errors = []
        posterior_errors_std = []
        prior_errors = []
        prior_errors_std = []
        # Calculate errors for each parameter count
        for n in n_parameters:
            # Get posterior errors
            error_dict = calculate_metric_errors(
                {n: posterior_df_dict[n]},
                prior_metrics_dict[n],
                default_metrics,
                metric,
                target,
                [n],
                loss_function,
            )
            posterior_errors.append(error_dict["posterior_errors"][0])  # Take first (only) value
            posterior_errors_std.append(error_dict["posterior_errors_std"][0])
            prior_errors.append(error_dict["prior_error"])
            prior_errors_std.append(error_dict["prior_error_std"])

        all_errors["posterior_errors"].append(posterior_errors)
        all_errors["prior_errors"].append(prior_errors)
        all_errors["default_errors"].append(error_dict["default_error"])

        _plot_error_with_bands(
            axes[idx],
            plot_x,
            posterior_errors,
            posterior_errors_std,
            prior_errors,
            prior_errors_std,
            error_dict["default_error"],
            error_dict["default_error_std"],
            use_log2_scale,
        )

        axes[idx].set_xlabel("Number of Parameter combinations")
        axes[idx].set_ylabel("Error")
        axes[idx].set_title(f"{loss_function} Error vs Parameters for {metric} (target = {target})")

    # Plot total errors
    total_posterior_errors = np.sum(all_errors["posterior_errors"], axis=0)
    total_prior_errors = np.sum(all_errors["prior_errors"], axis=0)

    _plot_error_with_bands(
        axes[-1],
        plot_x,
        total_posterior_errors,
        None,
        total_prior_errors,
        None,
        np.sum(all_errors["default_errors"]),
        None,
        use_log2_scale,
    )

    axes[-1].set_xlabel("Number of Parameter combinations")
    axes[-1].set_ylabel("Total Error")
    axes[-1].set_title(f"Total {loss_function} Error vs Parameters (All Metrics)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def main():
    import seaborn as sns

    metrics_names = ["symmetry"]  # , "cycle_length", "act"]
    target_metrics = {
        "symmetry": 0.8,
        # "cycle_length": 30.0,
        # "act": 0.6
    }
    prior_metrics_file = "ARCADE_OUTPUT/STEM_CELL/MS_ALL/MS_PRIOR_N512/final_metrics.csv"
    prior_metrics_df = pd.read_csv(prior_metrics_file)
    prior_metrics_statistics = calculate_metrics_statistics(prior_metrics_df, metrics_names)
    default_metrics = {metric: DEFAULT_METRICS[metric] for metric in target_metrics}

    loss_function = "absolute"

    if 1:
        parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL/MS_ALL/MS_POSTERIOR_N512/MS_POSTERIOR_10P"
        sample_sizes = [2**i for i in range(5, 6)]  # + [48]
        posterior_metrics_dict = {}
        for n in sample_sizes:
            posterior_df = pd.read_csv(f"{parameter_base_folder}/n{n}/final_metrics.csv")
            posterior_metrics_dict[n] = calculate_metrics_statistics(posterior_df, metrics_names)
        save_file = f"{parameter_base_folder}/error_vs_samples.png"
        plot_error_vs_samples(
            posterior_metrics_dict,
            prior_metrics_statistics,
            target_metrics,
            default_metrics,
            metrics_names,
            loss_function,
            save_file,
        )
    if 0:
        parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL/MS_POSTERIOR_N512"
        acceptance_percentages = [5, 10, 15, 20]
        posterior_metrics_dict = {}
        for acceptance_percentage in acceptance_percentages:
            posterior_df = pd.read_csv(
                f"{parameter_base_folder}/MS_POSTERIOR_{acceptance_percentage}P/n32/final_metrics.csv"
            )
            posterior_metrics_dict[acceptance_percentage] = calculate_metrics_statistics(
                posterior_df, metrics_names
            )

        save_file = f"{parameter_base_folder}/error_vs_acceptance.png"
        plot_error_vs_acceptance(
            posterior_metrics_dict,
            prior_metrics_statistics,
            target_metrics,
            default_metrics,
            metrics_names,
            loss_function,
            save_file,
        )
    if 0:
        parameter_base_folder = "ARCADE_OUTPUT/STEM_CELL/MS_POSTERIOR"
        number_of_parameters = [2**n for n in range(6, 11)]
        posterior_metrics_dict = {}
        prior_metrics_dict = {}

        for n in number_of_parameters:
            posterior_df = pd.read_csv(
                f"{parameter_base_folder}_N{n}/MS_POSTERIOR_10P/n32/final_metrics.csv"
            )
            posterior_metrics_dict[n] = calculate_metrics_statistics(posterior_df, metrics_names)
            prior_metrics_file = f"ARCADE_OUTPUT/STEM_CELL/MS_PRIOR_N{n}/final_metrics.csv"
            prior_metrics_df = pd.read_csv(prior_metrics_file)
            prior_metrics_dict[n] = calculate_metrics_statistics(prior_metrics_df, metrics_names)

        save_file = f"{parameter_base_folder}_N1024/error_vs_n_parameters.png"
        plot_error_vs_n_parameters(
            posterior_metrics_dict,
            prior_metrics_dict,
            target_metrics,
            default_metrics,
            metrics_names,
            loss_function="absolute",
            use_log2_scale=True,
            save_path=save_file,
        )
