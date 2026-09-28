import time
import subprocess
from pathlib import Path
import logging
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.stats import gaussian_kde, qmc, mode
import argparse
from inverse_design.common.enum import Target, Metric
from inverse_design.rf.abc_smc_rf_arcade import ABCSMCRF
from inverse_design.rf.abc_smc_rf_arcade_simplified import ABCSMCRF as ABCSMCRF_simplified
from inverse_design.analyze.source_metrics import (
    calculate_capillary_density,
    calculate_distance_between_points,
)
from inverse_design.analyze.simplify_model import analyze_abc_posterior_redundancy
from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.plotting.plot_dendrogram import (
    colorize_ward_clusters,
    colorize_ward_clusters_threshold,
)
from inverse_design.utils.s3_utils import (
    ensure_dir_exists,
    path_exists,
    save_json,
    read_csv,
    upload_file_to_s3,
)
from inverse_design.utils.utils import archive_directory
from inverse_design.utils.create_input_files import (
    _num_sobol_samples,
    set_particle_count,
)
from inverse_design.analyze.lr_predictor import CorrelationPredictor
from inverse_design.analyze.analyze_param_correlation import (
    analyze_peak_pairwise_correlations,
    analyze_parameter_chains,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
np.random.seed(42)

TARGET_RANGES = {
    "symmetry": (0.6, 1.2),
    "cycle_length": (10.0, 70.0),
    "act_ratio": (0.0, 1.0),
    "doub_time": (20.0, 100.0),
    "vol": (0.0, 10000.0),
    "colony_growth": (0.0, 100.0),
    "symmetry_std": (0.0, 0.01),
    "cycle_length_std": (0.0, 0.5),
    "act_ratio_std": (0.0, 0.01),
    "doub_time_std": (0.0, 0.5),
    "vol_std": (0.0, 1000.0),
    "colony_growth_std": (0.0, 10.0),
}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ABC-SMC-DRF on ARCADE model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--target", type=str, required=True, help="Path to target JSON file")
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Set defaults for missing values
    defaults = {
        "sobol_power": 9,
        "n_particles": None,  # overrides 2**sobol_power when set
        "run_name": None,  # overrides the derived output/input directory name
        "simplify_method": "linear",  # read unconditionally, even when simplify_model is false
        "n_group": 3,
        "n_min_sample": 5,
        "radius": 10,
        "margin": 2,
        "hex_size": 30,
        "simplify_model": True,
        "n_iterations": 5,
        "rf_type": "DRF",
        "n_trees": 50,
        "min_samples_leaf": 5,
        "random_state": 42,
        "criterion": "CART",
        "subsample_ratio": 0.5,
        "correlation_threshold": 0.8,
        "base_dir": "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast",
        "base_s3_dir": None,
        "base_input_dir": "../../../ARCADE_INPUT",
        "base_output_dir": "../../../ARCADE_OUTPUT",
    }

    for key, default_value in defaults.items():
        if key not in config:
            config[key] = default_value

    return config


def load_targets(target_path):
    """Load target values from JSON file"""
    with open(target_path, "r") as f:
        targets = json.load(f)

    # Convert to the format expected by the script
    target_names = []
    target_values = []

    for name, value in targets.items():
        target_names.append(name)
        target_values.append(value)

    return target_names, target_values


def prior_pdf(params, param_columns, param_ranges, config_params=None):
    """
    Evaluate the uniform prior density at the given parameters.

    Parameters:
    -----------
    params : array-like
        Array of parameter values for parameters being inferred,
        in the same order as they appear in INFER_PARAMS

    Returns:
    --------
    density : float
        Prior probability density at the given parameters (constant if valid, 0 if invalid)
    """
    if len(params) != len(param_ranges):
        raise ValueError(f"Expected {len(param_ranges)} parameters, got {len(params)}")
    if config_params["perturbed_config"] == "cellular":
        # Check if the number of parameters matches
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_index = param_columns.index(param_name)
            if params[param_index] < min_val or params[param_index] > max_val:
                return 0.0  # Parameter out of range, zero density
            else:
                return 1.0 / (max_val - min_val)
    else:
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_index = param_columns.index(param_name)
            if param_name in ["X_SPACING", "Y_SPACING"]:
                if config_params["point_based"] and param_name == "Y_SPACING":
                    original_y_spacing = int(params[param_index].split(":")[0])
                    if original_y_spacing < min_val or original_y_spacing > max_val:
                        return 0.0  # Parameter out of range, zero density
                else:
                    spacing = int(params[param_index].split(":")[-1])
                    if spacing < min_val or spacing > max_val:
                        return 0.0  # Parameter out of range, zero density
            elif param_name in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
                if params[param_index] < min_val or params[param_index] > max_val:
                    return 0.0  # Parameter out of range, zero density
    return 1.0 / (max_val - min_val)


_KERNEL_SCALE_LOGGED = set()


def _kernel_scale_factor(iteration, max_iterations):
    """Perturbation width for generation `iteration`.

    The published schedule is max(0.01, 0.1 * (1 - t / T)) with T the total
    generation count. Because T is also the loop bound, resuming a finished run
    at a larger T re-widens the kernel at the join. DED_KERNEL_SCHEDULE overrides
    the formula with an explicit comma-separated list indexed from t = 1, so a
    continuation can keep contracting. Iterations past the end of the list fall
    back to the formula, and an unset or empty variable reproduces the published
    behaviour exactly.
    """
    raw = os.environ.get("DED_KERNEL_SCHEDULE", "").strip()
    source = "published formula"
    value = max(0.01, 0.1 * (1 - iteration / max_iterations))
    if raw:
        for _sep in (";", ":", " "):
            raw = raw.replace(_sep, ",")
        schedule = [float(x) for x in raw.split(",") if x.strip()]
        if 1 <= iteration <= len(schedule):
            value = schedule[iteration - 1]
            source = "DED_KERNEL_SCHEDULE"
        else:
            source = "published formula (past end of DED_KERNEL_SCHEDULE)"
    if iteration not in _KERNEL_SCALE_LOGGED:
        _KERNEL_SCALE_LOGGED.add(iteration)
        logging.info(
            "iteration %d: perturbation scale_factor %.6g from %s",
            iteration, value, source,
        )
    return value


def perturbation_kernel(
    params, param_columns, param_ranges, iteration=1, max_iterations=5, seed=42, config_params=None
):
    perturbed_params = params.copy()
    scale_factor = _kernel_scale_factor(iteration, max_iterations)
    # Match the param_ranges with the parameter columns
    param_ranges = {col_name: param_ranges[col_name] for col_name in param_columns}
    if len(params) != len(param_ranges):
        raise ValueError(f"Expected {len(param_ranges)} parameters, got {len(params)}")
    if config_params["perturbed_config"] == "cellular":
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_index = param_columns.index(param_name)
            param_range = max_val - min_val
            scale = param_range * scale_factor
            perturbed_params[param_index] += np.random.normal(0, scale)
    elif config_params["perturbed_config"] == "source":
        for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
            param_index = param_columns.index(param_name)
            if param_name in ["X_SPACING", "Y_SPACING"]:
                if config_params["point_based"] and param_name == "Y_SPACING":
                    original_y_spacing = int(perturbed_params[param_index].split(":")[0])
                    perturb_value = config_params["y_interval"] * np.random.randint(-2, 3)
                    final_y_spacing = original_y_spacing + perturb_value
                    perturbed_params[param_index] = f"{final_y_spacing}:{final_y_spacing+1}"
                else:
                    if param_name == "X_SPACING":
                        x_spacing_value = float(
                            perturbed_params[param_index].split(":")[-1]
                        ) + np.random.randint(-4, 5)
                    else:
                        y_spacing_value = float(
                            perturbed_params[param_index].split(":")[-1]
                        ) + np.random.randint(-4, 5)
                        # swap x and y spacing if y spacing is smaller than x spacing
                        x_final = min(x_spacing_value, y_spacing_value)
                        y_final = max(x_spacing_value, y_spacing_value)
                        perturbed_params[param_index - 1] = "*:" + str(int(x_final))
                        perturbed_params[param_index] = "*:" + str(int(y_final))
            elif param_name in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
                param_range = max_val - min_val
                scale = param_range * scale_factor
                perturbed_params[param_index] += np.random.normal(0, scale)
    elif config_params["perturbed_config"] == "combined":
        # First handle cellular parameters
        cellular_params = {
            k: v
            for k, v in param_ranges.items()
            if k not in ["X_SPACING", "Y_SPACING", "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]
        }
        for i, (param_name, (min_val, max_val)) in enumerate(cellular_params.items()):
            param_index = param_columns.index(param_name)
            param_range = max_val - min_val
            scale = param_range * scale_factor
            perturbed_params[param_index] += np.random.normal(0, scale)

        # Then handle source parameters
        source_params = {
            k: v
            for k, v in param_ranges.items()
            if k in ["X_SPACING", "Y_SPACING", "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]
        }
        for i, (param_name, (min_val, max_val)) in enumerate(source_params.items()):
            param_index = param_columns.index(param_name)
            if param_name in ["X_SPACING", "Y_SPACING"]:
                if config_params["point_based"] and param_name == "Y_SPACING":
                    original_y_spacing = int(perturbed_params[param_index].split(":")[0])
                    perturb_value = config_params["y_interval"] * np.random.randint(-2, 3)
                    final_y_spacing = original_y_spacing + perturb_value
                    perturbed_params[param_index] = f"{final_y_spacing}:{final_y_spacing+1}"
                else:
                    if param_name == "X_SPACING":
                        x_spacing_value = float(
                            perturbed_params[param_index].split(":")[-1]
                        ) + np.random.randint(-4, 5)
                    else:
                        y_spacing_value = float(
                            perturbed_params[param_index].split(":")[-1]
                        ) + np.random.randint(-4, 5)
                        # swap x and y spacing if y spacing is smaller than x spacing
                        x_final = min(x_spacing_value, y_spacing_value)
                        y_final = max(x_spacing_value, y_spacing_value)
                        perturbed_params[param_index - 1] = "*:" + str(int(x_final))
                        perturbed_params[param_index] = "*:" + str(int(y_final))
            elif param_name in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
                param_range = max_val - min_val
                scale = param_range * scale_factor
                # print(f"scale: {scale}")
                # print(f"perturbed_params[param_index]: {perturbed_params[param_index]}")
                perturbed_params[param_index] += np.random.normal(0, scale)
    return perturbed_params


def plot_variable_importance(smc_rf, statistic_names, n_statistics, save_path=None):
    """Plot variable importance from the final iteration"""
    importance = smc_rf.get_variable_importance(t=None, n_statistics=n_statistics)
    # Ensure we have the right number of names
    if len(importance) != len(statistic_names):
        raise ValueError(
            f"Number of statistics ({len(importance)}) doesn't match number of names ({len(statistic_names)})"
        )

    plt.figure(figsize=(12, 6))
    plt.bar(statistic_names, importance)
    plt.xticks(rotation=45, ha="right")
    plt.title("Summary Statistic Importance")
    plt.ylabel("Importance")
    plt.xlabel("Summary Statistics")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_parameter_iterations(smc_rf, param_names, sobol_power=8, plot_kde=False, save_path=None):
    """Plot parameter distributions across iterations.

    Parameters:
    -----------
    smc_rf : ABCSMCRF
        The fitted ABC-SMC-RF object
    param_names : list
        List of parameter names
    plot_kde : bool, optional
        Whether to plot KDE (default: True)
    save_path : str, optional
        Path to save the figure
    """
    n_iterations = len(smc_rf.parameter_samples)
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, n_iterations, figsize=(12, 5 * n_params), squeeze=False)
    # Generate Sobol samples for prior
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=len(param_names), seed=42)
    n_samples = _num_sobol_samples(sobol_power)

    for idx, param_name in enumerate(param_names):
        param_idx = smc_rf.parameter_columns.index(param_name)
        min_val, max_val = smc_rf.param_ranges[param_name]

        for t in range(n_iterations):
            params, _, weights = smc_rf.get_iteration_results(t)
            param_values = params[:, param_idx]
            ax = axes[idx, t]
            ax.set_xlim(min_val, max_val)

            # Plot histogram
            ax.hist(
                param_values,
                bins=10,
                weights=weights,
                alpha=0.5,
                density=True,
                color="blue",
                label="Posterior",
                edgecolor="black",
            )

            # Calculate and plot KDE
            if plot_kde and len(param_values) > 1:
                try:
                    kde = gaussian_kde(param_values, weights=weights)
                    x_range = np.linspace(min_val, max_val, 200)
                    ax.plot(x_range, kde(x_range), "r-", lw=2, label="KDE")
                except np.linalg.LinAlgError:
                    print(f"Error computing KDE for {param_name} at iteration {t+1}")

            if idx == 0:
                ax.set_title(f"Iteration {t+1}")

            if t == 0:
                ax.set_ylabel(param_name)
                prior_samples = np.random.uniform(min_val, max_val, n_samples)
                ax.hist(
                    prior_samples, bins=20, alpha=0.5, density=True, color="gray", label="Prior"
                )

            if idx == n_params - 1:
                ax.set_xlabel("Value")

            if idx == 0 and t == 0:
                ax.legend()

    plt.suptitle("Parameter Distributions Across Iterations")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_parameter_iterations.png")
    else:
        plt.show()


def plot_statistic_iterations(smc_rf, target_names, target_values, plot_kde=True, save_path=None):
    """Plot statistic distributions across iterations.

    Parameters:
    -----------
    smc_rf : ABCSMCRF
        The fitted ABC-SMC-RF object
    target_names : list
        List of target statistic names
    target_values : list
        List of target values
    plot_kde : bool, optional
        Whether to plot KDE (default: True)
    save_path : str, optional
        Path to save the figure
    """
    n_iterations = len(smc_rf.parameter_samples)
    n_stats = len(target_names)
    fig, axes = plt.subplots(n_stats, n_iterations, figsize=(12, 5 * n_stats), squeeze=False)

    for idx, (stat_name, target_val) in enumerate(zip(target_names, target_values)):
        min_val, max_val = TARGET_RANGES[stat_name]

        for t in range(n_iterations):
            _, stats, weights = smc_rf.get_iteration_results(t)
            stat_values = stats[:, idx]
            # remove NaN or -inf, inf values
            weights = weights[~np.isnan(stat_values) & ~np.isinf(stat_values)]
            stat_values = stat_values[~np.isnan(stat_values) & ~np.isinf(stat_values)]

            ax = axes[idx, t]
            ax.set_xlim(min_val, max_val)

            # Plot histogram
            if t == 0:
                ax.hist(
                    stat_values,
                    bins=10,
                    weights=weights,
                    alpha=0.5,
                    color="gray",
                    label="Prior",
                    edgecolor="black",
                )
            else:
                ax.hist(
                    stat_values,
                    bins=10,
                    weights=weights,
                    alpha=0.5,
                    color="blue",
                    label="Posterior",
                    edgecolor="black",
                )

            # Add target line
            ax.axvline(target_val, color="g", linestyle="--", label="Target")
            ax.annotate(
                f"{target_val:.2f}",
                xy=(target_val, 0),
                xytext=(10, 10),
                textcoords="offset points",
                color="g",
                bbox=dict(facecolor="white", edgecolor="black", alpha=0.7),
            )
            # Calculate and plot KDE
            if plot_kde and len(stat_values) > 1:
                try:
                    kde = gaussian_kde(stat_values, weights=weights)
                    x_range = np.linspace(min_val, max_val, 200)
                    kde_values = kde(x_range)
                    ax.plot(x_range, kde_values, "r-", lw=2, label="KDE")

                    # Calculate and plot the mode of KDE
                    mode_idx = np.argmax(kde_values)
                    mode_x = x_range[mode_idx]
                    mode_y = kde_values[mode_idx]
                    ax.plot(mode_x, mode_y, "r^", markersize=10, label="KDE Mode")
                    ax.annotate(
                        f"{mode_x:.2f}",
                        xy=(mode_x, mode_y),
                        xytext=(10, 10),
                        textcoords="offset points",
                        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
                    )
                except np.linalg.LinAlgError and ValueError:
                    print(f"Error computing KDE for {stat_name} at iteration {t+1}")

            if idx == 0:
                ax.set_title(f"Iteration {t+1}")

            if t == 0:
                ax.set_ylabel(stat_name)

            if idx == n_stats - 1:
                ax.set_xlabel("Value")

            if idx == 0 and (t == 0 or t == 1):
                ax.legend()

    plt.suptitle("Statistics Distributions Across Iterations")
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}_statistic_iterations.png")
    else:
        plt.show()


def save_targets_to_json(target_names, target_values, output_file="targets.json"):
    """
    Save target names and their corresponding values to a JSON file.

    Parameters:
    -----------
    target_names : list
        List of target statistic names
    target_values : list
        List of target values
    output_file : str, optional
        Path to the output JSON file (default: "targets.json")
    """
    targets_dict = dict(zip(target_names, target_values))
    save_json(targets_dict, output_file, indent=4)


def run_example(config, target_names, target_values):
    """Run the ABC-SMC-DRF example on the ARCADE model"""
    targets = []
    for name, value in zip(target_names, target_values):
        targets.append(Target(metric=Metric.get(name), value=value, weight=1.0))
    n_statistics = len(target_names)
    print("\nRunning ABC-SMC-DRF...")
    start_time = time.time()

    # Extract configuration parameters
    sobol_power = config["sobol_power"]
    radius = config["radius"]
    margin = config["margin"]
    hex_size = config["hex_size"]
    side_length = hex_size / np.sqrt(3)
    simplify_model = config["simplify_model"]
    simplify_method = config["simplify_method"]
    n_group = config["n_group"]
    n_min_sample = config["n_min_sample"]
    correlation_threshold = config["correlation_threshold"]
    template_path = config["template_path"]
    base_dir = config["base_dir"]
    base_s3_dir = config["base_s3_dir"]
    base_input_dir = config["base_input_dir"]
    base_output_dir = config["base_output_dir"]
    n_cpu = config["n_cpu"]
    config_peak_name = config["peak_name"]
    mean_only = config["mean_only"]

    if mean_only:
        from inverse_design.config.parameter_config_mu_only import (
            PARAM_RANGES,
            SOURCE_PARAM_RANGES,
            PARAMS_DEFAULTS,
            SOURCE_PARAMS_DEFAULTS,
        )
    else:
        from inverse_design.config.parameter_config import (
            PARAM_RANGES,
            SOURCE_PARAM_RANGES,
            PARAMS_DEFAULTS,
            SOURCE_PARAMS_DEFAULTS,
        )
    PARAM_RANGES = PARAM_RANGES.copy()
    PARAMS_DEFAULTS = PARAMS_DEFAULTS.copy()
    PARAMS_DEFAULTS.update(SOURCE_PARAMS_DEFAULTS)

    input_configs = [
        {
            "perturbed_config": "cellular",
            "template_path": "sample_inputs/sample_cellular_v3.xml",
            "point_based": None,
            "y_interval": None,
            "radius_bound": None,
            "side_length": None,
        },
        {
            "perturbed_config": "source",
            "template_path": "sample_inputs/sample_source_v3.xml",
            "point_based": True,
            "y_interval": 4,
            "radius_bound": radius + margin,
            "side_length": side_length,
        },
        {
            "perturbed_config": "combined",
            "template_path": template_path,  # "sample_combined_v3.xml",#
            "point_based": False,
            "y_interval": 4,
            "radius_bound": radius + margin,
            "side_length": side_length,
        },
    ]
    n_samples = _num_sobol_samples(sobol_power)
    config_params = input_configs[2]
    if config_params["perturbed_config"] == "cellular":
        param_ranges = PARAM_RANGES.copy()
    elif config_params["perturbed_config"] == "source":
        param_ranges = SOURCE_PARAM_RANGES.copy()
    elif config_params["perturbed_config"] == "combined":
        param_ranges = {**PARAM_RANGES, **SOURCE_PARAM_RANGES}
    if config_params["point_based"]:
        param_ranges.pop("X_SPACING")
    print(f"Parameter ranges: {param_ranges}")
    source_type = "point" if config_params["point_based"] else "grid"
    jar_path = "models/arcade-logging-necrotic.jar"

    linear_model = False
    if simplify_model:
        if mean_only:
            input_dir = f"{base_input_dir}/abc_smc_rf_n{n_samples}_{config_params['perturbed_config']}_{source_type}_{simplify_method}_{correlation_threshold}_p{config_peak_name.split()[1]}_mean_only/"
            output_dir = f"{base_output_dir}/ABC_SMC_RF_N{n_samples}_{config_params['perturbed_config']}_{source_type}_{simplify_method}_{correlation_threshold}_p{config_peak_name.split()[1]}_mean_only/"
        else:
            input_dir = f"{base_input_dir}/abc_smc_rf_n{n_samples}_{config_params['perturbed_config']}_{source_type}_{simplify_method}_{correlation_threshold}_p{config_peak_name.split()[1]}/"
            output_dir = f"{base_output_dir}/ABC_SMC_RF_N{n_samples}_{config_params['perturbed_config']}_{source_type}_{simplify_method}_{correlation_threshold}_p{config_peak_name.split()[1]}/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        posterior_df = read_csv(f"{base_dir}/iter_4/all_param_df.csv")
        drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
        posterior_df.drop(columns=drop_cols, inplace=True)
        pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
            posterior_df, n_components=2
        )
        distances = np.sqrt(
            np.sum(
                (pca_result[:, :2] - peak_positions[int(config_peak_name.split()[1]) - 1]) ** 2,
                axis=1,
            )
        )
        closest_indices = np.argsort(distances)[:50]
        peak_data = posterior_df.iloc[closest_indices]
        if simplify_method == "dendrogram":  # dendrogram-based clustering
            correlation_matrix = np.corrcoef(peak_data.values, rowvar=False)
            (
                threshold,
                labels,
                valid_clusters,
                representative_parameters,
                redundant_params_dict,
                ylbls,
            ) = colorize_ward_clusters_threshold(
                data=correlation_matrix,
                param_names=np.array(peak_data.columns.tolist()),
                ward_threshold=correlation_threshold,
                save_path=f"{base_dir}/redundancy_analysis_threshold_{correlation_threshold}_p{config_peak_name.split()[1]}.png",
                verbose=False,
            )
            # Flatten the redundant_params dictionary
            redundant_params = [
                param for group in redundant_params_dict.values() for param in group
            ]
            for redundant_param in redundant_params:
                if redundant_param == "CAPILLARY_DENSITY":
                    X_SPACING, Y_SPACING = (
                        PARAMS_DEFAULTS["X_SPACING"],
                        PARAMS_DEFAULTS["Y_SPACING"],
                    )
                    param_ranges["X_SPACING"] = (X_SPACING, X_SPACING)
                    param_ranges["Y_SPACING"] = (Y_SPACING, Y_SPACING)
                    param_ranges[redundant_param] = (
                        PARAMS_DEFAULTS[redundant_param],
                        PARAMS_DEFAULTS[redundant_param],
                    )
                else:
                    param_ranges[redundant_param] = (
                        PARAMS_DEFAULTS[redundant_param],
                        PARAMS_DEFAULTS[redundant_param],
                    )

            individual_params = []
            for param in peak_data.columns:
                if param not in representative_parameters and param not in redundant_params:
                    individual_params.append(param)
            print(
                f"Representative: {representative_parameters}, shape: {len(representative_parameters)}"
            )
            print("-" * 50)
            print(f"redundant_params: {redundant_params}, shape: {len(redundant_params)}")
            print("-" * 50)
            print(f"individual_params: {individual_params}, shape: {len(individual_params)}")
            print("=" * 50)
            assert len(representative_parameters) + len(redundant_params) + len(
                individual_params
            ) == len(posterior_df.columns)

            save_json(
                {
                    "representative_parameters": representative_parameters,
                    "redundant_params": redundant_params_dict,
                    "individual_params": individual_params,
                },
                f"{output_dir}/redundancy_analysis_threshold_{correlation_threshold}_p{config_peak_name.split()[1]}.json",
                indent=4,
            )
        if simplify_method == "linear":  # r2-based clustering
            linear_model = True
            predictor = CorrelationPredictor(r_threshold=correlation_threshold)
            corr_matrix = predictor.compute_correlations(peak_data)
            edges = predictor.filter_edges()
            graphs = predictor.build_graphs(edges)
            predictor.predict_parameters(peak_data)
            lr_summary = predictor.save_prediction_summary_json(
                f"{output_dir}/lr_predictions_r{correlation_threshold}_p{config_peak_name.split()[1]}.json"
            )
            # predictor.visualize_graphs()
            if lr_summary["sampling_instructions"]["reduction_efficiency"] == "0.0%":
                linear_model = False
                print(f"ABC-SMC-DRF ends due to no chains found")
                return
    else:
        if mean_only:
            input_dir = f"{base_input_dir}/abc_smc_rf_n{n_samples}_{config_params['perturbed_config']}_{source_type}_only_mean/"
            output_dir = f"{base_output_dir}/ABC_SMC_RF_N{n_samples}_{config_params['perturbed_config']}_{source_type}_only_mean/"
        else:
            input_dir = f"{base_input_dir}/abc_smc_rf_n{n_samples}_{config_params['perturbed_config']}_{source_type}/"
            output_dir = f"{base_output_dir}/ABC_SMC_RF_N{n_samples}_{config_params['perturbed_config']}_{source_type}/"
        param_ranges = {k: v for k, v in param_ranges.items() if v[0] != v[1]}
        analysis_results = None
        chain_analysis = None

    # An explicit run_name overrides the name derived above. The derived name
    # encodes only particle count, perturbed config and source type, so it
    # cannot address runs whose directories carry extra qualifiers (the
    # published breast run is ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2).
    # Without this, resuming such a run silently starts a fresh one instead.
    # Output keeps the given case; input follows the existing lowercase convention.
    run_name = config["run_name"]
    if run_name:
        input_dir = f"{base_input_dir}/{run_name.lower()}/"
        output_dir = f"{base_output_dir}/{run_name}/"
        logging.info("run_name override: output=%s", output_dir)

    smc_rf_configs = {
        "n_iterations": config["n_iterations"],
        "sobol_power": sobol_power,
        "n_particles": config["n_particles"],
        "rf_type": config["rf_type"],
        "n_trees": config["n_trees"],
        "min_samples_leaf": config["min_samples_leaf"],
        "param_ranges": param_ranges,
        "random_state": config["random_state"],
        "criterion": config["criterion"],
        "subsample_ratio": config["subsample_ratio"],
        "n_cpu": n_cpu,
        "perturbation_kernel": perturbation_kernel,
        "prior_pdf": prior_pdf,
        "config_params": config_params,
    }
    if linear_model:
        smc_rf = ABCSMCRF_simplified(**smc_rf_configs, lr_summary=lr_summary)
    else:
        smc_rf = ABCSMCRF(
            **smc_rf_configs,
        )
    print(f"After simplification: Parameter ranges: {param_ranges}")
    timestamps = [
        "000720",
        "001440",
        "002160",
        "002880",
        "003600",
        "004320",
        "005040",
        "005760",
        "006480",
        "007200",
        "007920",
        "008640",
        "009360",
        "010080",
    ]
    # timestamps = timestamps[:6]
    ensure_dir_exists(output_dir)
    save_targets_to_json(target_names, target_values, f"{output_dir}targets.json")
    smc_rf.fit(
        target_names, target_values, input_dir, output_dir, jar_path, timestamps, config_peak_name
    )
    # Upload the results to S3
    if base_s3_dir:
        # Archive the input and output trees. The raw per-simulation ARCADE
        # dumps are deliberately left out of the output archive: they are ~33 GB
        # per generation, and the input XML pins every parameter value and the
        # seed range, so ARCADE v3.3.0 regenerates them exactly. What is kept is
        # what the analysis actually rests on -- the derived CSVs, the weights
        # and ess.json -- which is a few megabytes per generation.
        raw_arcade_output = ["*.CELLS.json", "*.LOCATIONS.json"]
        input_dir_zip = archive_directory(input_dir)
        output_dir_zip = archive_directory(output_dir, exclude=raw_arcade_output)
        stem = f"{config_params['perturbed_config']}_{source_type}_{simplify_method}_{correlation_threshold}_p{config_peak_name.split()[1]}"
        # The extension follows whichever archiver was available, so take it
        # from the produced file rather than assuming .zip.
        input_suffix = input_dir_zip[input_dir_zip.index(".", input_dir_zip.rfind("/")) :]
        output_suffix = output_dir_zip[output_dir_zip.index(".", output_dir_zip.rfind("/")) :]
        upload_file_to_s3(
            file_path=input_dir_zip,
            s3_destination=f"{base_s3_dir}/ARCADE_INPUT/abc_smc_rf_n{n_samples}_{stem}{input_suffix}",
        )
        upload_file_to_s3(
            file_path=output_dir_zip,
            s3_destination=f"{base_s3_dir}/ARCADE_OUTPUT/ABC_SMC_RF_N{n_samples}_{stem}{output_suffix}",
        )
        print(f"ABC-SMC-DRF completed in {time.time() - start_time:.2f} seconds")
        return
    return
    plot_dir = f"{output_dir}/PLOTS/"
    ensure_dir_exists(plot_dir)
    smc_rf.plot_tree(
        iteration=-1,
        feature_names=target_names,
        max_depth=10,
        target_values=target_values,
        save_path=f"{plot_dir}/arcade_tree",
    )
    print(f"ABC-SMC-DRF completed in {time.time() - start_time:.2f} seconds")
    return
    posterior_samples = smc_rf.posterior_sample(n_samples * 2)
    n_samples = posterior_samples.shape[0]
    print("\nParameter estimation results:")
    valid_parameters = read_csv(f"{output_dir}/iter_0/all_param_df.csv")
    valid_parameters.drop(columns=["input_folder"], inplace=True)
    if len(valid_parameters.columns) > len(smc_rf.param_ranges.keys()):
        valid_parameters = valid_parameters[smc_rf.param_ranges.keys()]
    if config_params["point_based"]:
        valid_parameters.drop(columns=["CAPILLARY_DENSITY"], inplace=True)
    else:
        valid_parameters.drop(columns=["DISTANCE_TO_CENTER"], inplace=True)
    parameter_columns = list(valid_parameters.columns)
    for i, param_name in enumerate(parameter_columns):
        if config_params["perturbed_config"] == "cellular":
            if 1:  # print mode
                unique_values, counts = np.unique(posterior_samples[:, i], return_counts=True)
                param_mode = unique_values[np.argmax(counts)]
                param_std = np.std(posterior_samples[:, i])
                print(f"{param_name}: {param_mode:.4f} ± {param_std:.4f}")
            else:  # print mean
                print(
                    f"{param_name}: {np.mean(posterior_samples[:, i]):.4f} ± {np.std(posterior_samples[:, i]):.4f}"
                )
        else:
            if param_name == "X_SPACING":
                continue
            elif param_name == "Y_SPACING":
                if config_params["point_based"]:
                    x_center = int((6 * config_params["radius_bound"] - 3) // 2)
                    y_center = int((6 * config_params["radius_bound"] - 3) // 2)
                    point_center = (x_center, y_center, 0)
                    source_sites = []
                    for j in range(n_samples):
                        source_sites.append(
                            np.array([x_center, posterior_samples[j][i].split(":")[1], 0]).astype(
                                int
                            )
                        )
                    distance_to_center = [
                        calculate_distance_between_points(
                            point_center,
                            source_site,
                            config_params["side_length"],
                            config_params["radius_bound"],
                        )
                        for source_site in source_sites
                    ]
                    if 1:  # print mode
                        # Calculate mode for distance to center
                        unique_values, counts = np.unique(distance_to_center, return_counts=True)
                        distance_mode = unique_values[np.argmax(counts)]
                        distance_std = np.std(distance_to_center)
                        print(f"DISTANCE_TO_CENTER: {distance_mode:.4f} ± {distance_std:.4f}")
                    else:  # print mean
                        print(
                            f"DISTANCE_TO_CENTER: {np.mean(distance_to_center):.4f} ± {np.std(distance_to_center):.4f}"
                        )
                else:
                    x_spacing_values = [
                        int(posterior_samples[j][i - 1].split(":")[1]) for j in range(n_samples)
                    ]
                    y_spacing_values = [
                        int(posterior_samples[j][i].split(":")[1]) for j in range(n_samples)
                    ]

                    capillary_densities = [
                        calculate_capillary_density(
                            radius + margin,
                            x_spacing,
                            y_spacing,
                            config_params["side_length"],
                        )
                        for x_spacing, y_spacing in zip(x_spacing_values, y_spacing_values)
                    ]
                    if 1:  # print mode
                        # Calculate mode for capillary density
                        unique_values, counts = np.unique(capillary_densities, return_counts=True)
                        capillary_mode = unique_values[np.argmax(counts)]
                        capillary_std = np.std(capillary_densities)
                        print(f"CAPILLARY_DENSITY: {capillary_mode:.4f} ± {capillary_std:.4f}")

                        # Calculate mode for x_spacing_values
                        unique_values, counts = np.unique(x_spacing_values, return_counts=True)
                        x_spacing_mode = unique_values[np.argmax(counts)]
                        x_spacing_std = np.std(x_spacing_values)
                        print(f"X_SPACING: {x_spacing_mode:.4f} ± {x_spacing_std:.4f}")

                        # Calculate mode for y_spacing_values
                        unique_values, counts = np.unique(y_spacing_values, return_counts=True)
                        y_spacing_mode = unique_values[np.argmax(counts)]
                        y_spacing_std = np.std(y_spacing_values)
                        print(f"Y_SPACING: {y_spacing_mode:.4f} ± {y_spacing_std:.4f}")
                    else:  # print mean
                        print(
                            f"CAPILLARY_DENSITY: {np.mean(capillary_densities):.4f} ± {np.std(capillary_densities):.4f}"
                        )
            else:
                if 1:  # print mode
                    unique_values, counts = np.unique(posterior_samples[:, i], return_counts=True)
                    param_mode = unique_values[np.argmax(counts)]
                    param_std = np.std(posterior_samples[:, i])
                    print(f"{param_name}: {param_mode:.12f} ± {param_std:.12f}")
                else:  # print mean
                    print(
                        f"{param_name}: {np.mean(posterior_samples[:, i]):.12f} ± {np.std(posterior_samples[:, i]):.12f}"
                    )
    # Plot results
    if config_params["perturbed_config"] == "cellular":
        param_names = ["AFFINITY", "COMPRESSION_TOLERANCE", "CELL_VOLUME_MU"]
    elif config_params["perturbed_config"] == "source":
        if config_params["point_based"]:
            param_names = ["Y_SPACING", "GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]
        else:
            param_names = [
                "X_SPACING",
                "Y_SPACING",
                "GLUCOSE_CONCENTRATION",
                "OXYGEN_CONCENTRATION",
            ]
    elif config_params["perturbed_config"] == "combined":
        param_names = [
            "COMPRESSION_TOLERANCE",
            "CELL_VOLUME_MU",
            "X_SPACING",
            "Y_SPACING",
            "GLUCOSE_CONCENTRATION",
            "OXYGEN_CONCENTRATION",
        ]
        param_names = [param for param in param_names if param in valid_parameters.columns]

    # plot_parameter_iterations(smc_rf, param_names, save_path=f"{plot_dir}/arcade_params_iterations")
    # plot_statistic_iterations(smc_rf, target_names, target_values, save_path=f"{plot_dir}/arcade_stats_iterations", plot_kde=False)
    # plot_variable_importance(smc_rf, target_names, n_statistics, save_path=f"{plot_dir}/arcade_variable_importance.png")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Load configuration and targets
    config = load_config(args.config)
    # Decouple the particle count from 2**sobol_power when the config asks for it.
    set_particle_count(config["n_particles"])
    if config["n_particles"] is not None:
        logging.info(
            "Particle count overridden to %d (sobol_power=%d would give %d)",
            config["n_particles"],
            config["sobol_power"],
            2 ** config["sobol_power"],
        )
    target_names, target_values = load_targets(args.target)

    # Run the example
    run_example(config, target_names, target_values)
