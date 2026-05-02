import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Optional
from scipy.stats import gaussian_kde


def _resolve_id_column(df: pd.DataFrame, metrics_name: str, id_column: Optional[str]) -> str:
    """Pick the column that identifies each row (ARCADE uses ``input_folder``)."""
    if id_column is not None:
        if id_column not in df.columns:
            raise KeyError(id_column)
        return id_column
    if "input_folder" in df.columns:
        return "input_folder"
    skip = {metrics_name, "states"}
    others = [c for c in df.columns if c not in skip]
    if len(others) == 1:
        return others[0]
    if "folder" in df.columns:
        return "folder"
    raise ValueError(
        "Cannot infer folder/id column (expected 'input_folder', 'folder', or exactly one "
        f"non-metric column). Columns: {list(df.columns)}. Pass id_column=..."
    )


def get_parameters_from_json(input_folder, parameter_list):
    """Extract parameters from json file."""
    # Find the base configuration file by looking for a .json file without CELL or LOCATION in the name
    json_files = list(Path(input_folder).glob("*.json"))
    config_file = next(f for f in json_files if "CELL" not in f.name and "LOCATION" not in f.name)
    with open(config_file, "r") as f:
        data = json.load(f)

    params = {}
    for param in parameter_list:
        # Remove any prefix (e.g., "metabolism/") from parameter name
        base_param = param.split("/")[-1]

        # Check different possible paths for the parameter
        possible_paths = [
            ("populations", "cancerous", param),  # Direct path
            ("populations", "cancerous", f"metabolism/{base_param}"),  # With metabolism prefix
            (
                "populations",
                "cancerous",
                f"proliferation/{base_param}",
            ),  # With proliferation prefix
            ("populations", "cancerous", f"signaling/{base_param}"),  # With proliferation prefix
        ]

        value = None
        for path in possible_paths:
            try:
                temp_value = data
                for key in path:
                    temp_value = temp_value[key]
                value = temp_value
                break
            except (KeyError, TypeError):
                continue

        if value is None:
            # print(f"Parameter {param} not found in any expected location in the JSON file")
            continue

        params[base_param] = value

    return params


def analyze_metric_percentiles(
    csv_file_path,
    metrics_name: str,
    percentile: float = 10,
    verbose: bool = True,
    id_column: Optional[str] = None,
):
    """
    Analyze rows where metrics fall into the bottom and top percentiles, optionally removing outliers first.

    Args:
        csv_file_path: Path to the aggregated results CSV file
        metrics_name: Name of the metric to analyze
        percentile: Percentile value between 0 and 50 (default: 10)
        verbose: Whether to print detailed analysis (default: True)
        id_column: Column naming each run/folder (default: ``input_folder`` if present, else inferred)

    Returns:
        tuple: (high_metric_folders, low_metric_folders, all_data)
    """
    if not 0 < percentile <= 50:
        raise ValueError("Percentile must be between 0 and 50")

    df = pd.read_csv(csv_file_path)
    folder_key = _resolve_id_column(df, metrics_name, id_column)

    # Calculate lower and upper percentiles on cleaned data
    lower_bound = df[metrics_name].quantile(percentile / 100)
    upper_bound = df[metrics_name].quantile(1 - percentile / 100)

    # Create a copy of the dataframe and add labels
    all_data = df.copy()
    all_data["percentile_label"] = "not_assigned"
    all_data.loc[df[metrics_name] <= lower_bound, "percentile_label"] = "low_metric"
    all_data.loc[df[metrics_name] >= upper_bound, "percentile_label"] = "high_metric"

    # Get the original outputs for backward compatibility
    low_metric_cases = df[df[metrics_name] <= lower_bound]
    high_metric_cases = df[df[metrics_name] >= upper_bound]
    high_metric_folders = high_metric_cases[folder_key].unique()
    low_metric_folders = low_metric_cases[folder_key].unique()

    if verbose:
        print(f"\nAnalysis for {metrics_name}")
        print("=" * 80)
        print(f"\nLow {metrics_name} cases ({percentile}%) (≤ {lower_bound:.3f}):")
        print(
            (
                low_metric_cases.drop(
                    columns=["states"] if "states" in low_metric_cases.columns else []
                )
            ).head()
        )
        print(f"\nHigh {metrics_name} cases ({percentile}%) (≥ {upper_bound:.3f}):")
        print(
            (
                high_metric_cases.drop(
                    columns=["states"] if "states" in high_metric_cases.columns else []
                )
            ).head()
        )
        print("\nLabel distribution:")
        print(all_data["percentile_label"].value_counts())

    return high_metric_folders, low_metric_folders, all_data


def collect_parameter_data(input_files, parameter_list, labels=None):
    """
    Collect parameter data from JSON files for a list of input folders.

    Args:
        input_files: List of input folder names
        parameter_list: List of parameters to extract
        labels: Optional list of labels corresponding to input_files

    Returns:
        pandas.DataFrame: DataFrame containing the parameter values, labels, and folder names
    """
    params_list = []
    for i, input_folder in enumerate(input_files):
        input_folder_path = Path(input_folder)
        params = get_parameters_from_json(input_folder_path, parameter_list)
        # Add folder name to params
        params["input_folder"] = str(input_folder_path.name)
        if labels is not None:
            params["percentile_label"] = labels[i]
        params_list.append(params)
    # sort params_list by numeric value in input_folder name
    params_list.sort(key=lambda x: int(x["input_folder"].split("_")[1]))
    return pd.DataFrame(params_list)


def remove_outliers(data: pd.DataFrame, iqr_multiplier: float = 1.5, verbose: bool = True):
    """
    Remove outliers from the data using the IQR method.

    Args:
        data: DataFrame containing the metrics
        iqr_multiplier: Multiplier for IQR to determine outlier threshold (default: 1.5)
        verbose: Whether to print analysis details (default: True)

    Returns:
        tuple: (cleaned_data, outlier_indices, outlier_data)
            - cleaned_data: DataFrame with outliers removed
            - outlier_indices: Indices of identified outliers
            - outlier_data: DataFrame containing only the outliers
    """
    # Calculate Q1, Q3, and IQR
    Q1 = data["value"].quantile(0.25)
    Q3 = data["value"].quantile(0.75)
    IQR = Q3 - Q1

    # Calculate outlier bounds
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR

    # Identify outliers
    outlier_mask = (data["value"] < lower_bound) | (data["value"] > upper_bound)
    outlier_indices = data[outlier_mask].index
    outlier_data = data.loc[outlier_indices]
    cleaned_data = data.loc[~outlier_mask]

    if verbose:
        print(f"\nOutlier Analysis for {data['metric'].unique()[0]}")
        print("=" * 80)
        print(f"Q1: {Q1:.3f}")
        print(f"Q3: {Q3:.3f}")
        print(f"IQR: {IQR:.3f}")
        print(f"Lower bound: {lower_bound:.3f}")
        print(f"Upper bound: {upper_bound:.3f}")
        print(f"Number of outliers removed: {len(outlier_indices)}")
        print(f"Remaining data points: {len(cleaned_data)}")

    return cleaned_data, outlier_indices, outlier_data


def calculate_percentage_error(values, target):
    """Return element-wise absolute percentage error relative to *target*."""
    from inverse_design.analyze.errors import absolute_percent_error

    return absolute_percent_error(values, target)


def calculate_metrics_statistics(metrics_df, metrics_names):
    """
    Calculate statistics (mode from KDE and std) for each metric.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics
        metrics_names (list): List of metric names to analyze

    Returns:
        dict: Dictionary containing mode and std for each metric
    """
    metrics_dict = {}
    for metric in metrics_names:
        values = metrics_df[metric].values

        # Fit KDE
        kde = gaussian_kde(values)

        # Find mode by evaluating KDE on a fine grid
        x_grid = np.linspace(min(values), max(values), 1000)
        kde_values = kde(x_grid)
        mode_idx = np.argmax(kde_values)
        mode = x_grid[mode_idx]

        # Calculate standard deviation
        std = np.std(values)

        metrics_dict[metric] = {"mode": mode, "metric": values, "std": std, "kde": kde}

    return metrics_dict
