import numpy as np
import pandas as pd
from typing import List
import json
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns

from inverse_design.plotting.theme import apply_publication_style


def load_data(param_file, metrics_file, target_name: List[str]):
    """
    Load parameter and metric data from CSV files and perform sensitivity analysis.

    Args:
        param_file (str): Path to parameters CSV file
        metrics_file (str): Path to metrics CSV file

    Returns:
        dict: Results from Sobol analysis
    """
    # Load data
    param_df = pd.read_csv(param_file)
    metrics_df = pd.read_csv(metrics_file)
    param_df = clean_param_df(param_df)
    metrics_df = clean_metrics_df(metrics_df, target_name)

    return param_df, metrics_df


def clean_param_df(param_df):
    constant_columns = param_df.columns[param_df.nunique() == 1]
    param_df = param_df.drop(columns=constant_columns)
    return param_df


def clean_metrics_df(metrics_df, target_name: List[str]):
    return metrics_df[target_name]


def perform_mi_analysis(param_df, metrics_df, parameter_names, n_neighbors=3, save_mi_json=None):
    """
    Perform sensitivity analysis using Mutual Information for strength of relationship
    and Spearman correlation for direction of effect.

    Args:
        param_df (pd.DataFrame): DataFrame of parameter values
        metrics_df (pd.DataFrame): DataFrame of output values
        parameter_names (list): List of parameter names
        n_neighbors (int): Number of neighbors for MI estimation

    Returns:
        dict: Dictionary containing sensitivity metrics
    """
    from sklearn.feature_selection import mutual_info_regression

    X = param_df.to_numpy()

    results = {}
    for metric in metrics_df.columns:
        Y = metrics_df[metric].to_numpy()

        # Calculate Mutual Information scores (without normalization)
        mi_scores = mutual_info_regression(X, Y, n_neighbors=n_neighbors, random_state=42)

        # Calculate Spearman correlations for direction
        spearman_corr = np.array([spearmanr(X[:, i], Y)[0] for i in range(X.shape[1])])
        results[metric] = {"MI": mi_scores, "spearman": spearman_corr, "names": parameter_names}
    if save_mi_json:
        results_json = {}
        for metric, result in results.items():
            results_json[metric] = {
                "MI": result["MI"].tolist(),
                "spearman": result["spearman"].tolist(),
                "names": result["names"],
            }

        with open(save_mi_json, "w") as f:
            json.dump(results_json, f)

    return results


def plot_mi_analysis(mi_results, save_path=None):
    """
    Plot sensitivity analysis results showing absolute importance.
    For each metric, creates a subplot showing:
    - Absolute MI values with direction (Spearman)
    - Only parameters with MI > 0.01
    Styled to match academic figure format.

    Args:
        mi_results (dict): Results from perform_mi_analysis
        save_path (str, optional): Path to save the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n_metrics = len(mi_results)

    apply_publication_style(font_size=12, axes_linewidth=1.0, tick_major_width=1.0)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6, 8))
    if n_metrics == 1:
        axes = np.array([axes])

    # Find max MI for consistent scaling
    max_mi = 0
    for col, (metric, results) in enumerate(mi_results.items()):
        mi_scores = results["MI"]
        # Only consider MI > 0.01 for scaling
        filtered_mi = mi_scores[mi_scores > 0.01]
        if len(filtered_mi) > 0 and np.max(filtered_mi) > max_mi:
            max_mi = np.max(filtered_mi)

    # Get all correlation values for colorbar range
    all_corrs = []
    for results in mi_results.values():
        mi_scores = results["MI"]
        spearman_corr = results["spearman"]
        # Only include correlations for MI > 0.01
        mask = mi_scores > 0.00
        all_corrs.extend(spearman_corr[mask])

    # Create custom colormap: skyblue for negative, lightcoral for positive
    import matplotlib.colors as mcolors
    from matplotlib.colorbar import ColorbarBase

    colors = ["#486b45", "white", "#545aab"]
    n_bins = 100
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    # norm = mcolors.Normalize(vmin=min(all_corrs), vmax=max(all_corrs))
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)

    for col, (metric, results) in enumerate(mi_results.items()):
        names = np.array(results["names"])
        mi_scores = results["MI"]
        spearman_corr = results["spearman"]

        # Filter parameters with MI > 0.01
        mask = mi_scores > 0.00
        filtered_names = names  # [mask]
        filtered_mi = mi_scores  # [mask]
        filtered_corr = spearman_corr  # [mask]

        # Skip if no parameters meet the threshold
        if len(filtered_mi) == 0:
            continue

        ax = axes[col]

        # Sort parameters by importance (ascending order for horizontal bars)
        sorted_idx = np.argsort(filtered_mi)
        sorted_idx = sorted_idx[-6:]
        pos = np.arange(len(filtered_names[sorted_idx]))

        # Create horizontal bar plot
        bars = ax.barh(pos, filtered_mi[sorted_idx], height=0.9)

        # Color bars based on Spearman correlation using the colormap
        for idx, bar in enumerate(bars):
            corr = filtered_corr[sorted_idx[idx]]
            # Get color from colormap based on correlation value
            color = cmap(norm(corr))
            bar.set_color(color)
            bar.set_edgecolor("black")
            bar.set_linewidth(1.75)

        # Set parameter names on y-axis
        ax.set_yticks(pos)
        ax.set_yticklabels([filtered_names[i] for i in sorted_idx], fontsize=12)

        # Set x-axis
        ax.set_xlabel("Mutual Information Score", fontsize=12, fontweight="bold")
        ax.set_xlim(-0.01, 1.01 * max_mi)
        ax.set_axisbelow(True)

        # Style the axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)

        # Ensure ticks are visible and bold
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.5)

        # Make tick labels bold
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation="vertical")
    cbar.ax.tick_params(labelsize=12)
    for label in cbar.ax.get_yticklabels():
        label.set_weight("bold")
    # cbar.set_label('Spearman Correlation (ρ)', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Adjust layout to accommodate colorbar
    plt.subplots_adjust(right=0.9)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300, facecolor="white")
        plt.close()
    else:
        plt.show()


def align_dataframes(param_df, metrics_df):
    """
    Align parameter and metrics dataframes based on input folders.
    Ensures both DataFrames have matching rows in the same order.

    Args:
        param_df (pd.DataFrame): DataFrame containing parameters
        metrics_df (pd.DataFrame): DataFrame containing metrics

    Returns:
        tuple: (aligned_param_df, aligned_metrics_df)
    """
    if len(param_df) != len(metrics_df):
        print(
            f"Warning: param_df and metrics_df have different lengths: {len(param_df)} != {len(metrics_df)}"
        )

    # Get common input folders and sort them
    common_folders = sorted(set(param_df["input_folder"]) & set(metrics_df["input_folder"]))

    # Filter and sort both DataFrames by input_folder
    param_df = param_df[param_df["input_folder"].isin(common_folders)].copy()
    metrics_df = metrics_df[metrics_df["input_folder"].isin(common_folders)].copy()

    param_df = param_df.set_index("input_folder").loc[common_folders].reset_index()
    metrics_df = metrics_df.set_index("input_folder").loc[common_folders].reset_index()

    # Drop input_folder columns
    param_df = param_df.drop(columns=["input_folder"])
    metrics_df = metrics_df.drop(columns=["input_folder"])

    return param_df, metrics_df


def fit_rf_model(
    param_df, metrics_df, metric_name, n_estimators=100, random_state=42, plot_performance=False
):
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    param_df = (param_df - param_df.mean()) / param_df.std()
    metrics_df = (metrics_df - metrics_df.mean()) / metrics_df.std()
    if plot_performance:
        train_df = param_df.sample(frac=0.8, random_state=random_state)
        test_df = param_df[~param_df.index.isin(train_df.index)]
        train_y = metrics_df.loc[train_df.index]
        test_y = metrics_df.loc[test_df.index]
        model.fit(train_df, train_y)
        Y_pred_train = model.predict(train_df)
        Y_pred_test = model.predict(test_df)
        plt.scatter(train_y, Y_pred_train)
        plt.scatter(test_y, Y_pred_test)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.savefig("rf.png")
        plt.show()
        print(f"R^2 (train) {metric_name}: {r2_score(train_y, Y_pred_train)}")
        print(f"R^2 (test) {metric_name}: {r2_score(test_y, Y_pred_test)}")
    else:
        model.fit(param_df, metrics_df)
        print(f"R^2 (train) {metric_name}: {r2_score(metrics_df, model.predict(param_df))}")
    return model


def perform_sobol_analysis(
    param_df,
    metrics_df,
    parameter_names,
    calc_second_order=True,
    save_sobol_json=None,
    plot_performance=True,
):
    """
    Perform Sobol sensitivity analysis.
    """
    from SALib.analyze import sobol
    from SALib.sample import sobol as sobol_sampler

    problem = {
        "num_vars": len(parameter_names),
        "names": np.array(parameter_names),
        "bounds": [[param_df[param].min(), param_df[param].max()] for param in parameter_names],
    }

    # Increase base sample size for more reliable second-order indices
    N = 2048  # Increased from 1024

    param_values = sobol_sampler.sample(problem, N, calc_second_order=calc_second_order)
    param_values_df = pd.DataFrame(param_values, columns=parameter_names)
    param_values_df = (param_values_df - param_values_df.mean()) / param_values_df.std()

    results = {}
    for metric in metrics_df.columns:
        model = fit_rf_model(
            param_df, metrics_df[metric], metric, plot_performance=plot_performance
        )
        Y = model.predict(param_values_df)
        Y += np.random.normal(0, 1e-10, Y.shape)
        Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)

        if calc_second_order:
            results[metric] = {
                "S1": Si["S1"],
                "ST": Si["ST"],
                "S2": Si["S2"],
                "names": parameter_names,
            }
        else:
            results[metric] = {"S1": Si["S1"], "ST": Si["ST"], "names": parameter_names}

    if save_sobol_json:
        results_json = {}
        for metric, result in results.items():
            results_json[metric] = {
                "S1": result["S1"].tolist(),
                "ST": result["ST"].tolist(),
                "S2": result["S2"].tolist() if calc_second_order else None,
                "names": result["names"],
            }

        with open(save_sobol_json, "w") as f:
            json.dump(results_json, f)

    return results


def plot_sobol_analysis(sobol_results, metrics_name, calc_second_order=True, save_path=None):
    """
    Plot Sobol sensitivity analysis results with indices sorted in decreasing order.
    """

    # Extract and clean indices (replace negative values with 0)
    S1 = np.maximum(sobol_results[metrics_name]["S1"], 0)
    ST = np.maximum(sobol_results[metrics_name]["ST"], 0)
    names = sobol_results[metrics_name]["names"]

    # Determine number of subplots based on calc_second_order
    if calc_second_order:
        fig = plt.figure(figsize=(20, 10))
        gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1.2])
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Sort indices by total order (ST) for consistent ordering
    ST_sorted_idx = np.argsort(ST)[::1]  # Reverse to get descending order
    names_sorted = [names[i] for i in ST_sorted_idx]
    S1_sorted = S1[ST_sorted_idx]
    ST_sorted = ST[ST_sorted_idx]

    # Plot First-Order Indices
    bars_S1 = ax1.barh(range(len(names)), S1_sorted)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names_sorted)
    ax1.set_xlabel("First-Order Index")
    ax1.set_title(f"First-Order Sobol Indices\nfor {metrics_name}")

    for i, bar in enumerate(bars_S1):
        width = bar.get_width()
        ax1.text(width + 0.01, i, f"{width:.3f}", va="center", ha="left")

    # Plot Total-Order Indices
    bars_ST = ax2.barh(range(len(names)), ST_sorted)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names_sorted)
    ax2.set_xlabel("Total-Order Index")
    ax2.set_title(f"Total-Order Sobol Indices\nfor {metrics_name}")

    for i, bar in enumerate(bars_ST):
        width = bar.get_width()
        ax2.text(width + 0.01, i, f"{width:.3f}", va="center", ha="left")

    # Plot Second-Order Indices if requested
    if calc_second_order:
        S2 = np.maximum(sobol_results[metrics_name]["S2"], 0)
        lower_triangle_S2 = np.triu(S2, k=1).T
        n_params = len(names)
        mask = np.zeros((n_params, n_params))
        mask[np.triu_indices_from(mask, k=0)] = True  # Mask upper triangle, k=1 to show diagonal
        sns.heatmap(
            lower_triangle_S2,
            mask=mask,
            xticklabels=names,
            yticklabels=names,
            ax=ax3,
            cmap="YlOrRd",
            vmin=0,
            vmax=np.max(lower_triangle_S2) if np.max(lower_triangle_S2) > 0 else 1,
            annot=True,
            fmt=".3e",  # Use scientific notation
            annot_kws={"size": 8},  # Reduce font size if needed
            cbar_kws={"label": "Second-Order Index"},
        )

        ax3.set_title(f"Second-Order Sobol Indices\nfor {metrics_name}")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


def remove_nan_rows(param_df, metrics_df):
    """
    Remove rows with NaN values in metrics_df and corresponding rows in param_df.

    Args:
        param_df (pd.DataFrame): DataFrame containing parameters
        metrics_df (pd.DataFrame): DataFrame containing metrics

    Returns:
        tuple: (cleaned_param_df, cleaned_metrics_df)
    """
    # Find rows without NaN values in metrics_df
    valid_rows = ~metrics_df.isna().any(axis=1)

    # Apply the same filtering to both DataFrames
    cleaned_param_df = param_df[valid_rows].reset_index(drop=True)
    cleaned_metrics_df = metrics_df[valid_rows].reset_index(drop=True)

    return cleaned_param_df, cleaned_metrics_df


def main():
    parameter_base_folder = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast/iter_0"
    param_file = f"{parameter_base_folder}/all_param_df.csv"
    metrics_file = f"{parameter_base_folder}/final_metrics.csv"
    target_metrics = json.load(open(f"{parameter_base_folder}/targets.json"))
    target_metrics_list = ["doub_time", "colony_growth", "act_ratio", "symmetry"]
    for i in range(len(target_metrics_list)):
        # target_metrics += [f"{metric}_std" for metric in target_metrics]
        alignment_columns = ["input_folder"]

        param_df, metrics_df = load_data(
            param_file, metrics_file, target_metrics + alignment_columns
        )
        param_df, metrics_df = align_dataframes(param_df, metrics_df)
        metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
        param_df, metrics_df = remove_nan_rows(param_df, metrics_df)
        param_df = param_df.drop(columns=["X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"])
        param_names = param_df.columns.tolist()
        # param_df.to_csv(f"{parameter_base_folder}/param_df.csv", index=False)
        # metrics_df.to_csv(f"{parameter_base_folder}/metrics_df.csv", index=False)

        # Perform both types of sensitivity analysis
        save_mi_json = f"{parameter_base_folder}/mi_analysis_{target_metrics[0]}.json"
        if 1:
            mi_results = perform_mi_analysis(
                param_df, metrics_df, param_names, save_mi_json=save_mi_json
            )
            plot_mi_analysis(
                mi_results,
                save_path=f"{parameter_base_folder}/mi_analysis_median_{target_metrics[0]}.png",
            )
