from matplotlib.ticker import ScalarFormatter
from itertools import combinations
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import warnings
import json
from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.plotting.theme import apply_publication_style

warnings.filterwarnings("ignore")


def analyze_peak_correlations(
    posterior_data,
    final_metrics_df,
    n_components=2,
    n_best_samples=50,
    save_path=None,
    peak_names=None,
):
    """
    Peak analysis focusing on Ridge coefficients and individual metric parity plots.

    Args:
        posterior_data: DataFrame containing posterior samples
        final_metrics_df: DataFrame containing simulated metrics corresponding to each parameter sample
        n_components: Number of PCA components (default: 2)
        n_best_samples: Number of best samples around each peak (default: 50)
        save_path: Path to save analysis results (optional)
        peak_names: Custom names for peaks (optional)

    Returns:
        dict: Analysis results for each peak
    """

    apply_publication_style(font_size=12, axes_linewidth=1.5)

    param_names = posterior_data.columns.tolist()

    # Prepare data
    data = posterior_data[param_names].values

    # Perform PCA and find peaks
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        data, n_components
    )

    n_peaks = len(peak_positions)
    if peak_names is None:
        peak_names = [f"Peak {i+1}" for i in range(n_peaks)]

    # Color mapping for metrics
    colors = {
        "symmetry": "#486b45",
        "doub_time": "#bb883b",
        "act_ratio": "#af1b0a",
        "colony_growth": "#545aab",
        "symmetry_std": "#679A63",
        "doub_time_std": "#ddc39d",
        "colony_growth_std": "#a9acd5",
        "act_ratio_std": "#d78d85",
    }

    # Peak colors
    peak_colors = ["lightcoral", "skyblue", "gold", "lightgreen", "plum", "orange"]

    analysis_results = {}

    # Calculate number of individual metric plots needed
    n_metrics = len(final_metrics_df.columns)

    # Process data for each peak
    for peak_idx in range(n_peaks):
        peak_name = peak_names[peak_idx]
        print(f"\n=== Analyzing {peak_name} ===")

        # Get samples around this peak
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[peak_idx]) ** 2, axis=1))
        closest_indices = np.argsort(distances)[:n_best_samples]
        peak_data = posterior_data.iloc[closest_indices][param_names]

        # Get corresponding metrics for these samples
        peak_metrics = final_metrics_df.iloc[closest_indices]

        # Initialize results for this peak
        peak_results = {
            "n_samples": len(closest_indices),
            "target_predictions": {},
            "parity_data": {},
        }

        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(peak_data)
        scaled_df = pd.DataFrame(scaled_data, columns=param_names)

        # LINEAR MODEL COEFFICIENTS FOR METRICS PREDICTION
        print(f"Computing Ridge regression for metrics prediction...")
        X_scaled = scaled_df
        y_metrics = peak_metrics.values

        # Store parity plot data
        parity_data = {}

        # Fit Ridge model
        model = Ridge(alpha=1.0)
        parity_data["Ridge"] = {}
        linear_results = {}

        try:
            # Fit model for each metric separately
            all_coeffs = []
            all_r2 = []

            for metric_idx, metric_name in enumerate(peak_metrics.columns):
                y_single = y_metrics[:, metric_idx]
                model_single = Ridge(alpha=1.0)
                model_single.fit(X_scaled, y_single)
                y_pred_single = model_single.predict(X_scaled)

                all_coeffs.append(model_single.coef_)
                r2_single = r2_score(y_single, y_pred_single)
                all_r2.append(r2_single)

                # Store individual metric parity data
                parity_data["Ridge"][metric_name] = {
                    "y_true": y_single,
                    "y_pred": y_pred_single,
                    "r2": r2_single,
                    "metric_name": metric_name,
                    "coefficients": dict(zip(param_names, model_single.coef_)),
                }

            coeffs = np.mean(all_coeffs, axis=0)  # Average coefficients across metrics
            r2 = np.mean(all_r2)  # Average R² across metrics

            linear_results["Ridge"] = {
                "coefficients": dict(zip(param_names, coeffs)),
                "r2_score": r2,
                "individual_metrics": {},
            }

            # Store individual metric results
            for metric_idx, metric_name in enumerate(peak_metrics.columns):
                y_single = y_metrics[:, metric_idx]
                model_single = Ridge(alpha=1.0)
                model_single.fit(X_scaled, y_single)
                linear_results["Ridge"]["individual_metrics"][metric_name] = {
                    "coefficients": dict(zip(param_names, model_single.coef_)),
                    "r2_score": r2_score(y_single, model_single.predict(X_scaled)),
                }

        except Exception as e:
            print(f"Warning: Ridge model failed: {e}")
            linear_results["Ridge"] = {
                "coefficients": {p: 0 for p in param_names},
                "r2_score": None,
                "individual_metrics": {},
            }
            parity_data["Ridge"] = {}

        peak_results["target_predictions"] = linear_results
        peak_results["parity_data"] = parity_data

        # Store results
        analysis_results[peak_name] = peak_results

        # Print summary for this peak
        print(f"  - Metric prediction performance (Ridge R²):")
        if "Ridge" in parity_data:
            ridge_parity = parity_data["Ridge"]
            for metric_name, data in ridge_parity.items():
                print(f"    {metric_name}: R² = {data['r2']:.3f}")

    # Create separate figures for Ridge coefficients and parity plots
    create_ridge_coefficient_plots(analysis_results, peak_names, save_path)
    create_parity_plots(analysis_results, peak_names, colors, save_path)

    return analysis_results


def create_ridge_coefficient_plots(analysis_results, peak_names, save_path):
    """Create Ridge coefficient plots in a separate figure."""
    n_peaks = len(peak_names)

    # Set up plotting for Ridge coefficients
    fig, axes = plt.subplots(n_peaks, 1, figsize=(6, 3 * n_peaks))
    if n_peaks == 1:
        axes = [axes]

    for peak_idx, peak_name in enumerate(peak_names):
        peak_results = analysis_results[peak_name]
        linear_results = peak_results["target_predictions"]

        # Plot Ridge coefficients (flipped x and y)
        ax_coeff = axes[peak_idx]
        ridge_coeffs = linear_results["Ridge"]["coefficients"]
        params_sorted = sorted(
            ridge_coeffs.keys(), key=lambda x: abs(ridge_coeffs[x]), reverse=True
        )
        coeffs_sorted = [ridge_coeffs[p] for p in params_sorted]

        colors_bars = ["red" if c < 0 else "blue" for c in coeffs_sorted]
        bars = ax_coeff.barh(
            range(len(params_sorted)),
            coeffs_sorted,
            color=colors_bars,
            alpha=0.7,
            edgecolor="black",
        )
        ax_coeff.set_yticks(range(len(params_sorted)))
        ax_coeff.set_yticklabels(params_sorted, fontsize=10)
        ax_coeff.axvline(x=0, color="black", linestyle="-", alpha=0.5)

        # Set consistent x limits
        ax_coeff.set_xlim(-1, 1)
        ax_coeff.tick_params(axis="both", which="major", labelsize=10, width=1.5)
        for label_tick in ax_coeff.get_xticklabels():
            label_tick.set_fontweight("bold")
        for label_tick in ax_coeff.get_yticklabels():
            label_tick.set_fontweight("bold")

    plt.tight_layout()
    if save_path:
        ridge_save_path = save_path.replace(".png", "_ridge_coefficients.png")
        plt.savefig(ridge_save_path, dpi=300, bbox_inches="tight")
        print(f"Ridge coefficient plot saved to: {ridge_save_path}")
    plt.show()


def create_parity_plots(analysis_results, peak_names, colors, save_path):
    """Create parity plots in a separate figure."""
    n_peaks = len(peak_names)

    # Set up plotting for parity plots
    fig, axes = plt.subplots(n_peaks, 1, figsize=(5, 3 * n_peaks))
    if n_peaks == 1:
        axes = [axes]

    for peak_idx, peak_name in enumerate(peak_names):
        peak_results = analysis_results[peak_name]
        parity_data = peak_results["parity_data"]

        # Combined metric parity plot with subplots
        ax_parity = axes[peak_idx]
        ridge_parity = parity_data.get("Ridge", {})

        if ridge_parity:
            # Calculate subplot layout for metrics within the parity plot
            n_metrics_actual = len(ridge_parity)
            n_cols = min(3, n_metrics_actual)  # Max 3 columns
            n_rows = int(np.ceil(n_metrics_actual / n_cols))

            # Remove the main axis ticks and labels
            ax_parity.set_xticks([])
            ax_parity.set_yticks([])

            # Style the axes
            ax_parity.spines["top"].set_visible(False)
            ax_parity.spines["right"].set_visible(False)
            ax_parity.spines["left"].set_visible(False)
            ax_parity.spines["bottom"].set_visible(False)

            # Create sub-subplots within the parity plot area
            for metric_idx, (metric_name, data) in enumerate(ridge_parity.items()):
                # Calculate position for this sub-subplot
                row = metric_idx // n_cols
                col = metric_idx % n_cols

                # Calculate the position within the main axis with padding to avoid overlap
                padding = 0.1  # Add padding between subplots
                left = col / n_cols + padding / 2
                bottom = 1 - (row + 1) / n_rows + padding / 2
                width = 1 / n_cols - padding
                height = 1 / n_rows - padding

                # Create inset axes (sub-subplot)
                sub_ax = ax_parity.inset_axes([left, bottom, width, height])

                # Get color for this metric
                metric_color = colors.get(metric_name, "#666666")

                # Add scatter plot
                sub_ax.scatter(
                    data["y_true"],
                    data["y_pred"],
                    alpha=0.7,
                    s=20,
                    color=metric_color,
                    facecolor="none",
                )

                # Add perfect prediction line
                if len(data["y_true"]) > 0:
                    min_val = min(np.min(data["y_true"]), np.min(data["y_pred"]))
                    max_val = max(np.max(data["y_true"]), np.max(data["y_pred"]))
                    min_val -= min_val * 0.1
                    max_val += max_val * 0.1
                    sub_ax.plot(
                        [min_val, max_val], [min_val, max_val], "k--", alpha=0.8, linewidth=1.5
                    )

                    # Set axis limits
                    sub_ax.set_xlim(min_val, max_val)
                    sub_ax.set_ylim(min_val, max_val)
                    # set the maximum decimal places to 2
                    sub_ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
                    sub_ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                    sub_ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

                # Make tick labels bold and thicker
                sub_ax.spines["top"].set_linewidth(1.0)
                sub_ax.spines["right"].set_linewidth(1.0)
                sub_ax.spines["left"].set_linewidth(1.0)
                sub_ax.spines["bottom"].set_linewidth(1.0)
                sub_ax.tick_params(axis="both", which="major", labelsize=9, width=1.5)
                for label_tick in sub_ax.get_xticklabels():
                    label_tick.set_fontweight("bold")
                for label_tick in sub_ax.get_yticklabels():
                    label_tick.set_fontweight("bold")
                sub_ax.grid(True, alpha=0.3)

                # Add R2 value to the plot
                r2_value = data["r2"]
                sub_ax.text(
                    0.05,
                    0.95,
                    f"R² = {r2_value:.2f}",
                    transform=sub_ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )

                # Make the plot square
                sub_ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    if save_path:
        parity_save_path = save_path.replace(".png", "_parity_plots.png")
        plt.savefig(parity_save_path, dpi=300, bbox_inches="tight")
        print(f"Parity plot saved to: {parity_save_path}")
    plt.show()


def calculate_pairwise_r2_and_plot(df, param_names, save_path=None):
    """
    Calculate pairwise R² values among all parameters and plot scatter plots
    for the top 9 highest R² pairs in a 3x3 grid.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing parameter values
    param_names : list
        List of parameter names to analyze
    title_prefix : str, optional
        Prefix for the plot title (e.g., "Prior" or "Posterior")

    Returns:
    --------
    r2_results : pandas.DataFrame
        DataFrame with columns ['param1', 'param2', 'r2'] sorted by R² values
    """

    # Calculate pairwise R² values
    r2_results = []

    for param1, param2 in combinations(param_names, 2):
        # Get non-null values for both parameters
        mask = df[param1].notna() & df[param2].notna()
        x = df.loc[mask, param1].values
        y = df.loc[mask, param2].values

        if len(x) > 1:  # Need at least 2 points for R²
            # Fit linear regression
            lr = LinearRegression()
            lr.fit(x.reshape(-1, 1), y)
            y_pred = lr.predict(x.reshape(-1, 1))

            # Calculate R²
            r2 = r2_score(y, y_pred)
            r2_results.append({"param1": param1, "param2": param2, "r2": r2})

    r2_df = pd.DataFrame(r2_results)
    r2_df = r2_df.sort_values("r2", ascending=False).reset_index(drop=True)

    print("=" * 50)
    for i in range(min(10, len(r2_df))):
        row = r2_df.iloc[i]
        print(f"{i+1:2d}. {row['param1']:20s} vs {row['param2']:20s} | R² = {row['r2']:.4f}")

    # Create 3x3 scatter plot grid for top 9 pairs
    fig, axes = plt.subplots(3, 3, figsize=(7, 7))

    for i in range(9):
        row = i // 3
        col = i % 3

        if i < len(r2_df):
            param1 = r2_df.iloc[i]["param1"]
            param2 = r2_df.iloc[i]["param2"]
            r2_val = r2_df.iloc[i]["r2"]

            # Get data for plotting
            mask = df[param1].notna() & df[param2].notna()
            x = df.loc[mask, param1]
            y = df.loc[mask, param2]

            # Create scatter plot
            axes[row, col].scatter(x, y, alpha=0.4, s=10, color="k", facecolor="none")

            # Add regression line
            if len(x) > 1:
                lr = LinearRegression()
                lr.fit(x.values.reshape(-1, 1), y.values)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = lr.predict(x_line.reshape(-1, 1))
                axes[row, col].plot(x_line, y_line, "k--", alpha=0.8)

            # Set labels and title
            axes[row, col].set_xlabel(param1, fontsize=10)
            axes[row, col].set_ylabel(param2, fontsize=10)
            axes[row, col].set_title(f"R² = {r2_val:.4f}", fontsize=10)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].text(
                0.05,
                0.95,
                f"R² = {r2_val:.4f}",
                transform=axes[row, col].transAxes,
                fontsize=9,
                fontweight="bold",
                ha="left",
                va="top",
            )
            for label_tick in axes[row, col].get_xticklabels():
                label_tick.set_fontweight("bold")
            for label_tick in axes[row, col].get_yticklabels():
                label_tick.set_fontweight("bold")

            axes[row, col].set_box_aspect(1)
        else:
            # Hide empty subplots
            axes[row, col].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()
    return r2_df


def analyze_peak_pairwise_correlations(
    posterior_data,
    n_components=2,
    n_best_samples=50,
    top_n_pairs=10,
    min_r2_threshold=0.1,
    peak_names=None,
    verbose=False,
):
    """
    Analyze peaks and find parameter pairs with high R² correlations for each peak.

    Args:
        posterior_data: DataFrame containing posterior samples
        n_components: Number of PCA components (default: 2)
        n_best_samples: Number of best samples around each peak (default: 50)
        top_n_pairs: Number of top R² pairs to analyze per peak (default: 10)
        min_r2_threshold: Minimum R² threshold for including pairs (default: 0.1)
        peak_names: Custom names for peaks (optional)

    Returns:
        dict: Analysis results containing peak data and parameter prediction models
    """

    param_names = posterior_data.columns.tolist()

    # Prepare data and perform PCA to find peaks
    data = posterior_data[param_names].values
    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        data, n_components
    )

    n_peaks = len(peak_positions)
    if peak_names is None:
        peak_names = [f"Peak {i+1}" for i in range(n_peaks)]

    analysis_results = {}

    # Process each peak
    for peak_idx in range(n_peaks):
        peak_name = peak_names[peak_idx]
        if verbose:
            print(f"\n=== Analyzing {peak_name} ===")

        # Get samples around this peak
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[peak_idx]) ** 2, axis=1))
        closest_indices = np.argsort(distances)[:n_best_samples]
        peak_data = posterior_data.iloc[closest_indices][param_names]

        # Calculate pairwise R² values for this peak
        r2_results = []
        regression_models = {}

        for param1, param2 in combinations(param_names, 2):
            # Get non-null values for both parameters
            mask = peak_data[param1].notna() & peak_data[param2].notna()
            x = peak_data.loc[mask, param1].values
            y = peak_data.loc[mask, param2].values

            if len(x) > 1:  # Need at least 2 points for R²
                # Fit linear regression (param1 -> param2)
                lr_12 = LinearRegression()
                lr_12.fit(x.reshape(-1, 1), y)
                y_pred = lr_12.predict(x.reshape(-1, 1))
                r2_12 = r2_score(y, y_pred)

                # Fit linear regression (param2 -> param1)
                lr_21 = LinearRegression()
                lr_21.fit(y.reshape(-1, 1), x)
                x_pred = lr_21.predict(y.reshape(-1, 1))
                r2_21 = r2_score(x, x_pred)

                # Use the average R² for ranking
                avg_r2 = (r2_12 + r2_21) / 2

                if avg_r2 >= min_r2_threshold:
                    r2_results.append(
                        {
                            "param1": param1,
                            "param2": param2,
                            "r2": avg_r2,
                            "r2_12": r2_12,  # param1 -> param2
                            "r2_21": r2_21,  # param2 -> param1
                        }
                    )

                    # Store both regression models
                    regression_models[f"{param1}_to_{param2}"] = {
                        "model": lr_12,
                        "r2": r2_12,
                        "coef": lr_12.coef_[0],
                        "intercept": lr_12.intercept_,
                        "input_param": param1,
                        "output_param": param2,
                    }

                    regression_models[f"{param2}_to_{param1}"] = {
                        "model": lr_21,
                        "r2": r2_21,
                        "coef": lr_21.coef_[0],
                        "intercept": lr_21.intercept_,
                        "input_param": param2,
                        "output_param": param1,
                    }

        # Sort by R² values
        r2_df = pd.DataFrame(r2_results)

        if not r2_df.empty:
            r2_df = r2_df.sort_values("r2", ascending=False).reset_index(drop=True)
            if verbose:
                print(f"Top {min(top_n_pairs, len(r2_df))} parameter pairs with highest R²:")
                for i in range(min(top_n_pairs, len(r2_df))):
                    row = r2_df.iloc[i]
                    print(
                        f"{i+1:2d}. {row['param1']:20s} vs {row['param2']:20s} | R² = {row['r2']:.4f}"
                    )

        # Create parameter predictor function
        def create_parameter_predictor(models):
            def predict_parameter(input_param, input_value, output_param):
                """
                Predict output parameter value given input parameter and its value.

                Args:
                    input_param: Name of input parameter
                    input_value: Value of input parameter
                    output_param: Name of output parameter to predict

                Returns:
                    dict: Prediction result with value, confidence (R²), and model info
                """
                model_key = f"{input_param}_to_{output_param}"

                if model_key not in models:
                    return {
                        "predicted_value": None,
                        "r2_score": None,
                        "error": f"No model available for {input_param} -> {output_param}",
                    }

                model_info = models[model_key]
                predicted_value = model_info["model"].predict([[input_value]])[0]

                return {
                    "predicted_value": predicted_value,
                    "r2_score": model_info["r2"],
                    "coefficient": model_info["coef"],
                    "intercept": model_info["intercept"],
                    "equation": f"{output_param} = {model_info['coef']:.4f} * {input_param} + {model_info['intercept']:.4f}",
                }

            return predict_parameter

        # Store results for this peak
        analysis_results[peak_name] = {
            "peak_position": peak_positions[peak_idx],
            "n_samples": len(closest_indices),
            "sample_indices": closest_indices,
            "peak_data": peak_data,
            "r2_pairs": r2_df if not r2_df.empty else pd.DataFrame(),
            "regression_models": regression_models,
            "parameter_predictor": create_parameter_predictor(regression_models),
            "available_predictions": list(regression_models.keys()),
        }

    return analysis_results


def get_parameter_prediction_summary(analysis_results):
    """
    Get a summary of all available parameter predictions across peaks.

    Args:
        analysis_results: Results from analyze_peak_pairwise_correlations

    Returns:
        dict: Summary of available predictions for each peak
    """
    summary = {}

    for peak_name, peak_data in analysis_results.items():
        summary[peak_name] = {"available_pairs": [], "prediction_matrix": {}}

        # Extract available parameter pairs
        for model_key in peak_data["available_predictions"]:
            input_param, output_param = model_key.split("_to_")
            pair_info = {
                "input_param": input_param,
                "output_param": output_param,
                "r2_score": peak_data["regression_models"][model_key]["r2"],
                "equation": f"{output_param} = {peak_data['regression_models'][model_key]['coef']:.4f} * {input_param} + {peak_data['regression_models'][model_key]['intercept']:.4f}",
            }
            summary[peak_name]["available_pairs"].append(pair_info)

            # Create prediction matrix
            if input_param not in summary[peak_name]["prediction_matrix"]:
                summary[peak_name]["prediction_matrix"][input_param] = {}
            summary[peak_name]["prediction_matrix"][input_param][output_param] = {
                "r2": peak_data["regression_models"][model_key]["r2"],
                "coef": peak_data["regression_models"][model_key]["coef"],
                "intercept": peak_data["regression_models"][model_key]["intercept"],
            }

    return summary


def plot_peak_pairwise_correlations(analysis_results, save_path=None, figsize=(10, 10)):
    """
    Plot the top 9 pairwise parameter correlations for each peak in a 3x3 grid.

    Args:
        analysis_results: Results from analyze_peak_pairwise_correlations
        save_path: Base path to save plots (optional). Will append peak names to filename
        figsize: Figure size for each peak plot (default: (15, 10))

    Returns:
        None (displays/saves plots)
    """

    apply_publication_style(font_size=12, axes_linewidth=1.5)

    n_peaks = len(analysis_results)

    for peak_name, peak_data in analysis_results.items():
        print(f"\nCreating pairwise correlation plot for {peak_name}...")

        r2_df = peak_data["r2_pairs"]
        peak_param_data = peak_data["peak_data"]

        if r2_df.empty:
            print(f"No significant correlations found for {peak_name}. Skipping plot.")
            continue

        # Create 3x3 subplot grid
        fig, axes = plt.subplots(3, 3, figsize=figsize)

        # Plot top 9 pairs
        for i in range(9):
            row = i // 3
            col = i % 3

            if i < len(r2_df):
                # Get parameter pair info
                pair_info = r2_df.iloc[i]
                param1 = pair_info["param1"]
                param2 = pair_info["param2"]
                r2_val = pair_info["r2"]

                # Get data for plotting
                mask = peak_param_data[param1].notna() & peak_param_data[param2].notna()
                x = peak_param_data.loc[mask, param1]
                y = peak_param_data.loc[mask, param2]

                # Create scatter plot
                axes[row, col].scatter(
                    x, y, s=25, color="k", facecolor="none", edgecolor="k", linewidth=1.0
                )

                # Add regression line
                if len(x) > 1:
                    from sklearn.linear_model import LinearRegression

                    lr = LinearRegression()
                    lr.fit(x.values.reshape(-1, 1), y.values)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = lr.predict(x_line.reshape(-1, 1))
                    axes[row, col].plot(x_line, y_line, "k--", linewidth=1, alpha=0.8)

                # Customize plot
                axes[row, col].set_xlabel(param1, fontsize=11, fontweight="bold")
                axes[row, col].set_ylabel(param2, fontsize=11, fontweight="bold")
                axes[row, col].grid(True, alpha=0.3)

                # Add R² text box
                axes[row, col].text(
                    0.05,
                    0.95,
                    f"R² = {r2_val:.2f}",
                    transform=axes[row, col].transAxes,
                    fontsize=10,
                    fontweight="bold",
                    ha="left",
                    va="top",
                )

                # Bold tick labels
                for label in axes[row, col].get_xticklabels() + axes[row, col].get_yticklabels():
                    label.set_fontweight("bold")

                # Set aspect ratio
                axes[row, col].set_box_aspect(1)

            else:
                # Hide empty subplots
                axes[row, col].set_visible(False)

        plt.tight_layout()

        # Save or show plot
        if save_path:
            peak_save_path = f"{save_path}_{peak_name.replace(' ', '_')}_pairwise_correlations.png"
            plt.savefig(peak_save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {peak_save_path}")
        else:
            plt.show()

        plt.close()


def create_correlation_heatmap(analysis_results, save_path=None, figsize=(12, 8)):
    """
    Create a heatmap showing R² values between all parameter pairs for each peak.

    Args:
        analysis_results: Results from analyze_peak_pairwise_correlations
        save_path: Base path to save heatmaps (optional)
        figsize: Figure size for each heatmap (default: (12, 8))

    Returns:
        None (displays/saves plots)
    """

    for peak_name, peak_data in analysis_results.items():
        print(f"\nCreating correlation heatmap for {peak_name}...")

        r2_df = peak_data["r2_pairs"]
        peak_param_data = peak_data["peak_data"]

        if r2_df.empty:
            print(f"No significant correlations found for {peak_name}. Skipping heatmap.")
            continue

        # Get all parameter names
        param_names = peak_param_data.columns.tolist()
        n_params = len(param_names)

        # Create correlation matrix
        correlation_matrix = np.zeros((n_params, n_params))

        # Fill diagonal with 1s (perfect correlation with self)
        np.fill_diagonal(correlation_matrix, 1.0)

        # Fill matrix with R² values
        for _, row in r2_df.iterrows():
            param1_idx = param_names.index(row["param1"])
            param2_idx = param_names.index(row["param2"])

            # Use the average R² for symmetry
            correlation_matrix[param1_idx, param2_idx] = row["r2"]
            correlation_matrix[param2_idx, param1_idx] = row["r2"]

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        import seaborn as sns

        sns.heatmap(
            correlation_matrix,
            xticklabels=param_names,
            yticklabels=param_names,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            center=0.5,
            square=True,
            cbar_kws={"label": "R² Score"},
            ax=ax,
        )

        ax.set_title(f"{peak_name} - Parameter Correlation Heatmap", fontsize=16, fontweight="bold")

        # Rotate labels for better readability
        plt.xticks(rotation=45, ha="right", fontweight="bold")
        plt.yticks(rotation=0, fontweight="bold")

        plt.tight_layout()

        if save_path:
            heatmap_save_path = f"{save_path}_{peak_name.replace(' ', '_')}_correlation_heatmap.png"
            plt.savefig(heatmap_save_path, dpi=300, bbox_inches="tight")
            print(f"Heatmap saved to: {heatmap_save_path}")
        else:
            plt.show()

        plt.close()


def analyze_parameter_chains(analysis_results, min_r2_threshold=0.6, verbose=False):
    """
    FIXED VERSION: Properly handles parameter classification without overlaps.
    """

    chain_analysis = {}

    for peak_name, peak_data in analysis_results.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"PARAMETER CHAIN ANALYSIS FOR {peak_name}")
            print(f"{'='*60}")

        r2_df = peak_data["r2_pairs"]

        # Get ALL parameters from the peak data
        peak_data_df = peak_data["peak_data"]
        all_params = set(peak_data_df.columns.tolist())

        if verbose:
            print(f"\nTotal parameters in peak: {len(all_params)}")

        if r2_df.empty:
            if verbose:
                print(f"No significant correlations found for {peak_name}")
            chain_analysis[peak_name] = create_empty_analysis(all_params)
            continue

        # Filter strong relationships
        strong_relationships = r2_df[r2_df["r2"] >= min_r2_threshold].copy()

        if strong_relationships.empty:
            if verbose:
                print(f"No relationships above R² = {min_r2_threshold} threshold")
            chain_analysis[peak_name] = create_empty_analysis(all_params)
            continue

        # Build parameter network
        param_network = defaultdict(list)
        params_with_relationships = set()

        for _, row in strong_relationships.iterrows():
            param1, param2 = row["param1"], row["param2"]
            r2_val = row["r2"]

            params_with_relationships.add(param1)
            params_with_relationships.add(param2)

            param_network[param1].append({"target": param2, "r2": r2_val})
            param_network[param2].append({"target": param1, "r2": r2_val})

        # Find chains using improved algorithm
        chains = find_optimal_chains(param_network, strong_relationships)

        # Classify parameters WITHOUT overlaps
        classification = classify_parameters_without_overlap(
            all_params, params_with_relationships, chains
        )

        if verbose:
            print_classification_results(classification, chains, strong_relationships)

        # Store results
        chain_analysis[peak_name] = {
            "strong_relationships": strong_relationships,
            "parameter_network": dict(param_network),
            "chains": chains,
            "sampling_strategy": classification,
            "total_params": len(all_params),
            "independent_count": len(classification["independent_params"]),
            "chain_count": len(chains),
            "derived_count": len(classification["derived_params"]),
            "peak_data": peak_data_df,  # Store for diagnostic
        }

    return chain_analysis


def find_optimal_chains(param_network, strong_relationships):
    """
    Find optimal parameter chains avoiding conflicts.
    """
    visited_global = set()
    chains = []

    # Sort parameters by number of connections (prefer highly connected as roots)
    param_connectivity = [(param, len(connections)) for param, connections in param_network.items()]
    param_connectivity.sort(key=lambda x: x[1], reverse=True)

    for param, _ in param_connectivity:
        if param not in visited_global:
            chain = build_chain_from_root(
                param, param_network, visited_global, strong_relationships
            )
            if len(chain) > 1:
                chains.append(
                    {
                        "sequence": chain,
                        "r2_values": get_chain_r2_values(chain, strong_relationships),
                        "root": chain[0],
                        "derived": chain[1:],
                    }
                )
                visited_global.update(chain)

    return chains


def build_chain_from_root(root, param_network, visited_global, strong_relationships):
    """
    Build a single chain from a root parameter.
    """
    chain = [root]
    current = root
    visited_in_chain = {root}

    while True:
        if current not in param_network:
            break

        # Find best unvisited connection
        candidates = [
            conn
            for conn in param_network[current]
            if conn["target"] not in visited_global and conn["target"] not in visited_in_chain
        ]

        if not candidates:
            break

        # Choose connection with highest R²
        best_connection = max(candidates, key=lambda x: x["r2"])
        next_param = best_connection["target"]

        chain.append(next_param)
        visited_in_chain.add(next_param)
        current = next_param

    return chain


def get_chain_r2_values(chain, strong_relationships):
    """
    Get R² values for chain links.
    """
    r2_values = []
    for i in range(len(chain) - 1):
        param1, param2 = chain[i], chain[i + 1]
        mask = (
            (strong_relationships["param1"] == param1) & (strong_relationships["param2"] == param2)
        ) | (
            (strong_relationships["param1"] == param2) & (strong_relationships["param2"] == param1)
        )
        if mask.any():
            r2_values.append(strong_relationships[mask]["r2"].iloc[0])
    return r2_values


def classify_parameters_without_overlap(all_params, params_with_relationships, chains):
    """
    Classify parameters ensuring no overlaps between categories.
    """
    # Start with all parameters
    unclassified = all_params.copy()

    # Extract chain information
    chain_roots = set()
    derived_params = set()

    for chain_info in chains:
        root = chain_info["root"]
        derived = set(chain_info["derived"])

        chain_roots.add(root)
        derived_params.update(derived)

        # Remove from unclassified
        unclassified.discard(root)
        unclassified -= derived

    # Remaining parameters are independent
    independent_params = unclassified

    # Double-check: ensure no overlaps
    assert len(chain_roots & derived_params) == 0, "Chain roots and derived overlap!"
    assert len(chain_roots & independent_params) == 0, "Chain roots and independent overlap!"
    assert len(derived_params & independent_params) == 0, "Derived and independent overlap!"

    # Verify we have all parameters
    total_classified = len(chain_roots) + len(derived_params) + len(independent_params)
    assert total_classified == len(
        all_params
    ), f"Missing parameters: {total_classified} != {len(all_params)}"

    return {
        "independent_params": list(independent_params),
        "chain_roots": list(chain_roots),
        "derived_params": list(derived_params),
        "chains": chains,
    }


def create_empty_analysis(all_params):
    """
    Create analysis result when no chains are found.
    """
    return {
        "strong_relationships": pd.DataFrame(),
        "parameter_network": {},
        "chains": [],
        "sampling_strategy": {
            "independent_params": list(all_params),
            "chain_roots": [],
            "derived_params": [],
            "chains": [],
        },
        "total_params": len(all_params),
        "independent_count": len(all_params),
        "chain_count": 0,
        "derived_count": 0,
    }


def print_classification_results(classification, chains, strong_relationships):
    """
    Print classification results with verification.
    """
    print(f"\n{'─'*60}")
    print("SAMPLING STRATEGY:")
    print(f"{'─'*60}")

    independent = classification["independent_params"]
    chain_roots = classification["chain_roots"]
    derived = classification["derived_params"]

    print(f"\n🎯 PARAMETERS TO SAMPLE: {len(independent) + len(chain_roots)}")
    if independent:
        print(f"   Independent parameters (len: {len(independent)}):")
        for param in sorted(independent):
            print(f"     • {param}")

    if chain_roots:
        print(f"   Chain root parameters (len: {len(chain_roots)}):")
        for param in sorted(chain_roots):
            print(f"     • {param}")

    print(f"\n🔗 PARAMETERS TO DERIVE: {len(derived)}")
    for param in sorted(derived):
        print(f"  • {param}")

    print(f"\n📋 DERIVATION SEQUENCE:")
    for i, chain_info in enumerate(chains):
        print(f"  Chain {i+1}: {' → '.join(chain_info['sequence'])}")
        r2_values = chain_info["r2_values"]
        for j in range(len(chain_info["sequence"]) - 1):
            param1, param2 = chain_info["sequence"][j], chain_info["sequence"][j + 1]
            r2_val = r2_values[j] if j < len(r2_values) else "N/A"
            print(f"    {param1} → {param2}: R² = {r2_val:.3f}")

    # Verification summary
    total_params = len(independent) + len(chain_roots) + len(derived)
    to_sample = len(independent) + len(chain_roots)
    reduction = len(derived) / total_params if total_params > 0 else 0

    print(f"\n📊 EFFICIENCY SUMMARY:")
    print(f"  Total parameters: {total_params}")
    print(f"  Parameters to sample: {to_sample}")
    print(f"  Parameters to derive: {len(derived)}")
    print(f"  Sampling reduction: {reduction:.1%}")

    # Verification
    print(f"\n✅ VERIFICATION:")
    print(f"  No overlaps: ✓")
    print(f"  All parameters classified: ✓")
    print(
        f"  Math checks out: {total_params} = {len(independent)} + {len(chain_roots)} + {len(derived)} ✓"
    )


def plot_lr_parameter_correlations(lr_summary_path, peak_data, save_path=None):
    """
    Plot scatter plots for parameter correlations using linear regression summary data.

    Args:
        lr_summary_path: Path to the JSON file containing linear regression predictions
        peak_data: Dictionary containing the actual parameter data for plotting
        save_path: Optional path to save the plot
    """

    # Load the linear regression summary
    with open(lr_summary_path, "r") as f:
        lr_data = json.load(f)

    models = lr_data["prediction_models"]["models"]

    if not models:
        print("No prediction models found in the summary")
        return

    # Calculate grid dimensions
    n_models = len(models)
    n_cols = min(4, n_models)  # Maximum 4 columns
    n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Handle single subplot case
    if n_models == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes] if n_cols > 1 else [[axes]]

    # Flatten axes array for easier indexing
    if n_rows > 1:
        axes_flat = axes.flatten()
    else:
        axes_flat = axes[0] if isinstance(axes[0], list) else axes
    # Create scatter plots for each model
    for i, model in enumerate(models):
        ax = axes_flat[i]

        # Extract parameter names
        predictor_name = model["predictor_parameter"]["name"]
        target_name = model["target_parameter"]["name"]

        # Check if data exists for both parameters
        if predictor_name not in peak_data or target_name not in peak_data:
            print(f"Warning: Data not found for {predictor_name} or {target_name}")
            ax.set_visible(False)
            continue

        # Get data for plotting
        x_data = peak_data[predictor_name]
        y_data = peak_data[target_name]

        # Scatter plot
        ax.scatter(x_data, y_data, s=25, color="k", facecolor="none", edgecolor="k", linewidth=1.0)

        # Add linear regression line using the model coefficients
        slope = model["linear_model"]["slope"]
        intercept = model["linear_model"]["intercept"]
        r_squared = model["prediction_quality"]["r_squared"]

        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, "k--", alpha=0.8, linewidth=1.5)

        # Format parameter names and split if too long
        xlabel = predictor_name.replace("_MU", " ").replace("_", " ").capitalize()
        if len(xlabel) > 14:
            words = xlabel.split()
            mid = len(words) // 2
            xlabel = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

        ylabel = target_name.replace("_MU", " ").replace("_", " ").capitalize()
        if len(ylabel) > 14:
            words = ylabel.split()
            mid = len(words) // 2
            ylabel = " ".join(words[:mid]) + "\n" + " ".join(words[mid:])

        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(1)

        # Set scientific notation for both axes
        ax.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))

        # Add R² text box
        r2_text = f"R² = {r_squared:.3f}"
        ax.text(
            0.05,
            0.95,
            r2_text,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
        )

        # Make tick labels bold
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    # Hide unused subplots
    for i in range(n_models, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", transparent=True)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


def plot_parameter_hierarchy(lr_summary_path, save_path=None):
    """
    Plot the parameter hierarchy showing root -> derived relationships.

    Args:
        lr_summary_path: Path to the JSON file containing linear regression predictions
        save_path: Optional path to save the plot
    """

    # Load the linear regression summary
    with open(lr_summary_path, "r") as f:
        lr_data = json.load(f)

    categorization = lr_data["parameter_categorization"]
    models = lr_data["prediction_models"]["models"]

    # Extract parameter information
    independent = [
        p["parameter_name"] for p in categorization["independent_parameters"]["parameters"]
    ]
    root = [p["parameter_name"] for p in categorization["root_parameters"]["parameters"]]

    print("Parameter Hierarchy Summary:")
    print("=" * 50)

    print(f"\nIndependent Parameters ({len(independent)}):")
    for param in independent:
        print(f"  • {param.replace('_MU', '').replace('_', ' ').title()}")

    print(f"\nRoot Parameters ({len(root)}):")
    for param in root:
        print(f"  • {param.replace('_MU', '').replace('_', ' ').title()}")

    print(f"\nPrediction Models ({len(models)}):")
    for model in models:
        predictor = model["predictor_parameter"]["name"]
        target = model["target_parameter"]["name"]
        r2 = model["prediction_quality"]["r_squared"]

        predictor_clean = predictor.replace("_MU", "").replace("_", " ").title()
        target_clean = target.replace("_MU", "").replace("_", " ").title()

        print(f"  {predictor_clean} → {target_clean} (R² = {r2:.3f})")

    reduction = lr_data["sampling_instructions"]["reduction_efficiency"]
    total_sample = lr_data["sampling_instructions"]["total_parameters_to_sample"]
    total_available = lr_data["sampling_instructions"]["total_parameters_available"]

    print(f"\nSampling Efficiency:")
    print(f"  Sample only {total_sample} out of {total_available} parameters")
    print(f"  Reduction: {reduction}")


def main():
    # Create all plots
    warnings.filterwarnings("ignore")

    # Load the data
    base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2"
    # base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N256_combined_grid_r2_0.7_p1"
    prior_df = pd.read_csv(f"{base_dir}/iter_0/all_param_df.csv")
    posterior_df = pd.read_csv(f"{base_dir}/iter_4/all_param_df.csv")
    final_metrics_df = pd.read_csv(f"{base_dir}/iter_4/final_metrics.csv")
    target_metrics = json.load(open(f"{base_dir}/targets.json"))
    # keep only the columns that are in the target_metrics
    final_metrics_df = final_metrics_df[target_metrics.keys()]

    drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
    prior_df.drop(columns=drop_cols, inplace=True)
    posterior_df.drop(columns=drop_cols, inplace=True)
    param_names = prior_df.columns.tolist()
    param_subset = prior_df.columns.tolist()
    param_subset = [param for param in param_subset]  # if not param.endswith("_SIGMA")]
    if 0:
        analysis_results = analyze_peak_correlations(
            posterior_df[param_subset],
            final_metrics_df,
            save_path=f"{base_dir}/posterior_plots/peak_correlation_analysis.png",
        )

    if 0:
        r2 = calculate_pairwise_r2_and_plot(
            posterior_df[param_subset],
            param_names,
            save_path=f"{base_dir}/posterior_plots/parameter_correlation.png",
        )
    if 1:
        analysis_results = analyze_peak_pairwise_correlations(
            posterior_df[param_subset], verbose=False
        )
        # summary = get_parameter_prediction_summary(analysis_results)
        # plot_peak_pairwise_correlations(analysis_results, save_path=f"{base_dir}/posterior_plots/peak_pairwise_correlations.png")
        # create_correlation_heatmap(analysis_results, save_path=f"{base_dir}/posterior_plots/correlation_heatmap.png")
        r2_thresholds = [0.9]  # , 0.8, 0.7,0.01]
        peak_name = "Peak 3"
        for r2_threshold in r2_thresholds:
            # chain_analysis = analyze_parameter_chains(analysis_results, min_r2_threshold=r2_threshold, verbose=True)
            # plot_chain_parameter_combinations(analysis_results, chain_analysis, peak_name, save_path=f"{base_dir}/posterior_plots/chain_parameter_combinations_r2_{r2_threshold}_p{peak_name.split()[1]}.png")
            correlation_base = f"../../../ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_linear_{r2_threshold}_p{int(peak_name.split()[1])}_mean_only/"
            pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
                posterior_df, n_components=2
            )
            distances = np.sqrt(
                np.sum(
                    (pca_result[:, :2] - peak_positions[int(peak_name.split()[1]) - 1]) ** 2, axis=1
                )
            )
            closest_indices = np.argsort(distances)[:50]
            peak_data = posterior_df.iloc[closest_indices]
            plot_lr_parameter_correlations(
                f"{correlation_base}/lr_predictions_r{r2_threshold}_p{int(peak_name.split()[1])}.json",
                peak_data,
                save_path=f"{correlation_base}/lr_parameter_correlations_r2_{r2_threshold}_p{int(peak_name.split()[1])}.png",
            )


if __name__ == "__main__":
    main()
