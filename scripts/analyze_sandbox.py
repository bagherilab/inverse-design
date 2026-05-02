import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings("ignore")

# Load the data
base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast"
prior_df = pd.read_csv(f"{base_dir}/iter_0/all_param_df.csv")
posterior_df = pd.read_csv(f"{base_dir}/iter_4/all_param_df.csv")
drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
prior_df.drop(columns=drop_cols, inplace=True)
posterior_df.drop(columns=drop_cols, inplace=True)
param_names = prior_df.columns.tolist()
prior_samples = prior_df.to_numpy()
posterior_samples = posterior_df.to_numpy()
n_samples, n_params = posterior_samples.shape


# Option 1: Create synthetic true_importance based on parameter shifts
# This is a reasonable approximation when we don't have ground truth
def create_synthetic_true_importance(shift_results):
    """Create synthetic true importance based on observable shifts"""
    # Combine standardized shift and uncertainty reduction
    importance = (shift_results["Standardized_Shift"] + shift_results["Uncertainty_Reduction"]) / 2
    # Normalize to sum to 1 (like a probability distribution)
    return importance / importance.sum()


# Option 2: Assume uniform importance (null hypothesis)
def create_uniform_importance(n_params):
    """Assume all parameters are equally important"""
    return np.ones(n_params) / n_params


# Option 3: Create importance based on posterior variance reduction
def create_variance_based_importance(prior_df, posterior_df):
    """Create importance based on variance reduction"""
    prior_var = prior_df.var()
    posterior_var = posterior_df.var()
    variance_reduction = (prior_var - posterior_var) / prior_var
    # Handle negative values (parameters that became more uncertain)
    variance_reduction = np.maximum(variance_reduction, 0)
    return (
        variance_reduction / variance_reduction.sum()
        if variance_reduction.sum() > 0
        else np.ones(len(variance_reduction)) / len(variance_reduction)
    )


# 1. Basic Parameter Shift Analysis
def calculate_parameter_shifts(prior_data, posterior_data):
    """Calculate various metrics of parameter shifts"""

    prior_stats = prior_data.describe()
    print(prior_stats.loc["std"])
    posterior_stats = posterior_data.describe()

    results = pd.DataFrame(index=param_names)

    # Absolute mean shift
    results["Mean_Shift"] = np.abs(posterior_stats.loc["mean"] - prior_stats.loc["mean"])

    # Standardized shift (by prior std)
    results["Standardized_Shift"] = results["Mean_Shift"] / prior_stats.loc["std"]

    # Relative shift (percentage change)
    results["Relative_Shift"] = np.abs(
        (posterior_stats.loc["mean"] - prior_stats.loc["mean"]) / (prior_stats.loc["mean"] + 1e-8)
    )

    # Uncertainty reduction
    results["Uncertainty_Reduction"] = (
        prior_stats.loc["std"] - posterior_stats.loc["std"]
    ) / prior_stats.loc["std"]

    # Effective sample size change (using variance ratio)
    results["Variance_Ratio"] = prior_stats.loc["std"] ** 2 / posterior_stats.loc["std"] ** 2

    # Kullback-Leibler divergence (approximate)
    results["KL_Divergence"] = (
        np.log(posterior_stats.loc["std"] / prior_stats.loc["std"])
        + (
            prior_stats.loc["std"] ** 2
            + (prior_stats.loc["mean"] - posterior_stats.loc["mean"]) ** 2
        )
        / (2 * posterior_stats.loc["std"] ** 2)
        - 0.5
    )

    return results


# 2. Modified Bayesian Sensitivity Analysis
def bayesian_sensitivity_analysis(prior_data, posterior_data, true_importance):
    """Perform Bayesian sensitivity analysis"""

    # Create synthetic output data based on parameter importance
    y_data = np.dot(posterior_samples, true_importance) + np.random.normal(0, 0.1, n_samples)

    # Calculate marginal variances
    marginal_variances = []
    for i, param in enumerate(param_names):
        # Hold all other parameters at their mean
        temp_samples = posterior_samples.copy()
        for j in range(n_params):
            if j != i:
                temp_samples[:, j] = np.mean(temp_samples[:, j])

        y_temp = np.dot(temp_samples, true_importance) + np.random.normal(0, 0.1, n_samples)
        marginal_variances.append(np.var(y_temp))

    # Calculate total variance
    total_variance = np.var(y_data)

    # Calculate sensitivity indices
    sensitivity_df = pd.DataFrame(
        {
            "Parameter": param_names,
            "First_Order_SI": np.array(marginal_variances)
            / (total_variance + 1e-10),  # Add small constant to avoid division by zero
            "Marginal_Variance": marginal_variances,
        }
    )

    # Normalize first-order sensitivity indices
    si_sum = sensitivity_df["First_Order_SI"].sum()
    if si_sum > 0:
        sensitivity_df["First_Order_SI_Normalized"] = sensitivity_df["First_Order_SI"] / si_sum
    else:
        sensitivity_df["First_Order_SI_Normalized"] = 1 / len(param_names)

    return sensitivity_df, y_data


# 3. Combined Importance Score
def calculate_combined_importance(shift_results, sensitivity_results, true_importance):
    """Calculate a combined importance score"""

    # Normalize all metrics to [0,1]
    metrics = ["Standardized_Shift", "Uncertainty_Reduction", "KL_Divergence"]
    normalized_shifts = shift_results[metrics].copy()

    for metric in metrics:
        metric_range = normalized_shifts[metric].max() - normalized_shifts[metric].min()
        if metric_range > 0:
            normalized_shifts[metric] = (
                normalized_shifts[metric] - normalized_shifts[metric].min()
            ) / metric_range
        else:
            normalized_shifts[metric] = 0.5  # If all values are the same, set to middle value

    # Calculate weighted importance score
    weights = {"Standardized_Shift": 0.4, "Uncertainty_Reduction": 0.3, "KL_Divergence": 0.3}

    combined_df = pd.DataFrame(index=param_names)
    combined_df["Shift_Score"] = sum(
        normalized_shifts[metric] * weights[metric] for metric in metrics
    )

    # Add sensitivity information if available
    if sensitivity_results is not None:
        combined_df["Sensitivity_Score"] = sensitivity_results.set_index("Parameter")[
            "First_Order_SI_Normalized"
        ]
        combined_df["Combined_Score"] = (
            combined_df["Shift_Score"] + combined_df["Sensitivity_Score"]
        ) / 2
    else:
        combined_df["Combined_Score"] = combined_df["Shift_Score"]

    # Add true importance for comparison
    combined_df["True_Importance"] = true_importance

    return combined_df.sort_values("Combined_Score", ascending=False)


# 4. Visualization Functions
def plot_parameter_importance_analysis(shift_results, sensitivity_results, combined_results):
    """Create comprehensive visualization of parameter importance"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # 1. Parameter shift magnitudes
    ax = axes[0]
    shift_results_sorted = shift_results.sort_values("Standardized_Shift", ascending=True)
    ax.barh(
        range(len(param_names)),
        shift_results_sorted["Standardized_Shift"],
        color="skyblue",
        alpha=0.7,
    )
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(shift_results_sorted.index)
    ax.set_xlabel("Standardized Shift")
    ax.set_title("Parameter Shift Magnitudes")

    # 2. Uncertainty reduction
    ax = axes[1]
    unc_reduction_sorted = shift_results.sort_values("Uncertainty_Reduction", ascending=True)
    ax.barh(
        range(len(param_names)),
        unc_reduction_sorted["Uncertainty_Reduction"],
        color="lightcoral",
        alpha=0.7,
    )
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(unc_reduction_sorted.index)
    ax.set_xlabel("Uncertainty Reduction")
    ax.set_title("Parameter Uncertainty Reduction")

    # 3. Sensitivity indices
    ax = axes[2]
    if sensitivity_results is not None:
        sens_sorted = sensitivity_results.sort_values("First_Order_SI", ascending=True)
        ax.barh(
            range(len(param_names)), sens_sorted["First_Order_SI"], color="lightgreen", alpha=0.7
        )
        ax.set_yticks(range(len(param_names)))
        ax.set_yticklabels(sens_sorted["Parameter"])
        ax.set_xlabel("First-Order Sensitivity Index")
        ax.set_title("Bayesian Sensitivity Analysis")
    else:
        ax.text(
            0.5,
            0.5,
            "No Output Data\nProvided",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Sensitivity Analysis (N/A)")

    # 4. Combined importance ranking
    ax = axes[3]
    combined_sorted = combined_results.reset_index().sort_values("Combined_Score", ascending=True)
    colors = plt.cm.viridis(
        combined_sorted["Combined_Score"] / (combined_sorted["Combined_Score"].max() + 1e-10)
    )
    bars = ax.barh(
        range(len(param_names)), combined_sorted["Combined_Score"], color=colors, alpha=0.8
    )
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(combined_sorted["index"])
    ax.set_xlabel("Combined Importance Score")
    ax.set_title("Combined Parameter Importance")

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=0, vmax=combined_sorted["Combined_Score"].max()),
    )
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Importance Score")

    # 5. Comparison with true importance
    ax = axes[4]
    comparison = combined_results.reset_index()
    ax.scatter(
        comparison["True_Importance"], comparison["Combined_Score"], s=100, alpha=0.7, c="purple"
    )

    # Add labels for points
    for i, txt in enumerate(comparison["index"]):
        ax.annotate(
            txt,
            (comparison["True_Importance"].iloc[i], comparison["Combined_Score"].iloc[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    # Add line of perfect agreement
    max_val = max(comparison["True_Importance"].max(), comparison["Combined_Score"].max())
    ax.plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="Perfect Agreement")

    # Calculate correlation
    correlation = comparison["True_Importance"].corr(comparison["Combined_Score"])
    ax.text(
        0.05,
        0.95,
        f"Correlation: {correlation:.3f}",
        transform=ax.transAxes,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("True Importance")
    ax.set_ylabel("Estimated Importance")
    ax.set_title("Estimated vs True Parameter Importance")
    ax.legend()

    # 6. KL divergence comparison
    ax = axes[5]
    kl_sorted = shift_results.sort_values("KL_Divergence", ascending=True)
    ax.barh(range(len(param_names)), kl_sorted["KL_Divergence"], color="orange", alpha=0.7)
    ax.set_yticks(range(len(param_names)))
    ax.set_yticklabels(kl_sorted.index)
    ax.set_xlabel("KL Divergence")
    ax.set_title("Information Gain (KL Divergence)")

    plt.tight_layout()
    return fig


# 5. Cross-correlation analysis
def plot_parameter_cross_correlation(prior_data, posterior_data):
    """Analyze how parameter correlations change"""

    prior_corr = prior_data.corr()
    posterior_corr = posterior_data.corr()
    correlation_change = np.abs(posterior_corr - prior_corr)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Prior correlations
    im1 = axes[0].imshow(prior_corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[0].set_title("Prior Parameter Correlations")
    axes[0].set_xticks(range(len(param_names)))
    axes[0].set_yticks(range(len(param_names)))
    axes[0].set_xticklabels(param_names, rotation=45, ha="right")
    axes[0].set_yticklabels(param_names)
    plt.colorbar(im1, ax=axes[0])

    # Posterior correlations
    im2 = axes[1].imshow(posterior_corr, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1].set_title("Posterior Parameter Correlations")
    axes[1].set_xticks(range(len(param_names)))
    axes[1].set_yticks(range(len(param_names)))
    axes[1].set_xticklabels(param_names, rotation=45, ha="right")
    axes[1].set_yticklabels(param_names)
    plt.colorbar(im2, ax=axes[1])

    # Correlation changes
    im3 = axes[2].imshow(correlation_change, cmap="Reds", vmin=0)
    axes[2].set_title("Correlation Change Magnitude")
    axes[2].set_xticks(range(len(param_names)))
    axes[2].set_yticks(range(len(param_names)))
    axes[2].set_xticklabels(param_names, rotation=45, ha="right")
    axes[2].set_yticklabels(param_names)
    plt.colorbar(im3, ax=axes[2])

    # Identify parameters with largest correlation changes
    upper_tri_mask = np.triu(np.ones_like(correlation_change, dtype=bool), k=1)
    if np.any(upper_tri_mask):
        # Find the indices where the maximum occurs in the masked array
        masked_correlation = correlation_change.copy()
        masked_correlation[~upper_tri_mask] = -np.inf  # Set lower triangle to -inf
        max_change_idx = np.unravel_index(np.argmax(masked_correlation), correlation_change.shape)
        max_change = correlation_change[max_change_idx]
        param1, param2 = param_names[max_change_idx[0]], param_names[max_change_idx[1]]
    else:
        max_change, param1, param2 = 0, param_names[0], param_names[1]

    plt.tight_layout()
    return fig, max_change, param1, param2


# Execute the analysis
print("Performing parameter importance analysis...")

# 1. Calculate parameter shifts
shift_results = calculate_parameter_shifts(prior_df, posterior_df)
print("\nTop 5 parameters by standardized shift:")
print(shift_results.sort_values("Standardized_Shift", ascending=False).head())

# 2. Create synthetic true importance (choose one of the methods)
# Option 1: Based on parameter shifts (recommended)
true_importance = create_synthetic_true_importance(shift_results)

# Option 2: Based on variance reduction
# true_importance = create_variance_based_importance(prior_df, posterior_df)

# Option 3: Uniform importance (null hypothesis)
# true_importance = create_uniform_importance(n_params)

print(f"\nUsing synthetic true importance based on parameter shifts")
print("Synthetic true importance values:")
for i, (param, importance) in enumerate(zip(param_names, true_importance)):
    print(f"  {param}: {importance:.3f}")

# 3. Perform Bayesian sensitivity analysis
sensitivity_results, y_data = bayesian_sensitivity_analysis(
    prior_samples, posterior_samples, true_importance
)
print("\nTop 5 parameters by sensitivity:")
print(sensitivity_results.sort_values("First_Order_SI", ascending=False).head())

# 4. Calculate combined importance
combined_results = calculate_combined_importance(
    shift_results, sensitivity_results, true_importance
)
print("\nCombined parameter importance ranking:")
print(
    combined_results[
        ["Shift_Score", "Sensitivity_Score", "Combined_Score", "True_Importance"]
    ].head()
)

# 5. Create visualizations
importance_fig = plot_parameter_importance_analysis(
    shift_results, sensitivity_results, combined_results
)
plt.figure(importance_fig.number)
plt.savefig("parameter_importance_analysis.png", dpi=300, bbox_inches="tight")
plt.show()

corr_fig, max_corr_change, param1, param2 = plot_parameter_cross_correlation(prior_df, posterior_df)
plt.figure(corr_fig.number)
plt.savefig("parameter_correlations.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"\nLargest correlation change: {max_corr_change:.3f} between {param1} and {param2}")

# 6. Summary recommendations
print("\nSUMMARY RECOMMENDATIONS:")
print("=" * 50)
print("1. Parameters ranked by shift magnitude:")
top_by_shift = shift_results.sort_values("Standardized_Shift", ascending=False).head(3)
for i, (param, row) in enumerate(top_by_shift.iterrows(), 1):
    print(f"   {i}. {param}: {row['Standardized_Shift']:.3f}")

print("\n2. Parameters ranked by uncertainty reduction:")
top_by_uncertainty = shift_results.sort_values("Uncertainty_Reduction", ascending=False).head(3)
for i, (param, row) in enumerate(top_by_uncertainty.iterrows(), 1):
    print(f"   {i}. {param}: {row['Uncertainty_Reduction']:.3f}")

print("\n3. Parameters ranked by combined importance:")
top_combined = combined_results.head(3)
for i, (param, row) in enumerate(top_combined.iterrows(), 1):
    print(f"   {i}. {param}: {row['Combined_Score']:.3f}")

print(
    "\n4. Correlation between estimated and synthetic true importance:",
    combined_results["True_Importance"].corr(combined_results["Combined_Score"]),
)

print("\nNOTE: Since true parameter importance is not known, we used synthetic")
print("importance based on observed parameter shifts. This provides a reasonable")
print("approximation for comparison purposes, but conclusions should be interpreted")
print("carefully in the context of your specific problem.")
