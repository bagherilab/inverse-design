import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import shapiro, jarque_bera, kstest
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import warnings
import json
from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.analyze.parameter_importance import (
    compute_gaussian_mi_matrix,
    compute_ksg_mi_matrix,
    plot_gaussian_mi_results,
    plot_ksg_mi_results,
)

warnings.filterwarnings("ignore")


def test_normality(data, method="shapiro"):
    """
    Test if data follows normal distribution.

    Parameters:
    -----------
    data : array-like
        1D array of data points
    method : str
        'shapiro', 'jarque_bera', or 'ks' (Kolmogorov-Smirnov)

    Returns:
    --------
    dict : Test results with p-value and interpretation
    """
    data = np.asarray(data).flatten()

    if method == "shapiro" and len(data) <= 5000:  # Shapiro-Wilk has sample size limit
        stat, p_value = shapiro(data)
        test_name = "Shapiro-Wilk"
    elif method == "jarque_bera":
        stat, p_value = jarque_bera(data)
        test_name = "Jarque-Bera"
    else:  # Kolmogorov-Smirnov
        stat, p_value = kstest(data, "norm", args=(np.mean(data), np.std(data)))
        test_name = "Kolmogorov-Smirnov"

    is_normal = p_value > 0.05

    return {
        "test": test_name,
        "statistic": stat,
        "p_value": p_value,
        "is_normal": is_normal,
        "interpretation": "Normal" if is_normal else "Non-normal",
    }


def test_linearity(x, y, method="pearson"):
    """
    Test if relationship between x and y is linear.

    Parameters:
    -----------
    x, y : array-like
        Data arrays
    method : str
        'pearson', 'spearman', or 'both'

    Returns:
    --------
    dict : Correlation results and linearity assessment
    """
    x, y = np.asarray(x).flatten(), np.asarray(y).flatten()

    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    # If Pearson and Spearman correlations are similar, relationship is likely linear
    linearity_score = abs(pearson_r) / (abs(spearman_r) + 1e-10)
    is_linear = linearity_score > 0.8 and abs(pearson_r - spearman_r) < 0.2

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "linearity_score": linearity_score,
        "is_linear": is_linear,
        "interpretation": "Linear" if is_linear else "Non-linear",
    }


def assess_sample_size(n_samples, n_features):
    """
    Assess if sample size is adequate for different MI methods.

    Parameters:
    -----------
    n_samples : int
        Number of samples
    n_features : int
        Number of features (dimensions)

    Returns:
    --------
    dict : Sample size assessment for different methods
    """
    # General guidelines
    min_gaussian = max(30, 5 * n_features)  # At least 5 samples per parameter
    min_ksg = max(50, 10 * n_features)  # KSG needs more samples

    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "adequate_for_gaussian": n_samples >= min_gaussian,
        "adequate_for_ksg": n_samples >= min_ksg,
        "recommended_gaussian": min_gaussian,
        "recommended_ksg": min_ksg,
        "overall_adequacy": (
            "Good"
            if n_samples >= min_ksg
            else "Limited" if n_samples >= min_gaussian else "Insufficient"
        ),
    }


def detect_outliers(data, method="iqr"):
    """
    Detect outliers in data.

    Parameters:
    -----------
    data : array-like
        1D array of data points
    method : str
        'iqr' or 'zscore'

    Returns:
    --------
    dict : Outlier detection results
    """
    data = np.asarray(data).flatten()

    if method == "iqr":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (data < lower_bound) | (data > upper_bound)
    else:  # zscore
        z_scores = np.abs(stats.zscore(data))
        outliers = z_scores > 3

    outlier_percentage = np.sum(outliers) / len(data) * 100

    return {
        "method": method,
        "n_outliers": np.sum(outliers),
        "outlier_percentage": outlier_percentage,
        "outlier_indices": np.where(outliers)[0],
        "has_many_outliers": outlier_percentage > 5,  # More than 5% outliers
    }


def comprehensive_suitability_test(X, Y, param_names=None, metric_names=None):
    """
    Comprehensive test to determine suitability for Gaussian vs KSG MI methods.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_params)
        Parameter values
    Y : array-like, shape (n_samples, n_metrics)
        Metric values
    param_names, metric_names : list, optional
        Names for parameters and metrics

    Returns:
    --------
    dict : Comprehensive assessment results
    """
    X, Y = np.asarray(X), np.asarray(Y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_samples, n_params = X.shape
    n_metrics = Y.shape[1]

    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]
    if metric_names is None:
        metric_names = [f"metric_{j}" for j in range(n_metrics)]

    results = {
        "sample_assessment": assess_sample_size(n_samples, n_params + n_metrics),
        "param_tests": {},
        "metric_tests": {},
        "pairwise_tests": {},
        "recommendations": {},
    }

    # Test each parameter
    for i, param_name in enumerate(param_names):
        param_data = X[:, i]
        results["param_tests"][param_name] = {
            "normality": test_normality(param_data),
            "outliers": detect_outliers(param_data),
        }

    # Test each metric
    for j, metric_name in enumerate(metric_names):
        metric_data = Y[:, j]
        results["metric_tests"][metric_name] = {
            "normality": test_normality(metric_data),
            "outliers": detect_outliers(metric_data),
        }

    # Test pairwise relationships
    for i, param_name in enumerate(param_names):
        for j, metric_name in enumerate(metric_names):
            pair_key = f"{param_name}_{metric_name}"
            results["pairwise_tests"][pair_key] = {"linearity": test_linearity(X[:, i], Y[:, j])}

    # Generate recommendations
    results["recommendations"] = generate_recommendations(results)

    return results


def generate_recommendations(test_results):
    """
    Generate method recommendations based on test results.
    """
    recommendations = {}

    # Overall sample size recommendation
    sample_info = test_results["sample_assessment"]

    # Count normality violations
    param_normal_count = sum(
        [1 for p in test_results["param_tests"].values() if p["normality"]["is_normal"]]
    )
    metric_normal_count = sum(
        [1 for m in test_results["metric_tests"].values() if m["normality"]["is_normal"]]
    )
    total_variables = len(test_results["param_tests"]) + len(test_results["metric_tests"])
    normality_ratio = (param_normal_count + metric_normal_count) / total_variables

    # Count linearity
    linear_count = sum(
        [1 for p in test_results["pairwise_tests"].values() if p["linearity"]["is_linear"]]
    )
    total_pairs = len(test_results["pairwise_tests"])
    linearity_ratio = linear_count / total_pairs if total_pairs > 0 else 0

    # Count outliers
    param_outlier_count = sum(
        [1 for p in test_results["param_tests"].values() if p["outliers"]["has_many_outliers"]]
    )
    metric_outlier_count = sum(
        [1 for m in test_results["metric_tests"].values() if m["outliers"]["has_many_outliers"]]
    )
    outlier_ratio = (param_outlier_count + metric_outlier_count) / total_variables

    # Generate overall recommendation
    gaussian_score = 0
    ksg_score = 0

    # Sample size factor
    if sample_info["adequate_for_ksg"]:
        ksg_score += 2
    if sample_info["adequate_for_gaussian"]:
        gaussian_score += 2

    # Normality factor
    gaussian_score += normality_ratio * 3
    ksg_score += (1 - normality_ratio) * 2

    # Linearity factor
    gaussian_score += linearity_ratio * 2
    ksg_score += (1 - linearity_ratio) * 3

    # Outlier factor
    gaussian_score -= outlier_ratio * 2
    ksg_score += outlier_ratio * 1  # KSG is more robust to outliers

    if gaussian_score > ksg_score:
        primary_method = "Gaussian"
        confidence = min(100, (gaussian_score / (gaussian_score + ksg_score)) * 100)
    else:
        primary_method = "KSG"
        confidence = min(100, (ksg_score / (gaussian_score + ksg_score)) * 100)

    recommendations["overall"] = {
        "primary_method": primary_method,
        "confidence": confidence,
        "gaussian_score": gaussian_score,
        "ksg_score": ksg_score,
        "use_both": abs(gaussian_score - ksg_score) < 1,  # Scores are close
        "rationale": {
            "normality_ratio": normality_ratio,
            "linearity_ratio": linearity_ratio,
            "outlier_ratio": outlier_ratio,
            "sample_adequacy": sample_info["overall_adequacy"],
        },
    }

    return recommendations


def plot_suitability_diagnostics(X, Y, param_names=None, metric_names=None, figsize=(15, 10)):
    """
    Create diagnostic plots to visualize data characteristics.
    """
    X, Y = np.asarray(X), np.asarray(Y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_params, n_metrics = X.shape[1], Y.shape[1]

    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]
    if metric_names is None:
        metric_names = [f"metric_{j}" for j in range(n_metrics)]

    # Create subplots
    fig = plt.figure(figsize=figsize)

    # 1. Distribution plots for parameters
    plt.subplot(2, 3, 1)
    for i, name in enumerate(param_names[:5]):  # Limit to first 5 for readability
        plt.hist(X[:, i], alpha=0.7, bins=20, label=name, density=True)
    plt.title("Parameter Distributions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()

    # 2. Distribution plots for metrics
    plt.subplot(2, 3, 2)
    for j, name in enumerate(metric_names[:5]):
        plt.hist(Y[:, j], alpha=0.7, bins=20, label=name, density=True)
    plt.title("Metric Distributions")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()

    # 3. Q-Q plots for normality (first parameter and metric)
    plt.subplot(2, 3, 3)
    stats.probplot(X[:, 0], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {param_names[0]}")

    plt.subplot(2, 3, 4)
    stats.probplot(Y[:, 0], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot: {metric_names[0]}")

    # 5. Scatter plot showing relationship (first param vs first metric)
    plt.subplot(2, 3, 5)
    plt.scatter(X[:, 0], Y[:, 0], alpha=0.6)
    plt.xlabel(param_names[0])
    plt.ylabel(metric_names[0])
    plt.title("Parameter-Metric Relationship")

    # Add trend line
    z = np.polyfit(X[:, 0], Y[:, 0], 1)
    p = np.poly1d(z)
    plt.plot(X[:, 0], p(X[:, 0]), "r--", alpha=0.8)

    # 6. Correlation heatmap
    plt.subplot(2, 3, 6)
    combined_data = np.hstack([X, Y])
    combined_names = param_names + metric_names
    corr_matrix = np.corrcoef(combined_data.T)

    im = plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im)
    plt.title("Correlation Matrix")
    plt.xticks(range(len(combined_names)), combined_names, rotation=45)
    plt.yticks(range(len(combined_names)), combined_names)

    plt.tight_layout()
    plt.show()


def print_suitability_report(test_results):
    """
    Print a comprehensive suitability report.
    """
    print("=" * 80)
    print("MUTUAL INFORMATION METHOD SUITABILITY REPORT")
    print("=" * 80)

    # Sample size assessment
    sample_info = test_results["sample_assessment"]
    print(f"\n📊 SAMPLE SIZE ASSESSMENT")
    print(f"   Samples: {sample_info['n_samples']}")
    print(f"   Features: {sample_info['n_features']}")
    print(f"   Adequacy: {sample_info['overall_adequacy']}")
    print(
        f"   Gaussian MI: {'✓' if sample_info['adequate_for_gaussian'] else '✗'} (need ≥{sample_info['recommended_gaussian']})"
    )
    print(
        f"   KSG MI: {'✓' if sample_info['adequate_for_ksg'] else '✗'} (need ≥{sample_info['recommended_ksg']})"
    )

    # Normality tests
    print(f"\n📈 NORMALITY TESTS")
    normal_count = 0
    total_vars = len(test_results["param_tests"]) + len(test_results["metric_tests"])

    for var_name, var_data in {
        **test_results["param_tests"],
        **test_results["metric_tests"],
    }.items():
        normality = var_data["normality"]
        status = "✓" if normality["is_normal"] else "✗"
        print(
            f"   {var_name}: {status} {normality['interpretation']} (p={normality['p_value']:.4f})"
        )
        if normality["is_normal"]:
            normal_count += 1

    print(f"   Normal variables: {normal_count}/{total_vars} ({normal_count/total_vars*100:.1f}%)")

    # Linearity tests
    print(f"\n📉 LINEARITY TESTS")
    linear_count = 0
    total_pairs = len(test_results["pairwise_tests"])

    for pair_name, pair_data in test_results["pairwise_tests"].items():
        linearity = pair_data["linearity"]
        status = "✓" if linearity["is_linear"] else "✗"
        print(
            f"   {pair_name}: {status} {linearity['interpretation']} (score={linearity['linearity_score']:.3f})"
        )
        if linearity["is_linear"]:
            linear_count += 1

    print(
        f"   Linear relationships: {linear_count}/{total_pairs} ({linear_count/total_pairs*100:.1f}%)"
    )

    # Outlier detection
    print(f"\n🎯 OUTLIER DETECTION")
    outlier_vars = 0
    for var_name, var_data in {
        **test_results["param_tests"],
        **test_results["metric_tests"],
    }.items():
        outliers = var_data["outliers"]
        status = "⚠️" if outliers["has_many_outliers"] else "✓"
        print(f"   {var_name}: {status} {outliers['outlier_percentage']:.1f}% outliers")
        if outliers["has_many_outliers"]:
            outlier_vars += 1

    # Recommendations
    rec = test_results["recommendations"]["overall"]
    print(f"\n🎯 RECOMMENDATIONS")
    print(f"   Primary Method: {rec['primary_method']} (confidence: {rec['confidence']:.1f}%)")
    print(f"   Use Both Methods: {'Yes' if rec['use_both'] else 'No'}")
    print(f"   Gaussian Score: {rec['gaussian_score']:.2f}")
    print(f"   KSG Score: {rec['ksg_score']:.2f}")

    print(f"\n📋 RATIONALE")
    rationale = rec["rationale"]
    print(f"   Normality: {rationale['normality_ratio']:.1%} of variables are normal")
    print(f"   Linearity: {rationale['linearity_ratio']:.1%} of relationships are linear")
    print(f"   Outliers: {rationale['outlier_ratio']:.1%} of variables have many outliers")
    print(f"   Sample Size: {rationale['sample_adequacy']}")

    print("\n💡 INTERPRETATION")
    if rec["primary_method"] == "Gaussian":
        print("   → Gaussian MI is recommended due to:")
        if rationale["normality_ratio"] > 0.7:
            print("     • Most variables follow normal distributions")
        if rationale["linearity_ratio"] > 0.7:
            print("     • Most relationships appear linear")
        if rationale["outlier_ratio"] < 0.2:
            print("     • Low outlier presence")
    else:
        print("   → KSG MI is recommended due to:")
        if rationale["normality_ratio"] < 0.5:
            print("     • Many variables are non-normal")
        if rationale["linearity_ratio"] < 0.5:
            print("     • Many relationships are non-linear")
        if rationale["outlier_ratio"] > 0.2:
            print("     • Significant outlier presence")

    if rec["use_both"]:
        print("   → Consider using both methods for comparison")

    print("=" * 80)


# Example usage of the MI method suitability testing

# Assuming you have your data ready
# X: parameter matrix (n_samples, n_params)
# Y: metrics matrix (n_samples, n_metrics)
# param_names: list of parameter names
# metric_names: list of metric names


def example_usage():
    """
    Example of how to use the suitability testing framework.
    """
    base_original_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_new"
    posterior_df = pd.read_csv(f"{base_original_dir}/iter_4/all_param_df.csv")
    final_metrics_df = pd.read_csv(f"{base_original_dir}/iter_4/final_metrics.csv")
    target_metrics = json.load(open(f"{base_original_dir}/targets.json"))
    final_metrics_df = final_metrics_df[target_metrics.keys()]
    drop_cols = [
        "input_folder",
        "X_SPACING",
        "Y_SPACING",
        "DISTANCE_TO_CENTER",
    ]  # , "CAPILLARY_DENSITY", "AUTOPHAGY_RATE_SIGMA"]
    posterior_df.drop(columns=drop_cols, inplace=True)
    param_names = posterior_df.columns
    metric_names = list(target_metrics.keys())

    pca_result, peak_positions, point_colors, pca, Z, X, Y, _ = perform_pca_and_find_peaks(
        posterior_df, n_components=2
    )
    target_peak_idx = 0
    # Assign each point to the nearest peak (same as your existing code)
    point_assignments = []
    if len(peak_positions) > 0:
        for point in pca_result[:, :2]:
            distances = [np.sqrt(np.sum((point - peak) ** 2)) for peak in peak_positions]
            nearest_peak = np.argmin(distances)
            point_assignments.append(nearest_peak)
        point_assignments = np.array(point_assignments)
    target_peak_mask = point_assignments == target_peak_idx
    posterior_df_filtered = posterior_df[target_peak_mask]
    Y = final_metrics_df[target_peak_mask]
    # drop if param ends with _SIGMA
    X = posterior_df_filtered.drop(
        columns=[col for col in posterior_df_filtered.columns if col.endswith("_SIGMA")]
    )
    param_names = [col for col in param_names if not col.endswith("_SIGMA")]
    # Step 1: Run comprehensive suitability test
    print("Running comprehensive suitability analysis...")
    test_results = comprehensive_suitability_test(X, Y, param_names, metric_names)

    # Step 2: Print detailed report
    print_suitability_report(test_results)

    # Step 3: Create diagnostic plots
    print("\nGenerating diagnostic plots...")
    plot_suitability_diagnostics(X, Y, param_names, metric_names)

    # Step 4: Get recommendation and proceed accordingly
    recommendation = test_results["recommendations"]["overall"]

    if recommendation["primary_method"] == "Gaussian":
        print("\n🎯 Proceeding with Gaussian MI estimation...")
        mi_results = compute_gaussian_mi_matrix(X, Y, param_names, metric_names)
        plot_gaussian_mi_results(mi_results)

    elif recommendation["primary_method"] == "KSG":
        print("\n🎯 Proceeding with KSG MI estimation...")
        mi_results = compute_ksg_mi_matrix(X, Y, param_names, metric_names)
        plot_ksg_mi_results(mi_results)

    if recommendation["use_both"]:
        print("\n🔄 Running both methods for comparison...")

        # Compute both
        gaussian_mi = compute_gaussian_mi_matrix(X, Y, param_names, metric_names)
        ksg_mi = compute_ksg_mi_matrix(X, Y, param_names, metric_names)

        # Plot comparison
        compare_mi_methods(gaussian_mi, ksg_mi)

    return test_results


def compare_mi_methods(gaussian_mi, ksg_mi):
    """
    Compare results from Gaussian and KSG MI methods.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Gaussian MI heatmap
    sns.heatmap(gaussian_mi, annot=True, cmap="viridis", ax=axes[0], fmt=".3f")
    axes[0].set_title("Gaussian MI")

    # Plot 2: KSG MI heatmap
    sns.heatmap(ksg_mi, annot=True, cmap="viridis", ax=axes[1], fmt=".3f")
    axes[1].set_title("KSG MI")

    # Plot 3: Difference heatmap
    diff = ksg_mi - gaussian_mi
    sns.heatmap(diff, annot=True, cmap="RdBu", center=0, ax=axes[2], fmt=".3f")
    axes[2].set_title("Difference (KSG - Gaussian)")

    plt.tight_layout()
    plt.show()

    # Print correlation between methods
    flat_gaussian = gaussian_mi.values.flatten()
    flat_ksg = ksg_mi.values.flatten()

    # Remove NaN values for correlation
    valid_mask = ~(np.isnan(flat_gaussian) | np.isnan(flat_ksg))
    if np.sum(valid_mask) > 0:
        correlation = np.corrcoef(flat_gaussian[valid_mask], flat_ksg[valid_mask])[0, 1]
        print(f"\n📊 Method Correlation: {correlation:.3f}")

        if correlation > 0.8:
            print("   → Methods agree well - either can be used")
        elif correlation > 0.5:
            print("   → Methods partially agree - consider using both")
        else:
            print("   → Methods disagree - data characteristics favor different approaches")


# Quick diagnostic function for rapid assessment
def quick_diagnostic(X, Y):
    """
    Quick diagnostic for immediate feedback.
    """
    print("🔍 QUICK DIAGNOSTIC")
    print("-" * 40)

    # Sample size
    n_samples = X.shape[0]
    n_features = X.shape[1] + Y.shape[1]
    print(f"Samples: {n_samples}, Features: {n_features}")

    if n_samples < 30:
        print("⚠️  Very small sample - consider more data")
    elif n_samples < 100:
        print("⚠️  Small sample - Gaussian MI may be unreliable")
    else:
        print("✓ Adequate sample size")

    # Test first parameter and metric for normality
    param_normal = test_normality(X[:, 0])["is_normal"]
    metric_normal = test_normality(Y[:, 0])["is_normal"]

    print(f"First param normal: {'✓' if param_normal else '✗'}")
    print(f"First metric normal: {'✓' if metric_normal else '✗'}")

    # Test linearity
    linearity = test_linearity(X[:, 0], Y[:, 0])["is_linear"]
    print(f"Linear relationship: {'✓' if linearity else '✗'}")

    # Quick recommendation
    if param_normal and metric_normal and linearity and n_samples >= 50:
        print("\n💡 Quick recommendation: Start with Gaussian MI")
    else:
        print("\n💡 Quick recommendation: Start with KSG MI")


# Usage pattern:
# 1. Run quick_diagnostic(X, Y) for immediate feedback
# 2. Run comprehensive_suitability_test() for detailed analysis
# 3. Use recommendations to choose method(s)
