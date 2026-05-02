"""Metric-contour and parallel-coordinate visualisations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.utils.utils import remove_outliers
from inverse_design.vis.utils import (
    create_parameter_order,
    find_best_sample_for_metric,
    load_redundancy_analysis,
)

__all__ = [
    # Utility re-exports (historical public API)
    "find_best_sample_for_metric",
    "load_redundancy_analysis",
    "create_parameter_order",
    # Plotting
    "plot_parallel_coordinates_with_change",
    "plot_pca_with_metric_contour",
    "plot_parallel_coordinates_multiple_groups_vs_posterior",
]


def plot_parallel_coordinates_with_change(
    prior_data: pd.DataFrame,
    posterior_data: pd.DataFrame,
    alpha: float = 0.1,
    n_components: int = 2,
    verbose: bool = False,
):
    """Parallel coordinates of per-sample percentage changes, coloured by PCA cluster."""
    param_names = posterior_data.columns.tolist()
    _, _, point_colors, _, _, _, _, peak_points = perform_pca_and_find_peaks(
        posterior_data[param_names].values, n_components
    )
    ref_values = prior_data[param_names].mean()
    posterior_df_copy = posterior_data[param_names].copy()
    for param in param_names:
        posterior_df_copy[param] = (
            (posterior_df_copy[param] - ref_values[param]) / ref_values[param]
        ) * 100

    if len(peak_points) > 0:
        peak_1_pct = {
            param: (
                (
                    pd.DataFrame(peak_points[0], columns=param_names)[param].mean()
                    - ref_values[param]
                )
                / ref_values[param]
            )
            * 100
            for param in param_names
        }
        param_names = sorted(param_names, key=lambda x: peak_1_pct[x], reverse=True)
        if verbose:
            for p in param_names:
                print(f"{p}: {peak_1_pct[p]:.2f}%")
        posterior_df_copy = posterior_df_copy[param_names]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    for i, (_, row) in enumerate(posterior_df_copy.iterrows()):
        ax.plot(
            range(len(param_names)),
            row[param_names].values,
            color=point_colors[i],
            alpha=alpha,
            linewidth=0.5,
        )
    ax.axhline(
        y=0, color="black", linestyle="--", alpha=0.7, linewidth=1.5, label="Reference (0% change)"
    )
    ax.set_ylabel("Percentage Change (%)", fontsize=10)
    ax.set_xticks(range(len(param_names)))
    ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fallback_colors = ["red", "green", "blue", "purple", "orange", "brown", "pink", "gray"]
    legend_elements = [
        plt.Line2D(
            [0],
            [0],
            color=fallback_colors[i % len(fallback_colors)],
            label=f"Peak {i + 1} ({len(pts)} points)",
            linewidth=2,
        )
        for i, pts in enumerate(peak_points)
        if len(pts) > 0
    ]
    legend_elements.append(
        plt.Line2D(
            [0], [0], color="black", linestyle="--", label="Reference (0% change)", linewidth=1.5
        )
    )
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))
    plt.tight_layout()
    return fig


def plot_pca_with_metric_contour(
    posterior_data_breast: pd.DataFrame,
    posterior_data_glio: pd.DataFrame,
    final_metrics_breast: pd.DataFrame,
    final_metrics_glio: pd.DataFrame,
    metric_name: str,
    cmap: str,
):
    """PCA projection coloured by interpolated metric values from two cohorts."""
    param_names = posterior_data_breast.columns.tolist()

    def _clean(metrics_df, params_df, col):
        valid = metrics_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        metric, outlier_idx = remove_outliers(valid, 1.5)
        return params_df.drop(outlier_idx), metric

    breast_data_clean, breast_metric = _clean(
        final_metrics_breast, posterior_data_breast, metric_name
    )
    glio_data_clean, glio_metric = _clean(final_metrics_glio, posterior_data_glio, metric_name)

    combined = np.vstack(
        [breast_data_clean[param_names].values, glio_data_clean[param_names].values]
    )
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaler.fit_transform(combined))
    n_breast = len(breast_data_clean)
    breast_pca = pca_result[:n_breast]
    glio_pca = pca_result[n_breast:]

    x_grid = np.linspace(pca_result[:, 0].min(), pca_result[:, 0].max(), 100)
    y_grid = np.linspace(pca_result[:, 1].min(), pca_result[:, 1].max(), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    values = np.concatenate([breast_metric, glio_metric])
    Z = griddata(pca_result[:, :2], values, (X, Y), method="cubic", fill_value=np.nan)
    Z = np.clip(Z, values.min(), values.max())

    fig, ax = plt.subplots(figsize=(12, 10))
    contour = ax.contourf(X, Y, Z, levels=20, cmap=cmap, alpha=0.6)
    cbar = plt.colorbar(contour, ax=ax, label=metric_name)
    cbar.mappable.set_clim(values.min(), values.max())
    ax.scatter(
        breast_pca[:, 0],
        breast_pca[:, 1],
        marker="o",
        alpha=0.3,
        s=20,
        label="Breast Cancer",
        edgecolors="red",
        facecolors="none",
    )
    ax.scatter(
        glio_pca[:, 0],
        glio_pca[:, 1],
        marker="s",
        alpha=0.3,
        s=20,
        label="Glioblastoma",
        edgecolors="blue",
        facecolors="none",
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_parallel_coordinates_multiple_groups_vs_posterior(
    posterior_df: pd.DataFrame,
    param_groups_dict: dict,
    param_subset,
    alpha: float = 0.5,
):
    """Parallel coordinates comparing multiple parameter groups against a posterior reference."""
    ref_values = posterior_df[param_subset].mean()
    pct_changes_dict = {}
    for group_name, group_df in param_groups_dict.items():
        pct = group_df[param_subset].copy()
        for p in param_subset:
            pct[p] = ((pct[p] - ref_values[p]) / ref_values[p]) * 100
        pct_changes_dict[group_name] = pct

    first_name = next(iter(param_groups_dict))
    sorted_params = pct_changes_dict[first_name].mean().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]
    for i, (group_name, pct) in enumerate(pct_changes_dict.items()):
        color = colors[i % len(colors)]
        for _, row in pct.iterrows():
            ax.plot(
                range(len(sorted_params)),
                row[sorted_params].values,
                color=color,
                alpha=alpha,
                linewidth=0.5,
                label=group_name if _ == 0 else "",
            )
    ax.axhline(
        y=0, color="black", linestyle="--", alpha=0.7, label="Reference (0% change)", linewidth=2
    )
    ax.set_ylabel("Percentage Change (%)", fontsize=12)
    ax.set_xticks(range(len(sorted_params)))
    ax.set_xticklabels(sorted_params, rotation=45, ha="right", fontsize=10)
    ax.set_title(f"Parameter Groups vs Posterior Reference (sorted by {first_name})", fontsize=14)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        dict(zip(labels, handles)).values(),
        dict(zip(labels, handles)).keys(),
        loc="upper right",
        bbox_to_anchor=(1.15, 1),
    )
    plt.tight_layout()
    return fig
