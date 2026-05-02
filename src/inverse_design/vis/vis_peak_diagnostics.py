"""Peak-focused posterior visualisations."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks
from inverse_design.plotting.colormap import LIGHT_PEAK_FILL_COLORS
from inverse_design.plotting.theme import apply_publication_style
from inverse_design.vis.utils import hatch_patterns, peak_markers, peak_colors

__all__ = [
    "plot_pca_with_peaks",
    "plot_all_peak_percentage_changes",
    "plot_all_peak_percentage_changes_grouped",
]

# Re-export so callers that do `from vis_peak_diagnostics import plot_pca_with_peaks` still work.
from inverse_design.vis.vis_posterior_overview import plot_pca_with_peaks  # noqa: F401


def plot_all_peak_percentage_changes(
    prior_data: pd.DataFrame,
    posterior_data: pd.DataFrame,
    n_components: int = 2,
):
    """Per-peak stacked bar chart of percentage changes from prior means."""
    param_names = posterior_data.columns.tolist()
    apply_publication_style(font_size=12, axes_linewidth=1.0)
    data = posterior_data[param_names].values
    pca_result, peak_positions, _, _, _, _, _, _ = perform_pca_and_find_peaks(data, n_components)
    prior_means = prior_data[param_names].mean().values

    n_peaks = len(peak_positions)
    all_pct_changes = []
    for i in range(n_peaks):
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[i]) ** 2, axis=1))
        point = data[np.argmin(distances)]
        all_pct_changes.extend(((point - prior_means) / prior_means) * 100)
    y_min = min(all_pct_changes) - 10
    y_max = max(all_pct_changes) + 10

    fig, axes = plt.subplots(n_peaks, 1, figsize=(9, 3 * n_peaks))
    if n_peaks == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[i]) ** 2, axis=1))
        point = data[np.argmin(distances)]
        pct_changes = ((point - prior_means) / prior_means) * 100
        ax.bar(
            range(len(param_names)),
            pct_changes,
            alpha=0.7,
            color="white",
            edgecolor="black",
            linewidth=1.5,
            hatch=hatch_patterns[i % len(hatch_patterns)],
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.set_xticks(range(len(param_names)))
        if i == n_peaks - 1:
            ax.set_xticklabels(param_names, rotation=45, ha="right", fontsize=10, fontweight="bold")
        else:
            ax.set_xticklabels([])
        if i == n_peaks // 2:
            ax.set_ylabel("Change (%)", fontsize=12, fontweight="bold")
        ax.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=2)
        ax.axhline(y=50, color="black", linestyle="--", alpha=0.2, linewidth=1)
        ax.axhline(y=-50, color="black", linestyle="--", alpha=0.2, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")
        if i == n_peaks - 1:
            for label in ax.get_xticklabels():
                label.set_fontweight("bold")
    plt.tight_layout()
    return fig


def plot_all_peak_percentage_changes_grouped(
    prior_data: pd.DataFrame,
    posterior_data: pd.DataFrame,
    n_components: int = 2,
    n_best_samples: int = 50,
    save_path: str | None = None,
):
    """Grouped bar chart: each parameter has one bar per peak, sorted by Peak 1 change."""
    param_names = posterior_data.columns.tolist()
    apply_publication_style(font_size=12, axes_linewidth=1.0)
    data = posterior_data[param_names].values
    colors = list(LIGHT_PEAK_FILL_COLORS)
    _hatch = ["///", "...", "|||", "---", "+++", "xxx"]
    pca_result, peak_positions, _, _, _, _, _, _ = perform_pca_and_find_peaks(data, n_components)
    prior_means = prior_data[param_names].mean().values
    n_peaks = len(peak_positions)

    peak_changes: dict = {}
    for i in range(n_peaks):
        distances = np.sqrt(np.sum((pca_result[:, :2] - peak_positions[i]) ** 2, axis=1))
        closest_indices = np.argsort(distances)[:n_best_samples]
        peak_pct_changes = []
        for idx in closest_indices:
            pct = ((data[idx] - prior_means) / prior_means) * 100
            valid = [0.0 if (np.isnan(v) or np.isinf(v)) else v for v in pct]
            peak_pct_changes.append(valid)
        peak_changes[f"Peak {i + 1}"] = np.median(np.array(peak_pct_changes), axis=0)

    sorted_indices = np.argsort(peak_changes["Peak 1"])[::-1]
    sorted_params = [param_names[i] for i in sorted_indices]
    peak_changes = {k: v[sorted_indices] for k, v in peak_changes.items()}

    fig, ax = plt.subplots(figsize=(12, 6))
    n_params = len(sorted_params)
    bar_width = 0.8 / n_peaks
    x_pos = np.arange(n_params)
    for i, (peak_name, changes) in enumerate(peak_changes.items()):
        bar_positions = x_pos + (i - n_peaks / 2 + 0.5) * bar_width
        ax.bar(
            bar_positions,
            changes,
            bar_width,
            label=peak_name,
            color="white",
            edgecolor=colors[i % len(colors)],
            linewidth=2,
            hatch=_hatch[i % len(_hatch)],
            alpha=0.8,
        )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.8, linewidth=2)
    ax.axhline(y=50, color="black", linestyle="--", alpha=0.2, linewidth=1)
    ax.axhline(y=-50, color="black", linestyle="--", alpha=0.2, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_xlabel("Parameter", fontsize=12, fontweight="bold")
    ax.set_ylabel("Change (%)", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sorted_params, rotation=45, ha="right", fontsize=10, fontweight="bold")
    ax.tick_params(axis="both", which="major", labelsize=10, width=1.5)
    for label in [*ax.get_yticklabels(), *ax.get_xticklabels()]:
        label.set_fontweight("bold")
    all_changes = np.concatenate(list(peak_changes.values()))
    ax.set_ylim(min(all_changes) - 10, max(all_changes) + 10)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
    return fig, ax
