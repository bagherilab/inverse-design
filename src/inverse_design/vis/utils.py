"""Shared constants and utility helpers for the vis package."""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from inverse_design.plotting.colormap import PEAK_COLORS

# ---------------------------------------------------------------------------
# Shared visual constants
# ---------------------------------------------------------------------------

hatch_patterns: List[str] = ["///", "...", "oo", "+++", "|||", "---"]
peak_markers: List[str] = ["o", "s", "^", "D", "*", "p"]
peak_colors: List[str] = list(PEAK_COLORS)

# ``matplotlib`` Line2D ``markersize`` (points) for peak markers (see simplification figures).
PEAK_MARKER_MARKERSIZE_PT: Dict[str, float] = {
    "o": 6.2,
    "s": 5.9,
    "^": 6.9,
    "D": 6.2,
    "*": 6.5,
    "p": 7.3,
}


def peak_marker_markersize(marker: str) -> float:
    """Return ``markersize`` in points for ``ax.plot(..., marker=...)`` / sizing scatter."""
    return PEAK_MARKER_MARKERSIZE_PT.get(marker, 8.0)


# ---------------------------------------------------------------------------
# Histogram utilities
# ---------------------------------------------------------------------------


def find_mode_bin(df: pd.DataFrame, n_bins: int = 30) -> dict:
    """Bin each column and return the bin with the highest count (mode bin).

    Args:
        df: DataFrame whose columns will be binned.
        n_bins: Number of equal-width bins per column.

    Returns:
        Dict keyed by column name, each value being a dict with keys
        ``mode_bin_center``, ``mode_bin_count``, ``mode_bin_range``,
        ``bin_edges``, ``bin_counts``, ``bin_centers``.
    """
    mode_bins: dict = {}
    for column in df.columns:
        data = df[column].dropna()
        if len(data) == 0:
            mode_bins[column] = {
                "mode_bin_center": None,
                "bin_edges": None,
                "bin_counts": None,
                "bin_centers": None,
            }
            continue
        bin_edges = np.linspace(data.min(), data.max(), n_bins + 1)
        bin_counts, bin_edges = np.histogram(data, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        mode_bin_idx = np.argmax(bin_counts)
        mode_bins[column] = {
            "mode_bin_center": bin_centers[mode_bin_idx],
            "mode_bin_count": bin_counts[mode_bin_idx],
            "mode_bin_range": (bin_edges[mode_bin_idx], bin_edges[mode_bin_idx + 1]),
            "bin_edges": bin_edges,
            "bin_counts": bin_counts,
            "bin_centers": bin_centers,
        }
    return mode_bins


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def find_best_sample_for_metric(final_metrics_df: pd.DataFrame, target_metrics: dict) -> int:
    """Return the row index whose normalised distance to *target_metrics* is smallest."""
    metric_names = list(target_metrics.keys())
    scaler = StandardScaler()
    normalised = scaler.fit_transform(final_metrics_df[metric_names].values)
    target_normalised = scaler.transform(np.array(list(target_metrics.values())).reshape(1, -1))[0]
    distances = [np.linalg.norm(row - target_normalised) for row in normalised]
    return int(np.argmin(distances))


# ---------------------------------------------------------------------------
# Redundancy-analysis helpers
# ---------------------------------------------------------------------------


def load_redundancy_analysis(file_paths: List[str]) -> Tuple[List[str], List[dict]]:
    """Load redundancy analysis JSON files and return the intersected individual params.

    Args:
        file_paths: Paths to redundancy-analysis JSON files (one per scenario).

    Returns:
        ``(individual_params, scenarios)`` where *individual_params* is the
        sorted intersection of ``individual_params`` across all files and
        *scenarios* is a list of per-file config dicts.
    """
    scenarios: List[dict] = []
    all_individual_params: List[set] = []

    for i, file_path in enumerate(file_paths):
        with open(file_path, "r") as fh:
            data = json.load(fh)
        scenarios.append(
            {
                "representative_parameters": data["representative_parameters"],
                "redundant_params": data["redundant_params"],
                "individual_params": data["individual_params"],
                "scenario_name": (
                    f'Scenario {i + 1}: {len(data["representative_parameters"])} rep groups'
                ),
            }
        )
        all_individual_params.append(set(data["individual_params"]))

    individual_params = sorted(set.intersection(*all_individual_params))
    return individual_params, scenarios


def create_parameter_order(
    individual_params: List[str],
    scenarios: List[dict],
    manual_rep_order: List[str],
) -> Tuple[List[str], dict, List[str], List[str]]:
    """Build a consistent parameter ordering for visualisation.

    Individual parameters come first; grouped parameters follow in the order
    given by *manual_rep_order* (representative first, then its redundant group).

    Returns:
        ``(all_param_names, param_to_index, representative_params, redundant_params)``
    """
    all_param_names = list(individual_params)
    representative_params: List[str] = []
    redundant_params: List[str] = []
    grouped_params: List[str] = []

    for scenario in scenarios[-1:]:
        for rep_param in manual_rep_order:
            if rep_param not in all_param_names and rep_param not in grouped_params:
                grouped_params.append(rep_param)
                representative_params.append(rep_param)
            for param in scenario["redundant_params"].get(rep_param, []):
                if param not in all_param_names and param not in grouped_params:
                    grouped_params.append(param)
                    redundant_params.append(param)

    all_param_names.extend(grouped_params)
    param_to_index = {p: i for i, p in enumerate(all_param_names)}
    return all_param_names, param_to_index, representative_params, redundant_params
