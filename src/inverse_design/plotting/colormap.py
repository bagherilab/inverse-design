"""Metric, peak, and PCA cluster colors shared across analysis and vis modules."""

from __future__ import annotations

from typing import Dict, Final, Mapping, Tuple

# Canonical ARCADE summary metrics (matches ABCMetricsComparison / aggregated_results).
DEFAULT_METRIC_COLORS: Final[Dict[str, str]] = {
    "symmetry": "#486b45",
    "doub_time": "#bb883b",
    "act_ratio": "#af1b0a",
    "colony_growth": "#545aab",
    "symmetry_std": "#679A63",
    "doub_time_std": "#ddc39d",
    "colony_growth_std": "#a9acd5",
    "act_ratio_std": "#d78d85",
}

# Extra names used in hyperparam / SIR / outcome plots (same hex family where applicable).
SUPPLEMENTAL_METRIC_COLORS: Final[Dict[str, str]] = {
    "n_cells": "#a790c1",
    "comp_time": "#FF4500",
    "black": "#000000",
    "peak_i": "#486b45",
    "time_to_peak": "#af1b0a",
    "final_r": "#bb883b",
    "area_i": "#545aab",
    "growth_rate": "#679A63",
}

# ColorBrewer / Tableau-style accents for PCA peaks and cluster assignment (vis.sandbox).
PEAK_COLORS: Final[Tuple[str, ...]] = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
)

# Same count as legacy ``perform_pca_and_find_peaks`` named colors (8 slots).
PCA_CLUSTER_COLORS: Final[Tuple[str, ...]] = PEAK_COLORS + ("#a65628", "#f781bf")

# Soft fills used for multi-peak bar panels in sandbox (not metric-mapped).
LIGHT_PEAK_FILL_COLORS: Final[Tuple[str, ...]] = (
    "lightcoral",
    "skyblue",
    "gold",
    "lightgreen",
    "plum",
    "orange",
)


def metric_color(name: str, *, default: str = "#7f7f7f") -> str:
    """Return a hex color for a metric or known alias (case-insensitive)."""
    key = name.strip().lower()
    if key in DEFAULT_METRIC_COLORS:
        return DEFAULT_METRIC_COLORS[key]
    if key in SUPPLEMENTAL_METRIC_COLORS:
        return SUPPLEMENTAL_METRIC_COLORS[key]
    return default


def merged_metric_colors() -> Dict[str, str]:
    """All metric + supplemental keys (mutate-safe copy)."""
    return {**DEFAULT_METRIC_COLORS, **SUPPLEMENTAL_METRIC_COLORS}


def peak_color(index: int) -> str:
    """Cyclic accent for peak index (PCA / markers)."""
    return PEAK_COLORS[index % len(PEAK_COLORS)]


def pca_cluster_color(index: int) -> str:
    """Color for points assigned to PCA density peak ``index``."""
    return PCA_CLUSTER_COLORS[index % len(PCA_CLUSTER_COLORS)]


def light_peak_fill(index: int) -> str:
    """Cyclic soft fill for bar / hatch peak panels."""
    return LIGHT_PEAK_FILL_COLORS[index % len(LIGHT_PEAK_FILL_COLORS)]


def metric_colors_for_keys(keys: Mapping[str, object] | Tuple[str, ...] | list[str]) -> list[str]:
    """Build a list of hex colors for an ordered sequence of metric names."""
    if isinstance(keys, Mapping):
        names = list(keys.keys())
    else:
        names = list(keys)
    return [metric_color(n) for n in names]


def sir_compartment_color(compartment: str) -> str:
    """SIR line colors aligned with metric palette (I→act_ratio, R→symmetry, S→colony_growth)."""
    c = compartment.strip().upper()
    if c == "I":
        return metric_color("act_ratio")
    if c == "R":
        return metric_color("symmetry")
    if c == "S":
        return metric_color("colony_growth")
    return metric_color(compartment, default="#7f7f7f")
