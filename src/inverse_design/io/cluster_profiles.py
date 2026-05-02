"""Load and normalize PCA cluster profile summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from inverse_design.config.parameter_config import PARAM_RANGES

_PROFILE_SKIP_KEYS = {
    "label",
    "kind",
    "cluster",
    "cluster_size",
    "cluster_fraction",
    "pc1",
    "pc2",
    "nearest_sample_index",
    "nearest_sample_pc_distance",
}


def load_targets(base_dir: Path, targets_json: Path | None = None) -> dict[str, Any]:
    """Load a target metrics JSON file from an explicit path or a run directory."""
    targets_path = targets_json or (base_dir / "targets.json")
    with open(targets_path, encoding="utf-8") as target_file:
        return json.load(target_file)


def _load_summary_entries(summary_json: Path) -> list[dict[str, Any]]:
    with open(summary_json, encoding="utf-8") as fh:
        return json.load(fh)


def load_cluster_profiles(summary_json: Path) -> dict[int, dict[str, float]]:
    """Load 0-based PCA peak inverse profiles from a summary JSON file."""
    profiles: dict[int, dict[str, float]] = {}
    for entry in _load_summary_entries(summary_json):
        if entry.get("kind") != "pca_peak_inverse":
            continue
        idx = int(entry["cluster"]) - 1
        profiles[idx] = {k: v for k, v in entry.items() if k not in _PROFILE_SKIP_KEYS}
    return profiles


def load_cluster_centroids(summary_json: Path) -> dict[int, tuple[float, float]]:
    """Return one-indexed cluster centroid coordinates from a summary JSON file."""
    centroids: dict[int, tuple[float, float]] = {}
    for entry in _load_summary_entries(summary_json):
        if entry.get("kind") != "pca_peak_inverse":
            continue
        cluster_k = int(entry["cluster"])
        centroids[cluster_k] = (float(entry["pc1"]), float(entry["pc2"]))
    return centroids


def normalize_profiles(
    profiles: dict[int, dict[str, float]],
    param_ranges: dict[str, tuple[float, float]] = PARAM_RANGES,
) -> dict[int, dict[str, float]]:
    """Map profile values to the normalized parameter range scale."""
    norm: dict[int, dict[str, float]] = {}
    for idx, vals in profiles.items():
        norm[idx] = {}
        for param, value in vals.items():
            if param not in param_ranges:
                continue
            lo, hi = param_ranges[param]
            if hi == lo:
                continue
            norm[idx][param] = float(np.clip((value - lo) / (hi - lo), -0.05, 1.05))
    return norm
