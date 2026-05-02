"""Helpers for building and consuming visualization cache payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from inverse_design.io.scenarios.load import CombinedGridN512Paths


def select_simplified_dirs(
    paths: CombinedGridN512Paths, simplified_method: str, peak_idx: int
) -> List[str]:
    """Return the 4 simplified run directories used by vis_simplified."""
    if simplified_method not in {"linear", "threshold"}:
        raise ValueError("simplified_method must be 'linear' or 'threshold'")

    if simplified_method == "linear":
        selected = list(paths.linear_model_dirs[peak_idx * 3 : (peak_idx + 1) * 3])
        selected.extend(paths.ext_linear_dirs[peak_idx : peak_idx + 1])
        return selected

    selected = list(paths.dendrogram_threshold_dirs[peak_idx * 3 : (peak_idx + 1) * 3])
    selected.extend(paths.ext_dendrogram_threshold_dirs[peak_idx : peak_idx + 1])
    return selected


def cache_payload_path(cache_root: Path, simplified_method: str, peak_idx: int) -> Path:
    """Path for a specific cache payload JSON."""
    return cache_root / f"{simplified_method}_p{peak_idx + 1}.json"


def load_cache_payload(cache_root: Path, simplified_method: str, peak_idx: int) -> Dict[str, Any]:
    """Read cache payload JSON for vis_simplified."""
    payload_file = cache_payload_path(cache_root, simplified_method, peak_idx)
    with open(payload_file, encoding="utf-8") as f:
        return json.load(f)
