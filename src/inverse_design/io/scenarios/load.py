"""Load experiment path bundles from ``registry.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _registry_yaml_path() -> Path:
    return Path(__file__).resolve().parent / "registry.yaml"


@lru_cache(maxsize=1)
def load_registry() -> Dict[str, Any]:
    """Parse and cache ``registry.yaml``."""
    with open(_registry_yaml_path(), encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class CombinedGridN512Paths:
    """Output directory roots for N512 combined-grid linear / dendrogram comparisons."""

    base_original_dir: str
    linear_model_dirs: Tuple[str, ...]
    dendrogram_threshold_dirs: Tuple[str, ...]
    ext_linear_dirs: Tuple[str, ...]
    ext_dendrogram_threshold_dirs: Tuple[str, ...]


def combined_grid_n512_breast() -> CombinedGridN512Paths:
    """Paths shared by ``ABCMetricsComparison`` and ``parameter_importance`` demo blocks."""
    exp = load_registry()["experiments"]["combined_grid_n512_breast"]
    return CombinedGridN512Paths(
        base_original_dir=exp["base_original_dir"],
        linear_model_dirs=tuple(exp["linear_model_dirs"]),
        dendrogram_threshold_dirs=tuple(exp["dendrogram_threshold_dirs"]),
        ext_linear_dirs=tuple(exp["ext_linear_dirs"]),
        ext_dendrogram_threshold_dirs=tuple(exp["ext_dendrogram_threshold_dirs"]),
    )
