"""On-disk layout for ARCADE / ABC-SMC-RF run directories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

FINAL_METRICS_CSV = "final_metrics.csv"
ALL_PARAM_DF_CSV = "all_param_df.csv"
TARGETS_JSON = "targets.json"


RunRoot = Union[str, Path]


@dataclass(frozen=True)
class ArcadeRunLayout:
    """Standard filenames and iteration subfolders under one run root."""

    run_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_root", Path(self.run_root))

    @classmethod
    def from_root(cls, root: RunRoot) -> "ArcadeRunLayout":
        return cls(Path(root))

    def iter_k_dir(self, k: int) -> Path:
        return self.run_root / f"iter_{k}"

    def iteration_subdir(self, dir_postfix: str) -> Path:
        """e.g. dir_postfix ``\"iter_3\"`` → ``run_root / \"iter_3\"``."""
        return self.run_root / dir_postfix

    def final_metrics_csv(self, dir_postfix: str) -> Path:
        return self.iteration_subdir(dir_postfix) / FINAL_METRICS_CSV

    def all_param_df_csv(self, dir_postfix: str) -> Path:
        return self.iteration_subdir(dir_postfix) / ALL_PARAM_DF_CSV

    def targets_json(self) -> Path:
        return self.run_root / TARGETS_JSON
