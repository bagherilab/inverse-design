"""Shared posterior loaders for sensitivity scripts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inverse_design.analyze.sensitivity_analysis import (
    align_dataframes,
    load_data,
    remove_nan_rows,
)


def load_posterior_data(
    data_dir: Path,
    target_metrics: list[str],
    drop_cols=(),
):
    """Load, align, clean, and optionally trim posterior parameter/metric frames."""
    param_df, metrics_df = load_data(
        data_dir / "all_param_df.csv",
        data_dir / "final_metrics.csv",
        target_metrics + ["input_folder"],
    )
    param_df, metrics_df = align_dataframes(param_df, metrics_df)
    metrics_df = metrics_df.replace([np.inf, -np.inf], np.nan)
    param_df, metrics_df = remove_nan_rows(param_df, metrics_df)
    param_df = param_df.drop(columns=list(drop_cols), errors="ignore")
    return param_df, metrics_df, param_df.columns.tolist()
