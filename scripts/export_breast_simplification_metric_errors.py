#!/usr/bin/env python3
"""Export breast experimental percent errors for full and simplified ARCADE models."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from combine_simplification_metrics import (
    DEFAULT_CLUSTER_DIR,
    DEFAULT_OUTPUT_DIR,
    EXP_TARGETS,
    LEVEL_LABELS,
    SIMP_METRICS,
    _add_std_columns,
    _drop_iqr_outliers,
    _orig_peak_csv_path,
    _signed_percent_error,
    _simp_run_dir,
)

MODEL_LEVELS = [
    ("full", "Full", "full", None, None),
    ("r09", LEVEL_LABELS["r09"], "r", "linear", "0.9"),
    ("r08", LEVEL_LABELS["r08"], "r", "linear", "0.8"),
    ("r07", LEVEL_LABELS["r07"], "r", "linear", "0.7"),
    ("t10", LEVEL_LABELS["t10"], "t", "dendrogram", "1.0"),
    ("t125", LEVEL_LABELS["t125"], "t", "dendrogram", "1.25"),
    ("t15", LEVEL_LABELS["t15"], "t", "dendrogram", "1.5"),
]
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "breast_simplification_metric_error_percentages.json"


def _finite_or_none(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value


def _round(value: Any, digits: int = 6) -> float | int | str | None:
    value = _finite_or_none(value)
    if isinstance(value, float):
        return round(value, digits)
    return value


def _metric_record(
    csv_path: Path,
    model_id: str,
    model_label: str,
    comparison_set: str,
    cluster_k: int,
    metric: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "model_id": model_id,
        "model": model_label,
        "comparison_set": comparison_set,
        "cluster": f"cluster_{cluster_k}",
        "cluster_index": cluster_k,
        "metric": metric,
        "target": _round(EXP_TARGETS[metric]),
        "source_csv": str(csv_path),
        "available": False,
    }
    if not csv_path.exists():
        record["missing_reason"] = "source_csv_missing"
        return record

    df = _add_std_columns(pd.read_csv(csv_path))
    if metric not in df.columns:
        record["missing_reason"] = "metric_column_missing"
        return record

    values = df[metric].replace([np.inf, -np.inf], np.nan).dropna()
    values_filtered = _drop_iqr_outliers(values)
    if len(values_filtered) == 0:
        record["missing_reason"] = "no_finite_values_after_iqr_filter"
        record["n_raw"] = int(len(values))
        record["n_used"] = 0
        return record

    target = EXP_TARGETS[metric]
    q25 = float(values_filtered.quantile(0.25))
    median = float(values_filtered.median())
    q75 = float(values_filtered.quantile(0.75))
    signed_error = _signed_percent_error(median, target)

    record.update(
        {
            "available": True,
            "n_raw": int(len(values)),
            "n_used": int(len(values_filtered)),
            "value_q25": _round(q25),
            "value_median": _round(median),
            "value_q75": _round(q75),
            "signed_error_percent_q25": _round(_signed_percent_error(q25, target)),
            "signed_error_percent_median": _round(signed_error),
            "signed_error_percent_q75": _round(_signed_percent_error(q75, target)),
            "absolute_error_percent_median": _round(abs(signed_error)),
        }
    )
    return record


def _csv_path_for_model(
    cluster_dir: Path,
    model_id: str,
    series: str | None,
    threshold: str | None,
    cluster_k: int,
) -> Path:
    if model_id == "full":
        return _orig_peak_csv_path(cluster_dir, cluster_k)
    if series is None or threshold is None:
        raise ValueError(f"Missing series/threshold for model {model_id}")
    run_dir = _simp_run_dir(series, threshold, cluster_k)
    if run_dir is None:
        run_name = f"ABC_SMC_RF_N512_combined_grid_{series}_{threshold}_p{cluster_k}_mean_only"
        return Path("/missing") / run_name / run_name / "iter_4" / "final_metrics_seed.csv"
    return run_dir / "final_metrics_seed.csv"


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _build_payload(cluster_dir: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for model_id, model_label, comparison_set, series, threshold in MODEL_LEVELS:
        for cluster_k in range(1, 5):
            csv_path = _csv_path_for_model(cluster_dir, model_id, series, threshold, cluster_k)
            for metric in SIMP_METRICS:
                records.append(
                    _metric_record(
                        csv_path,
                        model_id,
                        model_label,
                        comparison_set,
                        cluster_k,
                        metric,
                    )
                )

    cluster_summary = []
    for model_id, model_label, comparison_set, *_ in MODEL_LEVELS:
        for cluster_k in range(1, 5):
            rows = [
                row
                for row in records
                if row["model_id"] == model_id
                and row["cluster_index"] == cluster_k
                and row.get("available")
            ]
            errors = [row["absolute_error_percent_median"] for row in rows]
            cluster_summary.append(
                {
                    "model_id": model_id,
                    "model": model_label,
                    "comparison_set": comparison_set,
                    "cluster": f"cluster_{cluster_k}",
                    "cluster_index": cluster_k,
                    "available_metric_count": len(rows),
                    "mean_absolute_error_percent": _round(_mean(errors)),
                }
            )

    model_summary = []
    for model_id, model_label, comparison_set, *_ in MODEL_LEVELS:
        rows = [row for row in records if row["model_id"] == model_id and row.get("available")]
        by_metric = {}
        for metric in SIMP_METRICS:
            metric_rows = [row for row in rows if row["metric"] == metric]
            by_metric[metric] = {
                "mean_signed_error_percent": _round(
                    _mean([row["signed_error_percent_median"] for row in metric_rows])
                ),
                "mean_absolute_error_percent": _round(
                    _mean([row["absolute_error_percent_median"] for row in metric_rows])
                ),
                "available_cluster_count": len(metric_rows),
            }
        model_summary.append(
            {
                "model_id": model_id,
                "model": model_label,
                "comparison_set": comparison_set,
                "available_record_count": len(rows),
                "overall_mean_absolute_error_percent": _round(
                    _mean([row["absolute_error_percent_median"] for row in rows])
                ),
                "metrics": by_metric,
            }
        )

    return {
        "metadata": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "cluster_dir": str(cluster_dir),
            "metric_source": "final_metrics_seed.csv",
            "summary_statistic": "median after 1.5-IQR outlier filtering",
            "signed_error_percent_formula": (
                "(model_metric - breast_exp_target) / abs(breast_exp_target) * 100"
            ),
            "absolute_error_percent_formula": "abs(signed_error_percent)",
        },
        "breast_exp_targets": {metric: EXP_TARGETS[metric] for metric in SIMP_METRICS},
        "metric_order": SIMP_METRICS,
        "model_order": [level[1] for level in MODEL_LEVELS],
        "records": records,
        "cluster_summary": cluster_summary,
        "model_summary": model_summary,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export percent-error metrics for breast full/r/t simplified ARCADE models."
    )
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON_PATH)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.cluster_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.output_json}")
    print(f"Records: {len(payload['records'])}")


if __name__ == "__main__":
    main()
