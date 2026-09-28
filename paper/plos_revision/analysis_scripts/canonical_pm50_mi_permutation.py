#!/usr/bin/env python3
"""Fixed-size MI permutation test for the canonical Fig 3 P1--P4 partition."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from inverse_design.figures.combine_simplification_mi_table import _compute_mi, SIMP_METRICS


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO_ROOT
    / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50/iter_4"
)
DEFAULT_ASSIGNMENTS = REPO_ROOT / "analysis_outputs/pm50_fig3/canonical_peak_assignments.csv"
DEFAULT_SUMMARY = REPO_ROOT / "analysis_outputs/pm50_fig3/canonical_mi_permutation.csv"
DEFAULT_NULL = REPO_ROOT / "analysis_outputs/pm50_fig3/canonical_mi_permutation_null.csv"
DEFAULT_PROVENANCE = REPO_ROOT / "analysis_outputs/pm50_fig3/canonical_mi_permutation_provenance.json"
DOMINANT = 0.7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def r_theta(level: dict) -> dict[str, float]:
    return {
        parameter: float(np.mean([level[parameter][metric]["mi"] for metric in SIMP_METRICS]))
        for parameter in level
    }


def group_profiles(
    params: pd.DataFrame, metrics: pd.DataFrame, labels: np.ndarray
) -> tuple[np.ndarray, list[str]]:
    profiles: list[dict[str, float]] = []
    columns: list[str] | None = None
    for peak in range(1, 5):
        selected = params.loc[labels == peak].reset_index(drop=True)
        selected_inputs = set(selected["input_folder"])
        selected_metrics = metrics[metrics["input_folder"].isin(selected_inputs)].reset_index(
            drop=True
        )
        level, _ = _compute_mi(selected, selected_metrics)
        profile = r_theta(level)
        if columns is None:
            columns = sorted(profile)
        if set(profile) != set(columns):
            raise RuntimeError(f"MI parameter columns differ for P{peak}")
        profiles.append(profile)
    assert columns is not None
    return np.array([[profile[name] for name in columns] for profile in profiles]), columns


def statistics(profiles: np.ndarray) -> dict[str, float | int]:
    spread = float(
        np.mean(
            [
                np.linalg.norm(profiles[left] - profiles[right])
                for left, right in itertools.combinations(range(len(profiles)), 2)
            ]
        )
    )
    dominant = profiles > DOMINANT
    return {
        "spread": spread,
        "exclusive": int((dominant.sum(axis=0) == 1).sum()),
        "peakedness": float(np.mean(profiles.max(axis=1))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--perms", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--null", type=Path, default=DEFAULT_NULL)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()

    params_path = args.run / "all_param_df.csv"
    metrics_path = args.run / "final_metrics_seed.csv"
    params = pd.read_csv(params_path)
    metrics = pd.read_csv(metrics_path)
    assignments = pd.read_csv(args.assignments)
    if assignments["input_folder"].tolist() != params["input_folder"].astype(str).tolist():
        raise RuntimeError("canonical assignments are not row-aligned to the pm50 parent")
    labels = assignments["peak"].to_numpy(int)
    sizes = pd.Series(labels).value_counts().sort_index().to_dict()
    if sizes != {1: 310, 2: 315, 3: 162, 4: 237}:
        raise RuntimeError(f"unexpected canonical group sizes: {sizes}")

    observed_profiles, parameters = group_profiles(params, metrics, labels)
    observed = statistics(observed_profiles)
    rng = np.random.default_rng(args.seed)
    null_rows = []
    for index in range(args.perms):
        profiles, _ = group_profiles(params, metrics, rng.permutation(labels))
        null_rows.append({"permutation": index + 1, **statistics(profiles)})
        if (index + 1) % 25 == 0:
            print(f"completed {index + 1}/{args.perms}", flush=True)

    null = pd.DataFrame(null_rows)
    summary_rows = []
    for name in ("spread", "exclusive", "peakedness"):
        values = null[name].to_numpy(float)
        value = float(observed[name])
        summary_rows.append(
            {
                "statistic": name,
                "observed": value,
                "null_mean": float(values.mean()),
                "null_sd": float(values.std()),
                "null_p95": float(np.percentile(values, 95)),
                "p_one_sided": (1 + int((values >= value).sum())) / (1 + len(values)),
                "permutations": len(values),
            }
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(args.summary, index=False)
    null.to_csv(args.null, index=False)
    provenance = {
        "version": 1,
        "assignment_convention": "Fig 3 unweighted PCA/KDE nearest maximum",
        "assignment_file": str(args.assignments.relative_to(REPO_ROOT)),
        "assignment_file_sha256": sha256(args.assignments),
        "parameter_file": str(params_path.relative_to(REPO_ROOT)),
        "parameter_file_sha256": sha256(params_path),
        "metric_file": str(metrics_path.relative_to(REPO_ROOT)),
        "metric_file_sha256": sha256(metrics_path),
        "group_sizes": {str(key): int(value) for key, value in sizes.items()},
        "parameters": parameters,
        "metrics": list(SIMP_METRICS),
        "mi": "sklearn mutual_info_regression; random_state=42; min-max normalized within metric and subset",
        "dominant_threshold": DOMINANT,
        "permutations": args.perms,
        "random_seed": args.seed,
        "summary_file": str(args.summary.relative_to(REPO_ROOT)),
        "summary_file_sha256": sha256(args.summary),
        "null_file": str(args.null.relative_to(REPO_ROOT)),
        "null_file_sha256": sha256(args.null),
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "command": (
            "MPLCONFIGDIR=/tmp/inversedesign-mpl PYTHONPATH=../inverse_design/src "
            "python analysis_scripts/canonical_pm50_mi_permutation.py --perms 200"
        ),
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"saved {args.summary}, {args.null}, and {args.provenance}")


if __name__ == "__main__":
    main()
