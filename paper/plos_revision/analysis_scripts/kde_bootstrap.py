#!/usr/bin/env python3
"""Bootstrap recovery of the frozen unweighted Fig 3 density maxima."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bandwidth_stability_peaks import PAPER, detect_peaks, scott_factor


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO
    / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50/iter_4"
)
DEFAULT_ASSIGNMENTS = REPO / "analysis_outputs/pm50_fig3/canonical_peak_assignments.csv"
DEFAULT_RESULTS = REPO / "analysis_outputs/pm50_fig3/canonical_kde_bootstrap.csv"
DEFAULT_PROVENANCE = REPO / "analysis_outputs/pm50_fig3/canonical_kde_bootstrap_provenance.json"
B = 200
TOL = 1.0
SEED = 20260820


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def aligned_weights(run: Path, assignments: pd.DataFrame) -> np.ndarray:
    table = pd.read_csv(run / "weights.csv").set_index("input_index")["weight"]
    indexes = assignments.input_folder.str.extract(r"(\d+)").iloc[:, 0].astype(int)
    weights = indexes.map(table).fillna(0.0).to_numpy(float)
    if weights.sum() <= 0:
        raise RuntimeError("no positive saved random-forest weights")
    return weights / weights.sum()


def recovery(
    pc: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    scheme: str,
    factor: float,
    replicates: int,
) -> tuple[list[dict[str, float | int | str]], np.ndarray]:
    probabilities = weights if scheme == "weight-proportional" else None
    found: list[list[np.ndarray]] = [[] for _ in PAPER]
    peak_counts = []
    for replicate in range(replicates):
        index = rng.choice(len(pc), size=len(pc), replace=True, p=probabilities)
        centres, _ = detect_peaks(pc[index], factor=factor)
        peak_counts.append(len(centres))
        for peak, target in enumerate(PAPER, start=1):
            if len(centres) == 0:
                continue
            distances = np.linalg.norm(centres - target, axis=1)
            nearest = int(np.argmin(distances))
            if distances[nearest] <= TOL:
                found[peak - 1].append(centres[nearest])
        if (replicate + 1) % 25 == 0:
            print(f"{scheme}: completed {replicate + 1}/{replicates}", flush=True)

    rows: list[dict[str, float | int | str]] = []
    for peak, centres in enumerate(found, start=1):
        values = np.asarray(centres, dtype=float)
        rows.append(
            {
                "scheme": scheme,
                "peak": peak,
                "recovered": len(centres),
                "replicates": replicates,
                "recovery_fraction": len(centres) / replicates,
                "centroid_sd_pc1": float(values[:, 0].std()) if len(values) else np.nan,
                "centroid_sd_pc2": float(values[:, 1].std()) if len(values) else np.nan,
            }
        )
    return rows, np.asarray(peak_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--replicates", type=int, default=B)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()

    assignments = pd.read_csv(args.assignments)
    params = pd.read_csv(args.run / "all_param_df.csv")
    if assignments.input_folder.tolist() != params.input_folder.astype(str).tolist():
        raise RuntimeError("assignment rows are not aligned to parent parameter rows")
    pc = assignments[["pc1", "pc2"]].to_numpy(float)
    weights = aligned_weights(args.run, assignments)
    factor = scott_factor(len(pc))

    rows = []
    peak_count_summary = {}
    for scheme in ("uniform", "weight-proportional"):
        scheme_rows, peak_counts = recovery(
            pc, weights, np.random.default_rng(SEED), scheme, factor, args.replicates
        )
        rows.extend(scheme_rows)
        peak_count_summary[scheme] = {
            "mean": float(peak_counts.mean()),
            "min": int(peak_counts.min()),
            "max": int(peak_counts.max()),
        }

    result = pd.DataFrame(rows)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.results, index=False)
    provenance = {
        "version": 1,
        "assignment_convention": "frozen Fig 3 unweighted PCA/KDE",
        "assignments": str(args.assignments.relative_to(REPO)),
        "assignments_sha256": sha256(args.assignments),
        "params": str((args.run / "all_param_df.csv").relative_to(REPO)),
        "params_sha256": sha256(args.run / "all_param_df.csv"),
        "weights": str((args.run / "weights.csv").relative_to(REPO)),
        "weights_sha256": sha256(args.run / "weights.csv"),
        "detector": {"grid": 100, "maximum_filter": 5, "relative_density_threshold": 0.05},
        "scott_covariance_factor": factor,
        "matching_tolerance_pc_units": TOL,
        "replicates_per_scheme": args.replicates,
        "random_seed": SEED,
        "peak_count_summary": peak_count_summary,
        "results": str(args.results.relative_to(REPO)),
        "results_sha256": sha256(args.results),
        "script": str(Path(__file__).relative_to(REPO)),
        "script_sha256": sha256(Path(__file__)),
        "command": "python analysis_scripts/kde_bootstrap.py --replicates 200",
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
