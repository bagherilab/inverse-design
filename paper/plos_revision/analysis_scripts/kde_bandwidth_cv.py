#!/usr/bin/env python3
"""LOO-CV covariance factors for the frozen unweighted Fig 3 PCA cloud."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.special import logsumexp

from bandwidth_stability_peaks import PAPER, detect_peaks, scott_factor
from kde_bootstrap import aligned_weights


REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUN = (
    REPO
    / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50/iter_4"
)
DEFAULT_ASSIGNMENTS = REPO / "analysis_outputs/pm50_fig3/canonical_peak_assignments.csv"
DEFAULT_RESULTS = REPO / "analysis_outputs/pm50_fig3/canonical_kde_bandwidth_cv.csv"
DEFAULT_PROVENANCE = REPO / "analysis_outputs/pm50_fig3/canonical_kde_bandwidth_cv_provenance.json"
COARSE_FACTORS = np.round(np.arange(0.01, 0.501, 0.005), 4)
REFINE_HALF_WIDTH = 0.015
REFINE_STEP = 0.0005


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def whiten(pc: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    cov = np.cov(pc, rowvar=False, aweights=weights)
    return solve_triangular(cholesky(cov, lower=True), pc.T, lower=True).T


def loo_log_likelihood(
    z: np.ndarray, factors: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    n, dimension = z.shape
    squared = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(squared, np.inf)
    w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    with np.errstate(divide="ignore"):
        log_w = np.log(w)
    active = w > 0
    scores = []
    for factor in factors:
        log_kernel = (
            -0.5 * squared / factor**2
            - dimension * np.log(factor)
            - 0.5 * dimension * np.log(2 * np.pi)
        )
        log_density = logsumexp(log_kernel + log_w[None, :], axis=1)
        log_density -= np.log1p(-w)
        scores.append(float((w[active] * log_density[active]).sum()))
    return np.asarray(scores)


def select_factor(z: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float]:
    coarse_scores = loo_log_likelihood(z, COARSE_FACTORS, weights)
    coarse_best = float(COARSE_FACTORS[int(np.argmax(coarse_scores))])
    lower = max(REFINE_STEP, coarse_best - REFINE_HALF_WIDTH)
    fine = np.round(
        np.arange(lower, coarse_best + REFINE_HALF_WIDTH + REFINE_STEP / 2, REFINE_STEP), 4
    )
    fine_scores = loo_log_likelihood(z, fine, weights)
    index = int(np.argmax(fine_scores))
    return float(fine[index]), float(fine_scores[index])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()

    assignments = pd.read_csv(args.assignments)
    params = pd.read_csv(args.run / "all_param_df.csv")
    if assignments.input_folder.tolist() != params.input_folder.astype(str).tolist():
        raise RuntimeError("assignment rows are not aligned to parent parameter rows")
    pc = assignments[["pc1", "pc2"]].to_numpy(float)
    weights = aligned_weights(args.run, assignments)
    scott = scott_factor(len(pc))
    rows = [{"selection_rule": "Scott", "selected_h": scott, "relative_to_scott": 1.0}]
    for label, current_weights in (("LOO-CV, equal weights", None), ("LOO-CV, saved RF weights", weights)):
        z = whiten(pc, current_weights)
        best, score = select_factor(z, current_weights)
        centres, _ = detect_peaks(pc, factor=best, weights=current_weights)
        recovered = sum(
            len(centres) > 0 and np.linalg.norm(centres - target, axis=1).min() <= 1.0
            for target in PAPER
        )
        rows.append(
            {
                "selection_rule": label,
                "selected_h": best,
                "relative_to_scott": best / scott,
                "loo_log_likelihood": score,
                "detected_maxima": len(centres),
                "canonical_peaks_recovered": recovered,
            }
        )
    results = pd.DataFrame(rows)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results, index=False)
    provenance = {
        "version": 1,
        "assignment_convention": "frozen Fig 3 unweighted PCA/KDE",
        "assignments": str(args.assignments.relative_to(REPO)),
        "assignments_sha256": sha256(args.assignments),
        "params": str((args.run / "all_param_df.csv").relative_to(REPO)),
        "params_sha256": sha256(args.run / "all_param_df.csv"),
        "weights": str((args.run / "weights.csv").relative_to(REPO)),
        "weights_sha256": sha256(args.run / "weights.csv"),
        "results": str(args.results.relative_to(REPO)),
        "results_sha256": sha256(args.results),
        "script": str(Path(__file__).relative_to(REPO)),
        "script_sha256": sha256(Path(__file__)),
        "command": "python analysis_scripts/kde_bandwidth_cv.py",
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
