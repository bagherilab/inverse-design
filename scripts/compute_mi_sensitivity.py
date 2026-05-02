#!/usr/bin/env python3
"""Compute MI rankings for k=3, 5, and 10 and report rank stability."""

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from inverse_design.analyze.sensitivity_analysis import (
    perform_mi_analysis,
)
from inverse_design.io.posterior import load_posterior_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/N1024_breast/iter_4"
TARGET_METRICS = ["doub_time", "symmetry", "colony_growth"]
DROP_COLS = ["X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
K_VALUES = [3, 5, 10]
OUT_JSON = ROOT / "data/mi_sensitivity_results.json"


def main():
    param_df, metrics_df, param_names = load_posterior_data(DATA_DIR, TARGET_METRICS, DROP_COLS)

    results = {
        k: perform_mi_analysis(param_df, metrics_df, param_names, n_neighbors=k)
        for k in K_VALUES
    }

    output = {}
    for metric in TARGET_METRICS:
        print(f"\n=== {metric} ===")
        output[metric] = {}

        for k in K_VALUES:
            scores = results[k][metric]["MI"]
            top5 = [param_names[i] for i in np.argsort(-scores)[:5]]
            print(f"  k={k:2d}: {top5}")
            output[metric][f"k{k}"] = {"MI": scores.tolist(), "names": param_names}

        output[metric]["spearman"] = {}
        print()
        for i, k1 in enumerate(K_VALUES):
            for k2 in K_VALUES[i + 1 :]:
                scores_1 = results[k1][metric]["MI"]
                scores_2 = results[k2][metric]["MI"]
                rho, p_value = spearmanr(scores_1, scores_2)
                pair_key = f"k{k1}_vs_k{k2}"
                output[metric]["spearman"][pair_key] = {"rho": float(rho), "p": float(p_value)}
                print(f"  Spearman rho (k={k1} vs k={k2}): {rho:.4f}  p={p_value:.4e}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved full results to {OUT_JSON}")


if __name__ == "__main__":
    main()
