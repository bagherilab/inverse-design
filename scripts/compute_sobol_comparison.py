#!/usr/bin/env python3
"""Compute Sobol total-order indices and compare them with MI rankings."""

from pathlib import Path
import os

import numpy as np
from scipy.stats import spearmanr

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from inverse_design.analyze.sensitivity_analysis import (
    perform_mi_analysis,
    perform_sobol_analysis,
)
from inverse_design.io.posterior import load_posterior_data

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/N1024_breast/iter_4"
TARGET_METRICS = ["doub_time", "symmetry", "colony_growth"]
DROP_COLS = ["X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
SOBOL_JSON = ROOT / "data/sobol_results.json"
MI_JSON = ROOT / "data/mi_results_k3.json"


def main():
    param_df, metrics_df, param_names = load_posterior_data(DATA_DIR, TARGET_METRICS, DROP_COLS)
    SOBOL_JSON.parent.mkdir(parents=True, exist_ok=True)
    MI_JSON.parent.mkdir(parents=True, exist_ok=True)

    print(f"Posterior shape: {param_df.shape} (N samples x D params)")
    print("Training RF surrogates and computing Sobol indices...")

    sobol_results = perform_sobol_analysis(
        param_df,
        metrics_df,
        param_names,
        calc_second_order=False,
        save_sobol_json=SOBOL_JSON,
        plot_performance=False,
    )
    mi_results = perform_mi_analysis(
        param_df,
        metrics_df,
        param_names,
        n_neighbors=3,
        save_mi_json=MI_JSON,
    )

    print("\n=== Sobol ST vs MI ranking comparison ===")
    for metric in TARGET_METRICS:
        st_scores = np.array(sobol_results[metric]["ST"])
        mi_scores = mi_results[metric]["MI"]
        rho, p_value = spearmanr(st_scores, mi_scores)
        sobol_top5 = [param_names[i] for i in np.argsort(-st_scores)[:5]]
        mi_top5 = [param_names[i] for i in np.argsort(-mi_scores)[:5]]

        print(f"\n{metric}:")
        print(f"  Spearman rho (Sobol ST vs MI): {rho:.4f}  p={p_value:.4e}")
        print(f"  Sobol top-5: {sobol_top5}")
        print(f"  MI    top-5: {mi_top5}")

    print(f"\nSaved Sobol results to {SOBOL_JSON}")
    print(f"Saved MI results to {MI_JSON}")


if __name__ == "__main__":
    main()
