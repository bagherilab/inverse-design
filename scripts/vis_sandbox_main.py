"""Standalone script: generate posterior visualisation plots.

Previously the ``main()`` block inside ``vis/sandbox.py``.  Run from the repo
root after setting PYTHONPATH=src:

    python scripts/vis_sandbox_main.py
"""

import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_design.analyze.abc_metrics_comparison import analyze_iterations
from inverse_design.vis.vis_posterior_overview import plot_marginal_distributions, plot_pca_with_peaks
from inverse_design.vis.vis_parameter_comparison import plot_multi_boxcharts
from inverse_design.vis.utils import find_best_sample_for_metric


def main():
    warnings.filterwarnings("ignore")

    base_dir = "../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2"
    prior_df = pd.read_csv(f"{base_dir}/iter_0/all_param_df.csv")
    posterior_df = pd.read_csv(f"{base_dir}/iter_4/all_param_df.csv")
    final_metrics_df = pd.read_csv(f"{base_dir}/iter_4/final_metrics.csv")
    target_metrics = json.load(open(f"{base_dir}/targets.json"))

    drop_cols = ["input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"]
    prior_df.drop(columns=drop_cols, inplace=True)
    posterior_df.drop(columns=drop_cols, inplace=True)
    metrics_names = list(target_metrics.keys())
    best_sample_idx = find_best_sample_for_metric(final_metrics_df, target_metrics)

    os.makedirs(f"{base_dir}/posterior_plots", exist_ok=True)
    print("Generating visualisation plots...")

    marginal_fig = plot_marginal_distributions(posterior_df, best_sample_idx)
    plt.figure(marginal_fig.number)
    plt.savefig(f"{base_dir}/posterior_plots/marginal_distributions.png", dpi=300, bbox_inches="tight")

    plot_pca_with_peaks(
        posterior_df, n_components=2, save_path=f"{base_dir}/posterior_plots/pca_with_peaks.png"
    )
    print("All visualisations saved.")

    summary_metrics_paths = [f"{base_dir}/POSTERIOR_OUTPUTS/mean/iter_0/final_metrics_seed.csv"]
    summary_metrics_paths.append(f"{base_dir}/POSTERIOR_OUTPUTS/mode/iter_0/final_metrics_seed.csv")
    summary_metrics_paths.extend(
        [f"{base_dir}/POSTERIOR_OUTPUTS/peak_{i}/iter_0/final_metrics_seed.csv" for i in range(1, 5)]
    )
    scenario_names = ["Exp", "Mean", "Mode", "Peak 1", "Peak 2", "Peak 3", "Peak 4"]
    target_df = pd.DataFrame(list(target_metrics.items())).T
    target_df.columns = target_df.iloc[0]
    target_df = target_df.iloc[1:]
    plot_multi_boxcharts(
        target_df,
        summary_metrics_paths,
        scenario_names,
        save_path=f"{base_dir}/posterior_plots/pca_metrics_comparison.png",
    )


if __name__ == "__main__":
    main()
