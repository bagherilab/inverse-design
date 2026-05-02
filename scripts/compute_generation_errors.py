#!/usr/bin/env python3
"""Compute generation-wise total error and report gen3->gen4 percent change."""

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from inverse_design.analyze.abc_metrics_comparison import analyze_iterations
from inverse_design.analyze.utils.analyze_utils import calculate_percentage_error

SIR_DIR = Path("/home/pohaoc2/UW/bagherilab/SIR_OUTPUT/n3_t365_l30/n512")
SIR_METRICS = ["peak_I", "time_to_peak", "final_R", "area_I", "growth_rate"]
SIR_TARGETS = np.array([0.837117, 33.378653, 0.832338, 0.448831, 0.006877])

ARCADE_DIR = Path(
    "/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/"
    "ABC_SMC_RF_N512_combined_grid_breast"
)
ARCADE_METRICS = ["doub_time", "doub_time_std", "symmetry", "symmetry_std", "colony_growth"]
ARCADE_TARGETS = np.array([45.5, 13.79, 0.806, 0.067, 18.3])


def _require_iteration_files(folder, n_iters):
    missing = [
        folder / f"iter_{iteration}" / "final_metrics.csv" for iteration in range(n_iters)
        if not (folder / f"iter_{iteration}" / "final_metrics.csv").exists()
    ]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required final_metrics.csv files:\n{missing_list}")


def total_error_per_iter(folder, metric_names, targets, n_iters):
    _require_iteration_files(folder, n_iters)
    _, _, _, all_sim_metrics, _, _ = analyze_iterations(
        folder,
        metric_names,
        targets,
        n_iterations=n_iters,
        verbose=False,
        save_plots=False,
    )

    errors = []
    for metrics in all_sim_metrics:
        means = np.nanmean(metrics, axis=0)
        pct_errors = [
            calculate_percentage_error(means[i], targets[i]) for i in range(len(targets))
        ]
        errors.append(float(np.mean(pct_errors)))
    return errors


def pct_change(errors, gen_from, gen_to):
    return (errors[gen_from] - errors[gen_to]) / errors[gen_from] * 100


def _change_phrase(change):
    direction = "decreased" if change >= 0 else "increased"
    return f"{direction} by {abs(change):.0f}%"


def _interpretation(sir_change, arcade_change):
    if sir_change >= 0 and arcade_change >= 0:
        return "supporting convergence."
    return (
        "so generation-4 stopping should be described as a practical plateau rather than "
        "a monotonic error reduction."
    )


def _print_error_table(label, errors):
    for i, error in enumerate(errors):
        print(f"  {label} iter_{i}: {error:.2f}%")


def main():
    print("Computing SIR N=512 errors across 10 iterations...")
    sir_errors = total_error_per_iter(SIR_DIR, SIR_METRICS, SIR_TARGETS, n_iters=10)
    _print_error_table("SIR", sir_errors)
    sir_pct = pct_change(sir_errors, 3, 4)
    print(
        f"  => SIR gen3->gen4: {sir_errors[3]:.2f}% -> {sir_errors[4]:.2f}% "
        f"(Delta = {sir_pct:.1f}%)\n"
    )

    print("Computing ARCADE N=512 errors across 9 iterations...")
    arcade_errors = total_error_per_iter(ARCADE_DIR, ARCADE_METRICS, ARCADE_TARGETS, n_iters=9)
    _print_error_table("ARCADE", arcade_errors)
    arcade_pct = pct_change(arcade_errors, 3, 4)
    print(
        f"  => ARCADE gen3->gen4: {arcade_errors[3]:.2f}% -> {arcade_errors[4]:.2f}% "
        f"(Delta = {arcade_pct:.1f}%)"
    )

    print("\n--- Insert into results.tex lines 18-19 ---")
    print(
        f"The total error {_change_phrase(sir_pct)} (SIR) and {_change_phrase(arcade_pct)} "
        f"from generation 3 to generation 4, {_interpretation(sir_pct, arcade_pct)}"
    )


if __name__ == "__main__":
    main()
