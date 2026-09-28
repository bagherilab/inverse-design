#!/usr/bin/env python3
"""Close the reviewer seed-precision and Full symmetry-SD provenance checks.

The ARCADE aggregation code reports the median and population SD (``ddof=0``)
over ten seeds.  For each parameter set and metric, this script exhaustively
enumerates every 3-of-10 and 5-of-10 subset, compares the subset statistic with
the corresponding 10-seed statistic, and summarizes the absolute error.  Only
parameter sets with ten finite values for a metric enter its matched comparison.

It also reconstructs the two historical Full symmetry-SD error aggregates from
``body/metric_err_pct.json``.  The 29.9% value averages four numerical peaks;
36.95% averages peaks 1--3. Both use one representative Full parameter set per
peak and therefore document aggregation provenance only; the estimand-matched
ensemble comparison is computed separately by ``analysis_scripts/matched.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from reviewer_figure_style import (
    OKABE_ITO,
    PLOS_DOUBLE_WIDTH,
    apply_plos_style,
    assert_text_inside_figure,
    four_spines,
    in_axis_label,
    panel_label,
    save_figure,
)


REPO = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = REPO / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50/iter_4/final_metrics_seed.csv"
DEFAULT_ERRORS = REPO / "body/metric_err_pct.json"
DEFAULT_OUT = REPO / "analysis_outputs/reviewer_closeout"
DEFAULT_FIG = REPO / "body/updated_figures/figures/reviewer_closeout"

METRICS = ("doub_time", "symmetry", "colony_growth")
STATISTICS = ("median", "std_ddof0")
METRIC_COLORS = {
    "doub_time": OKABE_ITO["blue"],
    "symmetry": OKABE_ITO["green"],
    "colony_growth": OKABE_ITO["vermillion"],
}


def statistic(values: np.ndarray, name: str) -> float:
    if name == "median":
        return float(np.median(values))
    if name == "std_ddof0":
        return float(np.std(values, ddof=0))
    raise ValueError(name)


def enumerate_errors(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact per-parameter and global summaries.

    The exhaustive design contains more than two million subset estimates.  We
    retain their exact distribution in the aggregated quantiles rather than
    materializing a multi-gigabyte row-wise table.
    """
    per_parameter: list[dict[str, float | int | str]] = []
    global_errors: dict[tuple[str, str, int], list[np.ndarray]] = {}
    eligible: dict[str, int] = {}
    for metric in METRICS:
        groups = []
        for folder, group in seed_df.groupby("input_folder", sort=False):
            values = group[metric].to_numpy(float)
            if len(values) == 10 and np.isfinite(values).all():
                groups.append((folder, values))
        eligible[metric] = len(groups)
        combinations = {n: list(itertools.combinations(range(10), n)) for n in (3, 5, 10)}
        for folder, values in groups:
            for stat_name in STATISTICS:
                reference = statistic(values, stat_name)
                for n_seeds, choices in combinations.items():
                    estimates = np.fromiter(
                        (statistic(values[list(choice)], stat_name) for choice in choices),
                        dtype=float,
                        count=len(choices),
                    )
                    absolute = np.abs(estimates - reference)
                    relative = absolute / abs(reference) if reference != 0 else np.full_like(absolute, np.nan)
                    global_errors.setdefault((metric, stat_name, n_seeds), []).append(
                        np.column_stack([absolute, relative])
                    )
                    per_parameter.append(
                        {
                            "input_folder": folder,
                            "metric": metric,
                            "statistic": stat_name,
                            "n_seeds": n_seeds,
                            "subset_estimates": len(choices),
                            "reference_n10": reference,
                            "median_absolute_error": float(np.median(absolute)),
                            "q95_absolute_error": float(np.quantile(absolute, 0.95)),
                            "median_absolute_relative_error": float(np.nanmedian(relative)),
                            "q95_absolute_relative_error": float(np.nanquantile(relative, 0.95)),
                        }
                    )

    per_parameter_df = pd.DataFrame(per_parameter)
    summary_rows = []
    for (metric, stat_name, n_seeds), chunks in global_errors.items():
        values = np.concatenate(chunks, axis=0)
        summary_rows.append(
            {
                "metric": metric,
                "statistic": stat_name,
                "n_seeds": n_seeds,
                "eligible_parameter_sets": eligible[metric],
                "subset_estimates": len(values),
                "median_absolute_error": float(np.median(values[:, 0])),
                "q95_absolute_error": float(np.quantile(values[:, 0], 0.95)),
                "median_absolute_relative_error": float(np.nanmedian(values[:, 1])),
                "q95_absolute_relative_error": float(np.nanquantile(values[:, 1], 0.95)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    for metric, count in eligible.items():
        observed = summary.loc[summary.metric.eq(metric), "eligible_parameter_sets"].unique()
        assert len(observed) == 1 and observed[0] == count
    return per_parameter_df, summary


def symmetry_provenance(error_json: Path) -> tuple[pd.DataFrame, dict[str, float | list[int]]]:
    source = json.loads(error_json.read_text())
    rows = [
        record
        for record in source["records"]
        if record["model_id"] == "full" and record["metric"] == "symmetry_std"
    ]
    frame = pd.DataFrame(
        {
            "cluster_index": [int(row["cluster_index"]) for row in rows],
            "absolute_error_percent": [float(row["absolute_error_percent_median"]) for row in rows],
            "source_csv": [row["source_csv"] for row in rows],
        }
    ).sort_values("cluster_index")
    four = float(frame.absolute_error_percent.mean())
    first_three = float(frame.loc[frame.cluster_index.isin([1, 2, 3]), "absolute_error_percent"].mean())
    result: dict[str, float | list[int]] = {
        "all_four_peak_indices": [1, 2, 3, 4],
        "all_four_mean_absolute_error_percent": four,
        "matched_three_peak_indices": [1, 2, 3],
        "matched_three_mean_absolute_error_percent": first_three,
        "difference_percentage_points": first_three - four,
    }
    assert np.isclose(four, 29.946585, atol=1e-6)
    assert np.isclose(first_three, 36.953588, atol=1e-6)
    return frame, result


def plot_precision(summary: pd.DataFrame, output: Path, *, split_formats: bool = False) -> None:
    apply_plos_style()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(PLOS_DOUBLE_WIDTH, 2.45),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.10},
    )
    labels = {"doub_time": "Doubling time", "symmetry": "Symmetry", "colony_growth": "Colony growth"}
    panel_metrics = {
        "median": METRICS,
        "std_ddof0": ("doub_time", "symmetry"),
    }
    panel_titles = {
        "median": "Median summaries",
        "std_ddof0": "Population-SD summaries",
    }
    for panel, stat_name in enumerate(STATISTICS):
        axis = axes[panel]
        for metric in panel_metrics[stat_name]:
            values = summary[
                summary.metric.eq(metric) & summary.statistic.eq(stat_name)
            ].sort_values("n_seeds")
            x = values.n_seeds.to_numpy()
            axis.plot(
                x,
                100 * values.median_absolute_relative_error,
                marker="o",
                color=METRIC_COLORS[metric],
            )
        axis.set_xticks([3, 5, 10])
        axis.grid(False)
        in_axis_label(axis, panel_titles[stat_name])
        four_spines(axis)
        panel_label(axis, chr(ord("A") + panel), x=-0.12, y=1.02)
    axes[0].set_ylabel("Median absolute relative error (%)")
    axes[0].set_xlabel("Seeds retained")
    for axis, stat_name in zip(axes, STATISTICS):
        selected = summary[summary.statistic.eq(stat_name) & summary.metric.isin(panel_metrics[stat_name])]
        maximum = 100 * selected.median_absolute_relative_error.max()
        axis.set_ylim(-0.03 * maximum, 1.2 * maximum)

    metric_handles = [
        Line2D([0], [0], color=METRIC_COLORS[metric], label=labels[metric])
        for metric in METRICS
    ]
    summary_handles = []
    fig.legend(
        metric_handles + summary_handles,
        [handle.get_label() for handle in metric_handles + summary_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=False,
    )
    assert_text_inside_figure(fig)
    save_figure(fig, output, split_formats=split_formats)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-csv", type=Path, default=DEFAULT_SEEDS)
    parser.add_argument("--error-json", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIG)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    seed_df = pd.read_csv(args.seed_csv)
    per_parameter, summary = enumerate_errors(seed_df)
    per_parameter.to_csv(args.out / "seed_subsampling_per_parameter.csv", index=False)
    summary.to_csv(args.out / "seed_subsampling_summary.csv", index=False)
    plot_precision(summary, args.out / "seed_subsampling_precision")
    plot_precision(summary, args.figure_dir / "seed_subsampling_precision", split_formats=True)

    provenance_rows, provenance = symmetry_provenance(args.error_json)
    provenance_rows.to_csv(args.out / "full_symmetry_sd_cluster_errors.csv", index=False)
    (args.out / "full_symmetry_sd_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    report = {
        "seed_source": str(args.seed_csv.resolve()),
        "seed_rows": int(len(seed_df)),
        "aggregation_protocol": "median and population SD (numpy.std, ddof=0), matching SimulationMetrics._aggregate_timestamp_metrics",
        "subsampling_protocol": "all C(10,3), C(10,5), and C(10,10) subsets; only groups with exactly 10 finite metric values",
        "symmetry_sd_source": str(args.error_json.resolve()),
        "symmetry_sd_provenance": provenance,
        "seed_source_sha256": hashlib.sha256(args.seed_csv.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.out / "seed_and_symmetry_closeout.json").write_text(json.dumps(report, indent=2) + "\n")
    print(summary.to_string(index=False))
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
