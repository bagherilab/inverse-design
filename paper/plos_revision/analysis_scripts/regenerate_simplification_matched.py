#!/usr/bin/env python3
"""Regenerate Fig 4, S10 Fig, and S1 Data from the pm50 campaign.

The Full and reduced columns use the same estimand: metrics are summarized per
parameter vector across seeds and then summarized across parameter vectors.  A
missing generation-4 reduced run remains missing, so the upstream renderer
shows ``N/A`` rather than substituting the Full calibration.

The Full-cloud peak membership is loaded from the frozen Fig 3 unweighted
PCA/KDE assignment.  Linear-regression derived parameters are hidden using the JSON
for the same peak and correlation threshold, rather than one JSON shared by
all peaks.

The displayed pairwise-regression thresholds are 0.8, 0.75, and 0.7, ordered
from the most to the least restrictive threshold after Full.  Fig 4 displays
P1; the four-subset S10 Fig retains P1--P4, including P3.  S1 Data is rebuilt
from the matching peak-specific LR JSONs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = REPO_ROOT.parent / "inverse_design"
DEFAULT_DATA_ROOT = REPO_ROOT / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT"
DEFAULT_ASSIGNMENTS = (
    REPO_ROOT / "analysis_outputs/pm50_fig3/canonical_peak_assignments.csv"
)
DEFAULT_PROVENANCE = (
    REPO_ROOT / "analysis_outputs/pm50_fig3/canonical_assignments_provenance.json"
)
DEFAULT_FULL_OUT = (
    REPO_ROOT / "body/extended_data/figures/simplification/simplified_figure.png"
)
DEFAULT_SINGLE_OUT = (
    REPO_ROOT / "body/updated_figures/figures/simplification/simplified_figure_p1.png"
)
DEFAULT_METRIC_SUMMARY = (
    REPO_ROOT / "analysis_outputs/pm50_reduced_campaign/simplification_metric_summary.csv"
)
DEFAULT_MI_SUMMARY = (
    REPO_ROOT / "analysis_outputs/pm50_reduced_campaign/simplification_mi_summary.json"
)
DEFAULT_MI_RANK_COMPARISON = (
    REPO_ROOT / "analysis_outputs/pm50_reduced_campaign/simplification_mi_rank_comparison.csv"
)
DEFAULT_LR_EQUATIONS = REPO_ROOT / "body/extended_data/lr_equations_summary.csv"
DEFAULT_REGEN_PROVENANCE = (
    REPO_ROOT
    / "analysis_outputs/pm50_reduced_campaign/simplification_regeneration_provenance.json"
)
EXPECTED_PEAK_COUNTS = {1: 310, 2: 315, 3: 162, 4: 237}
MAIN_FIGURE_PEAK = 1

for path in (UPSTREAM / "scripts", UPSTREAM / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from inverse_design.figures import combine_simplification_metrics as metrics  # noqa: E402
from inverse_design.figures import combine_simplification_mi_table as mi  # noqa: E402

MEAN_METRICS = ("doub_time", "symmetry", "colony_growth")
LINEAR_LEVELS = [
    ("orig", None, None),
    ("r08", "linear", "0.8"),
    ("r075", "linear", "0.75"),
    ("r07", "linear", "0.7"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def install_display_levels() -> None:
    """Display linear thresholds from most to least restrictive after Full."""
    metrics.SIMP_LEVELS_R[:] = LINEAR_LEVELS
    metrics.LEVEL_LABELS.update(
        {"r07": "r=0.7", "r075": "r=0.75", "r08": "r=0.8"}
    )
    mi.ALL_LEVELS[:] = [
        *LINEAR_LEVELS,
        *mi.ALL_LEVELS[4:],
    ]
    mi.R_SERIES_KEYS[:] = ["r08", "r075", "r07"]
    mi.LEVEL_LABELS.update({"r07": "0.7", "r075": "0.75", "r08": "0.8"})


def _parameter_label(name: str) -> str:
    return name.removesuffix("_MU").replace("_", " ").title()


def write_lr_equations(path: Path, data_root: Path) -> None:
    """Write S1 Data from the JSON models used by the displayed linear arms."""
    rows: list[dict] = []
    for _level_key, _series, threshold in LINEAR_LEVELS[1:]:
        for peak_k in range(1, 5):
            run_name = (
                f"ABC_SMC_RF_N512_pm50_linear_{threshold}_p{peak_k}_mean_only"
            )
            json_path = data_root / run_name / f"lr_predictions_r{threshold}_p{peak_k}.json"
            if not json_path.is_file():
                raise FileNotFoundError(f"Missing linear-reduction provenance: {json_path}")
            with json_path.open(encoding="utf-8") as handle:
                models = json.load(handle).get("prediction_models", {}).get("models", [])
            for model in models:
                rows.append(
                    {
                        "r_threshold": float(threshold),
                        "peak": peak_k,
                        "predictor": _parameter_label(
                            model["predictor_parameter"]["name"]
                        ),
                        "target": _parameter_label(model["target_parameter"]["name"]),
                        "slope": round(float(model["linear_model"]["slope"]), 4),
                        "intercept": round(
                            float(model["linear_model"]["intercept"]), 4
                        ),
                        "R": round(
                            abs(float(model["prediction_quality"]["correlation_r"])), 3
                        ),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "r_threshold",
            "peak",
            "predictor",
            "target",
            "slope",
            "intercept",
            "R",
        ],
    ).to_csv(path, index=False)
    print(f"Saved {path}")


def _load_assignments(path: Path, parent_params: pd.DataFrame) -> pd.Series:
    assignments = pd.read_csv(path)
    required = {"input_folder", "peak"}
    if not required.issubset(assignments.columns):
        missing = ", ".join(sorted(required - set(assignments.columns)))
        raise ValueError(f"Peak assignment file is missing columns: {missing}")
    if assignments["input_folder"].duplicated().any():
        raise ValueError("Peak assignment file contains duplicate input_folder values")

    assignment_inputs = set(assignments["input_folder"])
    parent_inputs = set(parent_params["input_folder"])
    if assignment_inputs != parent_inputs:
        raise ValueError(
            "Peak assignments do not match the pm50 parent cloud: "
            f"assignments_only={len(assignment_inputs - parent_inputs)}, "
            f"parent_only={len(parent_inputs - assignment_inputs)}"
        )

    counts = assignments["peak"].value_counts().sort_index().to_dict()
    if counts != EXPECTED_PEAK_COUNTS:
        raise ValueError(f"Unexpected pm50 peak counts: {counts}")
    return assignments.set_index("input_folder")["peak"].astype(int)


def _per_set_summaries(seed_rows: pd.DataFrame) -> pd.DataFrame:
    """One row per parameter set: seed median for means and seed SD for SD targets."""
    grouped = seed_rows.groupby("input_folder")
    summary = grouped[list(MEAN_METRICS)].median()
    summary["doub_time_std"] = grouped["doub_time"].std()
    summary["symmetry_std"] = grouped["symmetry"].std()
    return summary


def _stats_from_summaries(summary: pd.DataFrame) -> dict:
    """Return median and IQR half-widths after the renderer's 1.5-IQR fence."""
    result: dict = {}
    for metric in metrics.SIMP_METRICS:
        if metric not in summary.columns:
            result[metric] = None
            continue
        values = summary[metric].replace([np.inf, -np.inf], np.nan).dropna()
        values = metrics._drop_iqr_outliers(values)
        if len(values) == 0:
            result[metric] = None
            continue
        q25 = float(values.quantile(0.25))
        median = float(values.median())
        q75 = float(values.quantile(0.75))
        result[metric] = (median, median - q25, q75 - median)
    return result


def _pm50_run_dir(data_root: Path, series: str, threshold: str, peak_k: int) -> Path | None:
    run_name = f"ABC_SMC_RF_N512_pm50_{series}_{threshold}_p{peak_k}_mean_only"
    run_dir = data_root / run_name / "iter_4"
    required = ("all_param_df.csv", "final_metrics_seed.csv")
    return run_dir if all((run_dir / name).is_file() for name in required) else None


def install_pm50_data(
    data_root: Path, assignments_path: Path
) -> tuple[Path, pd.Series]:
    """Point the upstream renderer at pm50 data and install matched summaries."""
    parent_run = data_root / "ABC_SMC_RF_N1024_pm50"
    parent_iter = parent_run / "iter_4"
    parent_params = pd.read_csv(parent_iter / "all_param_df.csv")
    parent_seed_rows = pd.read_csv(parent_iter / "final_metrics_seed.csv")
    assignments = _load_assignments(assignments_path, parent_params)

    full_summary = _per_set_summaries(parent_seed_rows)
    full_summary = full_summary.join(assignments.rename("peak"), how="inner")
    if len(full_summary) != len(assignments):
        raise ValueError(
            "Parent seed metrics do not cover every pm50 parameter vector: "
            f"summaries={len(full_summary)}, assignments={len(assignments)}"
        )

    def run_dir(series: str, threshold: str, peak_k: int) -> Path | None:
        return _pm50_run_dir(data_root, series, threshold, peak_k)

    def peak_stats_from_csv(csv_path: Path) -> dict:
        rows = pd.read_csv(csv_path)
        if "input_folder" not in rows.columns:
            return metrics._peak_stats_from_csv(csv_path)
        return _stats_from_summaries(_per_set_summaries(rows))

    def load_orig_peak_stats(_cluster_dir: Path, peak_k: int) -> dict | None:
        subset = full_summary[full_summary["peak"] == peak_k].drop(columns="peak")
        return _stats_from_summaries(subset) if len(subset) else None

    def load_orig_mi(
        _n1024_dir: Path, _summary_json: Path, peak_k: int
    ) -> tuple[mi.LevelMI, dict[str, float]]:
        selected_inputs = set(assignments[assignments == peak_k].index)
        selected_params = parent_params[parent_params["input_folder"].isin(selected_inputs)]
        selected_metrics = parent_seed_rows[
            parent_seed_rows["input_folder"].isin(selected_inputs)
        ]
        return mi._compute_mi(
            selected_params.reset_index(drop=True), selected_metrics.reset_index(drop=True)
        )

    metrics._simp_run_dir = run_dir
    metrics._peak_stats_from_csv = peak_stats_from_csv
    metrics._load_orig_peak_stats = load_orig_peak_stats
    mi._simp_run_dir = run_dir
    mi._load_orig_mi = load_orig_mi
    mi.CLUSTER_N.clear()
    mi.CLUSTER_N.update(EXPECTED_PEAK_COUNTS)

    original_draw = mi._draw_mi_cluster_block

    def draw_with_peak_specific_lr_mask(*args, **kwargs):
        peak_k = int(args[2] if len(args) > 2 else kwargs["peak_k"])
        hide_by_level: dict[str, frozenset[str]] = {}
        for level_key, series, threshold in mi.ALL_LEVELS:
            if level_key not in mi.R_SERIES_KEYS or series is None or threshold is None:
                continue
            run_name = f"ABC_SMC_RF_N512_pm50_{series}_{threshold}_p{peak_k}_mean_only"
            lr_json = data_root / run_name / f"lr_predictions_r{threshold}_p{peak_k}.json"
            hide_by_level[level_key] = mi._r_series_derived_param_names(lr_json)
        kwargs["r_series_hide_mi_by_level"] = hide_by_level
        return original_draw(*args, **kwargs)

    mi._draw_mi_cluster_block = draw_with_peak_specific_lr_mask
    return parent_run, assignments


def _renderer_args(
    parse_args,
    *,
    parent_run: Path,
    data_root: Path,
    provenance: Path,
    output_dir: Path,
):
    return parse_args(
        [
            "--cluster-dir",
            str(parent_run),
            "--n1024-dir",
            str(parent_run),
            "--summary-json",
            str(provenance),
            "--lr-arcade-root",
            str(data_root),
            "--output-dir",
            str(output_dir),
            "--no-export-mi-json",
        ]
    )


def write_metric_summary(path: Path, parent_run: Path) -> None:
    """Record the plotted centers and target-relative errors, including absent runs."""
    levels = [metrics.SIMP_LEVELS_R[0], *metrics.SIMP_LEVELS_R[1:], *metrics.SIMP_LEVELS_T[1:]]
    rows: list[dict] = []
    for peak_k in range(1, 5):
        for level_key, series, threshold in levels:
            if level_key == "orig":
                stats = metrics._load_orig_peak_stats(parent_run, peak_k)
            else:
                stats = metrics._load_peak_stats(metrics._simp_run_dir(series, threshold, peak_k))
            status = "complete" if stats is not None else "no_reduction"
            for metric_name in metrics.SIMP_METRICS:
                metric_stats = stats.get(metric_name) if stats is not None else None
                median = metric_stats[0] if metric_stats is not None else np.nan
                target = metrics.EXP_TARGETS[metric_name]
                signed_error = (
                    metrics._signed_percent_error(median, target)
                    if np.isfinite(median)
                    else np.nan
                )
                rows.append(
                    {
                        "peak": peak_k,
                        "level": level_key,
                        "series": series or "full",
                        "threshold": threshold or "full",
                        "status": status,
                        "metric": metric_name,
                        "median": median,
                        "target": target,
                        "signed_error_percent": signed_error,
                        "absolute_error_percent": abs(signed_error),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {path}")


def _mean_mi(level_data: dict, parameters: set[str]) -> dict[str, float]:
    return {
        parameter: float(
            np.mean([level_data[parameter][metric]["mi"] for metric in mi.SIMP_METRICS])
        )
        for parameter in parameters
    }


def write_mi_outputs(
    summary_path: Path,
    comparison_path: Path,
    *,
    parent_run: Path,
    provenance: Path,
    data_root: Path,
) -> None:
    """Persist plotted MI values and Full-versus-reduced ranking comparisons."""
    export_args = argparse.Namespace(
        n1024_dir=parent_run,
        summary_json=provenance,
        lr_arcade_root=data_root,
    )
    mi.export_mi_summary_to_json(summary_path, export_args)
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    summary["version"] = 2
    summary["description"] = (
        "Per-parameter mutual information for the pm50 parent cloud and independently "
        "recalibrated reduced generation-4 clouds. Parent particles use the frozen Fig 3 "
        "unweighted PCA/KDE nearest-maximum assignments. Pairwise-regression derived parameters remain in "
        "the raw MI values but are listed under display_masked_parameters and shown as an "
        "em dash in the figure. Empty levels are designed no-reduction outcomes."
    )
    summary["paths"] = {
        "data_root": str(data_root),
        "n1024_dir": str(parent_run),
        "peak_assignment_provenance": str(provenance),
    }
    missing_reductions: list[dict] = []
    comparison_rows: list[dict] = []

    for peak_k in range(1, 5):
        peak_record = summary["peaks"][str(peak_k)]
        peak_record["cluster_label"] = f"Numerical subset P{peak_k}"
        peak_record["display_masked_parameters"] = {}
        levels = peak_record["levels"]
        full_level = levels["orig"]
        for level_key, series, threshold in mi.ALL_LEVELS[1:]:
            reduced_level = levels.get(level_key, {})
            if not reduced_level:
                missing_reductions.append(
                    {
                        "peak": peak_k,
                        "level": level_key,
                        "series": series,
                        "threshold": threshold,
                    }
                )
                continue

            retained = set(reduced_level)
            if series == "linear":
                run_name = (
                    f"ABC_SMC_RF_N512_pm50_linear_{threshold}_p{peak_k}_mean_only"
                )
                lr_json = data_root / run_name / f"lr_predictions_r{threshold}_p{peak_k}.json"
                derived = mi._r_series_derived_param_names(lr_json)
                peak_record["display_masked_parameters"][level_key] = sorted(derived)
                retained -= set(derived)
            retained &= set(full_level)
            if len(retained) < 4:
                continue

            full_scores = _mean_mi(full_level, retained)
            reduced_scores = _mean_mi(reduced_level, retained)
            ordered = sorted(retained)
            rho = float(
                mi.spearmanr(
                    [full_scores[name] for name in ordered],
                    [reduced_scores[name] for name in ordered],
                ).statistic
            )
            full_ranks = {
                name: rank
                for rank, name in enumerate(
                    sorted(ordered, key=lambda item: -full_scores[item])
                )
            }
            reduced_ranks = {
                name: rank
                for rank, name in enumerate(
                    sorted(ordered, key=lambda item: -reduced_scores[item])
                )
            }
            comparison_rows.append(
                {
                    "peak": peak_k,
                    "level": level_key,
                    "series": series,
                    "threshold": threshold,
                    "retained_parameters": len(retained),
                    "spearman_rho": rho,
                    "moved_parameters": sum(
                        full_ranks[name] != reduced_ranks[name] for name in retained
                    ),
                    "full_top_stays_top": max(retained, key=full_scores.get)
                    == max(retained, key=reduced_scores.get),
                }
            )

    summary["missing_reductions"] = missing_reductions
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)
    print(f"Saved {summary_path}")
    print(f"Saved {comparison_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--full-out", type=Path, default=DEFAULT_FULL_OUT)
    parser.add_argument("--single-out", type=Path, default=DEFAULT_SINGLE_OUT)
    parser.add_argument("--metric-summary", type=Path, default=DEFAULT_METRIC_SUMMARY)
    parser.add_argument("--mi-summary", type=Path, default=DEFAULT_MI_SUMMARY)
    parser.add_argument("--mi-rank-comparison", type=Path, default=DEFAULT_MI_RANK_COMPARISON)
    parser.add_argument("--lr-equations", type=Path, default=DEFAULT_LR_EQUATIONS)
    parser.add_argument("--regeneration-provenance", type=Path, default=DEFAULT_REGEN_PROVENANCE)
    args = parser.parse_args()

    install_display_levels()
    parent_run, _ = install_pm50_data(args.data_root, args.assignments)
    write_metric_summary(args.metric_summary, parent_run)
    write_lr_equations(args.lr_equations, args.data_root)
    write_mi_outputs(
        args.mi_summary,
        args.mi_rank_comparison,
        parent_run=parent_run,
        provenance=args.provenance,
        data_root=args.data_root,
    )

    import combine_simplification_figure as full_figure
    import combine_simplification_figure_isolated as single_figure

    full_args = _renderer_args(
        full_figure._parse_args,
        parent_run=parent_run,
        data_root=args.data_root,
        provenance=args.provenance,
        output_dir=args.full_out.parent,
    )
    full_figure.render(full_args, out=args.full_out)

    single_args = _renderer_args(
        single_figure._parse_args,
        parent_run=parent_run,
        data_root=args.data_root,
        provenance=args.provenance,
        output_dir=args.single_out.parent,
    )
    single_figure.render(single_args, peaks=[f"P{MAIN_FIGURE_PEAK}"], out=args.single_out)

    input_paths = [
        args.assignments,
        args.provenance,
        parent_run / "iter_4/all_param_df.csv",
        parent_run / "iter_4/final_metrics_seed.csv",
    ]
    for _level, series, threshold in [*LINEAR_LEVELS[1:], *metrics.SIMP_LEVELS_T[1:]]:
        for peak_k in range(1, 5):
            run_dir = _pm50_run_dir(args.data_root, series, threshold, peak_k)
            if run_dir is not None:
                input_paths.extend([run_dir / "all_param_df.csv", run_dir / "final_metrics_seed.csv"])
            if series == "linear":
                run_name = f"ABC_SMC_RF_N512_pm50_linear_{threshold}_p{peak_k}_mean_only"
                input_paths.append(
                    args.data_root / run_name / f"lr_predictions_r{threshold}_p{peak_k}.json"
                )
    output_paths = [
        args.metric_summary,
        args.mi_summary,
        args.mi_rank_comparison,
        args.lr_equations,
        args.full_out,
        args.full_out.with_suffix(".pdf"),
        args.single_out,
        args.single_out.with_suffix(".pdf"),
    ]
    payload = {
        "version": 1,
        "assignment_convention": "frozen Fig 3 unweighted PCA/KDE nearest maximum",
        "peak_counts": {str(key): value for key, value in EXPECTED_PEAK_COUNTS.items()},
        "linear_threshold_display_order": [0.8, 0.75, 0.7],
        "main_figure_peak": MAIN_FIGURE_PEAK,
        "ward_thresholds": [1.0, 1.25, 1.5],
        "inputs": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)}
            for path in dict.fromkeys(input_paths)
        ],
        "outputs": [
            {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)}
            for path in output_paths
        ],
        "upstream_scripts": [
            {"path": str(path), "sha256": sha256(path)}
            for path in [Path(metrics.__file__), Path(mi.__file__)]
        ],
        "script": str(Path(__file__).relative_to(REPO_ROOT)),
        "script_sha256": sha256(Path(__file__)),
        "command": "python analysis_scripts/regenerate_simplification_matched.py",
    }
    args.regeneration_provenance.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {args.regeneration_provenance}")


if __name__ == "__main__":
    main()
