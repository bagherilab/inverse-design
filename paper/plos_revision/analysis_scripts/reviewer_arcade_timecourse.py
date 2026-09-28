#!/usr/bin/env python3
"""Build the canonical pm50 ARCADE time course from saved raw outputs.

The four series are the stored particles nearest to the frozen Fig 3
unweighted PCA/KDE maxima. Curves show the median and interquartile range over
ten saved seeds. No simulation is launched by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from reviewer_figure_style import (
    PEAK_COLORS,
    PEAK_MARKERS,
    PLOS_DOUBLE_WIDTH,
    apply_plos_style,
    assert_text_inside_figure,
    four_spines,
    panel_label,
    save_figure,
)

REPO = Path(__file__).resolve().parents[1]
LOCAL_IDESIGN = REPO.parent / "inverse_design"
IDESIGN = LOCAL_IDESIGN if LOCAL_IDESIGN.is_dir() else Path(os.environ.get("DED_INVERSE_DESIGN", "/home/pohaoc2/UW/bagherilab/inverse_design"))
DEFAULT_RUN = Path(os.environ.get("DED_HPC_ROOT", "/gscratch/cheme/chiu")) / "ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50/iter_4"
DEFAULT_SUMMARY = REPO / "analysis_outputs/pm50_fig3/summary.json"
DEFAULT_OUT = REPO / "analysis_outputs/reviewer_closeout"
DEFAULT_FIG = REPO / "body/updated_figures/figures/reviewer_closeout"
DEFAULT_EXPERIMENT = DEFAULT_OUT / "conger_ziskin_mda361_digitized.csv"
DEFAULT_DOUBLING_TIMES = DEFAULT_OUT / "nci60_breast_doubling_times.csv"
CONGER_REPORTED_SLOPE_UM_DAY = 18.3
EXPECTED_SEEDS = tuple(range(10))
EXPECTED_TIMES_MIN = tuple(range(0, 10080 + 720, 720))

FIGURE_HEIGHT_IN = 2.28
AXIS_SIZE_IN = 1.55
AXIS_BOTTOM_IN = 0.52
AXIS_LEFT_IN = 0.55
AXIS_GAP_IN = 0.68


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def representative_inputs(summary_path: Path) -> list[dict[str, int | float | str]]:
    rows = json.loads(summary_path.read_text())
    selected = []
    for row in rows:
        if row.get("cluster") is None:
            continue
        index = int(row["nearest_sample_index"])
        selected.append(
            {
                "mode": int(row["cluster"]),
                "input_folder": f"input_{index + 1}",
                "zero_based_parameter_row": index,
                "distance_to_kde_maximum_in_unweighted_pc_space": float(row["nearest_sample_pc_distance"]),
            }
        )
    if len(selected) != 4:
        raise ValueError(f"Expected four stored KDE representatives, found {len(selected)}")
    return selected


def extract(
    run_dir: Path,
    representatives: list[dict[str, int | float | str]],
    *,
    source_root: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    sys.path.insert(0, str(IDESIGN / "src"))
    from inverse_design.analyze.population_metrics import PopulationMetrics

    records: list[dict[str, float | int | str]] = []
    source_files: list[dict[str, str]] = []
    for representative in representatives:
        mode = int(representative["mode"])
        folder_name = str(representative["input_folder"])
        folder = run_dir / "inputs" / folder_name
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        cell_files = sorted(folder.glob("*.CELLS.json"))
        location_files = sorted(folder.glob("*.LOCATIONS.json"))
        if len(cell_files) != len(location_files):
            raise ValueError(f"Mismatched cell/location file counts in {folder}")
        location_by_stem = {path.name.replace(".LOCATIONS.json", ""): path for path in location_files}
        observed_seed_times = {
            (int(stem.rsplit("_", 2)[-2]), int(stem.rsplit("_", 2)[-1]))
            for stem in location_by_stem
        }
        expected_seed_times = {(seed, time) for seed in EXPECTED_SEEDS for time in EXPECTED_TIMES_MIN}
        if observed_seed_times != expected_seed_times:
            missing = sorted(expected_seed_times - observed_seed_times)
            extra = sorted(observed_seed_times - expected_seed_times)
            raise ValueError(
                f"Incomplete canonical time course in {folder}: "
                f"missing seed/time pairs={missing}, extra seed/time pairs={extra}"
            )
        for cell_path in cell_files:
            stem = cell_path.name.replace(".CELLS.json", "")
            location_path = location_by_stem.get(stem)
            if location_path is None:
                raise FileNotFoundError(f"No LOCATIONS pair for {cell_path}")
            parts = stem.rsplit("_", 2)
            seed = int(parts[-2])
            timestamp_min = int(parts[-1])
            cells = json.loads(cell_path.read_text())
            locations = json.loads(location_path.read_text())
            records.append(
                {
                    "mode": mode,
                    "input_folder": folder_name,
                    "seed": seed,
                    "timestamp_min": timestamp_min,
                    "time_days": timestamp_min / 1440.0,
                    "n_cells": len(cells),
                    "colony_diameter_um": PopulationMetrics.calculate_colony_diameter(cells, locations),
                    "colony_symmetry": PopulationMetrics.calculate_symmetry(cells, locations),
                }
            )
            source_files.extend(
                [
                    {
                        "path": str(Path(source_root) / cell_path.relative_to(run_dir))
                        if source_root
                        else str(cell_path.resolve()),
                        "sha256": sha256(cell_path),
                    },
                    {
                        "path": str(Path(source_root) / location_path.relative_to(run_dir))
                        if source_root
                        else str(location_path.resolve()),
                        "sha256": sha256(location_path),
                    },
                ]
            )
    frame = pd.DataFrame(records).sort_values(["mode", "seed", "timestamp_min"])
    expected_times = set(range(0, 10080 + 720, 720))
    for (mode, seed), group in frame.groupby(["mode", "seed"]):
        if set(group.timestamp_min) != expected_times:
            raise ValueError(f"Incomplete time course for mode {mode}, seed {seed}")
    return frame, source_files


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = ["n_cells", "colony_diameter_um", "colony_symmetry"]
    rows = []
    for (mode, folder, timestamp, day), group in frame.groupby(
        ["mode", "input_folder", "timestamp_min", "time_days"], sort=True
    ):
        for metric in metrics:
            values = group[metric].to_numpy(float)
            rows.append(
                {
                    "mode": mode,
                    "input_folder": folder,
                    "timestamp_min": timestamp,
                    "time_days": day,
                    "metric": metric,
                    "n_seeds": len(values),
                    "q25": np.quantile(values, 0.25),
                    "median": np.median(values),
                    "q75": np.quantile(values, 0.75),
                }
            )
    return pd.DataFrame(rows)


def fit_cell_count(summary: pd.DataFrame) -> pd.DataFrame:
    """Fit log-linear exponential curves to the 15 simulated median counts."""
    rows = []
    for peak in range(1, 5):
        sub = summary[(summary.metric == "n_cells") & (summary["mode"] == peak)].sort_values("time_days")
        time = sub.time_days.to_numpy(float)
        observed = sub["median"].to_numpy(float)
        growth_rate, log_intercept = np.polyfit(time, np.log(observed), 1)
        predicted = np.exp(log_intercept + growth_rate * time)
        residual = np.sum((observed - predicted) ** 2)
        total = np.sum((observed - observed.mean()) ** 2)
        rows.append(
            {
                "mode": peak,
                "metric": "n_cells",
                "model": "log N(t) = log(a) + b t",
                "intercept_cells": float(np.exp(log_intercept)),
                "growth_rate_day": float(growth_rate),
                "doubling_time_h": float(24.0 * np.log(2.0) / growth_rate),
                "r_squared_original_scale": float(1.0 - residual / total),
                "n_timepoints": len(time),
            }
        )
    return pd.DataFrame(rows)


def fit_colony_diameter(summary: pd.DataFrame) -> pd.DataFrame:
    """Fit linear growth rates to the 15 simulated median diameters."""
    rows = []
    for peak in range(1, 5):
        sub = summary[(summary.metric == "colony_diameter_um") & (summary["mode"] == peak)].sort_values(
            "time_days"
        )
        time = sub.time_days.to_numpy(float)
        observed = sub["median"].to_numpy(float)
        growth_rate, intercept = np.polyfit(time, observed, 1)
        predicted = intercept + growth_rate * time
        residual = np.sum((observed - predicted) ** 2)
        total = np.sum((observed - observed.mean()) ** 2)
        rows.append(
            {
                "mode": peak,
                "metric": "colony_diameter_um",
                "model": "D(t) = a + b t",
                "intercept_um": float(intercept),
                "colony_growth_um_day": float(growth_rate),
                "r_squared_original_scale": float(1.0 - residual / total),
                "n_timepoints": len(time),
            }
        )
    return pd.DataFrame(rows)


def fit_seed_variability(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate population SD across ten seed-specific kinetic fits."""
    rows = []
    for peak in range(1, 5):
        doubling_times = []
        colony_growth_rates = []
        for _, sub in frame.loc[frame["mode"] == peak].groupby("seed"):
            sub = sub.sort_values("time_days")
            time = sub.time_days.to_numpy(float)
            counts = sub.n_cells.to_numpy(float)
            if np.any(counts <= 0):
                raise ValueError(f"Non-positive Cell count prevents log fit for P{peak}")
            count_growth_rate, _ = np.polyfit(time, np.log(counts), 1)
            diameter_growth_rate, _ = np.polyfit(time, sub.colony_diameter_um.to_numpy(float), 1)
            doubling_times.append(24.0 * np.log(2.0) / count_growth_rate)
            colony_growth_rates.append(diameter_growth_rate)
        rows.extend(
            [
                {
                    "mode": peak,
                    "metric": "n_cells",
                    "n_seed_fits": len(doubling_times),
                    "seed_fit_population_sd": float(np.std(doubling_times, ddof=0)),
                },
                {
                    "mode": peak,
                    "metric": "colony_diameter_um",
                    "n_seed_fits": len(colony_growth_rates),
                    "seed_fit_population_sd": float(np.std(colony_growth_rates, ddof=0)),
                },
            ]
        )
    return pd.DataFrame(rows)


def _main_axes(fig: plt.Figure) -> list[plt.Axes]:
    axes = []
    for index in range(3):
        x_in = AXIS_LEFT_IN + index * (AXIS_SIZE_IN + AXIS_GAP_IN)
        axes.append(
            fig.add_axes(
                [
                    x_in / PLOS_DOUBLE_WIDTH,
                    AXIS_BOTTOM_IN / FIGURE_HEIGHT_IN,
                    AXIS_SIZE_IN / PLOS_DOUBLE_WIDTH,
                    AXIS_SIZE_IN / FIGURE_HEIGHT_IN,
                ]
            )
        )
    return axes


def _assert_layout(
    fig: plt.Figure,
    axes: list[plt.Axes],
    legend_a: plt.Legend,
    legend_b: plt.Legend,
    summary: pd.DataFrame,
) -> None:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = [axis.get_window_extent(renderer) for axis in axes]
    for index, box in enumerate(boxes):
        if abs(box.width - box.height) > 1.0:
            raise RuntimeError(f"Panel {index + 1} is not square: {box.width:.2f} x {box.height:.2f} px")
    if max(box.width for box in boxes) - min(box.width for box in boxes) > 1.0:
        raise RuntimeError("Main-panel plotting boxes do not have equal widths")
    for left, right in zip(boxes, boxes[1:]):
        if left.overlaps(right):
            raise RuntimeError("Adjacent main-panel plotting boxes overlap")

    legend_boxes = [legend_a.get_window_extent(renderer), legend_b.get_window_extent(renderer)]
    for panel, legend_box, name in zip(boxes[:2], legend_boxes, ("Panel A legend", "Panel B legend")):
        if not (
            panel.x0 <= legend_box.x0
            and legend_box.x1 <= panel.x1
            and panel.y0 <= legend_box.y0
            and legend_box.y1 <= panel.y1
        ):
            raise RuntimeError(f"{name} is not fully inside its panel")

    for axis, metric, occupied_box, name in (
        (axes[0], "n_cells", legend_boxes[0], "Panel A legend"),
        (axes[1], "colony_diameter_um", legend_boxes[1], "Panel B legend"),
    ):
        values = summary.loc[summary.metric == metric, ["time_days", "q25", "median", "q75"]]
        for ordinate in ("q25", "median", "q75"):
            display_points = axis.transData.transform(values[["time_days", ordinate]].to_numpy(float))
            if any(occupied_box.contains(x, y) for x, y in display_points):
                raise RuntimeError(f"{name} overlaps a displayed simulation summary point")

def plot(
    summary: pd.DataFrame,
    fits: pd.DataFrame,
    doubling_times: pd.DataFrame,
    output: Path,
    *,
    split_formats: bool = False,
) -> None:
    apply_plos_style()
    metrics = [
        ("n_cells", "Cell count"),
        ("colony_diameter_um", "Colony diameter (µm)"),
        ("colony_symmetry", "Symmetry"),
    ]
    fig = plt.figure(figsize=(PLOS_DOUBLE_WIDTH, FIGURE_HEIGHT_IN))
    axes = _main_axes(fig)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for peak in range(1, 5):
            sub = summary[(summary.metric == metric) & (summary["mode"] == peak)].sort_values("time_days")
            x = sub.time_days.to_numpy(float)
            med = sub["median"].to_numpy(float)
            q25 = sub.q25.to_numpy(float)
            q75 = sub.q75.to_numpy(float)
            axis.plot(
                x,
                med,
                color=PEAK_COLORS[peak - 1],
                linestyle="none" if metric == "n_cells" else "-",
                linewidth=0 if metric == "n_cells" else 0.75,
                marker=PEAK_MARKERS[peak - 1],
                markersize=2.7,
                markeredgecolor="black",
                markeredgewidth=0.25,
                label=f"P{peak}",
                zorder=4,
            )
            axis.fill_between(x, q25, q75, color=PEAK_COLORS[peak - 1], alpha=0.11, linewidth=0)
            if metric == "n_cells":
                fit = fits.loc[(fits.metric == "n_cells") & (fits["mode"] == peak)].iloc[0]
                fine_time = np.linspace(0, 7, 200)
                fitted_count = fit.intercept_cells * np.exp(fit.growth_rate_day * fine_time)
                axis.plot(
                    fine_time,
                    fitted_count,
                    color=PEAK_COLORS[peak - 1],
                    linewidth=0.85,
                    zorder=3,
                )
        axis.set_ylabel(ylabel)
        axis.set_xlim(0, 7)
        axis.margins(y=0.08)
        axis.set_xticks([0, 2, 4, 6])
        axis.grid(False)
        four_spines(axis)

    for label, axis in zip("ABC", axes):
        panel_label(axis, label, x=-0.28, y=1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fit_labels = [
        f"P{peak} ({fits.loc[(fits.metric == 'n_cells') & (fits['mode'] == peak), 'doubling_time_h'].iloc[0]:.0f} ± "
        f"{fits.loc[(fits.metric == 'n_cells') & (fits['mode'] == peak), 'seed_fit_population_sd'].iloc[0]:.0f})"
        for peak in range(1, 5)
    ]
    nci_handle = Line2D([], [], linestyle="none")
    nci_mean = doubling_times.doubling_time_h.mean()
    nci_std = doubling_times.doubling_time_h.std(ddof=0)
    nci_label = f"NCI exp. ({nci_mean:.1f} ± {nci_std:.1f})"
    legend_a = axes[0].legend(
        handles + [nci_handle],
        fit_labels + [nci_label],
        title="Fitted Doub (h)",
        loc="upper left",
        bbox_to_anchor=(0.025, 0.985),
        ncol=1,
        alignment="left",
        frameon=False,
        fontsize=5.0,
        title_fontsize=5.0,
        handlelength=1.25,
        handletextpad=0.35,
        labelspacing=0.25,
        borderaxespad=0.0,
    )
    axes[0].set_xlabel("Simulation time (days)")

    diameter_handles, _ = axes[1].get_legend_handles_labels()
    diameter_labels = [
        f"P{peak} ({fits.loc[(fits.metric == 'colony_diameter_um') & (fits['mode'] == peak), 'colony_growth_um_day'].iloc[0]:.1f} ± "
        f"{fits.loc[(fits.metric == 'colony_diameter_um') & (fits['mode'] == peak), 'seed_fit_population_sd'].iloc[0]:.1f})"
        for peak in range(1, 5)
    ]
    experimental_handle = Line2D([], [], linestyle="none")
    legend_b = axes[1].legend(
        diameter_handles + [experimental_handle],
        diameter_labels + ["MDA exp. (18.3)"],
        title="Fitted colony growth rate (µm/day)",
        loc="upper left",
        bbox_to_anchor=(0.025, 0.985),
        ncol=1,
        alignment="left",
        frameon=False,
        fontsize=4.5,
        title_fontsize=4.5,
        handlelength=1.0,
        handletextpad=0.3,
        labelspacing=0.2,
        borderaxespad=0.0,
    )

    _assert_layout(fig, axes, legend_a, legend_b, summary)
    assert_text_inside_figure(fig)
    save_figure(fig, output, split_formats=split_formats)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument(
        "--source-root",
        help="Persistent source path recorded in provenance when --run is a temporary local copy.",
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--doubling-times", type=Path, default=DEFAULT_DOUBLING_TIMES)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render from the tracked summary CSV without requiring the archived raw ARCADE run.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    experiment = pd.read_csv(args.experiment)
    doubling_times = pd.read_csv(args.doubling_times)
    if args.render_only:
        frame = pd.read_csv(args.out / "arcade_timecourse_source.csv")
        summary = pd.read_csv(args.out / "arcade_timecourse_summary.csv")
        provenance_path = args.out / "arcade_timecourse_provenance.json"
        provenance = json.loads(provenance_path.read_text())
        representatives = provenance["representatives"]
        source_files = provenance["source_files"]
    else:
        representatives = representative_inputs(args.summary)
        frame, source_files = extract(
            args.run,
            representatives,
            source_root=args.source_root,
        )
        summary = summarize(frame)
        frame.to_csv(args.out / "arcade_timecourse_source.csv", index=False)
        summary.to_csv(args.out / "arcade_timecourse_summary.csv", index=False)
        provenance = {
            "raw_output_root": args.source_root or str(args.run.resolve()),
            "representative_selection_source": str(args.summary.resolve().relative_to(REPO)),
            "representative_selection_source_sha256": sha256(args.summary),
            "representatives": representatives,
            "assignment_convention": "frozen Fig 3 unweighted PCA/KDE nearest maximum",
            "selection_interpretation": "stored particles nearest to the four canonical unweighted Fig 3 KDE maxima; no biological-regime interpretation",
            "measurement_times_min": list(EXPECTED_TIMES_MIN),
            "seeds_per_representative": len(EXPECTED_SEEDS),
            "metric_implementation": str((IDESIGN / "src/inverse_design/analyze/population_metrics.py").resolve()),
            "metric_implementation_sha256": sha256(
                IDESIGN / "src/inverse_design/analyze/population_metrics.py"
            ),
            "source_file_count": len(source_files),
            "source_files": source_files,
        }

    provenance.pop("displayed_references", None)
    fits = pd.concat([fit_cell_count(summary), fit_colony_diameter(summary)], ignore_index=True, sort=False)
    fits = fits.merge(fit_seed_variability(frame), on=["mode", "metric"], validate="one_to_one")
    fits.to_csv(args.out / "arcade_timecourse_fits.csv", index=False)
    provenance["figure_encoding"] = {
        "simulation": "markers are ten-seed medians and bands are IQRs; Panel A smooth curves are log-linear exponential fits to 15 simulated medians, while Panels B and C use thin segments between consecutive saved medians",
        "panel_a_comparison": "under the Fitted Doub (h) legend title, P1-P4 entries report each median-trajectory doubling-time fit plus the population SD across ten seed-specific fits; NCI exp. gives the mean plus population SD of seven experimental NCI-60 breast-cell-line doubling times as a text-only entry without a marker",
        "panel_b_comparison": "under the Fitted colony growth rate (µm/day) legend title, P1-P4 entries report each median-trajectory colony-diameter growth-rate fit plus the population SD across ten seed-specific fits; the published 18.3 µm/day MDA-361 experimental value is a text-only entry without a marker, and no experimental diameter trajectory is plotted",
        "published_data": "NCI doubling-time and MDA-361 colony-growth references are scalar values; no experimental trajectory exists for Panel C",
        "comparison_boundary": "Panel A and Panel B compare like scalar estimands without constructing or overlaying experimental time courses",
    }
    provenance["simulation_fits"] = {
        "scope": "Panel A cell-count doubling times and Panel B colony-diameter growth rates",
        "method": {
            "panel_a": "ordinary least squares of log median cell count versus time over 15 saved time points",
            "panel_b": "ordinary least squares of median colony diameter versus time over 15 saved time points",
            "variability": "population SD (ddof=0) across the corresponding kinetic metric fitted separately to each of ten seed-level 15-point time courses",
        },
        "rows": fits.to_dict(orient="records"),
    }
    provenance["experimental_doubling_times"] = {
        "source": "National Cancer Institute NCI-60 breast-cell-line table",
        "source_url": "https://dctd.cancer.gov/drug-discovery-development/assays/high-throughput-screening-services/nci60/submitting-compounds/nci60-cell-lines.pdf",
        "data": str(args.doubling_times.relative_to(REPO)),
        "data_sha256": sha256(args.doubling_times),
        "n_cell_lines": len(doubling_times),
        "median_h": float(doubling_times.doubling_time_h.median()),
        "mean_h": float(doubling_times.doubling_time_h.mean()),
        "population_sd_h": float(doubling_times.doubling_time_h.std(ddof=0)),
        "interpretation": "line-level scalar doubling times, not observed cell-count trajectories",
    }
    provenance["experimental_timecourse"] = {
        "source": "Conger AD, Ziskin MC. Growth of mammalian multicellular tumor spheroids. Cancer Research 43:556-560 (1983).",
        "pmid": "6848179",
        "source_url": "https://aacrjournals.org/cancerres/article-pdf/43/2/556/2864686/crs0430020556.pdf",
        "source_scan_sha256": "c25d2290c7e67b8d664ab2b5fec236c0a0f1063e46c96a1579018d8c90cef146",
        "source_location": "printed page 557, eight-line growth-curve chart identified as Chart 1 in the article text and table footnote",
        "series": "MDA-361 human breast epithelial carcinoma spheroids",
        "source_point_definition": "mean diameter of 12 or 24 spheroids; visible bars are S.E.; the source states omitted bars are smaller than the plotted symbol",
        "reported_slope_um_day": CONGER_REPORTED_SLOPE_UM_DAY,
        "digitized_data": str(args.experiment.relative_to(REPO)),
        "digitized_data_sha256": sha256(args.experiment),
        "digitization": {
            "rendering": "PDF page 2 rendered at 600 dpi",
            "x_calibration": "0-35 days from eight axis ticks",
            "y_calibration": "200-800 µm from seven MDA-361 ordinate ticks",
            "rounding": "time to 0.1 day and diameter/S.E. to 1 µm",
            "estimated_reading_uncertainty": "approximately ±0.2 day and ±5 µm, excluding experimental S.E.",
            "check": "linear fit to digitized means gives 17.4 µm/day; the article's tabulated 18.3 µm/day remains authoritative",
        },
    }
    plot(summary, fits, doubling_times, args.out / "arcade_timecourse")
    plot(summary, fits, doubling_times, args.figure_dir / "arcade_timecourse", split_formats=True)
    provenance["script"] = str(Path(__file__).resolve().relative_to(REPO))
    provenance["script_sha256"] = sha256(Path(__file__).resolve())
    if args.render_only:
        provenance["command"] = (
            "MPLCONFIGDIR=/tmp/inversedesign-mpl PYTHONPATH=../inverse_design/src "
            "python3 analysis_scripts/reviewer_arcade_timecourse.py --render-only"
        )
    else:
        source_root_option = f" --source-root {args.source_root}" if args.source_root else ""
        provenance["command"] = (
            "MPLCONFIGDIR=/tmp/inversedesign-mpl PYTHONPATH=../inverse_design/src "
            f"python3 analysis_scripts/reviewer_arcade_timecourse.py --run {args.run}"
            f"{source_root_option}"
        )
    output_paths = [
        args.out / "arcade_timecourse_source.csv",
        args.out / "arcade_timecourse_summary.csv",
        args.out / "arcade_timecourse_fits.csv",
        args.out / "arcade_timecourse.png",
        args.out / "arcade_timecourse.pdf",
        args.figure_dir / "png/arcade_timecourse.png",
        args.figure_dir / "pdf/arcade_timecourse.pdf",
    ]
    provenance["outputs"] = {
        str(path.relative_to(REPO)): sha256(path) for path in output_paths
    }
    (args.out / "arcade_timecourse_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(pd.DataFrame(representatives).to_string(index=False))
    print(
        f"rendered {len(summary)} simulation summaries, {len(fits)} time-course fits, "
        f"{len(doubling_times)} NCI doubling times, and {len(experiment)} digitized diameter points"
    )


if __name__ == "__main__":
    main()
