#!/usr/bin/env python3
"""Regenerate S2 Data and the illustrative S9 Fig from canonical pm50 P1--P4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inverse_design.plotting.plot_dendrogram import colorize_ward_clusters_threshold


REPO = Path(__file__).resolve().parents[1]
RUN = (
    REPO
    / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50/iter_4"
)
ASSIGNMENTS = REPO / "analysis_outputs/pm50_fig3/canonical_peak_assignments.csv"
S2_DATA = REPO / "body/extended_data/dendrogram_param_status.csv"
S9_FIG = REPO / "body/extended_data/figures/simplified_model_example/simplified_model_example.png"
PROVENANCE = REPO / "analysis_outputs/pm50_fig3/canonical_s2_s9_provenance.json"
THRESHOLDS = (1.0, 1.25, 1.5)
ILLUSTRATIVE_PEAK = 3
ILLUSTRATIVE_R = 0.8
META = {"input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def display(name: str) -> str:
    return name.replace("_MU", "").replace("_", " ").title().replace("Atp", "ATP")


def nearest50(params: pd.DataFrame, assignments: pd.DataFrame, peak: int) -> pd.DataFrame:
    provenance = json.loads(
        (REPO / "analysis_outputs/pm50_fig3/canonical_assignments_provenance.json").read_text()
    )
    folders = provenance["nearest_50_input_folders"][str(peak)]
    indexed = params.set_index("input_folder")
    return indexed.loc[folders].reset_index()


def ward_result(data: pd.DataFrame, threshold: float, ax=None):
    columns = [column for column in data.columns if column not in META]
    correlation = np.corrcoef(data[columns].to_numpy(float), rowvar=False)
    return columns, colorize_ward_clusters_threshold(
        correlation,
        np.asarray(columns),
        threshold,
        verbose=False,
        ax=ax,
    )


def regenerate_s2(params: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for threshold in THRESHOLDS:
        for peak in range(1, 5):
            selected = nearest50(params, assignments, peak)
            figure, axis = plt.subplots()
            columns, result = ward_result(selected, threshold, axis)
            plt.close(figure)
            representatives = set(result[3])
            redundant = result[4]
            fixed_to = {
                parameter: representative
                for representative, parameters in redundant.items()
                for parameter in parameters
            }
            for parameter in columns:
                rows.append(
                    {
                        "ward_threshold": threshold,
                        "peak": peak,
                        "parameter": display(parameter),
                        "status": "Fix" if parameter in fixed_to else "Perturb",
                        "group_representative": (
                            display(fixed_to[parameter])
                            if parameter in fixed_to
                            else display(parameter) if parameter in representatives else ""
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if len(result) != 3 * 4 * 19:
        raise RuntimeError(f"expected 228 S2 rows, found {len(result)}")
    result.to_csv(S2_DATA, index=False)
    return result


def linear_models() -> list[dict]:
    path = (
        REPO
        / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/"
        f"ABC_SMC_RF_N512_pm50_linear_{ILLUSTRATIVE_R}_p{ILLUSTRATIVE_PEAK}_mean_only/"
        f"lr_predictions_r{ILLUSTRATIVE_R}_p{ILLUSTRATIVE_PEAK}.json"
    )
    models = json.loads(path.read_text())["prediction_models"]["models"]
    return sorted(
        models,
        key=lambda model: abs(float(model["prediction_quality"]["correlation_r"])),
        reverse=True,
    )[:3]


def sig3(value: float) -> str:
    """Three significant figures; scientific notation outside [0.01, 1000)."""
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    if -2 <= exponent < 3:
        return f"{value:.{max(0, 2 - exponent)}f}"
    mantissa = value / 10**exponent
    return f"{mantissa:.2f}\\times10^{{{exponent}}}"


def draw_s9(params: pd.DataFrame, assignments: pd.DataFrame) -> list[dict]:
    selected = nearest50(params, assignments, ILLUSTRATIVE_PEAK)
    models = linear_models()
    figure = plt.figure(figsize=(10.5, 8.5))
    grid = figure.add_gridspec(2, 3, height_ratios=[1, 1.65], hspace=0.55, wspace=0.65)
    for index, model in enumerate(models):
        axis = figure.add_subplot(grid[0, index])
        predictor = model["predictor_parameter"]["name"]
        target = model["target_parameter"]["name"]
        slope = float(model["linear_model"]["slope"])
        intercept = float(model["linear_model"]["intercept"])
        correlation = float(model["prediction_quality"]["correlation_r"])
        x = selected[predictor].to_numpy(float)
        y = selected[target].to_numpy(float)
        order = np.argsort(x)
        axis.scatter(x, y, facecolors="none", edgecolors="#333333", s=24, linewidths=0.8)
        axis.plot(x[order], slope * x[order] + intercept, "--", color="#555555", linewidth=1.3)
        axis.set_xlabel(display(predictor), fontweight="bold")
        axis.set_ylabel(display(target))
        slope_text, intercept_text = sig3(slope), sig3(abs(intercept))
        sign = "+" if intercept >= 0 else "-"
        # Reserve headroom so the fit text sits inside the axes above the data.
        low, high = float(np.min(y)), float(np.max(y))
        span = high - low
        axis.set_ylim(low - 0.05 * span, high + 0.45 * span)
        axis.text(
            0.03,
            0.97,
            f"$R = {correlation:.2f}$\n$y = {slope_text}\\,x {sign} {intercept_text}$",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5),
        )
        axis.spines[["top", "right"]].set_visible(False)

    dend_axis = figure.add_subplot(grid[1, :])
    columns, result = ward_result(selected, 1.0, dend_axis)
    # Use the published CSV as the authoritative representative/perturbation list.
    status = pd.read_csv(S2_DATA)
    status = status[(status.peak == ILLUSTRATIVE_PEAK) & (status.ward_threshold == 1.0)]
    perturbed = set(status.loc[status.status.eq("Perturb"), "parameter"])
    def key(text):
        return " ".join(text.replace("_", " ").replace("*", "").lower().split())
    names = {key(display(name)): display(name) for name in columns}
    names.update({key(name): display(name) for name in columns})
    tick_labels, tick_weights = [], []
    for tick in dend_axis.get_yticklabels():
        name = names.get(key(tick.get_text()))
        if name is None:
            raise ValueError(f"Unmapped dendrogram label: {tick.get_text()}")
        tick_labels.append(name)
        tick_weights.append("bold" if name in perturbed else "normal")
    dend_axis.set_yticks(dend_axis.get_yticks(), labels=tick_labels)
    for tick, weight in zip(dend_axis.get_yticklabels(), tick_weights):
        tick.set_fontweight(weight)
    # The shared dendrogram helper bolds the x-axis; keep it at normal weight here.
    dend_axis.set_xlabel("Ward distance", fontsize=12, fontweight="normal")
    for label in dend_axis.get_xticklabels():
        label.set_fontweight("normal")
    dend_axis.figure.canvas.draw()

    figure.text(0.015, 0.965, "A", fontsize=21, fontweight="bold")
    dend_axis.text(-0.14, 1.03, "B", transform=dend_axis.transAxes, fontsize=21, fontweight="bold")
    figure.text(0.095, 0.965, "★", color="#ff7f0e", fontsize=23, va="center")
    figure.text(0.14, 0.965, "P3", fontsize=18, va="center")
    S9_FIG.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(S9_FIG, dpi=300, bbox_inches="tight")
    figure.savefig(S9_FIG.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    params_path = RUN / "all_param_df.csv"
    params = pd.read_csv(params_path)
    assignments = pd.read_csv(ASSIGNMENTS)
    if assignments.input_folder.tolist() != params.input_folder.astype(str).tolist():
        raise RuntimeError("canonical assignments are not aligned to parent parameter rows")
    # Published assignments are frozen; rendering must not reselect tied representatives.
    s2 = pd.read_csv(S2_DATA)
    models = draw_s9(params, assignments)
    model_path = (
        REPO
        / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/"
        f"ABC_SMC_RF_N512_pm50_linear_{ILLUSTRATIVE_R}_p{ILLUSTRATIVE_PEAK}_mean_only/"
        f"lr_predictions_r{ILLUSTRATIVE_R}_p{ILLUSTRATIVE_PEAK}.json"
    )
    provenance = {
        "version": 1,
        "assignment_convention": "frozen Fig 3 unweighted PCA/KDE nearest-50 subsets",
        "assignments": str(ASSIGNMENTS.relative_to(REPO)),
        "assignments_sha256": sha256(ASSIGNMENTS),
        "params": str(params_path.relative_to(REPO)),
        "params_sha256": sha256(params_path),
        "s2_data": str(S2_DATA.relative_to(REPO)),
        "s2_data_sha256": sha256(S2_DATA),
        "s2_rows": len(s2),
        "ward_thresholds": list(THRESHOLDS),
        "s9_peak": ILLUSTRATIVE_PEAK,
        "s9_linear_threshold": ILLUSTRATIVE_R,
        "s9_model_source": str(model_path.relative_to(REPO)),
        "s9_model_source_sha256": sha256(model_path),
        "s9_models": [model["linear_model"]["equation"] for model in models],
        "s9_figure": str(S9_FIG.relative_to(REPO)),
        "s9_figure_sha256": sha256(S9_FIG),
        "s9_figure_pdf": str(S9_FIG.with_suffix(".pdf").relative_to(REPO)),
        "s9_figure_pdf_sha256": sha256(S9_FIG.with_suffix(".pdf")),
        "script": str(Path(__file__).relative_to(REPO)),
        "script_sha256": sha256(Path(__file__)),
        "command": "python analysis_scripts/regenerate_pm50_s2_s9.py",
    }
    PROVENANCE.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {S2_DATA} ({len(s2)} rows), {S9_FIG}, and {PROVENANCE}")


if __name__ == "__main__":
    main()
