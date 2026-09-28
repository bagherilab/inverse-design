#!/usr/bin/env python3
"""Freeze the Fig 3 unweighted PCA/KDE P1--P4 particle assignments.

The output is the sole membership source for every downstream result that uses
the Fig 3 P1--P4 labels.  It deliberately reuses the same project function and
detector arguments as ``export_pm50_fig3_inputs.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn

from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARENT = (
    REPO_ROOT
    / "analysis_outputs/pm50_reduced_campaign/ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50/iter_4/all_param_df.csv"
)
DEFAULT_ASSIGNMENTS = REPO_ROOT / "analysis_outputs/pm50_fig3/canonical_peak_assignments.csv"
DEFAULT_PROVENANCE = (
    REPO_ROOT / "analysis_outputs/pm50_fig3/canonical_assignments_provenance.json"
)
FIG3_PROVENANCE = REPO_ROOT / "analysis_outputs/pm50_fig3/provenance.json"
FIG3_SUMMARY = REPO_ROOT / "analysis_outputs/pm50_fig3/summary.json"
PCA_IMPLEMENTATION = (
    REPO_ROOT.parent
    / "inverse_design/src/inverse_design/analyze/core/pca_peaks.py"
)
REDUCED_DRIVER = REPO_ROOT.parent / "inverse_design/src/inverse_design/rf/arcade_example.py"
DROP_COLUMNS = ("input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER")
EXPECTED_COUNTS = {1: 310, 2: 315, 3: 162, 4: 237}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _partition(frame: pd.DataFrame, threshold: float):
    numeric = frame.drop(columns=list(DROP_COLUMNS))
    pca_result, peak_positions, *_ = perform_pca_and_find_peaks(
        numeric,
        n_components=2,
        threshold_ratio=threshold,
        neighborhood_size=5,
        random_state=0,
    )
    squared = ((pca_result[:, None, :2] - peak_positions[None, :, :]) ** 2).sum(axis=2)
    assignments = np.argmin(squared, axis=1) + 1
    nearest_50 = {
        str(peak): np.argsort(squared[:, peak - 1])[:50].astype(int).tolist()
        for peak in range(1, len(peak_positions) + 1)
    }
    return pca_result[:, :2], peak_positions, assignments, nearest_50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    args = parser.parse_args()

    frame = pd.read_csv(args.parent)
    pc, peaks, assignments, nearest_50 = _partition(frame, 0.05)
    driver_pc, driver_peaks, driver_assignments, driver_nearest_50 = _partition(frame, 0.10)

    counts = pd.Series(assignments).value_counts().sort_index().to_dict()
    if counts != EXPECTED_COUNTS:
        raise RuntimeError(f"canonical Fig 3 counts changed: {counts}")
    if not np.array_equal(peaks, driver_peaks):
        raise RuntimeError("the reduced driver threshold changes the Fig 3 peak positions")
    if not np.array_equal(assignments, driver_assignments):
        raise RuntimeError("the reduced driver threshold changes the Fig 3 assignments")
    if nearest_50 != driver_nearest_50:
        raise RuntimeError("the reduced driver threshold changes a nearest-50 subset")

    fig3_summary = json.loads(FIG3_SUMMARY.read_text(encoding="utf-8"))
    recorded = np.array(
        [
            [row["pc1"], row["pc2"]]
            for row in fig3_summary
            if row["kind"] == "pca_peak_inverse"
        ]
    )
    # summary.json records coordinates rounded to ten decimal places.
    if not np.allclose(peaks, recorded, rtol=0, atol=1e-10):
        raise RuntimeError("computed maxima do not match the recorded Fig 3 summary")

    output = pd.DataFrame(
        {
            "source_row": np.arange(len(frame), dtype=int),
            "input_folder": frame["input_folder"].astype(str),
            "pc1": pc[:, 0],
            "pc2": pc[:, 1],
            "peak": assignments.astype(int),
        }
    )
    args.assignments.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.assignments, index=False, float_format="%.15g")

    source_rows = {
        str(peak): np.flatnonzero(assignments == peak).astype(int).tolist()
        for peak in range(1, 5)
    }
    provenance = {
        "version": 1,
        "assignment_convention": "Fig 3 unweighted PCA/KDE nearest maximum",
        "source_posterior": str(args.parent.relative_to(REPO_ROOT)),
        "source_posterior_sha256": sha256(args.parent),
        "fig3_provenance": str(FIG3_PROVENANCE.relative_to(REPO_ROOT)),
        "fig3_provenance_sha256": sha256(FIG3_PROVENANCE),
        "fig3_summary": str(FIG3_SUMMARY.relative_to(REPO_ROOT)),
        "fig3_summary_sha256": sha256(FIG3_SUMMARY),
        "assignment_file": str(args.assignments.relative_to(REPO_ROOT)),
        "assignment_file_sha256": sha256(args.assignments),
        "pca_implementation": str(PCA_IMPLEMENTATION.relative_to(REPO_ROOT.parent)),
        "pca_implementation_sha256": sha256(PCA_IMPLEMENTATION),
        "reduced_driver": str(REDUCED_DRIVER.relative_to(REPO_ROOT.parent)),
        "reduced_driver_sha256": sha256(REDUCED_DRIVER),
        "pca_parameters": [column for column in frame.columns if column not in DROP_COLUMNS],
        "drop_columns": list(DROP_COLUMNS),
        "n_components": 2,
        "random_state": 0,
        "kde": {
            "weighted": False,
            "covariance_factor": "scott",
            "grid": [100, 100],
            "maximum_filter": [5, 5],
            "relative_density_threshold": 0.05,
        },
        "peak_numbering": "row-major order returned by np.where on the Fig 3 density grid",
        "peak_centres": {
            str(index + 1): [float(value) for value in centre]
            for index, centre in enumerate(peaks)
        },
        "peak_counts": {str(key): int(value) for key, value in counts.items()},
        "peak_source_rows": source_rows,
        "nearest_50_source_rows": nearest_50,
        "nearest_50_input_folders": {
            peak: frame.iloc[rows]["input_folder"].astype(str).tolist()
            for peak, rows in nearest_50.items()
        },
        "reduced_driver_equivalence": {
            "driver_relative_density_threshold": 0.10,
            "same_peak_positions": True,
            "same_assignments": True,
            "same_nearest_50": True,
            "reason": "all four detected maxima exceed 0.10 of the maximum density",
        },
        "command": (
            "MPLCONFIGDIR=/tmp/inversedesign-mpl PYTHONPATH=../inverse_design/src "
            "python analysis_scripts/freeze_pm50_canonical_assignments.py"
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.assignments}")
    print(f"saved {args.provenance}")
    print("canonical counts=" + "/".join(str(counts[key]) for key in sorted(counts)))


if __name__ == "__main__":
    main()
