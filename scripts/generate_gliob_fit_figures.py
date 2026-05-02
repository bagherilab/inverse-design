#!/usr/bin/env python3
"""Generate glioblastoma ABC g0 vs g4 metric histograms into ``results/figures/gliob/``.

Uses :mod:`combine_fit_exp_figure` (same helpers as breast ``fit_exp``) with defaults
matching ``ref_figures/glioblastoma.png`` column order and y-axis scale.

Requires a completed ARCADE SMC run directory containing ``targets.json`` and
``iter_0`` / ``iter_4`` / ``final_metrics.csv`` (see ``ParameterSampler`` glio path).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLIO_BASE = (
    REPO_ROOT.parent / "ARCADE_OUTPUT" / "ABC_SMC_RF_N1024_combined_grid_glioblastoma"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "gliob"

# Same column order as ref_figures/glioblastoma.png (SI figure).
GLIO_PANEL_A_METRICS = [
    "doub_time",
    "doub_time_std",
    "symmetry",
    "symmetry_std",
    "colony_growth",
]


def parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Glioblastoma panel-A histograms (g0 vs g4); forwards extra flags to combine_fit_exp_figure.py."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=DEFAULT_GLIO_BASE,
        help="ARCADE SMC output with targets.json and iter_*/final_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for A_histograms.png and A_histogram_<metric>.png.",
    )
    parser.add_argument(
        "--hist-y-max",
        type=float,
        default=250.0,
        help="Panel A y-axis max (default 250 matches SI reference).",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, rest = parse_args(argv if argv is not None else sys.argv[1:])
    combine_script = REPO_ROOT / "scripts" / "combine_fit_exp_figure.py"
    cmd: list[str] = [
        sys.executable,
        str(combine_script),
        "--base-dir",
        str(args.base_dir),
        "--output-dir",
        str(args.output_dir),
        "--panel-a-only",
        "--hist-y-max",
        str(args.hist_y_max),
        "--panel-a-metrics",
        *GLIO_PANEL_A_METRICS,
    ]
    cmd.extend(rest)
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
