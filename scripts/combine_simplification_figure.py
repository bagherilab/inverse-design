#!/usr/bin/env python3
"""Simplification figure: MI table (top) plus metric bar grid (bottom)."""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/inverse_design_matplotlib")

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SRC_DIR = REPO_ROOT / "src"
for path in (SCRIPTS_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from combine_simplification_metrics import (  # noqa: E402
    DEFAULT_CLUSTER_DIR,
    plot_metrics_panel,
)
from combine_simplification_mi_table import (  # noqa: E402
    ARCADE_OUTPUT_DIR,
    DEFAULT_MI_SUMMARY_JSON,
    DEFAULT_N1024_DIR,
    DEFAULT_SUMMARY_JSON,
    export_mi_summary_to_json,
    plot_mi_table_panel,
)
from inverse_design.plotting.style import (  # noqa: E402
    DPI,
    FONT_LABEL,
    apply_style,
    savefig_both,
)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "figures" / "simplification"


def _resolve_output_path(output_dir: Path, out: Path | None) -> Path:
    if out is None:
        return output_dir / "simplified_figure.png"
    return out if out.is_absolute() else output_dir / out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined simplification figure: MI table + metric bar grid."
    )
    parser.add_argument("--cluster-dir", type=Path, default=DEFAULT_CLUSTER_DIR)
    parser.add_argument("--n1024-dir", type=Path, default=DEFAULT_N1024_DIR)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument(
        "--lr-arcade-root",
        type=Path,
        default=None,
        help=(
            "Where to find *linear_<r>* runs for per-column r MI masking (default "
            f"{ARCADE_OUTPUT_DIR})."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output filename or absolute path for the full comparison figure.",
    )
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--font-size", type=int, default=FONT_LABEL)
    parser.add_argument(
        "--export-mi-json",
        type=Path,
        default=DEFAULT_MI_SUMMARY_JSON,
        help=f"Write MI summary JSON (default: {DEFAULT_MI_SUMMARY_JSON}).",
    )
    parser.add_argument(
        "--no-export-mi-json",
        action="store_true",
        help="Skip writing data/summary_MI.json after the figure.",
    )
    return parser.parse_args(argv)


def render(args: argparse.Namespace, *, out: str | Path | None = None) -> Path:
    output_path = _resolve_output_path(args.output_dir, Path(out) if isinstance(out, str) else out)
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.50, 5.58), dpi=args.dpi, facecolor="white")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.97, bottom=0.03)
    outer = fig.add_gridspec(2, 1, height_ratios=[0.85, 1.0], hspace=0.10)

    mi_args = copy.copy(args)
    mi_args.font_size = max(args.font_size - 2, 5)
    plot_mi_table_panel(fig, outer[0], mi_args, panel_label="A")
    plot_metrics_panel(fig, outer[1], args, section_panel_labels=("B", "C"))

    savefig_both(fig, output_path)
    plt.close(fig)
    print(f"Saved {output_path}")
    return output_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    render(args, out=args.out)

    if not args.no_export_mi_json:
        export_mi_summary_to_json(args.export_mi_json, args)


if __name__ == "__main__":
    main()
