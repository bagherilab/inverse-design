"""Export ARCADE XML inputs from posterior PCA density clusters."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from inverse_design.vis.vis_posterior_overview import (
    DEFAULT_ARCADE_TEMPLATE,
    DEFAULT_METADATA_COLUMNS,
    DEFAULT_SOURCE_SIDE_LENGTH,
    export_pca_cluster_arcade_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create mean/mode/PCA-cluster ARCADE inputs from a posterior CSV."
    )
    parser.add_argument("posterior_csv", type=Path, help="Posterior all_param_df.csv file")
    parser.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_ARCADE_TEMPLATE,
        help="ARCADE XML template to patch",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/pca_cluster_arcade_inputs"),
        help="Directory for summary files and inputs/input_*.xml",
    )
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--threshold-ratio", type=float, default=0.1)
    parser.add_argument("--neighborhood-size", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--drop-column",
        action="append",
        default=list(DEFAULT_METADATA_COLUMNS),
        help="Column to exclude before PCA; can be passed multiple times",
    )
    parser.add_argument(
        "--side-length",
        type=float,
        default=DEFAULT_SOURCE_SIDE_LENGTH,
        help="Hex side length used when converting CAPILLARY_DENSITY to spacing",
    )
    parser.add_argument(
        "--clusters-only",
        action="store_true",
        help="Write only PCA peak cluster inputs, skipping posterior mean/mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    posterior_df = pd.read_csv(args.posterior_csv)
    summary_df = export_pca_cluster_arcade_inputs(
        posterior_df,
        output_dir=args.output_dir,
        template_path=args.template,
        n_components=args.n_components,
        threshold_ratio=args.threshold_ratio,
        neighborhood_size=args.neighborhood_size,
        n_bins=args.n_bins,
        random_state=args.random_state,
        drop_columns=tuple(args.drop_column),
        side_length=args.side_length,
        include_mean_mode=not args.clusters_only,
    )
    print(f"Wrote {len(summary_df)} representatives to {args.output_dir}")
    print(args.output_dir / "cluster_summary.txt")


if __name__ == "__main__":
    main()
