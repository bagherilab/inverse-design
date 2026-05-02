#!/usr/bin/env python3
"""Build minimal cache payloads for vis_simplified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_design.analyze.abc_metrics_comparison import analyze_iterations
from inverse_design.io.scenarios import combined_grid_n512_breast
from inverse_design.io import ArcadeRunLayout
from inverse_design.vis.vis_cache import cache_payload_path, select_simplified_dirs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["linear", "threshold"],
        default="linear",
        help="Simplification method used by vis_simplified.",
    )
    parser.add_argument("--peak-idx", type=int, default=2, help="Zero-based peak index.")
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=5,
        help="Number of ABC iterations to scan when selecting best-fit samples.",
    )
    parser.add_argument(
        "--cache-root",
        default="results/vis_cache",
        help="Directory where cache payload json will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = combined_grid_n512_breast()
    run_dirs = [paths.base_original_dir] + select_simplified_dirs(paths, args.method, args.peak_idx)

    base_layout = ArcadeRunLayout.from_root(paths.base_original_dir)
    with open(base_layout.targets_json(), encoding="utf-8") as target_file:
        target_metrics = json.load(target_file)
    metric_names = list(target_metrics.keys())

    best_fit_samples_idx_list = []
    for idx, run_dir in enumerate(run_dirs):
        (
            _iterations,
            _mean_errors,
            _best_errors,
            _all_simulation_metrics,
            _all_distance_results,
            best_fit_samples_idx,
        ) = analyze_iterations(
            run_dir,
            metric_names,
            target_metrics,
            n_iterations=args.n_iterations,
            verbose=False,
            find_peaks=idx == 0,
            target_peak=args.peak_idx,
        )
        best_fit_samples_idx_list.append(best_fit_samples_idx[-1])

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    payload_file = cache_payload_path(cache_root, args.method, args.peak_idx)
    payload = {
        "method": args.method,
        "peak_idx": args.peak_idx,
        "n_iterations": args.n_iterations,
        "metric_names": metric_names,
        "target_metrics": target_metrics,
        "base_original_dir": paths.base_original_dir,
        "simplified_posterior_dirs": run_dirs[1:],
        "best_fit_samples_idx_list": best_fit_samples_idx_list,
    }
    with open(payload_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote cache payload: {payload_file}")


if __name__ == "__main__":
    main()
