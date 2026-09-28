#!/usr/bin/env python3
"""Recompute the ARCADE predictive errors plotted in S4 Fig from the pm50 CSVs.

For each generation, every target metric is summarized by the median of the
particle values left after dropping values outside the 1.5 IQR fence; the error
is the absolute target-relative percent error of that median.  S4 Fig plots the
mean across the five targets with the population SD across the five targets.

Rows: generations 0--9 of ``ABC_SMC_RF_N1024_pm50`` and generation 4 of the main
``ABC_SMC_RF_N{128,256,512,1024}_pm50`` chains.  The script fails if any value
differs from the constants in ``regenerate_s3_empirical_stopping.py``.

Usage:
    arcade_generation_errors.py PM50_ROOT [--output CSV]

PM50_ROOT is the extracted ``pm50_campaign_csv_20260807.tar.gz``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import regenerate_s3_empirical_stopping as s4

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO / "analysis_outputs/reviewer_closeout/arcade_generation_errors.csv"


def drop_iqr_outliers(values: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return values
    return values[(values >= q25 - 1.5 * iqr) & (values <= q75 + 1.5 * iqr)]


def generation_error(run: Path, generation: int) -> tuple[float, float, dict[str, float]]:
    targets = json.loads((run / "targets.json").read_text())
    metrics = pd.read_csv(run / f"iter_{generation}/final_metrics.csv")
    errors = {
        name: abs(np.median(drop_iqr_outliers(metrics[name].to_numpy(float))) - target) / target * 100
        for name, target in targets.items()
    }
    values = np.array(list(errors.values()))
    return float(values.mean()), float(values.std()), errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pm50_root", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = []
    for generation in range(10):
        mean, sd, errors = generation_error(args.pm50_root / "ABC_SMC_RF_N1024_pm50", generation)
        rows.append(dict(panel="A", run="ABC_SMC_RF_N1024_pm50", N=1024, generation=generation, mae_pct=mean, sd_pct=sd, **errors))
    for n in s4.N_VALUES:
        mean, sd, errors = generation_error(args.pm50_root / f"ABC_SMC_RF_N{n}_pm50", 4)
        rows.append(dict(panel="B", run=f"ABC_SMC_RF_N{n}_pm50", N=int(n), generation=4, mae_pct=mean, sd_pct=sd, **errors))
    frame = pd.DataFrame(rows)

    checks = [
        (frame.query("panel == 'A'")["mae_pct"], s4.ARCADE_GENERATION_MEAN),
        (frame.query("panel == 'A'")["sd_pct"], s4.ARCADE_GENERATION_SD),
        (frame.query("panel == 'B'")["mae_pct"], s4.ARCADE_N_MEAN),
        (frame.query("panel == 'B'")["sd_pct"], s4.ARCADE_N_SD),
    ]
    for computed, published in checks:
        if not np.allclose(computed.to_numpy(), published, atol=5e-7, rtol=0):
            raise SystemExit(f"Mismatch with S4 Fig constants:\n{computed.to_numpy()}\n{published}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.6f")
    print(f"Wrote {args.output} ({len(frame)} rows); all values match S4 Fig")


if __name__ == "__main__":
    main()
