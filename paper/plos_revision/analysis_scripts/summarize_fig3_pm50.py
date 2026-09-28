#!/usr/bin/env python3
"""Reduce the six completed Fig. 3 pm50 ARCADE cases to cached metric CSVs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from inverse_design.analyze.analyze_single_folder import analyze_single_folder


CASES = ("mean", "mode", "cluster_1", "cluster_2", "cluster_3", "cluster_4")
TIMESTAMP_RE = re.compile(r"_(\d{6})\.CELLS\.json$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    metrics_root = args.campaign / "panel_c_metrics"
    manifest = {"cases": {}, "raw_retained": True}
    for case in CASES:
        raw = args.campaign / "CLUSTER" / case
        cells = sorted(raw.glob("*.CELLS.json"))
        locations = sorted(raw.glob("*.LOCATIONS.json"))
        if len(cells) != 150 or len(locations) != 150:
            raise RuntimeError(
                f"{case}: expected 150 CELLS and 150 LOCATIONS, got {len(cells)} and {len(locations)}"
            )
        timestamps = sorted(
            {match.group(1) for path in cells if (match := TIMESTAMP_RE.search(path.name))},
            key=int,
        )
        result = analyze_single_folder(raw, timestamps)
        output = metrics_root / case
        output.mkdir(parents=True, exist_ok=True)
        result["per_seed_df"].to_csv(output / "final_metrics_seed.csv", index=False)
        pd.DataFrame([result["final_metrics"]]).to_csv(output / "final_metrics.csv", index=False)
        manifest["cases"][case] = {
            "cell_json": len(cells),
            "location_json": len(locations),
            "timestamps": timestamps,
            "seed_rows": len(result["per_seed_df"]),
        }
    (metrics_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("summarized six cases into", metrics_root)


if __name__ == "__main__":
    main()
