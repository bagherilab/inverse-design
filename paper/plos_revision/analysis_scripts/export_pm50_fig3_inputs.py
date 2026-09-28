#!/usr/bin/env python3
"""Export the six Fig. 3C parameter sets from the bounded pm50 posterior."""

from __future__ import annotations

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from inverse_design.vis.vis_posterior_overview import export_pca_cluster_arcade_inputs


DROP_COLUMNS = ("input_folder", "X_SPACING", "Y_SPACING", "DISTANCE_TO_CENTER")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("posterior_csv", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    posterior = pd.read_csv(args.posterior_csv)
    summary = export_pca_cluster_arcade_inputs(
        posterior,
        output_dir=args.output_dir,
        template_path=args.template,
        n_components=2,
        threshold_ratio=0.05,
        neighborhood_size=5,
        n_bins=30,
        random_state=0,
        drop_columns=DROP_COLUMNS,
        include_mean_mode=True,
    )
    if len(summary) != 6 or (summary["kind"] == "pca_peak_inverse").sum() != 4:
        raise RuntimeError(
            "expected mean, mode, and four KDE maxima; got %d rows and %d maxima"
            % (len(summary), (summary["kind"] == "pca_peak_inverse").sum())
        )

    for xml_path in sorted((args.output_dir / "inputs").glob("input_*.xml")):
        series = ET.parse(xml_path).getroot().find(".//series")
        if series is None or series.get("start") != "0" or series.get("end") != "9":
            raise RuntimeError(f"{xml_path} is not a ten-seed input")

    provenance = {
        "source_posterior": str(args.posterior_csv),
        "source_posterior_sha256": sha256(args.posterior_csv),
        "template": str(args.template),
        "template_sha256": sha256(args.template),
        "pca_parameters": [
            column for column in posterior.select_dtypes("number").columns if column not in DROP_COLUMNS
        ],
        "drop_columns": list(DROP_COLUMNS),
        "n_components": 2,
        "kde": {
            "covariance_factor": "scott",
            "relative_density_threshold": 0.05,
            "grid": [100, 100],
            "maximum_filter": [5, 5],
        },
        "mode_bins": 30,
        "random_state": 0,
        "cases": ["mean", "mode", "cluster_1", "cluster_2", "cluster_3", "cluster_4"],
        "seeds": list(range(10)),
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("exported six ten-seed XMLs; peak sizes=" + ",".join(
        str(int(value)) for value in summary.loc[summary["kind"] == "pca_peak_inverse", "cluster_size"]
    ))


if __name__ == "__main__":
    main()
