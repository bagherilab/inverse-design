"""Tests for PCA cluster ARCADE input export."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from inverse_design.vis.vis_posterior_overview import (
    DEFAULT_ARCADE_TEMPLATE,
    export_pca_cluster_arcade_inputs,
    extract_pca_cluster_parameters,
    write_arcade_input_from_params,
)


def test_extract_pca_cluster_parameters_includes_mean_mode_and_clusters():
    rng = np.random.default_rng(4)
    cluster_a = rng.normal(
        loc=[2200, 5.0, 0.001, 0.1], scale=[20, 0.2, 0.00005, 0.01], size=(40, 4)
    )
    cluster_b = rng.normal(
        loc=[2600, 7.0, 0.0012, 0.12], scale=[20, 0.2, 0.00005, 0.01], size=(40, 4)
    )
    posterior_df = pd.DataFrame(
        np.vstack([cluster_a, cluster_b]),
        columns=[
            "CELL_VOLUME_MU",
            "COMPRESSION_TOLERANCE",
            "BASAL_ENERGY_MU",
            "LACTATE_RATE_MU",
        ],
    )

    summary_df = extract_pca_cluster_parameters(posterior_df)

    assert {"posterior_mean", "posterior_mode"}.issubset(set(summary_df["label"]))
    assert (summary_df["kind"] == "pca_peak_inverse").any()
    assert {"pc1", "pc2", "CELL_VOLUME_MU", "COMPRESSION_TOLERANCE"}.issubset(summary_df.columns)


def test_write_arcade_input_from_params_replaces_population_and_source_values(tmp_path):
    output_xml = tmp_path / "input.xml"

    write_arcade_input_from_params(
        {
            "CELL_VOLUME_MU": 1234.56,
            "CELL_VOLUME_SIGMA": 42.2,
            "COMPRESSION_TOLERANCE": 5.5,
            "SYNTHESIS_DURATION_MU": 700,
            "GLUCOSE_CONCENTRATION": 0.0075,
            "CAPILLARY_DENSITY": 242.23602484472045,
        },
        output_path=output_xml,
        template_path=DEFAULT_ARCADE_TEMPLATE,
    )

    root = ET.parse(output_xml).getroot()
    cell_volume = root.find(".//population.parameter[@id='CELL_VOLUME']").get("value")
    tolerance = root.find(".//population.parameter[@id='COMPRESSION_TOLERANCE']").get("value")
    synthesis = root.find(".//population.parameter[@id='proliferation/SYNTHESIS_DURATION']").get(
        "value"
    )
    glucose_values = [
        elem.get("value") for elem in root.findall(".//layer[@id='GLUCOSE']/layer.parameter")
    ]
    x_spacing = root.find(".//component.parameter[@id='X_SPACING']").get("value")

    assert cell_volume == "NORMAL(MU=1234.6,SIGMA=42.2)"
    assert tolerance == "5.500000"
    assert synthesis == "NORMAL(MU=700.000,SIGMA=20)"
    assert "0.0075" in glucose_values
    assert x_spacing.startswith("*:")


def test_export_pca_cluster_arcade_inputs_writes_summary_and_inputs(tmp_path):
    rng = np.random.default_rng(8)
    posterior_df = pd.DataFrame(
        rng.normal(size=(80, 4)),
        columns=[
            "CELL_VOLUME_MU",
            "COMPRESSION_TOLERANCE",
            "BASAL_ENERGY_MU",
            "LACTATE_RATE_MU",
        ],
    )
    posterior_df["CELL_VOLUME_MU"] = posterior_df["CELL_VOLUME_MU"] * 25 + 2300
    posterior_df["COMPRESSION_TOLERANCE"] = posterior_df["COMPRESSION_TOLERANCE"] * 0.3 + 5
    posterior_df["BASAL_ENERGY_MU"] = posterior_df["BASAL_ENERGY_MU"] * 0.00005 + 0.001
    posterior_df["LACTATE_RATE_MU"] = posterior_df["LACTATE_RATE_MU"] * 0.005 + 0.1
    posterior_df["DISTANCE_TO_CENTER"] = np.nan

    summary_df = export_pca_cluster_arcade_inputs(
        posterior_df,
        output_dir=tmp_path,
        template_path=DEFAULT_ARCADE_TEMPLATE,
    )

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "cluster_summary.txt").exists()
    assert (tmp_path / "parameter_log.csv").exists()
    assert len(list((tmp_path / "inputs").glob("input_*.xml"))) == len(summary_df)
