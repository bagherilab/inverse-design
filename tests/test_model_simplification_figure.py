import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


def _fake_corr(n=6, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((60, n))
    return np.corrcoef(x, rowvar=False)


FAKE_NAMES = np.array([f"PARAM_{i}_MU" for i in range(6)])


def test_colorize_ward_accepts_external_ax():
    from inverse_design.plotting.plot_dendrogram import (
        colorize_ward_clusters_threshold,
    )

    fig, ax = plt.subplots()
    result = colorize_ward_clusters_threshold(
        data=_fake_corr(),
        param_names=FAKE_NAMES,
        ward_threshold=0.5,
        ax=ax,
        verbose=False,
    )
    assert result is not None
    assert len(result) == 6
    plt.close("all")


def test_colorize_ward_standalone_still_works():
    from inverse_design.plotting.plot_dendrogram import (
        colorize_ward_clusters_threshold,
    )

    result = colorize_ward_clusters_threshold(
        data=_fake_corr(),
        param_names=FAKE_NAMES,
        ward_threshold=0.5,
        verbose=False,
    )
    assert result is not None
    plt.close("all")


def test_short_name_strips_mu_and_formats():
    from inverse_design.plotting.plot_model_simplification import short_name

    assert short_name("GLUCOSE_UPTAKE_RATE_MU") == "Glucose Uptake Rate"
    assert short_name("OXYGEN_CONCENTRATION") == "Oxygen Concentration"
    assert short_name("CAPILLARY_DENSITY") == "Capillary Density"


def test_sentence_case_name_formats_axis_labels():
    from inverse_design.plotting.plot_model_simplification import sentence_case_name

    assert sentence_case_name("GLUCOSE_UPTAKE_RATE_MU") == "Glucose uptake rate"
    assert sentence_case_name("COMPRESSION_TOLERANCE") == "Compression tolerance"
    assert sentence_case_name("ATP_PRODUCTION_RATE_MU") == "ATP production rate"


def test_load_lr_data_returns_sorted_models(tmp_path):
    from inverse_design.plotting.plot_model_simplification import load_lr_data

    payload = {
        "prediction_models": {
            "models": [
                {
                    "target_parameter": {"name": "PARAM_A_MU"},
                    "predictor_parameter": {"name": "PARAM_B_MU"},
                    "linear_model": {"slope": 0.5, "intercept": 0.1},
                    "prediction_quality": {"correlation_r": 0.85},
                },
                {
                    "target_parameter": {"name": "PARAM_C_MU"},
                    "predictor_parameter": {"name": "PARAM_D_MU"},
                    "linear_model": {"slope": 1.2, "intercept": -0.3},
                    "prediction_quality": {"correlation_r": 0.95},
                },
            ]
        }
    }
    lr_json = tmp_path / "lr.json"
    lr_json.write_text(json.dumps(payload))

    models = load_lr_data(lr_json)
    assert len(models) == 2
    assert models[0]["prediction_quality"]["correlation_r"] == pytest.approx(0.95)


def test_load_dend_data_returns_expected_keys(tmp_path):
    from inverse_design.plotting.plot_model_simplification import load_dend_data

    payload = {
        "representative_parameters": ["PARAM_A_MU"],
        "redundant_params": {"PARAM_A_MU": ["PARAM_B_MU"]},
        "individual_params": ["PARAM_C_MU"],
    }
    dend_json = tmp_path / "dend.json"
    dend_json.write_text(json.dumps(payload))

    rep, redundant, individual = load_dend_data(dend_json)
    assert rep == ["PARAM_A_MU"]
    assert redundant == {"PARAM_A_MU": ["PARAM_B_MU"]}
    assert individual == ["PARAM_C_MU"]


def test_draw_scatter_panels_does_not_raise():
    from inverse_design.plotting.plot_model_simplification import draw_scatter_panels

    models = [
        {
            "predictor_parameter": {"name": "PARAM_A_MU"},
            "target_parameter": {"name": "PARAM_B_MU"},
            "linear_model": {"slope": 0.5, "intercept": 0.1},
            "prediction_quality": {"correlation_r": 0.93},
        },
        {
            "predictor_parameter": {"name": "PARAM_C_MU"},
            "target_parameter": {"name": "PARAM_D_MU"},
            "linear_model": {"slope": -1.2, "intercept": -0.3},
            "prediction_quality": {"correlation_r": 0.88},
        },
        {
            "predictor_parameter": {"name": "PARAM_E_MU"},
            "target_parameter": {"name": "PARAM_F_MU"},
            "linear_model": {"slope": 0.8, "intercept": 0.05},
            "prediction_quality": {"correlation_r": 0.82},
        },
    ]
    rng = np.random.default_rng(0)
    posterior = pd.DataFrame(
        {
            "PARAM_A_MU": rng.uniform(0, 1, 50),
            "PARAM_B_MU": rng.uniform(0, 1, 50),
            "PARAM_C_MU": rng.uniform(0, 1, 50),
            "PARAM_D_MU": rng.uniform(0, 1, 50),
            "PARAM_E_MU": rng.uniform(0, 1, 50),
            "PARAM_F_MU": rng.uniform(0, 1, 50),
        }
    )
    fig, axes = plt.subplots(1, 3)
    draw_scatter_panels(axes, models[:3], posterior)
    assert "y =" in axes[0].texts[0].get_text()
    assert r"\times 10^" in axes[0].texts[0].get_text()
    assert "R = -0.88" in axes[1].texts[0].get_text()
    assert axes[0].get_box_aspect() == pytest.approx(1.0)
    plt.close("all")


def test_draw_equation_panel_does_not_raise():
    from inverse_design.plotting.plot_model_simplification import draw_equation_panel

    models = [
        {
            "predictor_parameter": {"name": f"PARAM_{chr(65 + i)}_MU"},
            "target_parameter": {"name": f"PARAM_{chr(66 + i)}_MU"},
            "linear_model": {"slope": 0.5 + i * 0.1, "intercept": 0.01 * i},
            "prediction_quality": {"correlation_r": 0.95 - i * 0.02},
        }
        for i in range(12)
    ]
    fig, ax = plt.subplots()
    draw_equation_panel(ax, models)
    plt.close("all")


def test_draw_dendrogram_panel_does_not_raise():
    from inverse_design.plotting.plot_model_simplification import draw_dendrogram_panel

    rng = np.random.default_rng(1)
    n = 6
    posterior = pd.DataFrame(
        rng.standard_normal((50, n)),
        columns=[f"PARAM_{i}_MU" for i in range(n)],
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    rep_params_out = draw_dendrogram_panel(ax, posterior, ward_threshold=0.5)
    assert isinstance(rep_params_out, list)
    assert list(ax.get_xticks()) == [0, 0.5, 1.0, 1.5, 2.0]
    assert ax.get_legend() is None
    assert any(label.get_fontweight() == "bold" for label in ax.get_yticklabels())
    assert any(label.get_color() != "black" for label in ax.get_yticklabels())
    plt.close("all")


def test_draw_status_panel_does_not_raise():
    from inverse_design.plotting.plot_model_simplification import draw_status_panel

    rep_params = ["PARAM_A_MU", "PARAM_C_MU"]
    redundant_params = {
        "PARAM_A_MU": ["PARAM_B_MU"],
        "PARAM_C_MU": ["PARAM_D_MU", "PARAM_E_MU"],
    }
    individual_params = ["PARAM_F_MU"]

    fig, ax = plt.subplots()
    draw_status_panel(ax, rep_params, redundant_params, individual_params)
    plt.close("all")
