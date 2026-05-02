"""Smoke tests for equifinality figure changes."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from inverse_design.vis.vis_parameter_comparison import plot_multi_boxcharts
from inverse_design.vis.vis_posterior_overview import plot_pca_with_peaks


@pytest.fixture
def posterior_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(size=(80, 6)), columns=[f"P{i}" for i in range(6)])


@pytest.fixture
def target_df():
    return pd.DataFrame(
        {
            "doub_time": [50.0],
            "doub_time_std": [3.0],
            "symmetry": [0.80],
            "symmetry_std": [0.03],
            "colony_growth": [20.0],
        }
    )


def test_pca_default_still_works(posterior_df):
    fig, _ = plot_pca_with_peaks(posterior_df)
    assert fig is not None
    plt.close(fig)


def test_pca_accepts_font_size(posterior_df):
    fig, _ = plot_pca_with_peaks(posterior_df, font_size=8)
    assert fig is not None
    plt.close(fig)


def test_pca_accepts_cluster_fill_alpha(posterior_df):
    fig, _ = plot_pca_with_peaks(posterior_df, cluster_fill_alpha=0.12)
    assert fig is not None
    plt.close(fig)


def test_pca_accepts_point_size(posterior_df):
    fig, _ = plot_pca_with_peaks(posterior_df, point_size=9, cluster_fill_alpha=0.0)
    assert fig is not None
    plt.close(fig)


def test_pca_accepts_fig_ax(posterior_df):
    fig, ax = plt.subplots()
    result_fig, _ = plot_pca_with_peaks(posterior_df, fig=fig, ax=ax)
    assert result_fig is fig
    plt.close(fig)


def test_boxcharts_default_still_works(target_df, tmp_path):
    csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "doub_time": [48.0, 51.0],
            "symmetry": [0.81, 0.79],
            "colony_growth": [19.0, 22.0],
        }
    ).to_csv(csv, index=False)
    fig, _ = plot_multi_boxcharts(
        target_df,
        [csv],
        ["EXP", "C1"],
        target_metrics=["doub_time", "symmetry", "colony_growth"],
    )
    assert fig is not None
    plt.close(fig)


def test_boxcharts_accepts_case_colors(target_df, tmp_path):
    csv = tmp_path / "metrics.csv"
    pd.DataFrame({"doub_time": [48.0], "symmetry": [0.81], "colony_growth": [19.0]}).to_csv(
        csv, index=False
    )
    fig, _ = plot_multi_boxcharts(
        target_df,
        [csv],
        ["EXP", "C1"],
        target_metrics=["doub_time", "symmetry", "colony_growth"],
        case_colors=["#4daf4a"],
    )
    assert fig is not None
    plt.close(fig)


def test_boxcharts_accepts_font_size(target_df, tmp_path):
    csv = tmp_path / "metrics.csv"
    pd.DataFrame({"doub_time": [48.0], "symmetry": [0.81], "colony_growth": [19.0]}).to_csv(
        csv, index=False
    )
    fig, _ = plot_multi_boxcharts(
        target_df,
        [csv],
        ["EXP", "C1"],
        target_metrics=["doub_time", "symmetry", "colony_growth"],
        font_size=8,
    )
    assert fig is not None
    plt.close(fig)


def test_boxcharts_accepts_iqr_error_and_marker_y(target_df, tmp_path):
    csv = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "doub_time": [45.0, 48.0, 51.0, 54.0],
            "symmetry": [0.76, 0.80, 0.84, 0.88],
            "colony_growth": [16.0, 19.0, 22.0, 25.0],
        }
    ).to_csv(csv, index=False)
    fig, _ = plot_multi_boxcharts(
        target_df,
        [csv],
        ["EXP", "C1"],
        target_metrics=["doub_time", "symmetry", "colony_growth"],
        error_stat="iqr",
        marker_label_y=-0.055,
    )
    assert fig is not None
    plt.close(fig)


def test_boxcharts_accepts_metric_titles(target_df, tmp_path):
    csv = tmp_path / "metrics.csv"
    pd.DataFrame({"doub_time": [48.0], "symmetry": [0.81], "colony_growth": [19.0]}).to_csv(
        csv, index=False
    )
    fig, axes = plot_multi_boxcharts(
        target_df,
        [csv],
        ["EXP", "C1"],
        target_metrics=["doub_time", "symmetry", "colony_growth"],
        metric_titles={"doub_time": "Doubling time"},
    )
    assert axes[0].get_title() == "Doubling time"
    plt.close(fig)
