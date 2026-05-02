"""Tests for plot_constraint_propagation_matrix."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from inverse_design.vis.vis_simulation_outcome import plot_constraint_propagation_matrix


@pytest.fixture
def run_data():
    rng = np.random.default_rng(42)
    n = 60

    def make_df(dt_mu, sym_mu, act_mu):
        return pd.DataFrame(
            {
                "doub_time": rng.normal(dt_mu, 5.0, n),
                "symmetry": rng.normal(sym_mu, 0.03, n),
                "act_ratio": rng.normal(act_mu, 0.05, n),
            }
        )

    return [
        ("Fit D=32", "doub_time", make_df(32, 0.90, 0.40)),
        ("Fit S=0.75", "symmetry", make_df(70, 0.75, 0.52)),
        ("Fit A=0.70", "act_ratio", make_df(50, 0.83, 0.70)),
        ("Fit all", None, make_df(88, 0.77, 0.58)),
    ]


def test_returns_fig_and_axes_grid(run_data):
    fig, axes_grid = plot_constraint_propagation_matrix(run_data)
    assert fig is not None
    assert len(axes_grid) == 4
    assert all(len(row) == 3 for row in axes_grid)
    plt.close(fig)


def test_accepts_target_metrics(run_data):
    targets = {"doub_time": 32.0, "symmetry": 0.75, "act_ratio": 0.70}
    fig, _ = plot_constraint_propagation_matrix(run_data, target_metrics=targets)
    plt.close(fig)


def test_no_fit_metric_row_draws_without_error(run_data):
    fig, axes_grid = plot_constraint_propagation_matrix(run_data)
    assert all(ax is not None for ax in axes_grid[3])
    plt.close(fig)


def test_accepts_subplot_spec(run_data):
    fig = plt.figure(figsize=(6, 4))
    from matplotlib import gridspec

    spec = gridspec.GridSpec(1, 1, figure=fig)[0]
    result_fig, _ = plot_constraint_propagation_matrix(run_data, fig=fig, subplot_spec=spec)
    assert result_fig is fig
    plt.close(fig)


def test_remove_outliers_flag(run_data):
    fig, _ = plot_constraint_propagation_matrix(run_data, remove_outliers_flag=False)
    plt.close(fig)
