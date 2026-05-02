import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from inverse_design.vis.vis_simplified import plot_simplification_figure


FAKE_PARAMS = ["alpha", "beta", "gamma", "delta", "epsilon"]
THRESH_LABELS = ["t=1.0", "t=1.25", "t=1.5"]


def _make_panel(color: str = "#4daf4a", seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    importance = rng.uniform(0.1, 1.0, len(FAKE_PARAMS))
    importance = (importance / importance.max()).tolist()
    sorted_idx = np.argsort(-np.array(importance))
    sorted_params = [FAKE_PARAMS[i] for i in sorted_idx]
    sorted_importance = [importance[i] for i in sorted_idx]
    return {
        "label": "Cluster 1",
        "color": color,
        "param_names": sorted_params,
        "importance": sorted_importance,
        "dropped": {
            "t=1.0": set(),
            "t=1.25": {"alpha"},
            "t=1.5": {"alpha", "beta"},
        },
        "errors": [0.15, 0.16, 0.19],
        "original_error": 0.14,
    }


def test_returns_fig_and_axes_dict():
    panels = [_make_panel("#4daf4a"), _make_panel("#984ea3", seed=7)]
    fig, axes_dict = plot_simplification_figure(panels, threshold_labels=THRESH_LABELS)
    assert isinstance(fig, plt.Figure)
    assert isinstance(axes_dict, dict)
    plt.close("all")


def test_axes_dict_has_expected_keys_two_clusters():
    panels = [_make_panel("#4daf4a"), _make_panel("#984ea3", seed=7)]
    fig, axes_dict = plot_simplification_figure(panels, threshold_labels=THRESH_LABELS)
    for key in ("imp_c0", "imp_c1", "grid_c0", "grid_c1", "accuracy"):
        assert key in axes_dict, f"Missing key: {key}"
    plt.close("all")


def test_single_cluster_panel_does_not_raise():
    panels = [_make_panel("#4daf4a")]
    fig, axes_dict = plot_simplification_figure(panels, threshold_labels=THRESH_LABELS)
    assert isinstance(fig, plt.Figure)
    assert "imp_c0" in axes_dict
    assert "grid_c0" in axes_dict
    assert "accuracy" in axes_dict
    plt.close("all")


def test_no_dropped_params_does_not_raise():
    panels = [_make_panel("#4daf4a"), _make_panel("#984ea3", seed=7)]
    for panel in panels:
        for threshold_label in THRESH_LABELS:
            panel["dropped"][threshold_label] = set()
    fig, _ = plot_simplification_figure(panels, threshold_labels=THRESH_LABELS)
    assert isinstance(fig, plt.Figure)
    plt.close("all")


def test_save_path_none_does_not_raise():
    panels = [_make_panel("#4daf4a"), _make_panel("#984ea3", seed=7)]
    fig, _ = plot_simplification_figure(panels, threshold_labels=THRESH_LABELS, save_path=None)
    plt.close("all")
