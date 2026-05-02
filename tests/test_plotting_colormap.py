"""plotting.colormap — metric / peak / PCA colors."""

from inverse_design.plotting.colormap import (
    DEFAULT_METRIC_COLORS,
    PEAK_COLORS,
    PCA_CLUSTER_COLORS,
    light_peak_fill,
    merged_metric_colors,
    metric_color,
    metric_colors_for_keys,
    pca_cluster_color,
    peak_color,
    sir_compartment_color,
)


def test_metric_color_symmetry_matches_canonical():
    assert metric_color("symmetry") == DEFAULT_METRIC_COLORS["symmetry"]
    assert metric_color("SYMMETRY") == DEFAULT_METRIC_COLORS["symmetry"]


def test_metric_color_unknown_returns_default():
    assert metric_color("not_a_metric_xyz", default="#abc") == "#abc"


def test_peak_color_cycles():
    assert peak_color(0) == PEAK_COLORS[0]
    assert peak_color(len(PEAK_COLORS)) == PEAK_COLORS[0]


def test_pca_cluster_color_has_eight_slots():
    assert len(PCA_CLUSTER_COLORS) == 8
    assert pca_cluster_color(7) == PCA_CLUSTER_COLORS[7]


def test_light_peak_fill_cycles():
    assert light_peak_fill(0) == "lightcoral"


def test_merged_metric_colors_includes_supplemental():
    m = merged_metric_colors()
    assert m["n_cells"] == "#a790c1"
    assert "peak_i" in m


def test_metric_colors_for_keys_order():
    keys = ("symmetry", "doub_time", "act_ratio")
    assert metric_colors_for_keys(keys) == [
        DEFAULT_METRIC_COLORS["symmetry"],
        DEFAULT_METRIC_COLORS["doub_time"],
        DEFAULT_METRIC_COLORS["act_ratio"],
    ]


def test_sir_compartment_matches_metric_hex():
    assert sir_compartment_color("I") == DEFAULT_METRIC_COLORS["act_ratio"]
    assert sir_compartment_color("R") == DEFAULT_METRIC_COLORS["symmetry"]
    assert sir_compartment_color("S") == DEFAULT_METRIC_COLORS["colony_growth"]
