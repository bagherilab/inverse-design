"""Scenario registry loader tests."""

from inverse_design.io.scenarios import combined_grid_n512_breast, load_registry


def test_load_registry_has_experiment():
    data = load_registry()
    assert "experiments" in data
    assert "combined_grid_n512_breast" in data["experiments"]


def test_combined_grid_n512_breast_counts():
    p = combined_grid_n512_breast()
    assert p.base_original_dir.endswith("combined_grid_breast_only_mean_2")
    assert len(p.linear_model_dirs) == 12
    assert len(p.dendrogram_threshold_dirs) == 12
    assert len(p.ext_linear_dirs) == 4
    assert len(p.ext_dendrogram_threshold_dirs) == 4
    assert "linear_0.9_p1" in p.linear_model_dirs[0]
