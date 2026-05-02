"""Tests for pure helpers in rf.rf_utils."""

import numpy as np

from inverse_design.rf.rf_utils import calculate_posterior_statistics


def test_calculate_posterior_statistics_shape_and_keys():
    rng = np.random.default_rng(42)
    posterior = rng.normal(size=(100, 2))
    true_vals = np.array([0.0, 1.0])
    stats = calculate_posterior_statistics(posterior, true_vals, parameter_names=["a", "b"])
    assert set(stats.keys()) == {"a", "b"}
    for name in ("a", "b"):
        assert "mean" in stats[name]
        assert "95%_CI" in stats[name]
        assert "in_95%_CI" in stats[name]
        assert len(stats[name]["95%_CI"]) == 2
