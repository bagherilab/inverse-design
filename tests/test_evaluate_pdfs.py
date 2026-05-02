"""Tests for analyze.evaluate.estimate_pdfs."""

import numpy as np

from inverse_design.analyze.evaluate import estimate_pdfs


def test_estimate_pdfs_empty_list():
    assert estimate_pdfs([]) == {}


def test_estimate_pdfs_single_sample():
    params = [{"a": 1.0, "b": 2.0}]
    out = estimate_pdfs(params)
    assert set(out.keys()) == {"a", "b"}
    assert out["a"]["mean"] == 1.0
    assert out["b"]["mean"] == 2.0
    assert out["a"]["cov"] == 0.0


def test_estimate_pdfs_kde_on_multiple_samples():
    rng = np.random.default_rng(0)
    params = [{"x": float(rng.normal())} for _ in range(30)]
    out = estimate_pdfs(params)
    assert out["x"]["kde"] is not None
    assert np.isclose(out["x"]["mean"], np.mean([p["x"] for p in params]))
