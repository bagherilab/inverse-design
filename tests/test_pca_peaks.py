"""Tests for PCA + peak helpers in analyze.core.pca_peaks."""

import numpy as np
import pytest

from inverse_design.analyze.core.pca_peaks import perform_pca_and_find_peaks


@pytest.fixture
def two_cluster_data():
    """100 x 5 array: two separated Gaussian blobs (enough for KDE)."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal((50, 5)) + np.array([5.0, 0.0, 0.0, 0.0, 0.0])
    b = rng.standard_normal((50, 5)) + np.array([-5.0, 0.0, 0.0, 0.0, 0.0])
    return np.vstack([a, b])


def test_perform_pca_and_find_peaks_returns_tuple_of_expected_length(two_cluster_data):
    out = perform_pca_and_find_peaks(two_cluster_data, n_components=2, random_state=0)
    assert len(out) == 8

    pca_result, peak_positions, point_colors, pca, Z, X, Y, peak_points = out

    assert pca_result.shape == (100, 2)
    assert pca.n_components_ == 2
    assert Z.shape == X.shape == Y.shape
    assert len(point_colors) == 100
    assert isinstance(peak_points, list)
    assert peak_positions.ndim == 2
    assert peak_positions.shape[1] == 2


def test_perform_pca_and_find_peaks_finds_at_least_one_peak(two_cluster_data):
    out = perform_pca_and_find_peaks(two_cluster_data, n_components=2, random_state=0)
    peak_positions = out[1]
    assert len(peak_positions) >= 1
