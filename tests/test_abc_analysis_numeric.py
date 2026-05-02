"""Tests for pure numeric helpers in analyze.abc_analysis."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from inverse_design.analyze.abc_analysis import (
    calculate_absolute_error,
    calculate_normalized_error,
    calculate_squared_error,
)


def test_calculate_absolute_error():
    v = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(calculate_absolute_error(v, 1.0), np.array([1.0, 0.0, 1.0]))


def test_calculate_squared_error():
    v = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_array_equal(calculate_squared_error(v, 0.0), np.array([1.0, 0.0, 4.0]))


def test_calculate_normalized_error_absolute():
    metric_values = np.array([10.0, 20.0, 30.0])
    err, scaler = calculate_normalized_error(metric_values, 20.0, loss_function="absolute")
    assert err.shape == (3, 1)
    assert isinstance(scaler, MinMaxScaler)
