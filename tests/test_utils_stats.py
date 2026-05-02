"""Tests for small utilities used in analysis pipelines."""

import numpy as np
import pandas as pd

from inverse_design.utils.utils import remove_outliers


def test_remove_outliers_empty_series():
    s = pd.Series([], dtype=float)
    filtered, idx = remove_outliers(s, iqr_multiplier=1.5)
    assert len(filtered) == 0
    assert idx == []


def test_remove_outliers_removes_extreme_rows():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 100.0], "b": [1.0, 2.0, 3.0, 4.0, 5.0]})
    filtered, outlier_idx = remove_outliers(df, iqr_multiplier=1.5)
    assert 100.0 not in filtered["a"].values
    assert len(outlier_idx) >= 1
