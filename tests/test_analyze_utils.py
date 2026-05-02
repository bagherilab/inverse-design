"""Tests for analyze.analyze_utils helpers."""

import json

import numpy as np
import pandas as pd
import pytest

from inverse_design.analyze.utils.analyze_utils import (
    analyze_metric_percentiles,
    get_parameters_from_json,
)


def test_analyze_metric_percentiles_invalid_percentile_raises(tmp_path):
    csv_path = tmp_path / "m.csv"
    pd.DataFrame({"score": [1, 2, 3, 4, 5]}).to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Percentile must be between"):
        analyze_metric_percentiles(str(csv_path), "score", percentile=60)


def test_analyze_metric_percentiles_labels(tmp_path):
    csv_path = tmp_path / "m.csv"
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0, 1, 50), rng.normal(10, 1, 50)])
    pd.DataFrame({"score": scores, "folder": [f"f{i}" for i in range(100)]}).to_csv(
        csv_path, index=False
    )
    high, low, all_data = analyze_metric_percentiles(
        str(csv_path), "score", percentile=10, verbose=False
    )
    assert "percentile_label" in all_data.columns
    assert len(high) >= 1
    assert len(low) >= 1


def test_get_parameters_from_json(tmp_path):
    cfg = {"populations": {"cancerous": {"affinity_mu": 1.23, "volume_mu": 4.56}}}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    out = get_parameters_from_json(str(tmp_path), ["affinity_mu", "volume_mu"])
    assert out["affinity_mu"] == 1.23
    assert out["volume_mu"] == 4.56
