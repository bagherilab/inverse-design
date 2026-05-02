"""Tests for MetricsBDM and MetricsFactory (BDM path)."""

import numpy as np
import pytest

from inverse_design.common.enum import Metric
from inverse_design.metrics.metrics import MetricsBDM, MetricsFactory
from inverse_design.models.bdm.grid import Grid


@pytest.fixture(autouse=True)
def _init_default_metrics():
    Metric.get("density")
    Metric.get("time_to_equilibrium")


@pytest.fixture
def simple_bdm_output():
    g0 = Grid(4)
    g1 = Grid(4)
    return {"time_points": np.array([0.0, 10.0]), "grid_states": [g0, g1]}


def test_metrics_factory_bdm(simple_bdm_output):
    m = MetricsFactory.create_metrics("BDM", simple_bdm_output)
    assert isinstance(m, MetricsBDM)


def test_metrics_factory_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        MetricsFactory.create_metrics("NOT_A_MODEL", {})


def test_metrics_bdm_density_in_range(simple_bdm_output):
    m = MetricsBDM(simple_bdm_output)
    d = m.calculate_density()
    assert 0.0 <= d <= 100.0


def test_metrics_bdm_calculate_metric_uses_enum(simple_bdm_output):
    Metric.get("density")
    Metric.get("time_to_equilibrium")
    m = MetricsBDM(simple_bdm_output)
    d = m.calculate_metric(Metric.DENSITY)
    assert isinstance(d, float)
