"""MetricsFactory ARCADE path."""

import pytest

from inverse_design.common.enum import Metric
from inverse_design.metrics.metrics import MetricsARCADE, MetricsFactory


@pytest.fixture(autouse=True)
def _init_arcade_metrics():
    Metric.get("growth_rate")
    Metric.get("symmetry")
    Metric.get("activity")


def test_metrics_factory_arcade():
    out = {"growth_rate": 0.1, "symmetry": 0.5, "activity": 0.9}
    m = MetricsFactory.create_metrics("ARCADE", out)
    assert isinstance(m, MetricsARCADE)


def test_metrics_arcade_calculate_metric_uses_enum():
    out = {"growth_rate": 1.0, "symmetry": 2.0, "activity": 3.0}
    m = MetricsARCADE(out)
    assert m.calculate_metric(Metric.GROWTH_RATE) == 1.0
    assert m.calculate_metric(Metric.SYMMETRY) == 2.0
    assert m.calculate_metric(Metric.ACTIVITY) == 3.0


def test_metrics_arcade_missing_key_raises():
    m = MetricsARCADE({})
    with pytest.raises(KeyError, match="growth_rate"):
        m.calculate_growth_rate()
