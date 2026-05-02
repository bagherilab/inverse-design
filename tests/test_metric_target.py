"""Tests for Metric and Target (used across ABC and ARCADE workflows)."""

import uuid

from inverse_design.common.enum import Metric, Target


def test_metric_get_symmetry():
    m = Metric.get("symmetry")
    assert m.value == "symmetry"
    assert Metric.get("symmetry") is m


def test_metric_get_creates_unknown_ephemeral():
    name = f"_test_metric_{uuid.uuid4().hex}"
    try:
        m = Metric.get(name)
        assert m.value == name
        assert Metric.get(name) is m
    finally:
        Metric._metrics.pop(name, None)


def test_target_dataclass():
    m = Metric.get("symmetry")
    t = Target(metric=m, value=0.8, weight=1.0)
    assert t.metric == m
    assert t.value == 0.8
    assert t.weight == 1.0
