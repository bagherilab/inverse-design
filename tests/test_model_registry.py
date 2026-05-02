"""Tests for ModelRegistry and registered model facades."""

import pytest

from inverse_design.common.enum import Metric
from inverse_design.models.model_base import ARCADEModel, BDMModel, ModelRegistry


@pytest.fixture(autouse=True)
def _init_metrics_for_targets():
    Metric.get("density")
    Metric.get("time_to_equilibrium")
    Metric.get("activity")


def test_registry_bdm():
    m = ModelRegistry.get_model("BDM")
    assert isinstance(m, BDMModel)
    keys = m.get_parameter_keys()
    assert "proliferate" in keys


def test_registry_arcade():
    m = ModelRegistry.get_model("ARCADE")
    assert isinstance(m, ARCADEModel)


def test_registry_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model type"):
        ModelRegistry.get_model("NOPE")


def test_bdm_default_targets_use_metrics():
    m = ModelRegistry.get_model("BDM")
    targets = m.get_default_targets()
    assert len(targets) >= 1
    assert all(isinstance(t.metric, Metric) for t in targets)
