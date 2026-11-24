"""UNIFORM MIN <= MAX constraints and the odd radius/margin warning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inverse_design.gui import _runner_core as core
from inverse_design.gui import constraints as order
from inverse_design.gui import run as gui_run
from inverse_design.gui import xml_template
from inverse_design.gui.spec import (
    Hyperparameters,
    MetricSpec,
    ParameterSpec,
    RunSpec,
    SimulationSettings,
    load_spec,
    save_spec,
)

from .test_run import ACCURACY_KEY, VOLUME_KEY, fake  # noqa: F401  (pytest fixture)

AGE_MIN_KEY = "population:cancerous:CELL_AGE.MIN"
AGE_MAX_KEY = "population:cancerous:CELL_AGE.MAX"

TEMPLATE = """<set prefix="t_">
    <series name="grid" start="0" end="1" ticks="2880" interval="720" radius="6" margin="2">
        <agents><populations><population id="cancerous" init="7" class="cancer">
            <population.parameter id="CELL_VOLUME" value="NORMAL(MU=2250,SIGMA=200)" />
            <population.parameter id="ACCURACY" value="0.8" />
            <population.parameter id="CELL_AGE" value="UNIFORM(MIN=100,MAX=300)" />
        </population></populations></agents>
    </series>
</set>
"""


def _detected(tmp_path: Path):
    path = tmp_path / "template.xml"
    path.write_text(TEMPLATE)
    return path, xml_template.detect_parameters(path)


def _age(min_range, max_range=None):
    params = [ParameterSpec(AGE_MIN_KEY, "CELL_AGE_MIN", 100.0, *min_range)]
    if max_range is not None:
        params.append(ParameterSpec(AGE_MAX_KEY, "CELL_AGE_MAX", 300.0, *max_range))
    return params


def test_constraint_found_for_both_or_one_perturbed_side(tmp_path):
    _, detected = _detected(tmp_path)
    (both,) = order.uniform_order_constraints(_age((50, 150), (150, 450)), detected)
    assert (both.low_label, both.high_label) == ("CELL_AGE_MIN", "CELL_AGE_MAX")
    assert both.holds({"CELL_AGE_MIN": 120, "CELL_AGE_MAX": 130})
    assert not both.holds({"CELL_AGE_MIN": 140, "CELL_AGE_MAX": 130})

    (one,) = order.uniform_order_constraints(_age((50, 400)), detected)
    assert one.high_label is None and one.high_fixed == 300.0
    assert one.holds({"CELL_AGE_MIN": 300}) and not one.holds({"CELL_AGE_MIN": 301})
    assert "fixed at 300" in one.describe()

    plain = [ParameterSpec(ACCURACY_KEY, "ACCURACY", 0.8, 0.4, 1.0)]
    assert order.uniform_order_constraints(plain, detected) == []


def test_order_problems_block_impossible_and_warn_on_overlap(tmp_path):
    _, detected = _detected(tmp_path)
    params = _age((50, 150), (150, 450))  # touching ranges: always valid
    assert order.order_problems(params, order.uniform_order_constraints(params, detected)) == (
        [],
        [],
    )
    params = _age((50, 400), (150, 450))
    problems, warnings = order.order_problems(
        params, order.uniform_order_constraints(params, detected)
    )
    assert problems == [] and "overlap" in warnings[0]
    params = _age((350, 400))  # MAX fixed at 300
    problems, _ = order.order_problems(params, order.uniform_order_constraints(params, detected))
    assert "can never hold" in problems[0]


def test_constrained_sobol_is_unchanged_without_constraints_and_valid_with_them(tmp_path):
    _, detected = _detected(tmp_path)
    ranges = {"CELL_AGE_MIN": (0.0, 1.0), "CELL_AGE_MAX": (0.0, 1.0)}
    labels = list(ranges)
    plain = core.constrained_sobol(ranges, labels, 5, 42, [])
    samples = core._sample_sobol_parameters(ranges, 5, 42)
    assert np.array_equal(plain, np.column_stack([samples[k] for k in labels]))

    params = [ParameterSpec(AGE_MIN_KEY, k, 0, 0, 1) for k in labels[:1]]
    params.append(ParameterSpec(AGE_MAX_KEY, labels[1], 0, 0, 1))
    cons = order.uniform_order_constraints(params, detected)
    drawn = core.constrained_sobol(ranges, labels, 5, 42, cons)
    assert drawn.shape == (32, 2) and (drawn[:, 0] <= drawn[:, 1]).all()


def test_box_prior_rejects_min_above_max(tmp_path):
    _, detected = _detected(tmp_path)
    params = _age((50, 400), (150, 450))
    cons = order.uniform_order_constraints(params, detected)
    ranges = {p.label: (p.min, p.max) for p in params}
    cols = ["CELL_AGE_MIN", "CELL_AGE_MAX"]
    assert core.box_prior_pdf(np.array([200.0, 300.0]), cols, ranges, constraints=cons) == 1.0
    assert core.box_prior_pdf(np.array([350.0, 300.0]), cols, ranges, constraints=cons) == 0.0
    assert core.box_prior_pdf(np.array([350.0, 300.0]), cols, ranges) == 1.0


def test_validate_against_plugins_reports_impossible_order(tmp_path):
    path, _ = _detected(tmp_path)
    spec = RunSpec(
        template_path=str(path),
        jar_path="arcade.jar",
        simulation=SimulationSettings(seeds=1, ticks=2880, interval=720, radius=6, margin=2),
        parameters=_age((350, 400)),
        metrics=[MetricSpec("n_cells", "builtin", 10.0)],
    )
    assert any("can never hold" in p for p in core.validate_against_plugins(spec))


def test_simulation_warnings_for_odd_radius_and_margin():
    sim = SimulationSettings(seeds=1, ticks=2880, interval=720, radius=5, margin=1)
    warnings = order.simulation_warnings(sim)
    assert len(warnings) == 2
    assert "rounds it up to 6" in warnings[0] and "rounds it up to 2" in warnings[1]
    sim.radius, sim.margin = 6, 2
    assert order.simulation_warnings(sim) == []


def test_full_run_keeps_min_below_max(tmp_path, fake, capsys):  # noqa: F811
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "template.xml").write_text(TEMPLATE)
    (run_dir / "arcade.jar").write_bytes(b"")
    spec = RunSpec(
        template_path="template.xml",
        jar_path="arcade.jar",
        simulation=SimulationSettings(seeds=1, ticks=2880, interval=720, radius=5, margin=2),
        parameters=[
            ParameterSpec(VOLUME_KEY, "CELL_VOLUME_MU", 2250.0, 1000.0, 3000.0),
            ParameterSpec(ACCURACY_KEY, "ACCURACY", 0.8, 0.2, 1.0),
            *_age((50, 400), (100, 450)),
        ],
        metrics=[MetricSpec("growth", "builtin", 2.0)],
        hyperparameters=Hyperparameters(
            n_particles=16, n_generations=3, n_trees=10, min_samples_leaf=3, n_cpu=1
        ),
    )
    save_spec(spec, run_dir / "run_spec.json")
    loaded = load_spec(run_dir / "run_spec.json")

    plan = gui_run.format_plan(loaded)
    assert "CELL_AGE_MIN <= CELL_AGE_MAX" in plan
    assert "radius 5 is odd" in plan and "overlap" in plan

    gui_run.run(loaded)
    for g in range(3):
        params = pd.read_csv(run_dir / "arcade_outputs" / f"iter_{g}" / "all_param_df.csv")
        assert len(params) == 16
        assert (params["CELL_AGE_MIN"] <= params["CELL_AGE_MAX"]).all(), g
