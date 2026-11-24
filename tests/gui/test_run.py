"""Tests for the GUI runner (``inverse_design.gui.run``) with a fake ARCADE model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inverse_design.gui import _runner_core as core
from inverse_design.gui import metrics_plugin, xml_template
from inverse_design.gui import run as gui_run
from inverse_design.gui.spec import (
    Hyperparameters,
    MetricSpec,
    ParameterSpec,
    RunSpec,
    SimulationSettings,
    load_spec,
    save_spec,
)

VOLUME_KEY = "population:cancerous:CELL_VOLUME.MU"
ACCURACY_KEY = "population:cancerous:ACCURACY"
BOUNDS = {"CELL_VOLUME_MU": (1000.0, 3000.0), "ACCURACY": (0.2, 1.0)}

TEMPLATE = """<set prefix="t_">
    <series name="grid" start="0" end="1" ticks="2880" interval="720" radius="5" margin="1">
        <agents><populations><population id="cancerous" init="7" class="cancer">
            <population.parameter id="CELL_VOLUME" value="NORMAL(MU=2250,SIGMA=200)" />
            <population.parameter id="ACCURACY" value="0.8" />
        </population></populations></agents>
    </series>
</set>
"""


class FakeArcade:
    """Stands in for xml writing, ARCADE and the metric plugin."""

    def __init__(self) -> None:
        self.simulated: list[tuple[str, int]] = []
        self.metric_calls = 0

    def write_particle_xml(self, template_path, out_path, values, simulation) -> None:
        end = simulation.seed_start + simulation.seeds - 1
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(
            f'<set><series name="grid" start="{simulation.seed_start}" end="{end}" '
            f'ticks="{simulation.ticks}" interval="{simulation.interval}"></series>'
            f"<!--{json.dumps(dict(values))}--></set>\n"
        )

    def simulate(self, inputs_dir, output_dir, jar_path, indices, n_cpu) -> None:
        for i in indices:
            xml = (Path(inputs_dir) / f"input_{i}.xml").read_text()
            values = json.loads(re.search(r"<!--(.*)-->", xml).group(1))
            end = int(re.search(r'end="(\d+)"', xml).group(1))
            ticks = int(re.search(r'ticks="(\d+)"', xml).group(1))
            sim = Path(output_dir) / "inputs" / f"input_{i}"
            sim.mkdir(parents=True, exist_ok=True)
            (sim / "values.json").write_text(json.dumps(values))
            (sim / f"t_grid_{end:04d}_{ticks:06d}.LOCATIONS.json").write_text("[]")
            self.simulated.append((Path(output_dir).name, i))

    def compute(self, sim_folder, builtin_names, custom_file=None, custom_names=None) -> dict:
        self.metric_calls += 1
        v = json.loads((Path(sim_folder) / "values.json").read_text())
        vol, acc = v[VOLUME_KEY], v[ACCURACY_KEY]
        noise = (hash(Path(sim_folder).name) % 100) / 1000.0
        out = {}
        if "growth" in builtin_names:
            out.update(growth=vol / 1000.0 + noise, growth_std=0.1 + acc)
        if custom_names and "spread" in custom_names:
            out.update(spread=acc * 2.0, spread_std=acc / 10.0 + noise)
        return out


@pytest.fixture
def fake(monkeypatch) -> FakeArcade:
    fake = FakeArcade()
    monkeypatch.setattr(xml_template, "write_particle_xml", fake.write_particle_xml)
    monkeypatch.setattr(metrics_plugin, "compute_particle_metrics", fake.compute)
    monkeypatch.setattr(core, "simulate_particles", fake.simulate)
    monkeypatch.setattr(core, "find_java", lambda: "/usr/bin/java")
    monkeypatch.setattr(core, "validate_against_plugins", lambda spec: [])
    return fake


def make_spec(run_dir: Path, n_particles: int = 16, n_generations: int = 2) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "template.xml").write_text(TEMPLATE)
    (run_dir / "arcade.jar").write_bytes(b"")
    (run_dir / "custom_metrics.py").write_text("def spread(seed):\n    return 0.0\n")
    spec = RunSpec(
        template_path="template.xml",
        jar_path="arcade.jar",
        simulation=SimulationSettings(seeds=2, ticks=2880, interval=720, radius=5, margin=1),
        parameters=[
            ParameterSpec(VOLUME_KEY, "CELL_VOLUME_MU", 2250.0, *BOUNDS["CELL_VOLUME_MU"]),
            ParameterSpec(ACCURACY_KEY, "ACCURACY", 0.8, *BOUNDS["ACCURACY"]),
        ],
        metrics=[
            MetricSpec("growth", "builtin", 2.0),
            MetricSpec("spread_std", "custom", 0.07),
        ],
        hyperparameters=Hyperparameters(
            n_particles=n_particles,
            n_generations=n_generations,
            n_trees=10,
            min_samples_leaf=3,
            n_cpu=1,
        ),
        custom_metrics_file="custom_metrics.py",
    )
    path = run_dir / "run_spec.json"
    save_spec(spec, path)
    return path


def test_full_run_layout_and_prior_box(tmp_path, fake, capsys):
    spec_path = make_spec(tmp_path / "run")
    final = gui_run.run(load_spec(spec_path))
    out_root = tmp_path / "run" / "arcade_outputs"
    in_root = tmp_path / "run" / "arcade_inputs"
    assert final == out_root / "final_particles.csv"

    labels = ["CELL_VOLUME_MU", "ACCURACY"]
    for g in (0, 1):
        assert len(list((in_root / f"iter_{g}" / "inputs").glob("input_*.xml"))) == 16
        params = pd.read_csv(out_root / f"iter_{g}" / "all_param_df.csv")
        assert list(params.columns) == labels + ["input_folder"]
        assert list(params["input_folder"]) == [f"input_{i}" for i in range(1, 17)]
        for label, (lo, hi) in BOUNDS.items():
            assert params[label].between(lo, hi).all(), (g, label)
        metrics = pd.read_csv(out_root / f"iter_{g}" / "final_metrics.csv")
        assert {"growth", "growth_std", "spread", "spread_std"} <= set(metrics.columns)
        assert metrics.columns[-1] == "input_folder"
        weights = pd.read_csv(out_root / f"iter_{g}" / "weights.csv")
        assert len(weights) == 16 and np.isclose(weights["weight"].sum(), 1.0)
        assert (out_root / f"iter_{g}" / "ess.json").exists()
    lineage = pd.read_csv(in_root / "iter_1" / "lineage.csv")
    assert len(lineage) == 16 and lineage["parent_input_index"].between(1, 16).all()
    # Generation 1 was drawn by the kernel, not copied from generation 0.
    p0 = pd.read_csv(out_root / "iter_0" / "all_param_df.csv")[labels].to_numpy()
    p1 = pd.read_csv(out_root / "iter_1" / "all_param_df.csv")[labels].to_numpy()
    assert not np.allclose(np.sort(p0, axis=0), np.sort(p1, axis=0))

    assert json.loads((out_root / "param_columns.json").read_text()) == {
        "CELL_VOLUME_MU": VOLUME_KEY,
        "ACCURACY": ACCURACY_KEY,
    }
    assert json.loads((out_root / "targets.json").read_text()) == {
        "growth": 2.0,
        "spread_std": 0.07,
    }
    particles = pd.read_csv(final)
    assert list(particles.columns) == [
        "generation",
        "input_folder",
        *labels,
        "weight",
        "growth",
        "spread_std",
    ]
    assert (particles["generation"] == 1).all() and len(particles) == 16
    assert len(fake.simulated) == 32

    printed = capsys.readouterr().out
    assert "final_particles.csv" in printed and "growth" in printed
    assert "posterior" not in printed.lower()


def test_resume_skips_finished_work_and_is_reproducible(tmp_path, fake):
    spec_path = make_spec(tmp_path / "run")
    first = pd.read_csv(gui_run.run(load_spec(spec_path)))
    n_sims, n_metrics = len(fake.simulated), fake.metric_calls

    second = pd.read_csv(gui_run.run(load_spec(spec_path)))
    assert len(fake.simulated) == n_sims and fake.metric_calls == n_metrics
    pd.testing.assert_frame_equal(first, second)

    # Lose one finished simulation and generation 1's tables: only that one re-runs.
    out_root = tmp_path / "run" / "arcade_outputs"
    (out_root / "iter_1" / "final_metrics.csv").unlink()
    sim = out_root / "iter_1" / "inputs" / "input_5"
    for f in sim.iterdir():
        f.unlink()
    sim.rmdir()
    third = pd.read_csv(gui_run.run(load_spec(spec_path)))
    assert fake.simulated[n_sims:] == [("iter_1", 5)]
    pd.testing.assert_frame_equal(first, third)


def test_partial_simulation_is_pruned_and_rerun(tmp_path, fake):
    spec_path = make_spec(tmp_path / "run", n_generations=1)
    assert gui_run.run(load_spec(spec_path), prepare_only=True).name == "iter_0"
    # A killed job leaves a folder without the last LOCATIONS file.
    partial = tmp_path / "run" / "arcade_outputs" / "iter_0" / "inputs" / "input_3"
    partial.mkdir(parents=True)
    gui_run.run(load_spec(spec_path))
    assert ("iter_0", 3) in fake.simulated and len(fake.simulated) == 16


def test_prepare_only_writes_next_generation_without_simulating(tmp_path, fake):
    run_dir = tmp_path / "run"
    spec_path = make_spec(run_dir, n_particles=20, n_generations=1)
    prepared = gui_run.run(load_spec(spec_path), prepare_only=True)
    assert prepared == run_dir / "arcade_inputs" / "iter_0"
    assert len(list((prepared / "inputs").glob("input_*.xml"))) == 20
    assert not fake.simulated
    assert not (run_dir / "arcade_outputs" / "iter_0" / "final_metrics.csv").exists()

    gui_run.run(load_spec(spec_path))  # simulate generation 0
    assert len(fake.simulated) == 20

    # Extend to two generations: prepare-only lays down iter_1 (with lineage) and stops.
    spec = load_spec(spec_path)
    spec.hyperparameters.n_generations = 2
    prepared = gui_run.run(spec, prepare_only=True)
    assert prepared == run_dir / "arcade_inputs" / "iter_1"
    assert len(list((prepared / "inputs").glob("input_*.xml"))) == 20
    assert (prepared / "lineage.csv").exists()
    assert len(fake.simulated) == 20
    log = pd.read_csv(prepared / "parameter_log.csv")
    for label, (lo, hi) in BOUNDS.items():
        assert log[label].between(lo, hi).all()

    final = gui_run.run(spec)
    assert len(fake.simulated) == 40
    assert set(pd.read_csv(final)["generation"]) == {1}


def test_changed_spec_refuses_to_resume(tmp_path, fake):
    spec_path = make_spec(tmp_path / "run", n_generations=1)
    gui_run.run(load_spec(spec_path))
    spec = load_spec(spec_path)
    spec.parameters[0].max = 4000.0
    with pytest.raises(core.SpecError, match="parameters"):
        gui_run.run(spec)


def test_dry_run_prints_plan_and_writes_nothing(tmp_path, fake, capsys):
    run_dir = tmp_path / "run"
    spec_path = make_spec(run_dir)
    assert gui_run.main(["--dry-run", str(run_dir)]) == 0
    out = capsys.readouterr().out
    assert "CELL_VOLUME_MU" in out and "spread_std" in out and "OK" in out
    assert "iter_1: not started" in out
    assert not (run_dir / "arcade_inputs").exists()
    assert not (run_dir / "arcade_outputs").exists()
    assert not fake.simulated


def test_dry_run_reports_problems(tmp_path, fake, monkeypatch, capsys):
    run_dir = tmp_path / "run"
    spec_path = make_spec(run_dir)
    spec = load_spec(spec_path)
    spec.parameters[1].min = 2.0
    spec.hyperparameters.rf_type = "RF"
    save_spec(spec, spec_path)
    (run_dir / "arcade.jar").unlink()
    monkeypatch.setattr(core, "find_java", lambda: None)
    assert gui_run.main(["--dry-run", str(spec_path)]) == 2
    err = capsys.readouterr().err
    assert "ACCURACY: min" in err
    assert "ARCADE JAR not found" in err
    assert "Java was not found" in err
    assert "rf_type 'RF'" in err


def test_main_runs_and_reports_errors(tmp_path, fake, capsys):
    spec_path = make_spec(tmp_path / "run", n_generations=1)
    assert gui_run.main([str(spec_path), "--n-cpu", "1"]) == 0
    assert "final_particles.csv" in capsys.readouterr().out
    assert gui_run.main([str(tmp_path / "missing.json")]) == 2


def test_all_failed_simulations_give_clear_error(tmp_path, fake, monkeypatch):
    spec_path = make_spec(tmp_path / "run", n_generations=1)
    monkeypatch.setattr(metrics_plugin, "compute_particle_metrics", lambda *a, **k: {})
    with pytest.raises(RuntimeError, match="none of the 16 simulations"):
        gui_run.run(load_spec(spec_path))
    # Nothing was marked as analysed, so a later run retries the metrics.
    assert not (tmp_path / "run" / "arcade_outputs" / "iter_0" / "final_metrics.csv").exists()


def test_box_prior_and_metric_requests():
    cols = ["a", "b"]
    ranges = {"a": (0.0, 1.0), "b": (10.0, 20.0)}
    assert core.box_prior_pdf(np.array([0.5, 15.0]), cols, ranges) == 1.0
    assert core.box_prior_pdf(np.array([0.5, 25.0]), cols, ranges) == 0.0
    assert core.box_prior_pdf(np.array([-0.1, 15.0]), cols, ranges) == 0.0
    metrics = [
        MetricSpec("doub_time", "builtin", 1.0),
        MetricSpec("doub_time_std", "builtin", 1.0),
        MetricSpec("diam_std", "custom", 1.0),
    ]
    assert core.metric_requests(metrics) == (["doub_time"], ["diam"])


def test_metric_setup_error_stops_the_run(tmp_path, fake, monkeypatch):
    spec_path = make_spec(tmp_path / "run", n_generations=1)

    def broken(*args, **kwargs):
        raise ValueError("Unknown built-in metric(s) ['growth']")

    monkeypatch.setattr(metrics_plugin, "compute_particle_metrics", broken)
    assert gui_run.main([str(spec_path)]) == 1
    assert not (tmp_path / "run" / "arcade_outputs" / "iter_0" / "final_metrics.csv").exists()
