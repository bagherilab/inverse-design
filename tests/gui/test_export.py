"""Export of a GUI run directory (pure Python, no Qt)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from inverse_design.gui import export, xml_template
from inverse_design.gui.spec import (
    CUSTOM_METRICS_FILENAME,
    SPEC_FILENAME,
    TEMPLATE_FILENAME,
    Hyperparameters,
    MetricSpec,
    ParameterSpec,
    RunPaths,
    SimulationSettings,
)

SAMPLE = (
    Path(__file__).resolve().parents[2] / "src/inverse_design/sample_inputs/sample_combined_v3.xml"
)
SIM = SimulationSettings(seeds=3, ticks=2880, interval=720, radius=8, margin=2, seed_start=0)


def _stubbed(func) -> bool:
    try:
        func()
    except NotImplementedError:
        return True
    except Exception:
        return False
    return False


@pytest.fixture
def template_writer(monkeypatch, tmp_path):
    """Use the real write_template_copy; fall back to a fake while it is a stub."""
    probe = tmp_path / "probe.xml"
    if _stubbed(lambda: xml_template.write_template_copy(SAMPLE, probe, SIM)):

        def fake(src: Path, out: Path, sim: SimulationSettings) -> None:
            Path(out).write_text(Path(src).read_text() + f"<!-- ticks={sim.ticks} -->\n")

        monkeypatch.setattr(xml_template, "write_template_copy", fake)
    probe.unlink(missing_ok=True)


def _spec(custom: Path | None = None, **kwargs):
    metrics = [MetricSpec("n_cells", "builtin", 100.0)]
    if custom is not None:
        metrics.append(MetricSpec("final_diameter_mm", "custom", 1.5))
    return export.build_spec(
        template_path=SAMPLE,
        jar_path=kwargs.pop("jar_path", "/opt/arcade/arcade.jar"),
        simulation=SIM,
        parameters=[
            ParameterSpec(
                "population:cancerous:CELL_VOLUME.MU", "CELL_VOLUME_MU", 2250.0, 1125.0, 3375.0
            )
        ],
        metrics=metrics,
        hyperparameters=Hyperparameters(n_particles=16, n_generations=2),
        paths=RunPaths("in", "out"),
        custom_metrics_file=custom,
    )


def _custom_file(tmp_path: Path) -> Path:
    path = tmp_path / "src" / "my_metrics.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("def final_diameter_mm(seed):\n    return 1.0\n")
    return path


def test_build_spec_keeps_source_paths():
    spec = _spec()
    assert spec.template_path == str(SAMPLE)
    assert spec.custom_metrics_file is None
    assert spec.validate() == []


def test_export_round_trip(tmp_path, template_writer):
    custom = _custom_file(tmp_path)
    spec = _spec(custom)
    run_dir = tmp_path / "run"
    spec_path = export.export_run(spec, run_dir, SAMPLE, custom)

    assert spec_path == (run_dir / SPEC_FILENAME).resolve()
    for name in (TEMPLATE_FILENAME, CUSTOM_METRICS_FILENAME, "run.sh", "README.txt"):
        assert (run_dir / name).is_file(), name
    assert (run_dir / CUSTOM_METRICS_FILENAME).read_text() == custom.read_text()

    loaded = export.load_run(spec_path)
    assert loaded.template_path == TEMPLATE_FILENAME
    assert loaded.custom_metrics_file == CUSTOM_METRICS_FILENAME
    assert loaded.resolve(loaded.template_path) == (run_dir / TEMPLATE_FILENAME).resolve()
    custom_copy = (run_dir / CUSTOM_METRICS_FILENAME).resolve()
    assert loaded.resolve(loaded.custom_metrics_file) == custom_copy
    assert loaded.parameters == spec.parameters
    assert loaded.metrics == spec.metrics
    assert loaded.hyperparameters == spec.hyperparameters
    assert loaded.simulation == spec.simulation
    assert loaded.paths == RunPaths("in", "out")


def test_run_script_and_readme(tmp_path, template_writer):
    run_dir = tmp_path / "run"
    export.export_run(_spec(), run_dir, SAMPLE)
    script = run_dir / "run.sh"
    assert os.access(script, os.X_OK)
    assert "python -m inverse_design.gui.run" in script.read_text()
    readme = (run_dir / "README.txt").read_text()
    assert export.run_command(run_dir.resolve()) in readme
    assert "random-forest weights" in readme
    assert "not a calibrated posterior" in readme
    assert not (run_dir / CUSTOM_METRICS_FILENAME).exists()


def test_template_gets_simulation_settings(tmp_path):
    if _stubbed(lambda: xml_template.read_simulation_settings(SAMPLE)):
        pytest.skip("xml_template is still a stub")
    run_dir = tmp_path / "run"
    export.export_run(_spec(), run_dir, SAMPLE)
    sim = xml_template.read_simulation_settings(run_dir / TEMPLATE_FILENAME)
    assert (sim.seeds, sim.ticks, sim.interval, sim.radius) == (3, 2880, 720, 8)


def test_relative_jar_becomes_absolute(tmp_path, template_writer, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec_path = export.export_run(_spec(jar_path="arcade.jar"), tmp_path / "run", SAMPLE)
    assert export.load_run(spec_path).jar_path == str((tmp_path / "arcade.jar").resolve())


def test_refuses_different_spec_unless_overwrite(tmp_path, template_writer):
    run_dir = tmp_path / "run"
    export.export_run(_spec(), run_dir, SAMPLE)
    export.export_run(_spec(), run_dir, SAMPLE)  # identical spec: allowed
    changed = replace(_spec(), hyperparameters=Hyperparameters(n_particles=32))
    with pytest.raises(export.RunDirConflict):
        export.export_run(changed, run_dir, SAMPLE)
    export.export_run(changed, run_dir, SAMPLE, overwrite=True)
    assert export.load_run(run_dir / SPEC_FILENAME).hyperparameters.n_particles == 32


def test_reexport_loaded_run_in_place(tmp_path, template_writer):
    custom = _custom_file(tmp_path)
    run_dir = tmp_path / "run"
    spec_path = export.export_run(_spec(custom), run_dir, SAMPLE, custom)
    loaded = export.load_run(spec_path)
    template = loaded.resolve(loaded.template_path)
    copy = loaded.resolve(loaded.custom_metrics_file)
    before = template.read_text()
    export.export_run(loaded, run_dir, template, copy)
    assert template.read_text().strip()
    assert copy.read_text() == custom.read_text()
    if not _stubbed(lambda: xml_template.read_simulation_settings(SAMPLE)):
        assert template.read_text() == before


def test_invalid_spec_and_missing_files(tmp_path, template_writer):
    bad = replace(_spec(), metrics=[])
    with pytest.raises(export.ExportError, match="target metric"):
        export.export_run(bad, tmp_path / "a", SAMPLE)
    with pytest.raises(export.ExportError, match="Template not found"):
        export.export_run(_spec(), tmp_path / "b", tmp_path / "missing.xml")
    custom_spec = _spec(tmp_path / "nope.py")
    with pytest.raises(export.ExportError, match="no custom metrics file"):
        export.export_run(custom_spec, tmp_path / "c", SAMPLE, None)
    with pytest.raises(export.ExportError, match="not found"):
        export.export_run(custom_spec, tmp_path / "d", SAMPLE, tmp_path / "nope.py")
    assert not (tmp_path / "a").exists()


def test_run_command_quotes_spaces():
    assert export.run_command(Path("/r")) == "python -m inverse_design.gui.run /r/run_spec.json"
    assert '"/my runs/x/run_spec.json"' in export.run_command(Path("/my runs/x"))
