"""Offscreen smoke test of the PySide6 setup GUI."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from inverse_design.gui import app as gui_app  # noqa: E402
from inverse_design.gui import export, metrics_plugin, xml_template  # noqa: E402
from inverse_design.gui.spec import (  # noqa: E402
    DetectedParam,
    MetricSpec,
    ParamRef,
    SimulationSettings,
)

SAMPLE = (
    Path(__file__).resolve().parents[2] / "src/inverse_design/sample_inputs/sample_combined_v3.xml"
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialogs(monkeypatch):
    """Record modal dialogs instead of blocking the test."""
    shown: list[tuple[str, str]] = []

    def record(kind):
        def _show(_parent, title, text, *args, **kwargs):
            shown.append((kind, f"{title}: {text}"))
            return QMessageBox.StandardButton.Yes

        return _show

    for kind in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, kind, record(kind))
    return shown


@pytest.fixture
def fake_backend(monkeypatch):
    """Deterministic template/metric functions so the test does not depend on other blocks."""
    detected = [
        DetectedParam(
            ParamRef("population", "cancerous", "CELL_VOLUME", field="MU"),
            2250.0,
            "CELL_VOLUME_MU",
            "NORMAL(MU=2250,SIGMA=200)",
        ),
        DetectedParam(ParamRef("population", "cancerous", "ACCURACY"), 0.8, "ACCURACY", "0.8"),
        DetectedParam(
            ParamRef("layer", "GLUCOSE", "CONCENTRATION", "generator"),
            0.005,
            "GLUCOSE_CONCENTRATION",
            "0.005",
        ),
    ]
    sim = SimulationSettings(seeds=10, ticks=10080, interval=720, radius=10, margin=2)
    monkeypatch.setattr(xml_template, "detect_parameters", lambda _p: list(detected))
    monkeypatch.setattr(xml_template, "read_simulation_settings", lambda _p: sim)
    monkeypatch.setattr(
        xml_template, "default_range", lambda v, _r: (0.0, 1.0) if v == 0 else (0.5 * v, 1.5 * v)
    )
    monkeypatch.setattr(
        xml_template,
        "write_template_copy",
        lambda src, out, s: Path(out).write_text(Path(src).read_text()),
    )
    monkeypatch.setattr(
        metrics_plugin,
        "builtin_metrics",
        lambda: [
            metrics_plugin.MetricInfo("n_cells", "Number of cells", "cells"),
            metrics_plugin.MetricInfo("doub_time", "Doubling time", "hr"),
        ],
    )
    return detected


def test_pages_switch(qapp, dialogs):
    window = gui_app.ArcadeSetupWindow()
    window.show()
    for index in (1, 2, 1, 0):
        window.go_to(index)
        qapp.processEvents()
        assert window.stage_pages.currentIndex() == index
        assert gui_app.STAGE_NAMES[index] in window.stage_indicator.text()
    assert window.validation_list.count() > 0  # problems listed with nothing loaded
    window.close()


def test_configure_and_export(qapp, dialogs, fake_backend, tmp_path):
    window = gui_app.ArcadeSetupWindow()
    assert window.load_template(SAMPLE)
    assert window.param_table.rowCount() == 3
    assert window.param_box.title() == "Detected parameters (3)"
    assert window.duration_ticks.text() == "= 10080 ticks"

    window.filter_edit.setText("glucose")
    assert [window.param_table.isRowHidden(r) for r in range(3)] == [True, True, False]
    window.filter_edit.setText("")
    window._set_visible_checks(True)
    window.param_table.item(2, 4).setText("0.001")

    jar = tmp_path / "arcade.jar"
    jar.write_text("")
    window.jar_edit.setText(str(jar))
    window.builtin_table.apply(
        [MetricSpec("n_cells", "builtin", 120.0), MetricSpec("doub_time_std", "builtin", 2.0)]
    )
    window.run_dir_edit.setText(str(tmp_path / "run"))
    window.go_to(2)
    qapp.processEvents()

    spec, problems, _ = window.collect_spec()
    assert problems == []
    assert [m.name for m in spec.metrics] == ["n_cells", "doub_time_std"]
    assert spec.parameters[2].min == pytest.approx(0.001)

    spec_path = window.export_run_dir()
    assert spec_path is not None and spec_path.is_file(), dialogs
    assert export.run_command(spec_path.parent) == window.command_edit.text()

    other = gui_app.ArcadeSetupWindow()
    assert other.load_run_spec(spec_path)
    assert len(other.selected_parameters()[0]) == 3
    assert other.collect_spec()[1] == []
    assert other.builtin_table.state()["doub_time_std"] == (True, "2")
    window.close()
    other.close()


def test_bad_template_reports_error(qapp, dialogs, tmp_path, monkeypatch):
    bad = tmp_path / "bad.xml"
    bad.write_text("<set><series/><series/></set>")

    def raise_error(_path):
        raise xml_template.TemplateError("needs exactly one <series>")

    monkeypatch.setattr(xml_template, "detect_parameters", raise_error)
    window = gui_app.ArcadeSetupWindow()
    assert not window.load_template(bad)
    assert any("exactly one" in text for _, text in dialogs)
    assert window.template_path is None
    window.close()


def test_real_sample_template(qapp, dialogs):
    try:
        xml_template.detect_parameters(SAMPLE)
    except NotImplementedError:
        pytest.skip("xml_template is still a stub")
    window = gui_app.ArcadeSetupWindow()
    assert window.load_template(SAMPLE), dialogs
    assert window.param_table.rowCount() > 10
    assert window.simulation_settings().ticks == 10080
    window.close()


def test_custom_metric_test_runs_in_thread(qapp, dialogs, monkeypatch, tmp_path):
    custom = tmp_path / "custom_metrics.py"
    custom.write_text("def final_diameter_mm(seed):\n    return 1.0\n")
    result = metrics_plugin.MetricTestResult(
        "final_diameter_mm", [1.0, float("nan")], ["seed 1: boom"], 1.0, 0.0
    )
    monkeypatch.setattr(
        metrics_plugin, "load_custom_metrics", lambda _p: {"final_diameter_mm": lambda s: 1.0}
    )
    monkeypatch.setattr(metrics_plugin, "test_custom_metrics", lambda _f, _c: [result])
    shown: list = []
    monkeypatch.setattr(gui_app.MetricTestDialog, "exec", lambda self: shown.append(self))

    window = gui_app.ArcadeSetupWindow()
    assert window.load_custom_file(custom)
    assert window.custom_table.names == ["final_diameter_mm"]
    window.start_metric_test(tmp_path)
    window.test_worker.wait(5000)
    for _ in range(20):
        qapp.processEvents()
        if shown:
            break
    assert shown and "1 seed error" in window.test_status.text()
    window.close()
