"""Desktop GUI (PySide6) that sets up an ARCADE calibration run directory.

Launch with ``python -m inverse_design.gui [template.xml | run_spec.json]``. The
GUI does not run the calibration: it exports a run directory that is run with
``python -m inverse_design.gui.run <run_dir>/run_spec.json``.
"""

from __future__ import annotations

import math
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Sequence

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QAction, QColor, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from inverse_design.gui import constraints, export, metrics_plugin, xml_template
from inverse_design.gui.spec import (
    SPEC_FILENAME,
    DetectedParam,
    Hyperparameters,
    MetricSpec,
    ParameterSpec,
    ParamRef,
    RunPaths,
    RunSpec,
    SimulationSettings,
)

TICKS_PER_DAY = 1440  # 1 ARCADE tick = 1 minute
XML_FILTER = "ARCADE XML template (*.xml);;All files (*)"
JAR_FILTER = "ARCADE JAR (*.jar);;All files (*)"
PY_FILTER = "Python file (*.py);;All files (*)"
SPEC_FILTER = f"Run spec ({SPEC_FILENAME} *.json);;All files (*)"
STAGE_NAMES = (
    "Screen 1 — Template & parameters",
    "Screen 2 — Target metrics",
    "Screen 3 — Run settings & export",
)
PRIMARY_STYLE = "font-weight: bold; background: #d8e8d2;"
SMALL_BUTTON_STYLE = "min-height: 0px; padding: 2px 10px;"
INVALID_BG = QColor("#f6d3d3")
CUSTOM_HELP = (
    "A custom metrics file is a plain Python file with one function per metric. Each "
    "function receives ONE seed of one particle (a SeedView with .timestamps, "
    ".cells(ts), .locations(ts), .days()) and returns a float. The pipeline takes the "
    "median across seeds (column <name>) and the SD across seeds (column <name>_std)."
)
CLUSTER_NOTE = (
    "The run directory is self-contained. On a cluster: copy the folder (or export it "
    "straight to a shared filesystem), activate a Python environment with inverse_design "
    "installed and Java available, check that jar_path in run_spec.json points to the "
    "cluster's ARCADE JAR, and call run.sh from a batch job (e.g. Slurm with "
    "--cpus-per-task set to n_cpu)."
)


def _fmt(value: float) -> str:
    """Compact number display that round-trips through float()."""
    return f"{value:.6g}"


def _parse_float(text: str) -> Optional[float]:
    try:
        value = float(text.strip())
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _fallback_range(value: float) -> tuple[float, float]:
    """[0.5x, 1.5x] (ordered), or [0, 1] for zero; used if default_range fails."""
    if value == 0:
        return 0.0, 1.0
    low, high = sorted((0.5 * value, 1.5 * value))
    return low, high


def _owner_text(ref: ParamRef) -> str:
    owner = " ".join(part for part in (ref.owner_tag, ref.owner_id) if part) or "(top level)"
    if ref.operation:
        owner += f" ({ref.operation})"
    return owner


def template_summary(path: Path) -> dict[str, list[str]]:
    """Ids of populations, layers, components and actions declared in a template."""
    summary: dict[str, list[str]] = {}
    root = ET.parse(path).getroot()
    for tag in ("population", "layer", "component", "action"):
        ids = [el.get("id", "") for el in root.iter(tag) if el.get("id")]
        if ids:
            summary[tag] = ids
    return summary


class DaysSpinBox(QDoubleSpinBox):
    """Days with four decimals but trailing zeros hidden (7 instead of 7.0000)."""

    def __init__(self, minimum: float, maximum: float) -> None:
        super().__init__()
        self.setDecimals(4)
        self.setRange(minimum, maximum)
        self.setSingleStep(0.5)
        self.setSuffix(" days")

    def textFromValue(self, value: float) -> str:  # noqa: N802 - Qt override
        return f"{value:.4f}".rstrip("0").rstrip(".")


class MetricTable(QTableWidget):
    """One row per metric; tick the median and/or SD (``_std``) statistic and set targets."""

    COLUMNS = (
        "Metric",
        "Unit",
        "Fit median",
        "Median target",
        "Fit SD (_std)",
        "SD target",
        "Description",
    )
    STATS = ((2, 3, ""), (4, 5, "_std"))

    def __init__(self) -> None:
        super().__init__(0, len(self.COLUMNS))
        self.setHorizontalHeaderLabels(self.COLUMNS)
        self.verticalHeader().setVisible(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.setAlternatingRowColors(True)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.names: list[str] = []

    def set_metrics(self, rows: Sequence[tuple[str, str, str]]) -> None:
        """Replace rows with ``(name, unit, description)``; keep ticks/targets by name."""
        previous = self.state()
        self.blockSignals(True)
        self.setRowCount(0)
        self.names = [name for name, _, _ in rows]
        for row, (name, unit, description) in enumerate(rows):
            self.insertRow(row)
            self._set(row, 0, name, editable=False)
            self._set(row, 1, unit, editable=False)
            for check_col, target_col, suffix in self.STATS:
                checked, target = previous.get(name + suffix, (False, ""))
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                check.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                self.setItem(row, check_col, check)
                self._set(row, target_col, target, editable=True)
            self._set(row, 6, description, editable=False)
            self.item(row, 6).setToolTip(description)
        self.blockSignals(False)

    def _set(self, row: int, col: int, text: str, editable: bool) -> None:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable
        item.setFlags(flags)
        self.setItem(row, col, item)

    def state(self) -> dict[str, tuple[bool, str]]:
        """``{column name: (checked, target text)}`` for every statistic."""
        out: dict[str, tuple[bool, str]] = {}
        for row, name in enumerate(self.names):
            for check_col, target_col, suffix in self.STATS:
                checked = self.item(row, check_col).checkState() == Qt.CheckState.Checked
                out[name + suffix] = (checked, self.item(row, target_col).text())
        return out

    def selected(self, source: str) -> tuple[list[MetricSpec], list[str]]:
        """Ticked statistics as MetricSpecs plus problems (missing/invalid targets)."""
        specs: list[MetricSpec] = []
        problems: list[str] = []
        for name, (checked, text) in self.state().items():
            if not checked:
                continue
            target = _parse_float(text)
            if target is None:
                problems.append(f"Metric {name}: enter a numeric target value.")
            else:
                spec = MetricSpec(name=name, source=source, target=target)  # type: ignore[arg-type]
                specs.append(spec)
        return specs, problems

    def apply(self, metrics: Sequence[MetricSpec]) -> list[str]:
        """Tick the given metrics and set targets; return names not in the table."""
        self.blockSignals(True)
        for row in range(self.rowCount()):
            for check_col, _, _ in self.STATS:
                self.item(row, check_col).setCheckState(Qt.CheckState.Unchecked)
        missing: list[str] = []
        for metric in metrics:
            base, col = metric.name, self.STATS[0]
            if base not in self.names and base.endswith("_std"):
                base, col = base[: -len("_std")], self.STATS[1]
            if base not in self.names:
                missing.append(metric.name)
                continue
            row = self.names.index(base)
            self.item(row, col[0]).setCheckState(Qt.CheckState.Checked)
            self.item(row, col[1]).setText(_fmt(metric.target))
        self.blockSignals(False)
        return missing

    def checked_count(self) -> int:
        return sum(1 for checked, _ in self.state().values() if checked)


class CustomMetricTestWorker(QThread):
    """Runs ``metrics_plugin.test_custom_metrics`` off the UI thread."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, sim_folder: Path, custom_file: Path) -> None:
        super().__init__()
        self.sim_folder = sim_folder
        self.custom_file = custom_file

    def run(self) -> None:
        try:
            results = metrics_plugin.test_custom_metrics(self.sim_folder, self.custom_file)
            self.succeeded.emit(results)
        except Exception:
            self.failed.emit(traceback.format_exc())


class MetricTestDialog(QDialog):
    """Per-seed values and errors of each custom metric on one sample particle."""

    def __init__(self, parent: QWidget, sim_folder: Path, results: list) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom metric test")
        self.resize(900, 520)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Sample output: {sim_folder}"))
        n_seeds = max((len(r.per_seed) for r in results), default=0)
        table = QTableWidget(len(results), 4 + n_seeds)
        table.setHorizontalHeaderLabels(
            ["Metric", "Median", "SD", "Failed seeds"] + [f"seed {i}" for i in range(n_seeds)]
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        errors: list[str] = []
        for row, res in enumerate(results):
            n_failed = sum(1 for v in res.per_seed if not math.isfinite(v))
            cells = [res.name, _fmt(res.median), _fmt(res.std), str(n_failed)]
            cells += [_fmt(v) for v in res.per_seed]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 3 and n_failed:
                    item.setBackground(INVALID_BG)
                table.setItem(row, col, item)
            errors += [f"[{res.name}] {err}" for err in res.errors]
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(table, 2)
        layout.addWidget(QLabel(f"Errors ({len(errors)})"))
        error_box = QPlainTextEdit("\n\n".join(errors) if errors else "No errors.")
        error_box.setReadOnly(True)
        layout.addWidget(error_box, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ArcadeSetupWindow(QMainWindow):
    """Three screens: template & parameters, target metrics, run settings & export."""

    DEFAULT_WINDOW_SIZE = (1400, 950)
    PARAM_COLUMNS = ("Perturb", "Parameter", "Owner", "XML value", "Min", "Max", "XML id = value")

    def __init__(self) -> None:
        super().__init__()
        self.template_path: Optional[Path] = None
        self.detected: list[DetectedParam] = []
        self.seed_start = 0
        self.custom_file: Optional[Path] = None
        self.custom_error: Optional[str] = None
        self.hidden_hyper = Hyperparameters()  # keeps fields the form does not show
        self.test_worker: Optional[CustomMetricTestWorker] = None
        self.last_export: Optional[Path] = None
        self._configure_window()
        self._build_ui()
        self._load_builtin_metrics()
        self._update_sim_labels()
        self.refresh_validation()

    # ------------------------------------------------------------------ window
    def _configure_window(self) -> None:
        self.setWindowTitle("ARCADE calibration setup")
        target_w, target_h = self.DEFAULT_WINDOW_SIZE
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            target_w = max(960, min(target_w, avail.width() - 40))
            target_h = max(760, min(target_h, avail.height() - 80))
        self.resize(target_w, target_h)
        self.setStyleSheet("""
            QMainWindow { background: #e6e6e6; }
            QGroupBox { background: #f4f4f4; border: 1px solid #858585; margin-top: 10px;
                        font-weight: bold; padding: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QGroupBox QLabel, QGroupBox QCheckBox { font-weight: normal; }
            QGroupBox QTableWidget, QGroupBox QListWidget, QGroupBox QPlainTextEdit,
            QGroupBox QSpinBox, QGroupBox QDoubleSpinBox, QGroupBox QComboBox,
            QGroupBox QHeaderView { font-weight: normal; }
            QLineEdit { background: white; border: 1px solid #9a9a9a; }
            QPushButton { background: #ececec; border: 1px solid #777; padding: 4px 10px;
                          min-height: 24px; font-weight: normal; }
            QPushButton:hover { background: #ffffff; }
            QPushButton:pressed { background: #c9c9c9; }
            QPushButton:disabled { color: #888; background: #dedede; border-color: #b3b3b3; }
            """)
        self.setStatusBar(QStatusBar())
        menu = self.menuBar().addMenu("&File")
        for text, handler, shortcut in (
            ("Open ARCADE template…", self.choose_template, "Ctrl+O"),
            ("Open existing run_spec.json…", self.choose_run_spec, "Ctrl+Shift+O"),
            ("Export run directory", self.export_run_dir, "Ctrl+E"),
            ("Quit", self.close, "Ctrl+Q"),
        ):
            action = QAction(text, self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            menu.addAction(action)

    def _build_ui(self) -> None:
        self.stage_pages = QStackedWidget()
        self.stage_pages.addWidget(self._page(self._build_screen1(), 0))
        self.stage_pages.addWidget(self._page(self._build_screen2(), 1))
        self.stage_pages.addWidget(self._page(self._build_screen3(), 2))
        self.stage_pages.currentChanged.connect(self._on_page_changed)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.stage_indicator = QLabel()
        self.stage_indicator.setStyleSheet(
            "padding: 6px 12px; background: #dfe6df; border-bottom: 1px solid #b0b0b0;"
            " font-size: 13px;"
        )
        layout.addWidget(self.stage_indicator)
        layout.addWidget(self.stage_pages)
        self.setCentralWidget(central)
        self._update_stage_indicator(0)
        self.statusBar().showMessage("Choose an ARCADE XML template to begin.")

    def _page(self, content: QWidget, index: int) -> QWidget:
        """Scrollable content plus a pinned bottom bar with Back/Continue."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)
        page_layout.addWidget(scroll)
        bar = QWidget()
        bar.setObjectName("nextBar")
        bar.setStyleSheet("#nextBar { border-top: 1px solid #b0b0b0; }")
        row = QHBoxLayout(bar)
        row.setContentsMargins(10, 6, 10, 8)
        if index > 0:
            back = QPushButton(f"← Back to Screen {index}")
            back.clicked.connect(lambda: self.go_to(index - 1))
            row.addWidget(back)
        row.addStretch(1)
        if index < len(STAGE_NAMES) - 1:
            forward = QPushButton(f"Continue to Screen {index + 2} →")
            forward.clicked.connect(lambda: self.go_to(index + 1))
            row.addWidget(forward)
        page_layout.addWidget(bar)
        return page

    @staticmethod
    def _content(title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(6)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 19px; font-weight: bold; color: #202020;")
        layout.addWidget(heading)
        sub = QLabel(subtitle)
        sub.setWordWrap(True)
        layout.addWidget(sub)
        return root, layout

    def go_to(self, index: int) -> None:
        self.stage_pages.setCurrentIndex(index)

    def _on_page_changed(self, index: int) -> None:
        self._update_stage_indicator(index)
        if index == 2:
            self.refresh_validation()

    def _update_stage_indicator(self, active: int) -> None:
        parts = []
        for index, name in enumerate(STAGE_NAMES):
            if index == active:
                parts.append(f"<span style='color:#202020; font-weight:700'>{name}</span>")
            else:
                parts.append(f"<span style='color:#9a9a9a'>{name}</span>")
        self.stage_indicator.setText("  →  ".join(parts))

    def _path_row(self, layout: QGridLayout, row: int, label: str, handler) -> QLineEdit:
        layout.addWidget(QLabel(label), row, 0)
        edit = QLineEdit()
        layout.addWidget(edit, row, 1)
        button = QPushButton("Browse…")
        button.setStyleSheet(SMALL_BUTTON_STYLE)
        button.clicked.connect(handler)
        layout.addWidget(button, row, 2)
        return edit

    # ---------------------------------------------------------------- screen 1
    def _build_screen1(self) -> QWidget:
        root, layout = self._content(
            STAGE_NAMES[0],
            "Pick an ARCADE XML template. Every numeric parameter in it is listed below; tick "
            "the ones to perturb and set their uniform prior. Unticked parameters stay at the "
            "template value.",
        )
        top = QHBoxLayout()
        top.addWidget(self._build_input_box(), 5)
        top.addWidget(self._build_summary_box(), 3)
        top.addWidget(self._build_sim_box(), 4)
        layout.addLayout(top)
        layout.addWidget(self._build_param_box(), 1)
        return root

    def _build_input_box(self) -> QGroupBox:
        box = QGroupBox("Input")
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        self.template_edit = self._path_row(grid, 0, "ARCADE XML template", self.choose_template)
        self.template_edit.setReadOnly(True)
        self.jar_edit = self._path_row(grid, 1, "ARCADE JAR", self.choose_jar)
        self.jar_edit.setPlaceholderText("path to arcade.jar (as seen where the run executes)")
        self.jar_edit.textChanged.connect(self.refresh_validation)
        open_run = QPushButton("Open existing run_spec.json…")
        open_run.clicked.connect(self.choose_run_spec)
        grid.addWidget(open_run, 2, 0, 1, 3)
        grid.setRowStretch(3, 1)
        return box

    def _build_summary_box(self) -> QGroupBox:
        box = QGroupBox("Template summary")
        column = QVBoxLayout(box)
        self.summary_label = QLabel("No template loaded.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.summary_label, 1)
        return box

    def _build_sim_box(self) -> QGroupBox:
        box = QGroupBox("Simulation settings (applied to every particle)")
        grid = QGridLayout(box)
        self.seeds_spin = QSpinBox()
        self.seeds_spin.setRange(1, 10000)
        self.seeds_spin.setValue(10)
        self.duration_spin = DaysSpinBox(0.0007, 100000)
        self.duration_spin.setValue(7)
        self.interval_spin = DaysSpinBox(0.0007, 100000)
        self.interval_spin.setValue(0.5)
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 100000)
        self.radius_spin.setValue(10)
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 100000)
        self.margin_spin.setValue(2)
        self.duration_ticks = QLabel()
        self.interval_ticks = QLabel()
        rows = (
            ("Seeds per particle", self.seeds_spin, None),
            ("Duration", self.duration_spin, self.duration_ticks),
            ("Snapshot interval", self.interval_spin, self.interval_ticks),
            ("Domain radius", self.radius_spin, None),
            ("Domain margin", self.margin_spin, None),
        )
        for row, (label, widget, extra) in enumerate(rows):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(widget, row, 1)
            if extra is not None:
                grid.addWidget(extra, row, 2)
            widget.valueChanged.connect(self._update_sim_labels)
        grid.setColumnStretch(1, 1)
        self.sim_status = QLabel()
        self.sim_status.setWordWrap(True)
        grid.addWidget(self.sim_status, len(rows), 0, 1, 3)
        return box

    def _build_param_box(self) -> QGroupBox:
        self.param_box = QGroupBox("Detected parameters (0)")
        column = QVBoxLayout(self.param_box)
        tools = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter by parameter or owner…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        tools.addWidget(QLabel("Filter:"))
        tools.addWidget(self.filter_edit, 1)
        perturb_all = QPushButton("Perturb all shown")
        perturb_all.setStyleSheet(SMALL_BUTTON_STYLE)
        perturb_all.clicked.connect(lambda: self._set_visible_checks(True))
        clear = QPushButton("Clear shown")
        clear.setStyleSheet(SMALL_BUTTON_STYLE)
        clear.clicked.connect(lambda: self._set_visible_checks(False))
        tools.addWidget(perturb_all)
        tools.addWidget(clear)
        self.param_count_label = QLabel("0 perturbed")
        tools.addWidget(self.param_count_label)
        column.addLayout(tools)

        self.param_table = QTableWidget(0, len(self.PARAM_COLUMNS))
        self.param_table.setHorizontalHeaderLabels(self.PARAM_COLUMNS)
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.setAlternatingRowColors(True)
        self.param_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.param_table.setMinimumHeight(240)
        header = self.param_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        for col in (4, 5):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.Interactive)
            self.param_table.setColumnWidth(col, 130)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        self.param_table.itemChanged.connect(self._on_param_item_changed)
        column.addWidget(self.param_table, 1)
        note = QLabel(
            "Min/Max define a uniform prior; defaults are 0.5x–1.5x of the XML value "
            "(fractions capped at 1). Hover a row for its XML id and raw value."
        )
        note.setWordWrap(True)
        column.addWidget(note)
        return self.param_box

    def _update_sim_labels(self) -> None:
        sim = self.simulation_settings()
        self.duration_ticks.setText(f"= {sim.ticks} ticks")
        self.interval_ticks.setText(f"= {sim.interval} ticks")
        problems = self._sim_problems(sim)
        warnings = constraints.simulation_warnings(sim)
        if problems:
            self.sim_status.setStyleSheet("color: #b00020;")
            self.sim_status.setText(" ".join(problems))
        elif warnings:
            self.sim_status.setStyleSheet("color: #8a5a00;")
            self.sim_status.setText(" ".join(warnings))
        else:
            snaps = sim.ticks // sim.interval
            self.sim_status.setStyleSheet("color: #2d6a2d;")
            self.sim_status.setText(
                f"{snaps} snapshot intervals per seed; seeds {sim.seed_start}–"
                f"{sim.seed_start + sim.seeds - 1}. 1 tick = 1 minute."
            )

    @staticmethod
    def _sim_problems(sim: SimulationSettings) -> list[str]:
        problems = []
        if sim.interval <= 0 or sim.ticks <= 0 or sim.ticks % sim.interval:
            problems.append("The snapshot interval must divide the duration.")
        elif sim.ticks // sim.interval < 2:
            problems.append("The duration must cover at least two snapshot intervals.")
        return problems

    def simulation_settings(self) -> SimulationSettings:
        return SimulationSettings(
            seeds=self.seeds_spin.value(),
            ticks=round(self.duration_spin.value() * TICKS_PER_DAY),
            interval=round(self.interval_spin.value() * TICKS_PER_DAY),
            radius=self.radius_spin.value(),
            margin=self.margin_spin.value(),
            seed_start=self.seed_start,
        )

    def set_simulation_settings(self, sim: SimulationSettings) -> None:
        self.seed_start = sim.seed_start
        self.seeds_spin.setValue(sim.seeds)
        self.duration_spin.setValue(sim.ticks / TICKS_PER_DAY)
        self.interval_spin.setValue(sim.interval / TICKS_PER_DAY)
        self.radius_spin.setValue(sim.radius)
        self.margin_spin.setValue(sim.margin)
        self._update_sim_labels()

    # --------------------------------------------------- screen 1: parameters
    def choose_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "ARCADE XML template", "", XML_FILTER)
        if path:
            self.load_template(Path(path))

    def choose_jar(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "ARCADE JAR", "", JAR_FILTER)
        if path:
            self.jar_edit.setText(path)

    def load_template(self, path: Path, *, apply_settings: bool = True) -> bool:
        """Detect parameters and settings; return False (after telling the user) on error."""
        path = Path(path).expanduser().resolve()
        try:
            detected = xml_template.detect_parameters(path)
            sim = xml_template.read_simulation_settings(path)
        except xml_template.TemplateError as exc:
            self._error("Cannot use this template", str(exc))
            return False
        except Exception as exc:
            self._error("Cannot read this template", f"{type(exc).__name__}: {exc}")
            return False
        self.template_path = path
        self.detected = list(detected)
        self.template_edit.setText(str(path))
        self.template_edit.setToolTip(str(path))
        if apply_settings:
            self.set_simulation_settings(sim)
        self._fill_param_table()
        self._fill_summary(path, sim)
        self.statusBar().showMessage(
            f"Loaded {path.name}: {len(self.detected)} numeric parameters detected."
        )
        self.refresh_validation()
        return True

    def _fill_summary(self, path: Path, sim: SimulationSettings) -> None:
        try:
            owners = template_summary(path)
        except Exception:
            owners = {}
        names = {
            "population": "Populations",
            "layer": "Layers",
            "component": "Components",
            "action": "Actions",
        }
        lines = [f"<b>{path.name}</b>"]
        for tag, label in names.items():
            if tag in owners:
                lines.append(f"{label} ({len(owners[tag])}): {', '.join(owners[tag])}")
        lines.append(f"Seeds: {sim.seeds} ({sim.seed_start}–{sim.seed_start + sim.seeds - 1})")
        days = sim.ticks / TICKS_PER_DAY
        lines.append(
            f"Duration: {days:g} days ({sim.ticks} ticks), snapshots every "
            f"{sim.interval / TICKS_PER_DAY:g} days"
        )
        lines.append(f"Domain: radius {sim.radius}, margin {sim.margin}")
        n_dist = sum(1 for d in self.detected if d.ref.field)
        lines.append(f"Numeric values: {len(self.detected)} ({n_dist} from NORMAL/UNIFORM fields)")
        self.summary_label.setText("<br>".join(lines))

    def _fill_param_table(self) -> None:
        table = self.param_table
        table.blockSignals(True)
        table.setRowCount(0)
        for row, dp in enumerate(self.detected):
            table.insertRow(row)
            try:
                low, high = xml_template.default_range(dp.xml_value, dp.ref)
            except Exception:
                low, high = _fallback_range(dp.xml_value)
            tip = f"{dp.ref.key}\nXML value: {dp.raw_value}"
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
            check.setCheckState(Qt.CheckState.Unchecked)
            table.setItem(row, 0, check)
            for col, text, editable in (
                (1, dp.label, False),
                (2, _owner_text(dp.ref), False),
                (3, _fmt(dp.xml_value), False),
                (4, _fmt(low), True),
                (5, _fmt(high), True),
                (6, f"{dp.ref.param_id} = {dp.raw_value}", False),
            ):
                item = QTableWidgetItem(text)
                flags = Qt.ItemFlag.ItemIsEnabled
                if editable:
                    flags |= Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsSelectable
                item.setFlags(flags)
                item.setToolTip(tip)
                table.setItem(row, col, item)
        table.blockSignals(False)
        self.param_box.setTitle(f"Detected parameters ({len(self.detected)})")
        self._apply_filter()
        self._refresh_param_count()

    def _apply_filter(self) -> None:
        needle = self.filter_edit.text().strip().lower()
        for row, dp in enumerate(self.detected):
            haystack = f"{dp.label} {dp.ref.key} {_owner_text(dp.ref)}".lower()
            self.param_table.setRowHidden(row, bool(needle) and needle not in haystack)

    def _set_visible_checks(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        self.param_table.blockSignals(True)
        for row in range(self.param_table.rowCount()):
            if not self.param_table.isRowHidden(row):
                self.param_table.item(row, 0).setCheckState(state)
        self.param_table.blockSignals(False)
        self._refresh_param_count()

    def _on_param_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() in (4, 5):
            self._mark_range(item.row())
        self._refresh_param_count()

    def _mark_range(self, row: int) -> bool:
        low = _parse_float(self.param_table.item(row, 4).text())
        high = _parse_float(self.param_table.item(row, 5).text())
        ok = low is not None and high is not None and low < high
        self.param_table.blockSignals(True)
        for col in (4, 5):
            item = self.param_table.item(row, col)
            item.setData(Qt.ItemDataRole.BackgroundRole, None if ok else INVALID_BG)
        self.param_table.blockSignals(False)
        return ok

    def _is_checked(self, row: int) -> bool:
        return self.param_table.item(row, 0).checkState() == Qt.CheckState.Checked

    def _refresh_param_count(self) -> None:
        n = sum(1 for row in range(self.param_table.rowCount()) if self._is_checked(row))
        self.param_count_label.setText(f"{n} perturbed")

    def selected_parameters(self) -> tuple[list[ParameterSpec], list[str]]:
        params: list[ParameterSpec] = []
        problems: list[str] = []
        for row, dp in enumerate(self.detected):
            if not self._is_checked(row):
                continue
            low = _parse_float(self.param_table.item(row, 4).text())
            high = _parse_float(self.param_table.item(row, 5).text())
            if low is None or high is None:
                problems.append(f"{dp.label}: Min and Max must be numbers.")
                continue
            params.append(ParameterSpec(dp.ref.key, dp.label, dp.xml_value, low, high))
        return params, problems

    def apply_parameters(self, params: Sequence[ParameterSpec]) -> list[str]:
        """Tick and set ranges for ``params``; return labels absent from the template."""
        rows = {dp.ref.key: row for row, dp in enumerate(self.detected)}
        self.param_table.blockSignals(True)
        for row in range(self.param_table.rowCount()):
            self.param_table.item(row, 0).setCheckState(Qt.CheckState.Unchecked)
        missing = []
        for p in params:
            row = rows.get(p.key)
            if row is None:
                missing.append(p.label)
                continue
            self.param_table.item(row, 0).setCheckState(Qt.CheckState.Checked)
            self.param_table.item(row, 4).setText(_fmt(p.min))
            self.param_table.item(row, 5).setText(_fmt(p.max))
        self.param_table.blockSignals(False)
        for row in range(self.param_table.rowCount()):
            self._mark_range(row)
        self._refresh_param_count()
        return missing

    # ---------------------------------------------------------------- screen 2
    def _build_screen2(self) -> QWidget:
        root, layout = self._content(
            STAGE_NAMES[1],
            "Choose the metrics the calibration matches and their target values. Every metric "
            "is summarized per particle by its median across seeds and its SD across seeds "
            "(the _std column); tick the statistics to match.",
        )
        builtin_box = QGroupBox("Built-in metrics")
        builtin_layout = QVBoxLayout(builtin_box)
        self.builtin_table = MetricTable()
        self.builtin_table.setMinimumHeight(200)
        self.builtin_table.itemChanged.connect(self._refresh_metric_count)
        builtin_layout.addWidget(self.builtin_table)
        layout.addWidget(builtin_box, 5)

        custom_box = QGroupBox("Custom metrics")
        custom_layout = QVBoxLayout(custom_box)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Custom metrics file"))
        self.custom_edit = QLineEdit()
        self.custom_edit.setReadOnly(True)
        self.custom_edit.setPlaceholderText("optional — a .py file with one function per metric")
        file_row.addWidget(self.custom_edit, 1)
        for text, handler in (
            ("Browse…", self.choose_custom_file),
            ("New from template…", self.new_custom_file),
            ("Reload", self.reload_custom_file),
            ("Clear", self.clear_custom_file),
        ):
            button = QPushButton(text)
            button.setStyleSheet(SMALL_BUTTON_STYLE)
            button.clicked.connect(handler)
            file_row.addWidget(button)
        custom_layout.addLayout(file_row)
        help_label = QLabel(CUSTOM_HELP)
        help_label.setWordWrap(True)
        custom_layout.addWidget(help_label)
        self.custom_status = QLabel("No custom metrics file.")
        self.custom_status.setWordWrap(True)
        custom_layout.addWidget(self.custom_status)
        self.custom_table = MetricTable()
        self.custom_table.setMinimumHeight(140)
        self.custom_table.itemChanged.connect(self._refresh_metric_count)
        custom_layout.addWidget(self.custom_table, 1)
        test_row = QHBoxLayout()
        self.test_button = QPushButton("Test on sample output…")
        self.test_button.setToolTip(
            "Pick one ARCADE particle output folder (the folder holding the "
            "*.CELLS.json / *.LOCATIONS.json files) and run every custom metric on each seed."
        )
        self.test_button.clicked.connect(self.test_custom_metrics)
        test_row.addWidget(self.test_button)
        self.test_status = QLabel()
        test_row.addWidget(self.test_status, 1)
        self.metric_count_label = QLabel()
        test_row.addWidget(self.metric_count_label)
        custom_layout.addLayout(test_row)
        layout.addWidget(custom_box, 3)
        return root

    def _load_builtin_metrics(self) -> None:
        try:
            infos = metrics_plugin.builtin_metrics()
        except Exception as exc:
            infos = []
            self.statusBar().showMessage(f"Built-in metrics unavailable: {exc!r}")
        self.builtin_table.set_metrics([(m.name, m.unit, m.description) for m in infos])
        self._refresh_metric_count()

    def _refresh_metric_count(self, *_args) -> None:
        n = self.builtin_table.checked_count() + self.custom_table.checked_count()
        self.metric_count_label.setText(f"{n} target statistics selected in total")

    def choose_custom_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Custom metrics file", "", PY_FILTER)
        if path:
            self.load_custom_file(Path(path))

    def new_custom_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "New custom metrics file", "custom_metrics.py", PY_FILTER
        )
        if not path:
            return
        try:
            metrics_plugin.write_custom_metrics_template(Path(path))
        except Exception as exc:
            self._error("Cannot write the example file", f"{type(exc).__name__}: {exc}")
            return
        if self.load_custom_file(Path(path)):
            QMessageBox.information(
                self,
                "Custom metrics file created",
                f"Wrote an example to\n{path}\n\nOpen it in your editor, write one function "
                "per metric, save, then click Reload.",
            )

    def reload_custom_file(self) -> None:
        if self.custom_file is None:
            self.statusBar().showMessage("No custom metrics file to reload.")
            return
        self.load_custom_file(self.custom_file)

    def clear_custom_file(self) -> None:
        self.custom_file = None
        self.custom_error = None
        self.custom_edit.clear()
        self.custom_table.set_metrics([])
        self.custom_status.setText("No custom metrics file.")
        self._refresh_metric_count()

    def load_custom_file(self, path: Path) -> bool:
        path = Path(path).expanduser().resolve()
        self.custom_file = path
        self.custom_edit.setText(str(path))
        self.custom_edit.setToolTip(str(path))
        try:
            functions = metrics_plugin.load_custom_metrics(path)
        except Exception as exc:
            self.custom_error = str(exc)
            self.custom_status.setStyleSheet("color: #b00020;")
            self.custom_status.setText(f"Cannot load {path.name}: {exc}")
            self.custom_table.set_metrics([])
            self._refresh_metric_count()
            return False
        self.custom_error = None
        rows = []
        for name, func in functions.items():
            doc = (getattr(func, "__doc__", None) or "").strip().splitlines()
            rows.append((name, "", doc[0] if doc else "custom function"))
        self.custom_table.set_metrics(rows)
        self.custom_status.setStyleSheet("color: #2d6a2d;")
        self.custom_status.setText(
            f"{len(rows)} metric function(s) in {path.name}: {', '.join(functions)}"
        )
        self._refresh_metric_count()
        return True

    def test_custom_metrics(self) -> None:
        if self.custom_file is None or self.custom_error:
            self._error("No usable custom metrics", "Load a custom metrics file first.")
            return
        folder = QFileDialog.getExistingDirectory(self, "ARCADE particle output folder")
        if folder:
            self.start_metric_test(Path(folder))

    def start_metric_test(self, folder: Path) -> None:
        assert self.custom_file is not None
        self.test_button.setEnabled(False)
        self.test_status.setText(f"Testing on {folder.name}…")
        worker = CustomMetricTestWorker(folder, self.custom_file)
        worker.succeeded.connect(lambda results: self._metric_test_done(folder, results))
        worker.failed.connect(self._metric_test_failed)
        worker.finished.connect(lambda: self.test_button.setEnabled(True))
        self.test_worker = worker
        worker.start()

    def _metric_test_done(self, folder: Path, results: list) -> None:
        n_err = sum(len(r.errors) for r in results)
        self.test_status.setText(
            f"Tested {len(results)} metric(s) on {folder.name}: {n_err} seed error(s)."
        )
        MetricTestDialog(self, folder, results).exec()

    def _metric_test_failed(self, details: str) -> None:
        self.test_status.setText("Test failed.")
        self._error("Custom metric test failed", details)

    # ---------------------------------------------------------------- screen 3
    def _build_screen3(self) -> QWidget:
        root, layout = self._content(
            STAGE_NAMES[2],
            "Set the calibration hyperparameters and where to write the run directory. The GUI "
            "only exports the run; it does not run ARCADE.",
        )
        top = QHBoxLayout()
        top.addWidget(self._build_hyper_box(), 1)
        top.addWidget(self._build_output_box(), 2)
        layout.addLayout(top)

        check_box = QGroupBox("Validation")
        check_layout = QVBoxLayout(check_box)
        self.validation_list = QListWidget()
        self.validation_list.setFixedHeight(120)
        check_layout.addWidget(self.validation_list)
        layout.addWidget(check_box)

        export_box = QGroupBox("Export and run")
        export_layout = QVBoxLayout(export_box)
        action_row = QHBoxLayout()
        recheck = QPushButton("Re-check")
        recheck.clicked.connect(self.refresh_validation)
        action_row.addWidget(recheck)
        open_run = QPushButton("Open existing run_spec.json…")
        open_run.clicked.connect(self.choose_run_spec)
        action_row.addWidget(open_run)
        action_row.addStretch(1)
        self.export_button = QPushButton("Export run directory")
        self.export_button.setStyleSheet(PRIMARY_STYLE)
        self.export_button.clicked.connect(self.export_run_dir)
        action_row.addWidget(self.export_button)
        export_layout.addLayout(action_row)
        self.export_status = QLabel("Not exported yet. Then run:")
        self.export_status.setWordWrap(True)
        export_layout.addWidget(self.export_status)
        command_row = QHBoxLayout()
        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        self.command_edit.setStyleSheet(
            "QLineEdit { font-family: Menlo, Consolas, monospace; background: #fbfbfb; }"
        )
        command_row.addWidget(self.command_edit, 1)
        copy = QPushButton("Copy")
        copy.setStyleSheet(SMALL_BUTTON_STYLE)
        copy.clicked.connect(self._copy_command)
        command_row.addWidget(copy)
        export_layout.addLayout(command_row)
        note = QLabel(CLUSTER_NOTE)
        note.setWordWrap(True)
        export_layout.addWidget(note)
        layout.addWidget(export_box)
        layout.addStretch(1)
        self._update_command()
        return root

    def _build_hyper_box(self) -> QGroupBox:
        box = QGroupBox("Hyperparameters")
        form = QFormLayout(box)
        hp = Hyperparameters()

        def spin(value: int, low: int, high: int) -> QSpinBox:
            widget = QSpinBox()
            widget.setRange(low, high)
            widget.setValue(value)
            widget.valueChanged.connect(self.refresh_validation)
            return widget

        self.particles_spin = spin(hp.n_particles, 2, 10_000_000)
        self.generations_spin = spin(hp.n_generations, 1, 1000)
        self.trees_spin = spin(hp.n_trees, 1, 100_000)
        self.leaf_spin = spin(hp.min_samples_leaf, 1, 100_000)
        self.subsample_spin = QDoubleSpinBox()
        self.subsample_spin.setRange(0.01, 1.0)
        self.subsample_spin.setSingleStep(0.05)
        self.subsample_spin.setDecimals(2)
        self.subsample_spin.setValue(hp.subsample_ratio)
        # The pipeline's DRF always subsamples half the particles per tree
        # (ABCSMCRF._build_rf_model never passes subsample_ratio on), so the
        # field is shown for transparency but cannot be changed yet.
        self.subsample_spin.setEnabled(False)
        self.subsample_spin.setToolTip("Fixed at 0.5 by the current DRF implementation.")
        self.random_state_spin = spin(hp.random_state, 0, 2**31 - 1)
        self.cpu_spin = spin(hp.n_cpu, 1, 4096)
        self.rf_combo = QComboBox()
        # The RF (per-parameter regressor) branch of ABCSMCRF is not usable for
        # ARCADE runs yet; the runner rejects it, so only DRF is offered.
        self.rf_combo.addItems(["DRF"])
        self.rf_combo.setEnabled(False)
        self.rf_combo.setToolTip("Distributional random forest, as in the paper.")
        for label, widget in (
            ("Particles per generation", self.particles_spin),
            ("Generations", self.generations_spin),
            ("Trees (n_trees)", self.trees_spin),
            ("min_samples_leaf", self.leaf_spin),
            ("subsample_ratio", self.subsample_spin),
            ("random_state", self.random_state_spin),
            ("CPUs (n_cpu)", self.cpu_spin),
            ("Random-forest type", self.rf_combo),
        ):
            form.addRow(label, widget)
        return box

    def _build_output_box(self) -> QGroupBox:
        box = QGroupBox("Output")
        grid = QGridLayout(box)
        grid.setColumnStretch(1, 1)
        self.run_dir_edit = self._path_row(grid, 0, "Run directory", self.choose_run_dir)
        self.run_dir_edit.setPlaceholderText("folder to create, e.g. ~/runs/my_calibration")
        self.run_dir_edit.textChanged.connect(self._update_command)
        self.run_dir_edit.textChanged.connect(self.refresh_validation)
        paths = RunPaths()
        self.input_dir_edit = QLineEdit(paths.input_dir)
        self.output_dir_edit = QLineEdit(paths.output_dir)
        for row, (label, edit) in enumerate(
            (
                ("ARCADE input subfolder", self.input_dir_edit),
                ("ARCADE output subfolder", self.output_dir_edit),
            ),
            start=1,
        ):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1, 1, 2)
            edit.textChanged.connect(self.refresh_validation)
        layout_note = QLabel(
            "The run directory receives run_spec.json, template.xml, custom_metrics.py (if "
            "used), run.sh and README.txt. Subfolders are relative to it: inputs get "
            "iter_<g>/inputs/input_<i>.xml; outputs get ARCADE results plus per-generation "
            "final_metrics.csv and all_param_df.csv."
        )
        layout_note.setWordWrap(True)
        grid.addWidget(layout_note, 3, 0, 1, 3)
        grid.setRowStretch(4, 1)
        return box

    def choose_run_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Run directory")
        if path:
            self.run_dir_edit.setText(path)

    def _run_dir(self) -> Optional[Path]:
        text = self.run_dir_edit.text().strip()
        return Path(text).expanduser().resolve() if text else None

    def _update_command(self) -> None:
        run_dir = self._run_dir()
        self.command_edit.setText(export.run_command(run_dir or Path("<run_dir>")))
        self.command_edit.setCursorPosition(0)
        self.command_edit.setToolTip(self.command_edit.text())

    def _copy_command(self) -> None:
        QGuiApplication.clipboard().setText(self.command_edit.text())
        self.statusBar().showMessage("Command copied to the clipboard.", 4000)

    def hyperparameters(self) -> Hyperparameters:
        return Hyperparameters(
            n_particles=self.particles_spin.value(),
            n_generations=self.generations_spin.value(),
            n_trees=self.trees_spin.value(),
            min_samples_leaf=self.leaf_spin.value(),
            subsample_ratio=round(self.subsample_spin.value(), 4),
            random_state=self.random_state_spin.value(),
            n_cpu=self.cpu_spin.value(),
            rf_type=self.rf_combo.currentText(),  # type: ignore[arg-type]
            criterion=self.hidden_hyper.criterion,
        )

    def set_hyperparameters(self, hp: Hyperparameters) -> None:
        self.hidden_hyper = hp
        self.particles_spin.setValue(hp.n_particles)
        self.generations_spin.setValue(hp.n_generations)
        self.trees_spin.setValue(hp.n_trees)
        self.leaf_spin.setValue(hp.min_samples_leaf)
        self.subsample_spin.setValue(hp.subsample_ratio)
        self.random_state_spin.setValue(hp.random_state)
        self.cpu_spin.setValue(hp.n_cpu)
        self.rf_combo.setCurrentText(hp.rf_type)

    # ------------------------------------------------------------ spec + export
    def collect_spec(self) -> tuple[Optional[RunSpec], list[str], list[str]]:
        """Current GUI state as a RunSpec (source paths) plus blocking problems and warnings."""
        problems: list[str] = []
        warnings: list[str] = []
        if self.template_path is None:
            problems.append("Choose an ARCADE XML template (Screen 1).")
        jar = self.jar_edit.text().strip()
        if not jar:
            problems.append("Set the ARCADE JAR path (Screen 1).")
        elif not Path(jar).expanduser().is_file():
            warnings.append(
                "The ARCADE JAR does not exist on this computer; fine if the run executes "
                "elsewhere, but check jar_path there."
            )
        params, param_problems = self.selected_parameters()
        problems += param_problems
        builtin, builtin_problems = self.builtin_table.selected("builtin")
        custom, custom_problems = self.custom_table.selected("custom")
        problems += builtin_problems + custom_problems
        if self.custom_file is not None and self.custom_error:
            msg = f"Custom metrics file does not load: {self.custom_error}"
            (problems if custom else warnings).append(msg)
        if self._run_dir() is None:
            problems.append("Choose a run directory.")
        in_dir = self.input_dir_edit.text().strip()
        out_dir = self.output_dir_edit.text().strip()
        if not in_dir or not out_dir:
            problems.append("Input and output subfolder names must not be empty.")
        elif in_dir == out_dir:
            problems.append("Input and output subfolders must differ.")
        spec = export.build_spec(
            template_path=self.template_path or "",
            jar_path=jar,
            simulation=self.simulation_settings(),
            parameters=params,
            metrics=builtin + custom,
            hyperparameters=self.hyperparameters(),
            paths=RunPaths(input_dir=in_dir, output_dir=out_dir),
            custom_metrics_file=self.custom_file,
        )
        for problem in spec.validate():
            if problem not in problems:
                problems.append(problem)
        order_problems, order_warnings = constraints.order_problems(
            params, constraints.uniform_order_constraints(params, self.detected)
        )
        problems += order_problems
        warnings += order_warnings + constraints.simulation_warnings(spec.simulation)
        return spec, problems, warnings

    def refresh_validation(self, *_args) -> None:
        if not hasattr(self, "validation_list"):
            return
        spec, problems, warnings = self.collect_spec()
        self.validation_list.clear()
        for text in problems:
            item = QListWidgetItem(f"✗  {text}")
            item.setForeground(QColor("#b00020"))
            self.validation_list.addItem(item)
        for text in warnings:
            item = QListWidgetItem(f"⚠  {text}")
            item.setForeground(QColor("#8a5a00"))
            self.validation_list.addItem(item)
        if not problems and spec is not None:
            item = QListWidgetItem(
                f"✓  Ready to export: {len(spec.parameters)} parameters, "
                f"{len(spec.metrics)} target statistics, {spec.hyperparameters.n_particles} "
                f"particles x {spec.hyperparameters.n_generations} generations x "
                f"{spec.simulation.seeds} seeds."
            )
            item.setForeground(QColor("#2d6a2d"))
            self.validation_list.addItem(item)

    def export_run_dir(self) -> Optional[Path]:
        spec, problems, _ = self.collect_spec()
        self.refresh_validation()
        if problems or spec is None or self.template_path is None:
            self.go_to(2)
            self._error("Cannot export yet", "Fix these first:\n- " + "\n- ".join(problems))
            return None
        run_dir = self._run_dir()
        assert run_dir is not None
        custom = self.custom_file if self.custom_file and not self.custom_error else None
        try:
            try:
                spec_path = export.export_run(spec, run_dir, self.template_path, custom)
            except export.RunDirConflict as exc:
                answer = QMessageBox.question(
                    self,
                    "Overwrite run directory?",
                    f"{exc}\n\nOverwrite its run_spec.json, template.xml, run.sh and README.txt?",
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return None
                spec_path = export.export_run(
                    spec, run_dir, self.template_path, custom, overwrite=True
                )
        except Exception as exc:
            self._error("Export failed", f"{type(exc).__name__}: {exc}")
            return None
        self.last_export = spec_path
        self._update_command()
        self.export_status.setText(f"Exported to {spec_path.parent}. Run it with:")
        self.export_status.setStyleSheet("color: #2d6a2d; font-weight: bold;")
        self.statusBar().showMessage(f"Exported {spec_path}")
        return spec_path

    def choose_run_spec(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open run_spec.json", "", SPEC_FILTER)
        if path:
            self.load_run_spec(Path(path))

    def load_run_spec(self, path: Path) -> bool:
        """Repopulate every screen from an exported run_spec.json."""
        try:
            spec = export.load_run(path)
        except Exception as exc:
            self._error("Cannot open run spec", f"{type(exc).__name__}: {exc}")
            return False
        template = spec.resolve(spec.template_path)
        if template is None or not self.load_template(template, apply_settings=False):
            return False
        self.set_simulation_settings(spec.simulation)
        self.jar_edit.setText(spec.jar_path)
        missing = self.apply_parameters(spec.parameters)
        custom_path = spec.resolve(spec.custom_metrics_file)
        if custom_path is not None:
            self.load_custom_file(custom_path)
        else:
            self.clear_custom_file()
        missing += self.builtin_table.apply([m for m in spec.metrics if m.source == "builtin"])
        missing += self.custom_table.apply([m for m in spec.metrics if m.source == "custom"])
        self._refresh_metric_count()
        self.set_hyperparameters(spec.hyperparameters)
        self.input_dir_edit.setText(spec.paths.input_dir)
        self.output_dir_edit.setText(spec.paths.output_dir)
        if spec.base_dir is not None:
            self.run_dir_edit.setText(str(spec.base_dir))
            self.export_status.setText(
                f"Opened the run in {spec.base_dir}; export again to save changes. Run it with:"
            )
        self.refresh_validation()
        if missing:
            self._error(
                "Some entries were not found",
                "These parameters/metrics from the run spec are not available and were "
                "skipped:\n- " + "\n- ".join(missing),
            )
        self.statusBar().showMessage(f"Opened {path}")
        return True

    def _error(self, title: str, message: str) -> None:
        self.statusBar().showMessage(f"{title}: {message.splitlines()[0] if message else ''}")
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.test_worker is not None and self.test_worker.isRunning():
            self.test_worker.wait(2000)
        super().closeEvent(event)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Launch the GUI; an optional argument opens a template (.xml) or run_spec.json."""
    args = list(sys.argv[1:] if argv is None else argv)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = ArcadeSetupWindow()
    window.show()
    if args:
        path = Path(args[0])
        if path.suffix.lower() == ".json":
            window.load_run_spec(path)
        else:
            window.load_template(path)
    return app.exec()
