"""Built-in and user-defined (custom) per-seed metrics for GUI runs.

Contract used by the GUI (``app.py``) and the runner (``run.py``). Owner: the
"metrics" work block. Implement every function below; keep signatures.

A custom metrics file is a plain Python file. Every top-level function whose
name does not start with ``_`` and that takes exactly one positional argument
is a metric. It receives a :class:`SeedView` for ONE seed of ONE particle and
returns a float. For each metric ``m`` the particle table gets ``m`` (median
across seeds) and ``m_std`` (SD across seeds), matching the built-in
aggregation in ``analyze.save_aggregated_results.SimulationMetrics``.
An exception or a non-finite return for a seed makes that seed's value NaN;
the particle's value is NaN if every seed is NaN.
"""

from __future__ import annotations

import inspect
import json
import logging
import math
import re
import sys
import traceback
import types
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ``<group>_<name>_<seed>_<timestamp>.<CELLS|LOCATIONS>.json``; group/name may contain "_".
_FILE_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<seed>\d+)_(?P<ts>\d+)\.(?P<kind>CELLS|LOCATIONS)\.json$"
)
_MINUTES_PER_DAY = 60 * 24
# Columns of the per-particle table that a custom metric may not shadow.
_RESERVED_NAMES = frozenset({"input_folder", "states", "colony_growth_r"})
_MODULE_PREFIX = "_inverse_design_custom_metrics_"


@dataclass
class MetricInfo:
    name: str  # column name without _std
    description: str
    unit: str  # e.g. "hr", "-", "um/day"


@dataclass
class MetricTestResult:
    """Result of running one custom metric on a sample simulation folder."""

    name: str
    per_seed: list[float]  # NaN where the function failed
    errors: list[str]  # one formatted traceback/summary per failing seed
    median: float
    std: float


class SeedView:
    """Read-only, lazily loaded view of one seed's ARCADE output files.

    Files are ``<group>_<name>_<seed:04d>_<timestamp:06d>.{CELLS,LOCATIONS}.json``
    in the particle's output folder (see ``analyze.utils.file_utils``).

    ``timestamps`` lists the snapshots that have both a CELLS and a LOCATIONS
    file. Loaded data are cached and shared between metrics: treat them as
    read-only (copy before modifying).
    """

    seed: int
    timestamps: list[str]  # sorted ascending, e.g. ["000000", "000720", ...]

    def __init__(self, folder: Path, seed: int, files: dict[str, dict[str, Path]]) -> None:
        """``files`` maps timestamp -> {"CELLS": path, "LOCATIONS": path}."""
        self.folder = Path(folder)
        self.seed = int(seed)
        self._files = {
            ts: kinds for ts, kinds in files.items() if "CELLS" in kinds and "LOCATIONS" in kinds
        }
        self.timestamps = sorted(self._files, key=int)
        self._cache: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def __repr__(self) -> str:
        n_ts = len(self.timestamps)
        return f"SeedView(seed={self.seed}, timestamps={n_ts}, folder={self.folder})"

    @property
    def last_timestamp(self) -> str:
        """The final snapshot's timestamp."""
        if not self.timestamps:
            raise ValueError(f"seed {self.seed} has no complete snapshots in {self.folder}")
        return self.timestamps[-1]

    def _load(self, kind: str, timestamp: str) -> list[dict[str, Any]]:
        key = (kind, timestamp)
        if key not in self._cache:
            if timestamp not in self._files:
                raise KeyError(
                    f"seed {self.seed} has no snapshot at timestamp {timestamp!r}; "
                    f"available: {self.timestamps}"
                )
            with open(self._files[timestamp][kind]) as fh:
                self._cache[key] = json.load(fh)
        return self._cache[key]

    def cells(self, timestamp: str) -> list[dict[str, Any]]:
        """Parsed CELLS.json for this seed at ``timestamp`` (cached)."""
        return self._load("CELLS", timestamp)

    def locations(self, timestamp: str) -> list[dict[str, Any]]:
        """Parsed LOCATIONS.json for this seed at ``timestamp`` (cached)."""
        return self._load("LOCATIONS", timestamp)

    def days(self, timestamps: Optional[Sequence[str]] = None) -> list[float]:
        """Timestamps converted to days (ticks are minutes)."""
        if timestamps is None:
            timestamps = self.timestamps
        return [int(ts) / _MINUTES_PER_DAY for ts in timestamps]


# Every non-``states`` column of ``SimulationMetrics.analyze_simulation(...)["final_metrics"]``
# (CELLULAR_METRICS + POPULATION_METRICS at the last snapshot, plus doub_time and
# colony_growth), except the colony_growth_r fit diagnostic.
_BUILTINS: tuple[MetricInfo, ...] = (
    MetricInfo(
        "doub_time",
        "Population doubling time from the first to the last snapshot (inf if no growth)",
        "hr",
    ),
    MetricInfo(
        "colony_growth",
        "Colony growth rate: slope of a linear fit of colony diameter vs time over all snapshots",
        "um/day",
    ),
    MetricInfo(
        "symmetry",
        "Colony symmetry at the last snapshot: fraction of hexagonally symmetric positions "
        "that are occupied (1 = perfectly symmetric)",
        "-",
    ),
    MetricInfo(
        "act_ratio",
        "Fraction of active (proliferative or migratory) cells at the last snapshot",
        "-",
    ),
    MetricInfo("n_cells", "Number of cells at the last snapshot", "cells"),
    MetricInfo(
        "colony_diameter",
        "Colony diameter at the last snapshot (mean hexagonal extent x 30 um)",
        "um",
    ),
    MetricInfo("vol", "Mean volume of non-necrotic cells at the last snapshot", "um^3"),
    MetricInfo(
        "cycle_length",
        "Mean cell-cycle length at the last snapshot (cells with a completed cycle)",
        "hr",
    ),
    MetricInfo("age", "Mean age of non-necrotic cells at the last snapshot", "min"),
    MetricInfo(
        "activity",
        "Active vs inactive balance at the last snapshot: 2*active/(active+inactive) - 1, "
        "inactive = necrotic or apoptotic",
        "-",
    ),
    MetricInfo("shannon", "Shannon diversity of cell states at the last snapshot", "-"),
)


def builtin_metrics() -> list[MetricInfo]:
    """Metrics the existing pipeline already computes per particle.

    Must match the columns that ``compute_particle_metrics`` returns for
    built-ins (each also has a ``_std`` column).
    """
    return [MetricInfo(m.name, m.description, m.unit) for m in _BUILTINS]


def _builtin_names() -> set[str]:
    return {m.name for m in _BUILTINS}


def _is_metric_function(name: str, obj: Any, module_name: str) -> bool:
    """Public function defined in the file itself with exactly one required positional arg."""
    if name.startswith("_") or not inspect.isfunction(obj) or obj.__module__ != module_name:
        return False
    try:
        params = list(inspect.signature(obj).parameters.values())
    except (TypeError, ValueError):
        return False
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    variadic = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    required = [p for p in params if p.default is p.empty and p.kind not in variadic]
    return len(required) == 1 and required[0].kind in positional


def _user_frame_hint(tb: Optional[types.TracebackType], filename: str) -> str:
    """`` (file.py line N)`` for the deepest traceback frame inside the user's file."""
    frames = [f for f in traceback.extract_tb(tb) if f.filename == filename]
    if not frames:
        return ""
    return f" ({Path(filename).name} line {frames[-1].lineno})"


def _format_exception(exc: BaseException, filename: str) -> str:
    """Last traceback line (``Type: message``) plus the user-file line number."""
    summary = traceback.format_exception_only(type(exc), exc)[-1].strip()
    return summary + _user_frame_hint(exc.__traceback__, filename)


def load_custom_metrics(path: Path) -> dict[str, Callable[["SeedView"], float]]:
    """Import ``path`` and return its metric functions by name.

    Raise ValueError (message shown in the GUI) on import/syntax errors or when
    no metric function is found, or when a name collides with a built-in.

    The file runs once per call as a fresh module with a unique name. That name
    is in ``sys.modules`` only while the file executes (so dataclasses and
    similar work) and is removed afterwards. Functions are returned in file order.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Custom metrics file not found: {path}")
    if path.suffix != ".py":
        raise ValueError(f"Custom metrics file must be a .py file: {path}")
    filename = str(path.resolve())
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read custom metrics file {path}: {exc}") from exc
    try:
        code = compile(source, filename, "exec")
    except SyntaxError as exc:
        where = f" (line {exc.lineno})" if exc.lineno else ""
        raise ValueError(f"Syntax error in {path.name}{where}: {exc.msg}") from exc

    module_name = _MODULE_PREFIX + uuid.uuid4().hex
    module = types.ModuleType(module_name)
    module.__file__ = filename
    sys.modules[module_name] = module
    try:
        exec(code, module.__dict__)
    except (Exception, SystemExit) as exc:
        raise ValueError(
            f"Error while loading {path.name}: {_format_exception(exc, filename)}"
        ) from exc
    finally:
        sys.modules.pop(module_name, None)

    metrics = {
        name: obj
        for name, obj in vars(module).items()
        if _is_metric_function(name, obj, module_name)
    }
    if not metrics:
        raise ValueError(
            f"No metric functions found in {path.name}. Define top-level functions that "
            "take one argument (a SeedView) and return a number, e.g. "
            "'def final_diameter_mm(seed): ...'."
        )
    builtins = _builtin_names()
    for name in metrics:
        if name in builtins:
            raise ValueError(
                f"Custom metric '{name}' in {path.name} has the same name as a built-in "
                "metric; rename the function."
            )
        if name in _RESERVED_NAMES or name.endswith("_std"):
            raise ValueError(
                f"Custom metric name '{name}' in {path.name} is reserved (names ending in "
                "'_std' hold the SD across seeds); rename the function."
            )
    return metrics


_TEMPLATE = '''\
"""Custom target metrics for an ARCADE calibration run.

How this file is used
---------------------
* Every top-level function below whose name does not start with "_" and that
  takes ONE argument is a metric. The function name becomes the metric name.
* The argument, ``seed``, gives access to ONE seed (replicate) of ONE simulated
  parameter set:
      seed.seed                 seed number (int)
      seed.timestamps           snapshot times as strings, ascending
                                (ARCADE ticks = minutes, e.g. "000000", "000720")
      seed.last_timestamp       the final snapshot, same as seed.timestamps[-1]
      seed.days(timestamps)     snapshot times converted to days (all if omitted)
      seed.cells(t)             list of cell dicts at snapshot t (CELLS.json):
                                "id", "state", "volume", "age", "cycles", ...
      seed.locations(t)         list of location dicts at snapshot t
                                (LOCATIONS.json): "coordinate" [u, v, w, z], "ids"
  Treat the returned lists as read-only; they are shared between metrics.
* Return a single number. The calibration takes the median across seeds as
  the metric (column ``name``) and the SD across seeds as ``name_std``.
  If the function raises or returns NaN/inf for a seed, that seed is ignored;
  if it fails for every seed, the parameter set is dropped as invalid.
* Names must not clash with built-in metrics (doub_time, symmetry, ...) and
  must not end in "_std". Functions starting with "_" are private helpers.
* In the GUI, use "Test on sample output" to run these functions on an
  existing ARCADE output folder before starting a run.

Example goal: "tumor size asymptotes between 1 and 2 mm diameter in
simulations without angiogenesis". Two metrics express it:
    final_diameter_mm       target e.g. 1.5   (size at the end)
    late_growth_mm_per_day  target 0          (no growth late on = plateau)
"""

import math

from inverse_design.gui.metric_helpers import colony_diameter_um, count_state, n_cells, slope


def final_diameter_mm(seed):
    """Colony diameter at the last snapshot, in millimetres."""
    locations = seed.locations(seed.last_timestamp)
    return colony_diameter_um(locations) / 1000.0


def late_growth_mm_per_day(seed):
    """Slope of colony diameter vs time over the last quarter of the snapshots.

    Units are mm/day. A value near 0 means the colony has stopped growing
    (a plateau); a positive value means it is still expanding.
    """
    n_late = max(2, math.ceil(len(seed.timestamps) / 4))  # at least two points for a slope
    late = seed.timestamps[-n_late:]
    diameters_mm = [colony_diameter_um(seed.locations(t)) / 1000.0 for t in late]
    return slope(seed.days(late), diameters_mm)


# More ideas (uncomment to use; each becomes a metric):
#
# def necrotic_fraction(seed):
#     """Fraction of necrotic cells at the last snapshot."""
#     cells = seed.cells(seed.last_timestamp)
#     return count_state(cells, "NECROTIC") / max(n_cells(cells), 1)
#
# def _private_helper(x, y):  # leading "_" or two arguments: not a metric
#     return x + y
'''


def write_custom_metrics_template(path: Path) -> None:
    """Write a commented example file (final colony diameter in mm and the
    late-time diameter slope in mm/day) that users edit.

    Overwrites an existing file; parent directories are created.
    """
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_TEMPLATE, encoding="utf-8")


def _scan_output_files(sim_folder: Path) -> dict[int, dict[str, dict[str, Path]]]:
    """``{seed: {timestamp: {"CELLS"|"LOCATIONS": path}}}`` for one particle folder.

    Raises ValueError when the folder mixes several simulations (file prefixes),
    because seeds would then be ambiguous.
    """
    found: dict[int, dict[str, dict[str, Path]]] = {}
    prefixes: set[str] = set()
    for entry in sorted(sim_folder.iterdir()):
        match = _FILE_RE.match(entry.name)
        if match is None or not entry.is_file():
            continue
        prefixes.add(match["prefix"])
        by_ts = found.setdefault(int(match["seed"]), {})
        by_ts.setdefault(match["ts"], {}).setdefault(match["kind"], entry)
    if len(prefixes) > 1:
        shown = ", ".join(sorted(prefixes)[:3]) + (", ..." if len(prefixes) > 3 else "")
        raise ValueError(
            f"{sim_folder} holds output of {len(prefixes)} different simulations ({shown}); "
            "choose the output folder of a single parameter set."
        )
    return found


def iter_seed_views(sim_folder: Path) -> list[SeedView]:
    """One SeedView per seed found in an ARCADE particle output folder.

    Seeds are in ascending order; an empty list means no output files. Raises
    ValueError if ``sim_folder`` is not a directory or mixes several simulations.
    """
    sim_folder = Path(sim_folder).expanduser()
    if not sim_folder.is_dir():
        raise ValueError(f"Simulation output folder not found: {sim_folder}")
    return [
        SeedView(sim_folder, seed, files)
        for seed, files in sorted(_scan_output_files(sim_folder).items())
    ]


def _evaluate(
    name: str, func: Callable[[SeedView], float], seed: SeedView, filename: str
) -> tuple[float, Optional[str]]:
    """One metric on one seed: ``(value, None)`` or ``(nan, error summary)``."""
    try:
        raw = func(seed)
    except (Exception, SystemExit) as exc:
        return math.nan, f"{name}, seed {seed.seed}: {_format_exception(exc, filename)}"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return math.nan, (
            f"{name}, seed {seed.seed}: returned {type(raw).__name__} {raw!r:.60}, "
            "expected a number"
        )
    if not math.isfinite(value):
        return math.nan, f"{name}, seed {seed.seed}: returned non-finite value {value}"
    return value, None


def _aggregate(values: Sequence[float]) -> tuple[float, float]:
    """Median and population SD (ddof=0, as ``np.std`` in the built-ins) of finite values."""
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.nan, math.nan
    return float(np.median(finite)), float(np.std(finite))


def test_custom_metrics(sim_folder: Path, custom_file: Path) -> list[MetricTestResult]:
    """Run every custom metric on every seed of ``sim_folder`` (GUI "Test" button).

    Errors inside a metric function are reported in ``MetricTestResult.errors``
    and never raised. ValueError is raised only for a missing folder, a folder
    without ARCADE output files, or a custom file that cannot be loaded.
    """
    metrics = load_custom_metrics(custom_file)
    seeds = iter_seed_views(sim_folder)
    if not seeds:
        raise ValueError(
            f"No ARCADE output files (*.CELLS.json / *.LOCATIONS.json) in {sim_folder}"
        )
    filename = str(Path(custom_file).expanduser().resolve())
    results = []
    for name, func in metrics.items():
        per_seed: list[float] = []
        errors: list[str] = []
        for seed in seeds:
            value, error = _evaluate(name, func, seed, filename)
            per_seed.append(value)
            if error is not None:
                errors.append(error)
        median, std = _aggregate(per_seed)
        results.append(MetricTestResult(name, per_seed, errors, median, std))
    return results


# Keep pytest from collecting the GUI helper above as a test when it is imported by name.
test_custom_metrics.__test__ = False  # type: ignore[attr-defined]


def _base_names(names: Optional[Sequence[str]]) -> list[str]:
    """Strip a trailing ``_std`` and drop duplicates, keeping order."""
    out: list[str] = []
    for name in names or ():
        base = name[: -len("_std")] if name.endswith("_std") else name
        if base not in out:
            out.append(base)
    return out


def _builtin_values(sim_folder: Path, names: Sequence[str]) -> dict[str, float]:
    """Built-ins via ``SimulationMetrics.analyze_simulation`` over every snapshot found."""
    from inverse_design.analyze.save_aggregated_results import SimulationMetrics

    timestamps = sorted(
        {ts for files in _scan_output_files(sim_folder).values() for ts in files}, key=int
    )
    if len(timestamps) < 2:
        raise ValueError(f"need at least two snapshots in {sim_folder}, found {len(timestamps)}")
    calculator = SimulationMetrics(str(sim_folder), str(sim_folder))
    final = calculator.analyze_simulation(sim_folder, timestamps)["final_metrics"]
    out: dict[str, float] = {}
    for name in names:
        out[name] = float(final[name])
        out[name + "_std"] = float(final[name + "_std"])
    return out


def compute_particle_metrics(
    sim_folder: Path,
    builtin_names: Sequence[str],
    custom_file: Optional[Path] = None,
    custom_names: Optional[Sequence[str]] = None,
) -> dict[str, float]:
    """Metrics for one particle folder: ``{name: median, name + "_std": sd}``.

    ``builtin_names``/``custom_names`` are base names without ``_std``. Built-ins
    reuse the existing pipeline (``SimulationMetrics.analyze_simulation`` over all
    snapshot timestamps) so values match the paper's definitions.

    ``custom_names=None`` computes every metric in ``custom_file``; a trailing
    ``_std`` on a requested name is ignored. Unknown names or an unloadable
    custom file raise ValueError (a setup error). A missing or empty output
    folder, or a failed built-in analysis, yields NaN values (an invalid
    particle) instead of raising. Built-ins are rounded to 3 decimals as in
    ``final_metrics.csv``; custom values are not rounded.
    """
    sim_folder = Path(sim_folder).expanduser()
    builtins = _base_names(builtin_names)
    unknown = [n for n in builtins if n not in _builtin_names()]
    if unknown:
        raise ValueError(
            f"Unknown built-in metric(s) {unknown}; available: {sorted(_builtin_names())}"
        )

    customs: dict[str, Callable[[SeedView], float]] = {}
    custom_path = Path(custom_file).expanduser() if custom_file is not None else None
    if custom_path is not None:
        loaded = load_custom_metrics(custom_path)
        wanted = list(loaded) if custom_names is None else _base_names(custom_names)
        missing = [n for n in wanted if n not in loaded]
        if missing:
            raise ValueError(
                f"Custom metric(s) {missing} not found in {custom_path.name}; "
                f"available: {list(loaded)}"
            )
        customs = {n: loaded[n] for n in wanted}
    elif _base_names(custom_names):
        raise ValueError("Custom metrics were requested but no custom metrics file was given.")

    result: dict[str, float] = {}
    for name in builtins + list(customs):
        result[name] = math.nan
        result[name + "_std"] = math.nan

    seeds = iter_seed_views(sim_folder) if sim_folder.is_dir() else []
    if not seeds:
        logger.warning("No ARCADE output in %s; metrics set to NaN", sim_folder)
        return result

    if builtins:
        try:
            result.update(_builtin_values(sim_folder, builtins))
        except Exception as exc:  # noqa: BLE001 - an unreadable particle is invalid, not fatal
            logger.warning("Built-in metrics failed for %s: %s", sim_folder, exc)

    if customs and custom_path is not None:
        filename = str(custom_path.resolve())
        for name, func in customs.items():
            values = []
            for seed in seeds:
                value, error = _evaluate(name, func, seed, filename)
                if error is not None:
                    logger.debug("%s: %s", sim_folder, error)
                values.append(value)
            result[name], result[name + "_std"] = _aggregate(values)
    return result
