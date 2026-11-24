"""Run specification shared by the ARCADE GUI and the GUI runner.

The GUI writes a ``run_spec.json`` with :func:`save_spec`; the runner
(``python -m inverse_design.gui.run <run_dir>/run_spec.json``) reads it with
:func:`load_spec`. Every other module in ``inverse_design.gui`` exchanges data
through the dataclasses defined here, so this file is the contract between
them. Bump ``SPEC_VERSION`` when the JSON layout changes incompatibly.

Relative paths inside a spec are resolved against the directory that holds the
``run_spec.json`` file (see :meth:`RunSpec.resolve`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

SPEC_VERSION = 1
SPEC_FILENAME = "run_spec.json"
TEMPLATE_FILENAME = "template.xml"
CUSTOM_METRICS_FILENAME = "custom_metrics.py"

# Distribution fields that a parameter reference may point at.
DistField = Literal["MU", "SIGMA", "MIN", "MAX"]


@dataclass(frozen=True)
class ParamRef:
    """Location of one numeric value inside an ARCADE XML template.

    A value lives in an element whose tag ends with ``.parameter`` (for
    example ``population.parameter``, ``layer.parameter``,
    ``component.parameter``, ``patch.parameter``) and that has ``id`` and
    ``value`` attributes.

    owner_tag:  tag of the nearest ancestor that carries an ``id`` attribute
                (``population``, ``layer``, ``component``, ``action``, ...),
                or ``""`` when there is none (e.g. ``patch.parameter``).
    owner_id:   that ancestor's ``id`` (``""`` when owner_tag is ``""``).
    param_id:   the parameter element's ``id`` (e.g.
                ``metabolism/GLUCOSE_UPTAKE_RATE``).
    operation:  the parameter element's ``operation`` attribute, or ``""``.
                Needed because a layer may hold the same ``id`` for several
                operations.
    field:      ``None`` when ``value`` is a plain number; ``"MU"``/``"SIGMA"``
                for ``NORMAL(MU=..,SIGMA=..)``; ``"MIN"``/``"MAX"`` for
                ``UNIFORM(MIN=..,MAX=..)``.
    """

    owner_tag: str
    owner_id: str
    param_id: str
    operation: str = ""
    field: Optional[DistField] = None

    @property
    def key(self) -> str:
        """Stable, unique string form used as a JSON key and CSV column.

        Format: ``owner_tag:owner_id:param_id[@operation][.FIELD]``, e.g.
        ``population:cancerous:metabolism/GLUCOSE_UPTAKE_RATE.MU`` or
        ``layer:GLUCOSE:CONCENTRATION@generator``.
        """
        key = f"{self.owner_tag}:{self.owner_id}:{self.param_id}"
        if self.operation:
            key += f"@{self.operation}"
        if self.field:
            key += f".{self.field}"
        return key

    @classmethod
    def from_key(cls, key: str) -> "ParamRef":
        owner_tag, owner_id, rest = key.split(":", 2)
        field_name: Optional[str] = None
        head, dot, tail = rest.rpartition(".")
        if dot and tail in ("MU", "SIGMA", "MIN", "MAX"):
            rest, field_name = head, tail
        operation = ""
        if "@" in rest:
            rest, operation = rest.split("@", 1)
        return cls(owner_tag, owner_id, rest, operation, field_name)  # type: ignore[arg-type]


@dataclass
class DetectedParam:
    """One numeric value found in a template (output of template detection)."""

    ref: ParamRef
    xml_value: float
    label: str  # short human/CSV name, e.g. GLUCOSE_UPTAKE_RATE_MU; unique per template
    raw_value: str  # the full original ``value`` attribute, e.g. "NORMAL(MU=1.12,SIGMA=0.0896)"


@dataclass
class SimulationSettings:
    """Settings on the template's ``<series>`` element, shared by every particle.

    ``seeds`` is ``end - start + 1``; the writer keeps ``start`` and sets
    ``end = start + seeds - 1``. ``ticks`` and ``interval`` are in ARCADE
    ticks (1 tick = 1 minute, so 1440 ticks = 1 day).
    """

    seeds: int
    ticks: int
    interval: int
    radius: int
    margin: int
    seed_start: int = 0


@dataclass
class ParameterSpec:
    """A perturbed parameter: uniform prior on [min, max]."""

    key: str  # ParamRef.key
    label: str
    xml_value: float
    min: float
    max: float

    @property
    def ref(self) -> ParamRef:
        return ParamRef.from_key(self.key)


@dataclass
class MetricSpec:
    """A target metric.

    ``name`` is a column of the per-particle metric table: a built-in name from
    ``metrics_plugin.builtin_metrics()`` or a custom function name, optionally
    with the ``_std`` suffix (SD across seeds). ``source`` says which.
    """

    name: str
    source: Literal["builtin", "custom"]
    target: float


@dataclass
class Hyperparameters:
    n_particles: int = 1024
    n_generations: int = 5
    n_trees: int = 50
    min_samples_leaf: int = 5
    subsample_ratio: float = 0.5
    random_state: int = 42
    n_cpu: int = 8
    rf_type: Literal["DRF", "RF"] = "DRF"
    criterion: str = "CART"


@dataclass
class RunPaths:
    """Where the runner writes ARCADE inputs and outputs.

    ``input_dir`` receives ``iter_<g>/inputs/input_<i>.xml``; ``output_dir``
    receives ARCADE outputs ``iter_<g>/inputs/input_<i>/...`` plus per-generation
    ``final_metrics.csv`` and ``all_param_df.csv`` (same layout as
    ``inverse_design.io.ArcadeRunLayout``).
    """

    input_dir: str = "arcade_inputs"
    output_dir: str = "arcade_outputs"


@dataclass
class RunSpec:
    template_path: str  # usually TEMPLATE_FILENAME, a copy inside the run dir
    jar_path: str
    simulation: SimulationSettings
    parameters: list[ParameterSpec]
    metrics: list[MetricSpec]
    hyperparameters: Hyperparameters = field(default_factory=Hyperparameters)
    paths: RunPaths = field(default_factory=RunPaths)
    custom_metrics_file: Optional[str] = None  # usually CUSTOM_METRICS_FILENAME
    model: Literal["arcade"] = "arcade"
    version: int = SPEC_VERSION
    # Directory holding run_spec.json; set by load_spec, not serialized.
    base_dir: Optional[Path] = field(default=None, compare=False, repr=False)

    def resolve(self, path: Optional[str]) -> Optional[Path]:
        """Resolve a spec path against ``base_dir`` (absolute paths pass through)."""
        if path is None:
            return None
        p = Path(path).expanduser()
        if p.is_absolute() or self.base_dir is None:
            return p
        return (self.base_dir / p).resolve()

    def validate(self) -> list[str]:
        """Return human-readable problems; empty list means the spec is usable."""
        problems: list[str] = []
        if not self.parameters:
            problems.append("Select at least one parameter to perturb.")
        if not self.metrics:
            problems.append("Select at least one target metric.")
        for p in self.parameters:
            if not p.min < p.max:
                problems.append(f"{p.label}: min ({p.min}) must be below max ({p.max}).")
        if any(m.source == "custom" for m in self.metrics) and not self.custom_metrics_file:
            problems.append("Custom metrics are selected but no custom metrics file is set.")
        sim = self.simulation
        if sim.seeds < 1:
            problems.append("Seeds per particle must be at least 1.")
        if sim.interval <= 0 or sim.ticks <= 0 or sim.ticks % sim.interval != 0:
            problems.append("Duration must be a positive multiple of the snapshot interval.")
        if sim.ticks // max(sim.interval, 1) < 2:
            problems.append("Duration must cover at least two snapshot intervals.")
        hp = self.hyperparameters
        if hp.n_particles < 2:
            problems.append("Number of particles must be at least 2.")
        if hp.n_generations < 1:
            problems.append("Number of generations must be at least 1.")
        if not 0 < hp.subsample_ratio <= 1:
            problems.append("Subsample ratio must be in (0, 1].")
        return problems

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("base_dir", None)
        return d

    @classmethod
    def from_dict(cls, d: dict, base_dir: Optional[Path] = None) -> "RunSpec":
        d = dict(d)
        if d.get("version", SPEC_VERSION) != SPEC_VERSION:
            raise ValueError(f"Unsupported run_spec version {d.get('version')}")
        return cls(
            template_path=d["template_path"],
            jar_path=d["jar_path"],
            simulation=SimulationSettings(**d["simulation"]),
            parameters=[ParameterSpec(**p) for p in d["parameters"]],
            metrics=[MetricSpec(**m) for m in d["metrics"]],
            hyperparameters=Hyperparameters(**d.get("hyperparameters", {})),
            paths=RunPaths(**d.get("paths", {})),
            custom_metrics_file=d.get("custom_metrics_file"),
            model=d.get("model", "arcade"),
            version=d.get("version", SPEC_VERSION),
            base_dir=base_dir,
        )


def save_spec(spec: RunSpec, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2) + "\n")


def load_spec(path: Path) -> RunSpec:
    path = Path(path).resolve()
    return RunSpec.from_dict(json.loads(path.read_text()), base_dir=path.parent)
