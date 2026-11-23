"""Read and write ARCADE XML templates generically (no hard-coded parameter list).

Contract used by the GUI (``app.py``) and the runner (``run.py``). Owner: the
"XML template" work block.

Formats follow ARCADE's own parser (``arcade.core.sim.Series`` and
``arcade.core.util.MiniBox``):

* Distribution values must match ``NAME(K=v,K=v)`` where every ``v`` is
  ``-?[0-9.]+``; scientific notation inside a distribution is *not* accepted
  (ARCADE then reads the parameter as NaN). Plain numbers accept ``1.5``,
  ``15`` and ``15E-1`` (upper-case ``E``, no ``+``). The writer therefore
  always emits positional notation.
* Missing ``<series>`` attributes fall back to ARCADE's defaults: ``start=0``,
  ``end=0``, ``ticks=100``, ``interval=50``, ``radius=1`` and ``margin=0``.
  (ARCADE patch models round odd ``radius``/``margin`` up to the next even
  number.)
* When one owner lists the same parameter twice, ARCADE keeps the last one;
  detection reports that value and the writer updates every copy.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterator, Mapping, Optional

from inverse_design.gui.spec import DetectedParam, ParamRef, SimulationSettings

# ARCADE MiniBox.NUMBER_REGEX (plain numbers).
_NUMBER_RE = re.compile(r"^(-?\d*\.\d*)$|^(-?\d+)$|^(-?\d+E-?\d+)$|^(-?\d*\.\d*E-?\d+)$")
# ARCADE Series.DISTRIBUTION_REGEX.
_DISTRIBUTION_RE = re.compile(r"^([A-Z_]+)\(([A-Z]+)=(-?[\d.]+)(?:,([A-Z]+)=(-?[\d.]+))*\)$")
_DIST_FIELD_RE = re.compile(r"([A-Z]+)=(-?[\d.]+)")
_DIST_FIELDS = ("MU", "SIGMA", "MIN", "MAX")

# ARCADE defaults for <series> attributes (core/parameter.xml, patch/parameter.patch.xml).
_SERIES_DEFAULTS = {"start": 0, "end": 0, "ticks": 100, "interval": 50, "radius": 1, "margin": 0}

_FRACTION_HINTS = ("FRACTION", "ACCURACY", "AFFINITY", "PREFERENCE", "RATIO")
_SIGNIFICANT_DIGITS = 10
_PROLOG_RE = re.compile(r"^﻿?(?:\s*(?:<\?.*?\?>|<!--.*?-->|<!DOCTYPE[^>]*>))*\s*", re.DOTALL)


class TemplateError(ValueError):
    """Raised for a template the GUI cannot use; message is shown to the user."""


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


@dataclass
class _Template:
    root: ET.Element
    series: ET.Element
    prolog: str  # text before the root element (XML declaration, comments)
    trailing_newline: bool


@dataclass
class _Occurrence:
    elem: ET.Element
    ref: ParamRef
    value: float


def _load(template_path: Path) -> _Template:
    path = Path(template_path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise TemplateError(f"Cannot read template file {path}: {exc}") from exc
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    try:
        parser.feed(text)
        root = parser.close()
    except ET.ParseError as exc:
        raise TemplateError(f"Template {path.name} is not valid XML: {exc}") from exc
    series = list(root.iter("series"))
    if not series:
        raise TemplateError(
            f"Template {path.name} has no <series> element; an ARCADE input needs exactly one."
        )
    if len(series) > 1:
        raise TemplateError(
            f"Template {path.name} has {len(series)} <series> elements; "
            "the GUI supports templates with exactly one."
        )
    prolog_match = _PROLOG_RE.match(text)
    prolog = prolog_match.group(0).lstrip("﻿") if prolog_match else ""
    return _Template(root, series[0], prolog, text.endswith("\n"))


def _save(template: _Template, out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = template.prolog + ET.tostring(template.root, encoding="unicode")
    if template.trailing_newline and not text.endswith("\n"):
        text += "\n"
    out_path.write_text(text, encoding="utf-8")


def _parse_value(raw: str) -> list[tuple[Optional[str], float]]:
    """Numeric entries of a ``value`` attribute as ``[(field, value), ...]``; ``[]`` if none."""
    if _NUMBER_RE.match(raw):
        try:
            return [(None, float(raw))]
        except ValueError:  # "." or "-." match the ARCADE regex but are not numbers
            return []
    out: list[tuple[Optional[str], float]] = []
    if _DISTRIBUTION_RE.match(raw):
        for name, text in _DIST_FIELD_RE.findall(raw):
            if name in _DIST_FIELDS:
                try:
                    out.append((name, float(text)))
                except ValueError:
                    continue
    return out


def _iter_parameter_elements(root: ET.Element) -> Iterator[tuple[ET.Element, str, str]]:
    """Yield ``(element, owner_tag, owner_id)`` for every ``*.parameter`` with id and value."""

    def walk(elem: ET.Element, owner: tuple[str, str]) -> Iterator[tuple[ET.Element, str, str]]:
        for child in elem:
            if not isinstance(child.tag, str):  # comments / processing instructions
                continue
            if child.tag.endswith(".parameter"):
                if child.get("id") is not None and child.get("value") is not None:
                    yield child, owner[0], owner[1]
            child_owner = (child.tag, child.get("id", "")) if child.get("id") else owner
            yield from walk(child, child_owner)

    yield from walk(root, ("", ""))


def _scan(root: ET.Element) -> list[_Occurrence]:
    """All numeric occurrences in document order (duplicate keys possible)."""
    found: list[_Occurrence] = []
    for elem, owner_tag, owner_id in _iter_parameter_elements(root):
        param_id, operation = elem.get("id", ""), elem.get("operation", "")
        for field, value in _parse_value(elem.get("value", "")):
            ref = ParamRef(owner_tag, owner_id, param_id, operation, field)  # type: ignore
            if ParamRef.from_key(ref.key) != ref:
                # Ids containing ":" / "@" / a ".MU"-like suffix cannot be addressed by key.
                continue
            found.append(_Occurrence(elem, ref, value))
    return found


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #


def _clean(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").upper()


def _label(ref: ParamRef, level: int) -> str:
    """Label of increasing specificity: 0 short, 1 +owner id, 2 +full id path, 3 +owner tag."""
    name = ref.param_id if level >= 2 else ref.param_id.rsplit("/", 1)[-1]
    parts = [name]
    if ref.operation:
        parts.append(ref.operation)
    if ref.field:
        parts.append(ref.field)
    # Layers share parameter names (GLUCOSE, OXYGEN, ...): always name the layer.
    if ref.owner_id and (level >= 1 or ref.owner_tag == "layer"):
        parts.insert(0, ref.owner_id)
    if level >= 3 and ref.owner_tag:
        parts.insert(0, ref.owner_tag)
    label = _clean("_".join(parts)) or "PARAM"
    if label[0].isdigit():  # e.g. population id "8": keep labels identifier-like
        label = _clean(f"{ref.owner_tag or 'P'}_{label}")
    return label


def _unique_labels(refs: list[ParamRef]) -> list[str]:
    """Short labels, made more specific only for the refs whose labels clash."""
    levels = [0] * len(refs)
    while True:
        labels = [_label(ref, lvl) for ref, lvl in zip(refs, levels)]
        groups: dict[str, list[int]] = {}
        for i, label in enumerate(labels):
            groups.setdefault(label, []).append(i)
        clashing = [i for idx in groups.values() if len(idx) > 1 for i in idx]
        if not clashing or all(levels[i] >= 3 for i in clashing):
            break
        for i in clashing:
            levels[i] = min(levels[i] + 1, 3)
    # Ids that differ only in punctuation: number the repeats.
    used: set[str] = set()
    for i, label in enumerate(labels):
        candidate, n = label, 2
        while candidate in used:
            candidate, n = f"{label}_{n}", n + 1
        used.add(candidate)
        labels[i] = candidate
    return labels


# --------------------------------------------------------------------------- #
# Writing helpers
# --------------------------------------------------------------------------- #


def format_arcade_number(value: float, significant: int = _SIGNIFICANT_DIGITS) -> str:
    """Positional (non-scientific) number string that ARCADE accepts everywhere.

    ``2250.0 -> "2250"``, ``8e-05 -> "0.00008"``; rounded to ``significant`` digits.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Cannot write non-finite value {value!r} to an ARCADE template")
    if number == 0:
        return "0"
    text = format(Decimal(f"{number:.{significant}g}"), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _set_field(elem: ET.Element, field: Optional[str], value: float) -> None:
    number = format_arcade_number(value)
    if field is None:
        elem.set("value", number)
        return
    raw = elem.get("value", "")
    new, count = re.subn(rf"(?<=[(,]){field}=-?[\d.]+", f"{field}={number}", raw, count=1)
    if count != 1:  # guarded by the key index; kept as a safety net
        raise KeyError(f"Field {field} not found in value {raw!r}")
    elem.set("value", new)


def _apply_simulation(series: ET.Element, simulation: SimulationSettings) -> None:
    minimums = {"seeds": 1, "ticks": 1, "interval": 1, "radius": 1, "margin": 0, "seed_start": 0}
    ints: dict[str, int] = {}
    for name, minimum in minimums.items():
        value = getattr(simulation, name)
        if isinstance(value, bool) or float(value) != int(float(value)):
            raise ValueError(f"Simulation setting {name} must be an integer, got {value!r}")
        ints[name] = int(float(value))
        if ints[name] < minimum:
            raise ValueError(f"Simulation setting {name} must be >= {minimum}, got {value!r}")
    series.set("start", str(ints["seed_start"]))
    series.set("end", str(ints["seed_start"] + ints["seeds"] - 1))
    for name in ("ticks", "interval", "radius", "margin"):
        series.set(name, str(ints[name]))


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def detect_parameters(template_path: Path) -> list[DetectedParam]:
    """Every numeric value in the template that can be perturbed.

    Scans all elements whose tag ends with ``.parameter`` and that carry ``id``
    and ``value``. A plain number gives one DetectedParam (``field=None``);
    ``NORMAL(MU=a,SIGMA=b)`` gives two (MU, SIGMA); ``UNIFORM(MIN=a,MAX=b)``
    gives two (MIN, MAX). Non-numeric values (e.g. ``"*:2"``, ``"hex"``) are
    skipped. Order follows document order. Labels are unique short names
    (``GLUCOSE_UPTAKE_RATE_MU``; prefixed with the owner id only when needed
    to stay unique; layer parameters always carry the layer id, e.g.
    ``GLUCOSE_CONCENTRATION_GENERATOR``).
    """
    template = _load(template_path)
    by_key: dict[str, _Occurrence] = {}
    for occ in _scan(template.root):
        by_key[occ.ref.key] = occ  # dict keeps first-seen order; ARCADE keeps the last value
    occurrences = list(by_key.values())
    labels = _unique_labels([occ.ref for occ in occurrences])
    return [
        DetectedParam(
            ref=occ.ref, xml_value=occ.value, label=label, raw_value=occ.elem.get("value", "")
        )
        for occ, label in zip(occurrences, labels)
    ]


def read_simulation_settings(template_path: Path) -> SimulationSettings:
    """Seeds/ticks/interval/radius/margin from the (single) ``<series>`` element.

    Raise TemplateError if there is not exactly one ``<series>``. Missing
    attributes take ARCADE's defaults.
    """
    template = _load(template_path)
    values: dict[str, int] = {}
    for name, default in _SERIES_DEFAULTS.items():
        raw = template.series.get(name)
        if raw is None:
            values[name] = default
            continue
        parsed = _parse_value(raw.strip())
        if len(parsed) != 1 or parsed[0][0] is not None:
            raise TemplateError(f'<series> attribute {name}="{raw}" is not a number.')
        values[name] = int(parsed[0][1])  # ARCADE truncates like Double.intValue()
    if values["end"] < values["start"]:
        raise TemplateError(
            f"<series> end ({values['end']}) is smaller than start ({values['start']})."
        )
    return SimulationSettings(
        seeds=values["end"] - values["start"] + 1,
        ticks=values["ticks"],
        interval=values["interval"],
        radius=values["radius"],
        margin=values["margin"],
        seed_start=values["start"],
    )


def write_particle_xml(
    template_path: Path,
    out_path: Path,
    values: Mapping[str, float],
    simulation: SimulationSettings,
) -> None:
    """Write one ARCADE input file.

    ``values`` maps ParamRef.key -> value; only those entries change, every
    other attribute and the rest of the document stay as in the template.
    For distribution fields only the named field changes
    (``NORMAL(MU=2250,SIGMA=200)`` with MU=3000 -> ``NORMAL(MU=3000,SIGMA=200)``).
    ``simulation`` overwrites the ``<series>`` attributes (end = seed_start +
    seeds - 1). Raise KeyError for a key that is not in the template.
    Numbers are written in positional notation with 10 significant digits.
    """
    template = _load(template_path)
    index: dict[str, list[_Occurrence]] = {}
    for occ in _scan(template.root):
        index.setdefault(occ.ref.key, []).append(occ)
    unknown = [key for key in values if key not in index]
    if unknown:
        raise KeyError(f"Parameter(s) not found in template {Path(template_path).name}: {unknown}")
    for key, value in values.items():
        for occ in index[key]:
            _set_field(occ.elem, occ.ref.field, value)
    _apply_simulation(template.series, simulation)
    _save(template, out_path)


def write_template_copy(
    template_path: Path, out_path: Path, simulation: SimulationSettings
) -> None:
    """Copy the template with ``simulation`` applied (the run dir's template.xml)."""
    write_particle_xml(template_path, out_path, {}, simulation)


def default_range(xml_value: float, ref: ParamRef) -> tuple[float, float]:
    """Default uniform prior: [0.5x, 1.5x] of the XML value.

    For a zero value fall back to (0.0, 1.0); for negative values keep
    min < max. Fractions bounded by one stay within [0, 1] when the XML value
    is in [0, 1] and the parameter id suggests a fraction (FRACTION, ACCURACY,
    AFFINITY, PREFERENCE, RATIO) -- clip max to 1.0.
    """
    x = float(xml_value)
    if x == 0:
        return (0.0, 1.0)
    low, high = sorted((0.5 * x, 1.5 * x))
    if 0.0 <= x <= 1.0 and any(hint in ref.param_id.upper() for hint in _FRACTION_HINTS):
        high = min(high, 1.0)
    return (low, high)
