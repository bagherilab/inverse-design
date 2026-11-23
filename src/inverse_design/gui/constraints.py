"""Cross-parameter checks shared by the GUI and the runner.

* ``UNIFORM(MIN=..,MAX=..)`` pairs: when either field is perturbed, every
  particle must keep ``MIN <= MAX``. The unperturbed side keeps its template
  value. The runner rejects proposals that break the order (generation 0
  keeps the first valid points of a longer Sobol sequence), and a spec whose
  ranges can never satisfy it is refused up front.
* ``<series radius margin>``: ARCADE's ``PatchInputBuilder`` adds 1 to an odd
  radius or margin, so the simulated domain would differ from the spec. The
  GUI and the dry run warn about it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from inverse_design.gui.spec import DetectedParam, ParameterSpec, ParamRef, SimulationSettings

_PARTNER = {"MIN": "MAX", "MAX": "MIN"}


@dataclass(frozen=True)
class OrderConstraint:
    """``low <= high`` for one UNIFORM pair; a side is a perturbed label or a fixed value."""

    low_label: Optional[str]
    low_fixed: float
    high_label: Optional[str]
    high_fixed: float
    low_name: str
    high_name: str

    def _value(self, label: Optional[str], fixed: float, values: Mapping[str, float]) -> float:
        return float(values[label]) if label is not None else fixed

    def holds(self, values: Mapping[str, float]) -> bool:
        low = self._value(self.low_label, self.low_fixed, values)
        high = self._value(self.high_label, self.high_fixed, values)
        return low <= high

    def describe(self) -> str:
        low = self.low_label or f"{self.low_name} (fixed at {self.low_fixed:g})"
        high = self.high_label or f"{self.high_name} (fixed at {self.high_fixed:g})"
        return f"{low} <= {high}"


def uniform_order_constraints(
    parameters: Sequence[ParameterSpec], detected: Iterable[DetectedParam]
) -> list[OrderConstraint]:
    """One constraint per UNIFORM pair with at least one perturbed field."""
    by_key = {d.ref.key: d for d in detected}
    perturbed = {p.key: p for p in parameters}
    constraints: list[OrderConstraint] = []
    seen: set[str] = set()
    for p in parameters:
        ref = ParamRef.from_key(p.key)
        if ref.field not in _PARTNER:
            continue
        min_ref = ParamRef(ref.owner_tag, ref.owner_id, ref.param_id, ref.operation, "MIN")
        max_ref = ParamRef(ref.owner_tag, ref.owner_id, ref.param_id, ref.operation, "MAX")
        if min_ref.key in seen:
            continue
        seen.add(min_ref.key)
        sides = []
        for side_ref in (min_ref, max_ref):
            spec = perturbed.get(side_ref.key)
            det = by_key.get(side_ref.key)
            if spec is not None:
                sides.append((spec.label, spec.xml_value, spec.label))
            elif det is not None:
                sides.append((None, det.xml_value, det.label))
            else:  # partner missing from the template: nothing to order against
                sides = []
                break
        if len(sides) == 2:
            (lo_label, lo_fixed, lo_name), (hi_label, hi_fixed, hi_name) = sides
            constraints.append(
                OrderConstraint(lo_label, lo_fixed, hi_label, hi_fixed, lo_name, hi_name)
            )
    return constraints


def template_order_constraints(
    parameters: Sequence[ParameterSpec], template_path: Optional[Path]
) -> list[OrderConstraint]:
    """:func:`uniform_order_constraints` with the partners read from ``template_path``."""
    if not any(ParamRef.from_key(p.key).field in _PARTNER for p in parameters):
        return []
    from inverse_design.gui import xml_template  # local import keeps spec-level imports light

    detected = xml_template.detect_parameters(Path(template_path)) if template_path else []
    return uniform_order_constraints(parameters, detected)


def _bounds(
    label: Optional[str], fixed: float, ranges: Mapping[str, tuple[float, float]]
) -> tuple[float, float]:
    return ranges[label] if label is not None else (fixed, fixed)


def order_problems(
    parameters: Sequence[ParameterSpec], constraints: Sequence[OrderConstraint]
) -> tuple[list[str], list[str]]:
    """(blocking problems, warnings) for the UNIFORM constraints under the chosen ranges.

    Blocking: no point of the ranges satisfies MIN <= MAX. Warning: the ranges
    overlap, so part of the box is rejected and the effective prior is the box
    minus the MIN > MAX corner.
    """
    ranges = {p.label: (float(p.min), float(p.max)) for p in parameters}
    problems: list[str] = []
    warnings: list[str] = []
    for c in constraints:
        low_min, low_max = _bounds(c.low_label, c.low_fixed, ranges)
        high_min, high_max = _bounds(c.high_label, c.high_fixed, ranges)
        if low_min > high_max:
            problems.append(
                f"UNIFORM order {c.describe()} can never hold: the lowest MIN "
                f"({low_min:g}) is above the highest MAX ({high_max:g})."
            )
        elif low_max > high_min:
            warnings.append(
                f"UNIFORM order {c.describe()}: the ranges overlap, so particles with "
                f"MIN > MAX are rejected."
            )
    return problems, warnings


def order_mask(
    values: np.ndarray, labels: Sequence[str], constraints: Sequence[OrderConstraint]
) -> np.ndarray:
    """Boolean mask over the rows of ``values`` (columns = ``labels``) that satisfy all."""
    values = np.asarray(values, dtype=float)
    mask = np.ones(values.shape[0], dtype=bool)
    index = {label: i for i, label in enumerate(labels)}
    for c in constraints:
        low = values[:, index[c.low_label]] if c.low_label is not None else c.low_fixed
        high = values[:, index[c.high_label]] if c.high_label is not None else c.high_fixed
        mask &= np.asarray(low <= high)
    return mask


def simulation_warnings(sim: SimulationSettings) -> list[str]:
    """Settings ARCADE silently changes: odd radius or margin is rounded up by one."""
    warnings = []
    for name in ("radius", "margin"):
        value = getattr(sim, name)
        if value % 2:
            warnings.append(
                f"Domain {name} {value} is odd; ARCADE rounds it up to {value + 1}, "
                f"so simulations use {name} {value + 1}."
            )
    return warnings
