"""Small helpers for user-written custom metric files.

Import them at the top of your custom metrics file, e.g.::

    from inverse_design.gui.metric_helpers import colony_diameter_um, slope

Every helper works on the plain Python data that ``SeedView.cells(t)`` and
``SeedView.locations(t)`` return (the parsed ARCADE ``CELLS.json`` and
``LOCATIONS.json`` lists). Helpers imported into a metrics file are never
mistaken for metrics: only functions defined in the file itself count.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Union

import numpy as np

#: Cell states that ARCADE writes in the ``state`` field of ``CELLS.json``.
CELL_STATES = (
    "UNDEFINED",
    "QUIESCENT",
    "MIGRATORY",
    "PROLIFERATIVE",
    "APOPTOTIC",
    "NECROTIC",
    "SENESCENT",
)

#: Hexagonal lattice spacing used by the built-in colony-diameter metric (um).
HEX_SIZE_UM = 30.0


def colony_diameter_um(
    locations: Sequence[dict[str, Any]], hex_size_um: float = HEX_SIZE_UM
) -> float:
    """Colony diameter in micrometres from one ``LOCATIONS.json`` snapshot.

    Same definition as the built-in ``colony_diameter`` metric: the mean extent
    of occupied hexagonal coordinates along the u, v and w axes, times the hex
    size. Returns 0.0 for an empty colony. Divide by 1000 for millimetres.
    """
    from inverse_design.analyze.population_metrics import PopulationMetrics

    return float(PopulationMetrics.calculate_colony_diameter([], list(locations), C=hex_size_um))


def slope(x: Sequence[float], y: Sequence[float]) -> float:
    """Least-squares slope of ``y`` against ``x`` (units of y per unit of x).

    Raises ValueError for fewer than two points, unequal lengths, or constant x.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.shape != ya.shape or xa.ndim != 1:
        raise ValueError(f"slope needs two 1-D sequences of equal length: {xa.shape}, {ya.shape}")
    if xa.size < 2:
        raise ValueError("slope needs at least two points")
    dx = xa - xa.mean()
    denom = float(np.dot(dx, dx))
    if denom == 0.0:
        raise ValueError("slope is undefined when all x values are equal")
    return float(np.dot(dx, ya - ya.mean()) / denom)


def n_cells(cells: Sequence[dict[str, Any]]) -> int:
    """Number of cells in one ``CELLS.json`` snapshot (all states)."""
    return len(cells)


def count_state(cells: Sequence[dict[str, Any]], state: Union[str, Iterable[str]]) -> int:
    """Number of cells whose ``state`` is ``state`` (or any of several states).

    ``state`` is case-insensitive, e.g. ``"necrotic"`` or
    ``("PROLIFERATIVE", "MIGRATORY")``. Unknown state names raise ValueError so
    typos do not silently count zero cells. Valid names: :data:`CELL_STATES`.
    """
    wanted = {state.upper()} if isinstance(state, str) else {s.upper() for s in state}
    unknown = wanted.difference(CELL_STATES)
    if unknown:
        raise ValueError(
            f"Unknown cell state(s) {sorted(unknown)}; expected one of {list(CELL_STATES)}"
        )
    return sum(1 for cell in cells if cell.get("state") in wanted)
