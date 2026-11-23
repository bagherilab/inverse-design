"""Tests for inverse_design.gui.metrics_plugin and metric_helpers.

The fixture is a synthetic ARCADE particle folder with the real CELLS/LOCATIONS
JSON schema (copied from an ARCADE v2 output): three seeds, three snapshots
(0, 0.5, 1 day). Seeds 0 and 1 grow hexagonal rings of radius 0, 1, 2; seed 2
stops at radius 1. Colony diameter for radius r is 30 * (2r + 1) um.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from inverse_design.gui import metric_helpers as mh
from inverse_design.gui import metrics_plugin as mp

TIMESTAMPS = ["000000", "000720", "001440"]
RADII = {0: [0, 1, 2], 1: [0, 1, 2], 2: [0, 1, 1]}
STATES = ["PROLIFERATIVE", "QUIESCENT", "MIGRATORY", "NECROTIC"]


def _hex_positions(radius: int) -> list[tuple[int, int, int]]:
    return [
        (u, v, -u - v)
        for u in range(-radius, radius + 1)
        for v in range(-radius, radius + 1)
        if max(abs(u), abs(v), abs(u + v)) <= radius
    ]


def _write_snapshot(folder: Path, seed: int, ts: str, radius: int) -> None:
    positions = _hex_positions(radius)
    cells, locations = [], []
    for i, (u, v, w) in enumerate(positions, start=1):
        cells.append(
            {
                "id": i,
                "parent": 0,
                "pop": 1,
                "age": 100 * i,
                "divisions": 50,
                "state": STATES[i % len(STATES)],
                "volume": 2000.0 + i,
                "height": 8.7,
                "criticals": [2117.64, 13.05],
                "cycles": [1200 + 60 * seed] if i % 2 else [],
            }
        )
        locations.append({"coordinate": [u, v, w, 0], "ids": [i]})
    stem = f"gui_test_{seed:04d}_{ts}"
    (folder / f"{stem}.CELLS.json").write_text(json.dumps(cells))
    (folder / f"{stem}.LOCATIONS.json").write_text(json.dumps(locations))


@pytest.fixture
def particle(tmp_path: Path) -> Path:
    folder = tmp_path / "input_0"
    folder.mkdir()
    for seed, radii in RADII.items():
        for ts, radius in zip(TIMESTAMPS, radii):
            _write_snapshot(folder, seed, ts, radius)
    (folder / "gui_test.json").write_text("{}")  # ARCADE also writes a summary file
    return folder


@pytest.fixture
def template(tmp_path: Path) -> Path:
    path = tmp_path / "metrics" / "custom_metrics.py"
    mp.write_custom_metrics_template(path)
    return path


def _write(tmp_path: Path, source: str, name: str = "user_metrics.py") -> Path:
    path = tmp_path / name
    path.write_text(source)
    return path


# --- SeedView ---------------------------------------------------------------------------


def test_seed_views_discover_seeds_and_timestamps(particle: Path) -> None:
    seeds = mp.iter_seed_views(particle)
    assert [s.seed for s in seeds] == [0, 1, 2]
    assert all(s.timestamps == TIMESTAMPS for s in seeds)
    assert seeds[0].last_timestamp == "001440"
    assert seeds[0].days() == [0.0, 0.5, 1.0]
    assert seeds[0].days(["001440"]) == [1.0]


def test_seed_view_loads_and_caches(particle: Path) -> None:
    seed = mp.iter_seed_views(particle)[0]
    cells = seed.cells("001440")
    assert len(cells) == 19 and {"id", "state", "volume", "cycles"} <= set(cells[0])
    assert seed.cells("001440") is cells
    locations = seed.locations("000720")
    assert len(locations) == 7 and seed.locations("000720") is locations
    with pytest.raises(KeyError, match="no snapshot"):
        seed.cells("999999")


def test_seed_view_skips_incomplete_snapshots(particle: Path) -> None:
    (particle / "gui_test_0000_002160.CELLS.json").write_text("[]")  # no LOCATIONS file
    seed = mp.iter_seed_views(particle)[0]
    assert seed.timestamps == TIMESTAMPS


def test_iter_seed_views_rejects_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        mp.iter_seed_views(tmp_path / "nope")
    assert mp.iter_seed_views(tmp_path) == []


def test_iter_seed_views_rejects_mixed_simulations(particle: Path) -> None:
    (particle / "other_run_0000_000000.CELLS.json").write_text("[]")
    with pytest.raises(ValueError, match="2 different simulations"):
        mp.iter_seed_views(particle)


# --- helpers ----------------------------------------------------------------------------


def test_helpers() -> None:
    locations = [{"coordinate": [u, v, w, 0], "ids": [1]} for u, v, w in _hex_positions(2)]
    assert mh.colony_diameter_um(locations) == pytest.approx(150.0)
    assert mh.colony_diameter_um([]) == 0.0
    assert mh.slope([0, 1, 2], [1, 3, 5]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        mh.slope([1], [1])
    with pytest.raises(ValueError):
        mh.slope([1, 1], [1, 2])
    cells = [{"state": "NECROTIC"}, {"state": "PROLIFERATIVE"}, {"state": "MIGRATORY"}]
    assert mh.n_cells(cells) == 3
    assert mh.count_state(cells, "necrotic") == 1
    assert mh.count_state(cells, ("PROLIFERATIVE", "MIGRATORY")) == 2
    with pytest.raises(ValueError, match="Unknown cell state"):
        mh.count_state(cells, "NECROSIS")


# --- built-in list ----------------------------------------------------------------------


def test_builtin_metrics_list() -> None:
    names = [m.name for m in mp.builtin_metrics()]
    assert len(names) == len(set(names))
    assert {"doub_time", "colony_growth", "symmetry", "act_ratio", "n_cells"} <= set(names)
    assert "states" not in names and "colony_growth_r" not in names
    assert all(m.description and m.unit for m in mp.builtin_metrics())


# --- loading custom files ---------------------------------------------------------------


def test_template_loads_two_metrics_with_expected_values(particle: Path, template: Path) -> None:
    metrics = mp.load_custom_metrics(template)
    # Imported helpers (colony_diameter_um, slope, ...) are not metrics.
    assert list(metrics) == ["final_diameter_mm", "late_growth_mm_per_day"]
    seeds = mp.iter_seed_views(particle)
    assert [metrics["final_diameter_mm"](s) for s in seeds] == pytest.approx([0.15, 0.15, 0.09])
    late = [metrics["late_growth_mm_per_day"](s) for s in seeds]
    assert late == pytest.approx([0.12, 0.12, 0.0])


def test_load_does_not_leave_module_in_sys_modules(template: Path) -> None:
    before = {n for n in sys.modules if n.startswith(mp._MODULE_PREFIX)}
    mp.load_custom_metrics(template)
    mp.load_custom_metrics(template)
    after = {n for n in sys.modules if n.startswith(mp._MODULE_PREFIX)}
    assert after == before


def test_load_supports_dataclasses_and_skips_non_metrics(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "from dataclasses import dataclass\n"
        "@dataclass\nclass Box:\n    x: int = 1\n"
        "def _private(seed):\n    return 1.0\n"
        "def two_args(a, b):\n    return 1.0\n"
        "def with_default(seed, scale=2.0):\n    return scale * Box().x\n",
    )
    assert list(mp.load_custom_metrics(path)) == ["with_default"]


@pytest.mark.parametrize(
    ("source", "match"),
    [
        ("def doub_time(seed):\n    return 1.0\n", "same name as a built-in"),
        ("def size_std(seed):\n    return 1.0\n", "reserved"),
        ("def f(seed)\n    return 1.0\n", r"Syntax error .*line 1"),
        ("x = 1\n", "No metric functions"),
        ("import not_a_real_module_xyz\n", r"ModuleNotFoundError.*line 1"),
        ("def f(seed):\n    return 1.0\nraise RuntimeError('boom')\n", "RuntimeError: boom"),
    ],
)
def test_load_errors_are_value_errors(tmp_path: Path, source: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        mp.load_custom_metrics(_write(tmp_path, source))


def test_load_rejects_missing_and_non_python_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        mp.load_custom_metrics(tmp_path / "missing.py")
    with pytest.raises(ValueError, match=r"\.py"):
        mp.load_custom_metrics(_write(tmp_path, "def f(seed): return 1\n", "metrics.txt"))


# --- "Test on sample output" -------------------------------------------------------------


def test_test_custom_metrics_reports_values(particle: Path, template: Path) -> None:
    results = {r.name: r for r in mp.test_custom_metrics(particle, template)}
    diam = results["final_diameter_mm"]
    assert diam.per_seed == pytest.approx([0.15, 0.15, 0.09])
    assert diam.errors == []
    assert diam.median == pytest.approx(0.15)
    assert diam.std == pytest.approx(np.std([0.15, 0.15, 0.09]))


def test_test_custom_metrics_captures_errors(particle: Path, tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "def flaky(seed):\n"
        "    if seed.seed == 1:\n"
        "        return 1 / 0\n"
        "    return float(seed.seed)\n"
        "def always_fails(seed):\n"
        "    return seed.cells('123456')\n"
        "def not_finite(seed):\n"
        "    return float('inf') if seed.seed == 0 else 2.0\n"
        "def wrong_type(seed):\n"
        "    return None\n",
    )
    results = {r.name: r for r in mp.test_custom_metrics(particle, path)}

    flaky = results["flaky"]
    assert flaky.per_seed[0] == 0.0 and math.isnan(flaky.per_seed[1]) and flaky.per_seed[2] == 2.0
    assert flaky.median == pytest.approx(1.0) and flaky.std == pytest.approx(1.0)
    assert len(flaky.errors) == 1
    assert "flaky, seed 1" in flaky.errors[0]
    assert "ZeroDivisionError" in flaky.errors[0] and "line 3" in flaky.errors[0]

    fails = results["always_fails"]
    assert all(math.isnan(v) for v in fails.per_seed) and len(fails.errors) == 3
    assert math.isnan(fails.median) and math.isnan(fails.std)
    assert "KeyError" in fails.errors[0]

    assert results["not_finite"].per_seed[1:] == [2.0, 2.0]
    assert "non-finite" in results["not_finite"].errors[0]
    assert "expected a number" in results["wrong_type"].errors[0]


def test_test_custom_metrics_invalid_inputs(particle: Path, template: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        mp.test_custom_metrics(tmp_path / "missing", template)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No ARCADE output"):
        mp.test_custom_metrics(empty, template)
    with pytest.raises(ValueError, match="not found"):
        mp.test_custom_metrics(particle, tmp_path / "missing.py")


# --- compute_particle_metrics -----------------------------------------------------------


def test_compute_builtins_match_existing_pipeline(particle: Path) -> None:
    names = [m.name for m in mp.builtin_metrics()]
    out = mp.compute_particle_metrics(particle, names)
    assert set(out) == {n for name in names for n in (name, name + "_std")}
    assert out["n_cells"] == 19 and out["colony_diameter"] == pytest.approx(150.0)
    assert out["colony_growth"] == pytest.approx(120.0)  # um/day, median over seeds
    expected_doub = 1440 * math.log(2) / math.log(19) / 60
    assert out["doub_time"] == pytest.approx(expected_doub, abs=1e-3)
    assert out["n_cells_std"] == pytest.approx(np.std([19, 19, 7]), abs=1e-3)


def test_compute_builtins_and_customs(particle: Path, template: Path) -> None:
    out = mp.compute_particle_metrics(
        particle, ["doub_time", "symmetry_std"], template, ["final_diameter_mm"]
    )
    assert list(out) == [
        "doub_time",
        "doub_time_std",
        "symmetry",
        "symmetry_std",
        "final_diameter_mm",
        "final_diameter_mm_std",
    ]
    assert out["final_diameter_mm"] == pytest.approx(0.15)
    assert out["final_diameter_mm_std"] == pytest.approx(np.std([0.15, 0.15, 0.09]))
    assert all(math.isfinite(v) for v in out.values())


def test_compute_all_customs_when_names_omitted(particle: Path, template: Path) -> None:
    out = mp.compute_particle_metrics(particle, [], template)
    assert set(out) == {
        "final_diameter_mm",
        "final_diameter_mm_std",
        "late_growth_mm_per_day",
        "late_growth_mm_per_day_std",
    }
    assert out["late_growth_mm_per_day"] == pytest.approx(0.12)


def test_compute_runs_custom_file_once(particle: Path, tmp_path: Path) -> None:
    counter = tmp_path / "count.txt"
    path = _write(
        tmp_path,
        f"with open({str(counter)!r}, 'a') as fh:\n"
        "    fh.write('x')\n"
        "def m(seed):\n"
        "    return 1.0\n",
    )
    mp.compute_particle_metrics(particle, ["n_cells"], path, ["m"])
    assert counter.read_text() == "x"


def test_compute_nan_handling(particle: Path, tmp_path: Path) -> None:
    path = _write(tmp_path, "def bad(seed):\n    raise ValueError('no')\n")
    out = mp.compute_particle_metrics(particle, ["n_cells"], path)
    assert math.isnan(out["bad"]) and math.isnan(out["bad_std"])
    assert out["n_cells"] == 19

    missing = mp.compute_particle_metrics(tmp_path / "crashed", ["n_cells"], path)
    assert set(missing) == {"n_cells", "n_cells_std", "bad", "bad_std"}
    assert all(math.isnan(v) for v in missing.values())


def test_compute_setup_errors(particle: Path, template: Path) -> None:
    with pytest.raises(ValueError, match="Unknown built-in"):
        mp.compute_particle_metrics(particle, ["not_a_metric"])
    with pytest.raises(ValueError, match="not found in"):
        mp.compute_particle_metrics(particle, [], template, ["nope"])
    with pytest.raises(ValueError, match="no custom metrics file"):
        mp.compute_particle_metrics(particle, [], None, ["final_diameter_mm"])
