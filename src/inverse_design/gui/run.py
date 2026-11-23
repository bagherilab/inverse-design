"""Run an exported GUI setup: ``python -m inverse_design.gui.run <run_dir>/run_spec.json``.

Runs the paper's random-forest-weighted sequential calibration for ARCADE (DRF weights,
the published perturbation kernel, a Sobol first generation) with the parameters and
metrics chosen in the GUI. The output is a generation-N particle set with random-forest
weights; the sampler omits the proposal-density correction, so the weights are not
calibrated probabilities.

Re-running the same command resumes: finished simulations and finished generations are
skipped. ``--prepare-only`` writes the next generation's ARCADE inputs and exits (run the
simulations elsewhere, then re-run this command); ``--dry-run`` checks the setup and
prints the plan without writing anything.

Layout (paths from the spec, relative to the run directory)::

    <input_dir>/iter_<g>/inputs/input_<i>.xml      ARCADE inputs
    <input_dir>/iter_<g>/parameter_log.csv         parameter values per input
    <input_dir>/iter_<g>/lineage.csv               parent of each particle (g >= 1)
    <output_dir>/iter_<g>/inputs/input_<i>/        ARCADE outputs
    <output_dir>/iter_<g>/all_param_df.csv         parameters (labels) + input_folder
    <output_dir>/iter_<g>/final_metrics.csv        metrics + input_folder
    <output_dir>/iter_<g>/weights.csv, ess.json    random-forest weights and ESS
    <output_dir>/param_columns.json                label -> template key
    <output_dir>/targets.json                      metric -> target
    <output_dir>/final_particles.csv               last generation: labels, weight, metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from inverse_design.gui import _runner_core as core
from inverse_design.gui import constraints as constraints_mod
from inverse_design.gui._runner_core import GuiRunLayout, PrepareOnlyStop, SpecError
from inverse_design.gui.spec import SPEC_FILENAME, RunSpec, load_spec

logger = logging.getLogger("inverse_design.gui.run")

# Snapshot sections that may change between resumed invocations.
_RESUMABLE_HYPERPARAMETERS = ("n_cpu", "n_generations")


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------


def check_spec(spec: RunSpec, *, need_java: bool = True) -> list[str]:
    """Every reason ``spec`` cannot be run (empty list = OK)."""
    problems = list(spec.validate())
    labels = [p.label for p in spec.parameters]
    keys = [p.key for p in spec.parameters]
    names = [m.name for m in spec.metrics]
    for what, items in (("parameter label", labels), ("parameter key", keys), ("metric", names)):
        dupes = sorted({x for x in items if items.count(x) > 1})
        if dupes:
            problems.append(f"Duplicate {what}(s): {', '.join(dupes)}.")
    if core.INPUT_FOLDER in labels:
        problems.append(f"{core.INPUT_FOLDER!r} cannot be used as a parameter label.")
    for m in spec.metrics:
        if not math.isfinite(m.target):
            problems.append(f"{m.name}: target must be a finite number.")
    hp = spec.hyperparameters
    if hp.rf_type != "DRF":
        problems.append(f"rf_type {hp.rf_type!r} is not supported; the ARCADE pipeline uses DRF.")
    if hp.criterion not in ("CART", "MMD"):
        problems.append(f"criterion must be CART or MMD, not {hp.criterion!r}.")
    if hp.n_trees < 1 or hp.min_samples_leaf < 1 or hp.n_cpu < 1:
        problems.append("n_trees, min_samples_leaf and n_cpu must be at least 1.")

    template = spec.resolve(spec.template_path)
    if template is None or not template.is_file():
        problems.append(f"Template not found: {template}")
    jar = spec.resolve(spec.jar_path)
    if jar is None or not jar.is_file():
        problems.append(f"ARCADE JAR not found: {jar}")
    custom = spec.resolve(spec.custom_metrics_file)
    if custom is not None and not custom.is_file():
        problems.append(f"Custom metrics file not found: {custom}")
    if need_java and core.find_java() is None:
        problems.append("Java was not found on PATH; ARCADE needs Java 17 or newer.")
    if not problems:
        problems += core.validate_against_plugins(spec)
    return problems


def _snapshot(spec: RunSpec) -> dict:
    d = spec.to_dict()
    return {
        "simulation": d["simulation"],
        "parameters": d["parameters"],
        "metrics": d["metrics"],
        "hyperparameters": {
            k: v for k, v in d["hyperparameters"].items() if k not in _RESUMABLE_HYPERPARAMETERS
        },
    }


def _check_resume_compatible(spec: RunSpec, layout: GuiRunLayout) -> None:
    """Refuse to resume a run directory that was started with a different setup."""
    path = layout.spec_snapshot
    if not path.exists():
        return
    old = json.loads(path.read_text())
    new = _snapshot(spec)
    changed = [k for k in new if old.get(k) != new[k]]
    if changed:
        raise SpecError(
            [
                f"{layout.output_root} was started with different {', '.join(changed)}. "
                "Use a new output directory, or restore the original settings to resume."
            ]
        )


# --------------------------------------------------------------------------------------
# Plan / progress
# --------------------------------------------------------------------------------------


def _generation_status(layout: GuiRunLayout, g: int, n: int) -> str:
    if layout.final_metrics(g).exists():
        return "analysed"
    if core.inputs_complete(layout, g, n):
        done = sum(layout.sim_dir(g, i).is_dir() for i in range(1, n + 1))
        return f"inputs written, {done}/{n} simulation folders"
    return "not started"


def format_plan(spec: RunSpec) -> str:
    """Human-readable summary of what ``run`` would do."""
    sim, hp = spec.simulation, spec.hyperparameters
    layout = GuiRunLayout.from_spec(spec)
    java = core.find_java()
    lines = [
        "ARCADE calibration plan",
        f"  template      {spec.resolve(spec.template_path)}",
        f"  ARCADE JAR    {spec.resolve(spec.jar_path)}",
        f"  java          {java or 'NOT FOUND'}",
    ]
    if spec.custom_metrics_file:
        lines.append(f"  custom file   {spec.resolve(spec.custom_metrics_file)}")
    lines += [
        f"  inputs        {layout.input_root}",
        f"  outputs       {layout.output_root}",
        "",
        f"Parameters ({len(spec.parameters)}, uniform ranges; others keep template values):",
    ]
    width = max([len(p.label) for p in spec.parameters] + [5])
    for p in spec.parameters:
        lines.append(
            f"  {p.label:<{width}}  [{p.min:.6g}, {p.max:.6g}]  template {p.xml_value:.6g}"
            f"  ({p.key})"
        )
    template = spec.resolve(spec.template_path)
    try:
        constraints = constraints_mod.template_order_constraints(spec.parameters, template)
    except Exception:  # unreadable template: validation reports it
        constraints = []
    if constraints:
        lines.append("UNIFORM order constraints (proposals breaking them are rejected):")
        lines += [f"  {c.describe()}" for c in constraints]
    lines.append(f"Metrics ({len(spec.metrics)}):")
    width = max([len(m.name) for m in spec.metrics] + [5])
    for m in spec.metrics:
        lines.append(f"  {m.name:<{width}}  target {m.target:.6g}  ({m.source})")
    n, gens = hp.n_particles, hp.n_generations
    lines += [
        "Simulation:",
        f"  {sim.seeds} seeds per particle, {sim.ticks / 1440:g} days, snapshots every "
        f"{sim.interval / 60:g} h, radius {sim.radius}, margin {sim.margin}",
        "Sampler (random-forest-weighted sequential calibration):",
        f"  {n} particles x {gens} generations = {n * gens} particles, "
        f"{n * gens * sim.seeds} ARCADE replicates",
        f"  generation 0: Sobol draw{'' if core.is_power_of_two(n) else ' (not a power of 2)'}"
        f"; later: {hp.rf_type} weights ({hp.n_trees} trees, min leaf {hp.min_samples_leaf}, "
        f"{hp.criterion}) + perturbation kernel inside the ranges",
        f"  random_state {hp.random_state}, {hp.n_cpu} CPUs",
    ]
    warnings = constraints_mod.simulation_warnings(sim)
    warnings += constraints_mod.order_problems(spec.parameters, constraints)[1]
    if warnings:
        lines.append("Warnings:")
        lines += [f"  ! {w}" for w in warnings]
    lines += [
        "Progress:",
    ]
    lines += [f"  iter_{g}: {_generation_status(layout, g, n)}" for g in range(gens)]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------------------


def _write_run_metadata(spec: RunSpec, layout: GuiRunLayout) -> None:
    layout.output_root.mkdir(parents=True, exist_ok=True)
    layout.input_root.mkdir(parents=True, exist_ok=True)
    layout.param_columns_json.write_text(
        json.dumps({p.label: p.key for p in spec.parameters}, indent=2) + "\n"
    )
    layout.targets_json.write_text(
        json.dumps({m.name: m.target for m in spec.metrics}, indent=2) + "\n"
    )
    if not layout.spec_snapshot.exists():
        layout.spec_snapshot.write_text(json.dumps(_snapshot(spec), indent=2) + "\n")


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of ``values`` under ``weights`` (NaN values ignored)."""
    mask = np.isfinite(values) & np.isfinite(weights)
    values, weights = values[mask], weights[mask]
    if len(values) == 0 or weights.sum() <= 0:
        return float("nan")
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cum, 0.5 * cum[-1])])


def write_final_particles(model: core.GuiABCSMCRF, spec: RunSpec, layout: GuiRunLayout) -> Path:
    """Write the last generation's particles (labels, weight, target metrics)."""
    g = len(model.parameter_samples) - 1
    labels = [p.label for p in spec.parameters]
    df = pd.DataFrame(np.asarray(model.parameter_samples[-1], dtype=float), columns=labels)
    df.insert(0, core.INPUT_FOLDER, [f"input_{i}" for i in model.particle_ids[-1]])
    df["weight"] = np.asarray(model.weights[-1], dtype=float)
    metric_names = [m.name for m in spec.metrics]
    metrics = pd.read_csv(layout.final_metrics(g))[[core.INPUT_FOLDER, *metric_names]]
    df = df.merge(metrics, on=core.INPUT_FOLDER, how="left")
    df.insert(0, "generation", g)
    core.write_csv_atomic(df, layout.final_particles_csv)
    return layout.final_particles_csv


def format_summary(spec: RunSpec, layout: GuiRunLayout, particles: pd.DataFrame) -> str:
    from inverse_design.rf.smc_provenance import effective_sample_size

    g = int(particles["generation"].iloc[0]) if len(particles) else 0
    w = particles["weight"].to_numpy(dtype=float)
    lines = [
        f"Generation-{g} particle set with random-forest weights: {len(particles)} particles, "
        f"ESS {effective_sample_size(w):.1f}",
        f"  written to {layout.final_particles_csv}",
        "  (The sampler omits the proposal-density correction; the weights are not "
        "calibrated probabilities.)",
        "",
    ]
    width = max([len(m.name) for m in spec.metrics] + [6])
    lines.append(f"  {'metric':<{width}}  {'target':>12}  {'median':>12}  {'wtd median':>12}")
    for m in spec.metrics:
        v = particles[m.name].to_numpy(dtype=float)
        median = float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")
        lines.append(
            f"  {m.name:<{width}}  {m.target:>12.6g}  {median:>12.6g}  "
            f"{weighted_median(v, w):>12.6g}"
        )
    return "\n".join(lines)


def run(spec: RunSpec, *, prepare_only: bool = False, n_cpu: Optional[int] = None) -> Path:
    """Run (or resume) the calibration described by ``spec``.

    Returns ``final_particles.csv``; with ``prepare_only`` returns the input directory of
    the generation whose inputs were just prepared (or ``final_particles.csv`` when every
    generation is already finished). Raises :class:`SpecError` for an unusable spec.
    """
    problems = check_spec(spec, need_java=not prepare_only)
    if problems:
        raise SpecError(problems)
    layout = GuiRunLayout.from_spec(spec)
    _check_resume_compatible(spec, layout)
    _write_run_metadata(spec, layout)

    from inverse_design.rf import arcade_example  # noqa: F401  (seeds np.random on import)
    from inverse_design.utils import create_input_files as cif

    hp = spec.hyperparameters
    model = core.GuiABCSMCRF(spec, layout, prepare_only=prepare_only, n_cpu=n_cpu)
    # The paper's kernel draws from numpy's global RNG; seed it so runs are reproducible.
    np.random.seed(hp.random_state)
    previous_override = cif.get_particle_count()
    # A power-of-two count keeps the paper's random_base2 Sobol path.
    cif.set_particle_count(None if core.is_power_of_two(hp.n_particles) else hp.n_particles)
    try:
        model.fit(
            [m.name for m in spec.metrics],
            [float(m.target) for m in spec.metrics],
            str(layout.input_root),
            str(layout.output_root),
            str(spec.resolve(spec.jar_path)),
            [],
            "gui",
        )
    except PrepareOnlyStop as stop:
        print(
            f"Prepared iter_{stop.generation}: {stop.pending} simulations pending.\n"
            f"  ARCADE inputs: {stop.input_dir / 'inputs'}\n"
            f"  Run them (outputs go to {layout.iter_out(stop.generation) / 'inputs'}), "
            "then re-run this command."
        )
        return stop.input_dir
    finally:
        cif.set_particle_count(previous_override)

    path = write_final_particles(model, spec, layout)
    if prepare_only:
        print("All generations are already finished; nothing to prepare.")
    print(format_summary(spec, layout, pd.read_csv(path)))
    return path


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m inverse_design.gui.run",
        description="Run an ARCADE calibration exported by the GUI (resumes automatically).",
    )
    p.add_argument("spec", type=Path, help=f"path to {SPEC_FILENAME} (or its run directory)")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="check the setup, print the plan, and exit"
    )
    mode.add_argument(
        "--prepare-only",
        action="store_true",
        help="write the next generation's ARCADE inputs and exit without simulating",
    )
    p.add_argument("--n-cpu", type=int, default=None, help="override the spec's n_cpu")
    p.add_argument("-q", "--quiet", action="store_true", help="only log warnings")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    spec_path = args.spec / SPEC_FILENAME if args.spec.is_dir() else args.spec
    try:
        spec = load_spec(spec_path)
    except FileNotFoundError:
        print(f"error: run spec not found: {spec_path}", file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError) as exc:
        print(f"error: cannot read run spec {spec_path}: {exc}", file=sys.stderr)
        return 2
    if args.n_cpu is not None:
        if args.n_cpu < 1:
            print("error: --n-cpu must be at least 1", file=sys.stderr)
            return 2
        spec.hyperparameters.n_cpu = args.n_cpu

    if args.dry_run:
        print(format_plan(spec))
        problems = check_spec(spec, need_java=True)
        if problems:
            print("\nProblems:\n" + "\n".join(f"  - {p}" for p in problems), file=sys.stderr)
            return 2
        print("\nOK: the setup is runnable.")
        return 0

    try:
        run(spec, prepare_only=args.prepare_only)
    except SpecError as exc:
        print(f"error: the run spec cannot be run:\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; re-run the same command to resume.", file=sys.stderr)
        return 130
    except (RuntimeError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
