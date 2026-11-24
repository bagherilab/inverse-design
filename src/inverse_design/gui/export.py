"""Build a :class:`RunSpec` from GUI state and export a self-contained run directory.

Pure Python (no Qt) so it can be tested and scripted. A run directory holds::

    run_dir/
        run_spec.json       # what to run (paths relative to run_dir)
        template.xml        # the ARCADE template with the simulation settings applied
        custom_metrics.py   # copy of the user's custom metrics file (optional)
        run.sh              # python -m inverse_design.gui.run <run_dir>/run_spec.json
        README.txt          # what the files are and how to run them
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

from inverse_design.gui import xml_template
from inverse_design.gui.spec import (
    CUSTOM_METRICS_FILENAME,
    SPEC_FILENAME,
    TEMPLATE_FILENAME,
    Hyperparameters,
    MetricSpec,
    ParameterSpec,
    RunPaths,
    RunSpec,
    SimulationSettings,
    load_spec,
    save_spec,
)

RUN_SCRIPT_FILENAME = "run.sh"
README_FILENAME = "README.txt"


class ExportError(ValueError):
    """Raised when a run directory cannot be written; message is shown to the user."""


class RunDirConflict(ExportError):
    """``run_dir`` already holds a different run_spec.json (retry with overwrite=True)."""


def run_command(run_dir: Path) -> str:
    """Shell command that runs an exported run directory."""
    spec_path = Path(run_dir) / SPEC_FILENAME
    text = str(spec_path)
    if any(ch.isspace() for ch in text):
        text = f'"{text}"'
    return f"python -m inverse_design.gui.run {text}"


def build_spec(
    *,
    template_path: Path | str,
    jar_path: Path | str,
    simulation: SimulationSettings,
    parameters: Iterable[ParameterSpec],
    metrics: Iterable[MetricSpec],
    hyperparameters: Optional[Hyperparameters] = None,
    paths: Optional[RunPaths] = None,
    custom_metrics_file: Optional[Path | str] = None,
) -> RunSpec:
    """Assemble a RunSpec from GUI state (source paths, not yet the run-dir copies)."""
    return RunSpec(
        template_path=str(template_path),
        jar_path=str(jar_path),
        simulation=simulation,
        parameters=list(parameters),
        metrics=list(metrics),
        hyperparameters=hyperparameters or Hyperparameters(),
        paths=paths or RunPaths(),
        custom_metrics_file=str(custom_metrics_file) if custom_metrics_file else None,
    )


def _absolute_jar(jar_path: str) -> str:
    if not jar_path:
        return jar_path
    return str(Path(jar_path).expanduser().resolve())


def _check_target(run_dir: Path, new_spec: RunSpec, overwrite: bool) -> None:
    if run_dir.exists() and not run_dir.is_dir():
        raise ExportError(f"{run_dir} exists and is not a directory.")
    existing = run_dir / SPEC_FILENAME
    if overwrite or not existing.exists():
        return
    try:
        old = load_spec(existing)
    except Exception as exc:  # unreadable spec: treat as a different run
        raise RunDirConflict(
            f"{run_dir} already holds an unreadable {SPEC_FILENAME} ({exc}). "
            "Choose another directory or allow overwriting."
        ) from exc
    if old.to_dict() != new_spec.to_dict():
        raise RunDirConflict(
            f"{run_dir} already holds a different {SPEC_FILENAME}. "
            "Choose another directory or allow overwriting."
        )


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _write_template(source: Path, dest: Path, simulation: SimulationSettings) -> None:
    """Write via a temp file so re-exporting a loaded run (source == dest) is safe."""
    fd, tmp_name = tempfile.mkstemp(suffix=".xml", dir=dest.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        xml_template.write_template_copy(source, tmp, simulation)
        tmp.chmod(0o644)  # mkstemp creates 0600; the run dir is meant to be shared
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def _run_script() -> str:
    return (
        "#!/usr/bin/env bash\n"
        "# Run the ARCADE calibration exported by the inverse_design GUI.\n"
        "# Activate the Python environment that has inverse_design installed first.\n"
        "set -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'python -m inverse_design.gui.run "$HERE/run_spec.json" "$@"\n'
    )


def _readme(spec: RunSpec, run_dir: Path) -> str:
    hp = spec.hyperparameters
    sim = spec.simulation
    custom = (
        f"  {CUSTOM_METRICS_FILENAME:<18} copy of your custom metric functions\n"
        if spec.custom_metrics_file
        else ""
    )
    return (
        "ARCADE calibration run directory (exported by the inverse_design GUI)\n"
        "=====================================================================\n\n"
        "Files\n"
        f"  {SPEC_FILENAME:<18} run specification: parameters and uniform priors, target\n"
        f"  {'':<18} metrics, hyperparameters, paths (relative to this folder)\n"
        f"  {TEMPLATE_FILENAME:<18} ARCADE XML template with the simulation settings applied\n"
        f"{custom}"
        f"  {RUN_SCRIPT_FILENAME:<18} runs the calibration\n\n"
        "Run\n"
        f"  {run_command(run_dir)}\n"
        f"  (or: bash {RUN_SCRIPT_FILENAME})\n\n"
        "The command needs Python with inverse_design installed and Java for ARCADE.\n"
        "On a cluster, copy this folder, edit jar_path in run_spec.json if the ARCADE JAR\n"
        "lives elsewhere there, and call run.sh from your batch script (e.g. a Slurm job\n"
        f"requesting {hp.n_cpu} CPUs).\n\n"
        "Setup\n"
        f"  {len(spec.parameters)} perturbed parameters, {len(spec.metrics)} target metrics "
        "(median or SD across seeds)\n"
        f"  {hp.n_particles} particles x {hp.n_generations} generations, "
        f"{sim.seeds} seeds per particle\n"
        f"  ARCADE inputs  -> {spec.paths.input_dir}/\n"
        f"  ARCADE outputs -> {spec.paths.output_dir}/\n\n"
        "Result\n"
        f"  After {hp.n_generations} generations the run produces the final-generation\n"
        "  particle set with random-forest weights. The sampler omits the proposal-density\n"
        "  correction of ABC-SMC, so these weighted particles are not a calibrated posterior.\n"
    )


def export_run(
    spec: RunSpec,
    run_dir: Path,
    source_template: Path,
    source_custom_file: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Write a self-contained run directory and return the path of its run_spec.json.

    Raises ExportError when the spec is invalid or a source file is missing, and
    RunDirConflict when ``run_dir`` already holds a different run_spec.json and
    ``overwrite`` is False. Re-exporting an identical spec is allowed.
    """
    run_dir = Path(run_dir).expanduser().resolve()
    source_template = Path(source_template).expanduser().resolve()
    custom_src = Path(source_custom_file).expanduser().resolve() if source_custom_file else None

    problems = spec.validate()
    if problems:
        raise ExportError("The run setup is not valid:\n- " + "\n- ".join(problems))
    if not source_template.is_file():
        raise ExportError(f"Template not found: {source_template}")
    uses_custom = any(m.source == "custom" for m in spec.metrics)
    if uses_custom and custom_src is None:
        raise ExportError("Custom metrics are selected but no custom metrics file is set.")
    if custom_src is not None and not custom_src.is_file():
        raise ExportError(f"Custom metrics file not found: {custom_src}")

    out = replace(
        spec,
        template_path=TEMPLATE_FILENAME,
        jar_path=_absolute_jar(spec.jar_path),
        custom_metrics_file=CUSTOM_METRICS_FILENAME if custom_src else None,
        base_dir=None,
    )
    _check_target(run_dir, out, overwrite)

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_template(source_template, run_dir / TEMPLATE_FILENAME, out.simulation)
    custom_dest = run_dir / CUSTOM_METRICS_FILENAME
    if custom_src is not None and not _same_file(custom_src, custom_dest):
        shutil.copyfile(custom_src, custom_dest)

    spec_path = run_dir / SPEC_FILENAME
    save_spec(out, spec_path)
    script = run_dir / RUN_SCRIPT_FILENAME
    script.write_text(_run_script())
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (run_dir / README_FILENAME).write_text(_readme(out, run_dir))
    return spec_path


def load_run(run_spec_path: Path) -> RunSpec:
    """Load a run_spec.json (paths resolve against its folder via ``RunSpec.resolve``)."""
    return load_spec(Path(run_spec_path))
