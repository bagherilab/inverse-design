"""Engine behind ``python -m inverse_design.gui.run``.

:class:`GuiABCSMCRF` subclasses the paper's :class:`~inverse_design.rf.abc_smc_rf_arcade.ABCSMCRF`
and replaces only the ARCADE-specific steps: writing input XMLs from a generic template,
running the simulations, and reading metrics/parameters from the GUI's metric plugin.
``fit``, ``_build_rf_model`` (DRF), ``_compute_weights``, ``_remove_invalid_rows``,
``normalize_statistics`` and the weights/ESS provenance are inherited unchanged; the
perturbation kernel is the paper's ``arcade_example.perturbation_kernel`` and generation 0
uses the paper's Sobol helper.

The result of a run is a generation-N particle set with random-forest weights. The sampler
omits the proposal-density correction, so the weights are not calibrated probabilities.
"""

from __future__ import annotations

import concurrent.futures
import functools
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import qmc

from inverse_design.gui import constraints as order, metrics_plugin, xml_template
from inverse_design.gui.spec import MetricSpec, ParameterSpec, RunSpec, SimulationSettings
from inverse_design.io import ArcadeRunLayout
from inverse_design.rf.abc_smc_rf_arcade import ABCSMCRF
from inverse_design.rf.shard_support import (
    generation_is_analysed,
    prune_incomplete_outputs,
    sim_is_complete,
)
from inverse_design.rf.smc_provenance import (
    ESS_FILENAME,
    LINEAGE_FILENAME,
    WEIGHTS_FILENAME,
    particle_ids_from_folders,
    write_lineage,
)
from inverse_design.utils.create_input_files import (
    _sample_sobol_parameters,
    _scale_sobol_values,
)

logger = logging.getLogger(__name__)

PARAMETER_LOG_CSV = "parameter_log.csv"
PARAM_COLUMNS_JSON = "param_columns.json"
FINAL_PARTICLES_CSV = "final_particles.csv"
SPEC_SNAPSHOT_JSON = "run_spec_used.json"
INPUT_FOLDER = "input_folder"
STD_SUFFIX = "_std"

# The paper's kernel branches on config_params["perturbed_config"]; "cellular" is the
# branch for plain continuous parameters, which is what every GUI parameter is.
KERNEL_CONFIG: dict[str, Any] = {"perturbed_config": "cellular", "point_based": None}


class SpecError(ValueError):
    """The spec (or its files) cannot be run; ``problems`` lists every reason."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("\n".join(f"- {p}" for p in self.problems))


class PrepareOnlyStop(Exception):
    """Raised to stop ``fit`` once a generation's inputs are on disk (``--prepare-only``)."""

    def __init__(self, generation: int, input_dir: Path, pending: int):
        self.generation = generation
        self.input_dir = input_dir
        self.pending = pending
        super().__init__(f"iter_{generation}: {pending} simulations pending")


# --------------------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class GuiRunLayout:
    """File locations of a GUI run; per-generation names follow ``ArcadeRunLayout``."""

    input_root: Path
    output_root: Path

    @classmethod
    def from_spec(cls, spec: RunSpec) -> "GuiRunLayout":
        return cls(spec.resolve(spec.paths.input_dir), spec.resolve(spec.paths.output_dir))

    @property
    def _out(self) -> ArcadeRunLayout:
        return ArcadeRunLayout.from_root(self.output_root)

    def iter_in(self, g: int) -> Path:
        return ArcadeRunLayout.from_root(self.input_root).iter_k_dir(g)

    def iter_out(self, g: int) -> Path:
        return self._out.iter_k_dir(g)

    def input_xml(self, g: int, i: int) -> Path:
        return self.iter_in(g) / "inputs" / f"input_{i}.xml"

    def sim_dir(self, g: int, i: int) -> Path:
        return self.iter_out(g) / "inputs" / f"input_{i}"

    def parameter_log(self, g: int) -> Path:
        return self.iter_in(g) / PARAMETER_LOG_CSV

    def lineage(self, g: int) -> Path:
        return self.iter_in(g) / LINEAGE_FILENAME

    def final_metrics(self, g: int) -> Path:
        return self._out.final_metrics_csv(f"iter_{g}")

    def all_param_df(self, g: int) -> Path:
        return self._out.all_param_df_csv(f"iter_{g}")

    def weights(self, g: int) -> Path:
        return self.iter_out(g) / WEIGHTS_FILENAME

    def ess(self, g: int) -> Path:
        return self.iter_out(g) / ESS_FILENAME

    @property
    def targets_json(self) -> Path:
        return self._out.targets_json()

    @property
    def param_columns_json(self) -> Path:
        return self.output_root / PARAM_COLUMNS_JSON

    @property
    def final_particles_csv(self) -> Path:
        return self.output_root / FINAL_PARTICLES_CSV

    @property
    def spec_snapshot(self) -> Path:
        return self.output_root / SPEC_SNAPSHOT_JSON


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def metric_base_name(name: str) -> str:
    """``doub_time_std`` -> ``doub_time``; other names pass through."""
    return name[: -len(STD_SUFFIX)] if name.endswith(STD_SUFFIX) else name


def metric_requests(metrics: Sequence[MetricSpec]) -> tuple[list[str], list[str]]:
    """Unique base names to request from the plugin, split into (builtin, custom)."""
    builtin: list[str] = []
    custom: list[str] = []
    for m in metrics:
        bucket = builtin if m.source == "builtin" else custom
        base = metric_base_name(m.name)
        if base not in bucket:
            bucket.append(base)
    return builtin, custom


def sobol_power_for(n_particles: int) -> int:
    """Smallest power of two holding ``n_particles`` (exact for power-of-two counts)."""
    return max(0, int(np.ceil(np.log2(n_particles))))


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def box_prior_pdf(
    params: np.ndarray,
    param_columns: Sequence[str],
    param_ranges: Mapping[str, tuple[float, float]],
    config_params: Optional[Mapping[str, Any]] = None,
    *,
    constraints: Sequence[order.OrderConstraint] = (),
) -> float:
    """Uniform prior on the [min, max] box: 1.0 inside, 0.0 outside.

    Every perturbed parameter is range-checked (Step 8 of the sampler), and every
    UNIFORM pair in ``constraints`` must keep MIN <= MAX. Only positivity matters to the
    sampler (``prior_density > 0``), so the constant density is not computed (a product of
    many 1/range factors can underflow to zero).
    """
    columns = list(param_columns)
    for name, (lo, hi) in param_ranges.items():
        value = float(params[columns.index(name)])
        if not lo <= value <= hi:
            return 0.0
    if constraints:
        values = {name: float(params[i]) for i, name in enumerate(columns)}
        if not all(c.holds(values) for c in constraints):
            return 0.0
    return 1.0


def constrained_sobol(
    param_ranges: Mapping[str, tuple[float, float]],
    labels: Sequence[str],
    sobol_power: int,
    seed: int,
    constraints: Sequence[order.OrderConstraint],
    max_doublings: int = 10,
) -> np.ndarray:
    """Generation-0 Sobol draw whose rows all satisfy ``constraints``.

    The first draw is the paper's (``_sample_sobol_parameters``, which honours the
    particle-count override). If some rows break MIN <= MAX, the same scrambled sequence
    (same seed) is extended and the first valid rows, in sequence order, are kept.
    """
    samples = _sample_sobol_parameters(dict(param_ranges), sobol_power, seed)
    values = np.column_stack([samples[label] for label in labels])
    n_rows = len(values)
    mask = order.order_mask(values, labels, constraints)
    if mask.all():
        return values
    bounds = [param_ranges[label] for label in labels]
    for doubling in range(1, max_doublings + 1):
        unit = qmc.Sobol(d=len(labels), scramble=True, seed=seed).random(n=n_rows * 2**doubling)
        values = np.column_stack(
            [_scale_sobol_values(unit[:, j], lo, hi) for j, (lo, hi) in enumerate(bounds)]
        )
        valid = values[order.order_mask(values, labels, constraints)]
        if len(valid) >= n_rows:
            return valid[:n_rows]
    raise RuntimeError(
        f"Only {len(valid)} of {len(values)} Sobol points keep MIN <= MAX for the UNIFORM "
        "parameters; widen or separate their ranges."
    )


def write_csv_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` so a reader never sees a half-written file (files double as markers)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def inputs_complete(layout: GuiRunLayout, g: int, n: int) -> bool:
    """True when generation ``g`` has its parameter log and all ``n`` input XMLs."""
    log = layout.parameter_log(g)
    if not log.exists():
        return False
    try:
        if len(pd.read_csv(log)) != n:
            return False
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return False
    return all(layout.input_xml(g, i).exists() for i in range(1, n + 1))


def find_java() -> Optional[str]:
    """Path of the ``java`` executable on PATH, or None."""
    return shutil.which("java")


# --------------------------------------------------------------------------------------
# Model-specific steps (module-level so tests can replace them)
# --------------------------------------------------------------------------------------


def write_generation_inputs(
    template: Path,
    layout: GuiRunLayout,
    g: int,
    values: np.ndarray,
    parameters: Sequence[ParameterSpec],
    simulation: SimulationSettings,
) -> None:
    """Write ``input_<i>.xml`` for every row of ``values``, then ``parameter_log.csv``.

    The parameter log is written last and atomically: its presence marks the inputs as
    complete, and it is the source of the parameter values used in ``all_param_df.csv``.
    It records the values as written to the XML (``format_arcade_number`` rounding), the
    same way the paper's pipeline reads parameters back from its input files.
    """
    layout.iter_in(g).joinpath("inputs").mkdir(parents=True, exist_ok=True)
    written = np.array(
        [[float(xml_template.format_arcade_number(v)) for v in row] for row in values],
        dtype=float,
    ).reshape(len(values), len(parameters))
    for i, row in enumerate(written, start=1):
        particle = {p.key: float(v) for p, v in zip(parameters, row)}
        xml_template.write_particle_xml(template, layout.input_xml(g, i), particle, simulation)
    log = pd.DataFrame(written, columns=[p.label for p in parameters])
    log.insert(0, "file_name", [f"input_{i}.xml" for i in range(1, len(values) + 1)])
    write_csv_atomic(log, layout.parameter_log(g))
    logger.info("Wrote %d ARCADE inputs to %s", len(values), layout.iter_in(g) / "inputs")


def simulate_particles(
    inputs_dir: Path, output_dir: Path, jar_path: Path, indices: Sequence[int], n_cpu: int
) -> None:
    """Run ARCADE for ``inputs_dir/input_<i>.xml`` (i in ``indices``) via the paper's runner."""
    if find_java() is None:
        raise RuntimeError("Java was not found on PATH; ARCADE needs Java 17 or newer.")
    if not Path(jar_path).is_file():
        raise RuntimeError(f"ARCADE JAR not found: {jar_path}")
    from inverse_design.examples.run_simulations import run_simulations

    # run_simulations logs to $DED_LOG_DIR/simulation_run.log (default: the cwd).
    restore = "DED_LOG_DIR" not in os.environ
    if restore:
        os.environ["DED_LOG_DIR"] = str(Path(output_dir).parent)
    try:
        run_simulations(
            input_dir=str(inputs_dir),
            output_dir=str(output_dir),
            jar_path=str(jar_path),
            max_workers=n_cpu,
            running_index=list(indices),
        )
    finally:
        if restore:
            os.environ.pop("DED_LOG_DIR", None)


_MetricJob = tuple[Path, Path, list[str], Optional[Path], list[str]]


def _metrics_for_folder(job: _MetricJob) -> dict:
    folder, input_xml, builtin, custom_file, custom = job
    # Only finished simulations: a crashed run leaves a truncated trajectory, and the
    # metric plugin caches folder listings per process.
    if not sim_is_complete(folder, input_xml):
        if folder.is_dir():
            logger.warning("Simulation %s did not finish; treating it as invalid", folder.name)
        return {}
    try:
        return dict(
            metrics_plugin.compute_particle_metrics(
                folder, builtin, custom_file if custom else None, custom
            )
        )
    except ValueError as exc:  # setup error (unknown metric, bad custom file): stop the run
        raise RuntimeError(f"Metric setup error: {exc}") from exc
    except Exception as exc:  # one broken particle must not abort a generation
        logger.warning("Metrics failed for %s: %s", folder.name, exc)
        return {}


def compute_metrics_table(
    folders: Sequence[Path],
    input_xmls: Sequence[Path],
    builtin: list[str],
    custom_file: Optional[Path],
    custom: list[str],
    columns: Sequence[str],
    n_cpu: int,
) -> pd.DataFrame:
    """One row per folder (NaN where a simulation failed), columns sorted + ``input_folder``."""
    jobs = [(Path(f), Path(x), builtin, custom_file, custom) for f, x in zip(folders, input_xmls)]
    if n_cpu > 1 and len(jobs) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_cpu, len(jobs))) as pool:
            rows = list(pool.map(_metrics_for_folder, jobs))
    else:
        rows = [_metrics_for_folder(job) for job in jobs]
    df = pd.DataFrame(rows, index=range(len(jobs)), dtype=float)
    missing = [c for c in columns if c not in df.columns]
    if missing and any(rows):
        logger.warning("Metric plugin returned no column(s) %s; treating as NaN", missing)
    for c in missing:
        df[c] = np.nan
    df = df[sorted(df.columns)]
    df[INPUT_FOLDER] = [Path(f).name for f in folders]
    return df


def validate_against_plugins(spec: RunSpec) -> list[str]:
    """Check parameter keys against the template and metric names against the plugins.

    Checks whose plugin function is not implemented yet are skipped.
    """
    problems: list[str] = []
    template = spec.resolve(spec.template_path)
    if template is not None and template.is_file():
        try:
            keys = {d.ref.key for d in xml_template.detect_parameters(template)}
            for p in spec.parameters:
                if p.key not in keys:
                    problems.append(f"{p.label}: key {p.key!r} is not in the template.")
        except NotImplementedError:
            pass
        except Exception as exc:
            problems.append(f"Cannot read template {template}: {exc}")
    if template is not None and template.is_file():
        try:
            constraints = order.template_order_constraints(spec.parameters, template)
            problems += order.order_problems(spec.parameters, constraints)[0]
        except Exception as exc:  # the template problem is already reported above
            logger.debug("Skipping UNIFORM order check: %s", exc)
    builtin, custom = metric_requests(spec.metrics)
    if builtin:
        try:
            known = {m.name for m in metrics_plugin.builtin_metrics()}
            problems += [f"Unknown built-in metric {b!r}." for b in builtin if b not in known]
        except NotImplementedError:
            pass
    custom_file = spec.resolve(spec.custom_metrics_file)
    if custom and custom_file is not None and custom_file.is_file():
        try:
            known = set(metrics_plugin.load_custom_metrics(custom_file))
            problems += [
                f"Custom metric {c!r} is not defined in {custom_file.name}."
                for c in custom
                if c not in known
            ]
        except NotImplementedError:
            pass
        except ValueError as exc:
            problems.append(f"Custom metrics file: {exc}")
    return problems


# --------------------------------------------------------------------------------------
# The sampler
# --------------------------------------------------------------------------------------


class GuiABCSMCRF(ABCSMCRF):
    """ABCSMCRF with parameters/metrics from a :class:`RunSpec` instead of the paper's lists.

    Overrides only the model-specific steps; see the module docstring.
    """

    def __init__(
        self,
        spec: RunSpec,
        layout: GuiRunLayout,
        *,
        prepare_only: bool = False,
        n_cpu: Optional[int] = None,
    ):
        from inverse_design.rf.arcade_example import perturbation_kernel

        hp = spec.hyperparameters
        param_ranges = {p.label: (float(p.min), float(p.max)) for p in spec.parameters}
        template = spec.resolve(spec.template_path)
        self.order_constraints = order.template_order_constraints(spec.parameters, template)
        super().__init__(
            n_iterations=hp.n_generations,
            sobol_power=sobol_power_for(hp.n_particles),
            n_particles=hp.n_particles,
            rf_type=hp.rf_type,
            n_trees=hp.n_trees,
            min_samples_leaf=hp.min_samples_leaf,
            param_ranges=param_ranges,
            random_state=hp.random_state,
            criterion=hp.criterion,  # type: ignore[arg-type]
            subsample_ratio=hp.subsample_ratio,
            n_cpu=n_cpu or hp.n_cpu,
            perturbation_kernel=perturbation_kernel,
            prior_pdf=functools.partial(box_prior_pdf, constraints=self.order_constraints),
            config_params=dict(KERNEL_CONFIG),
        )
        # The parent filters the paper's source-site names out of param_list; GUI labels
        # are plain columns, so keep every selected parameter.
        self.source_param_list = []
        self.param_list = list(param_ranges)
        self.spec = spec
        self.layout = layout
        self.template = template
        self.custom_file = spec.resolve(spec.custom_metrics_file)
        self.builtin_metrics, self.custom_metrics = metric_requests(spec.metrics)
        self.prepare_only = prepare_only
        self.sobol_seed = hp.random_state

    # -- input writing -----------------------------------------------------------------

    def _write_inputs(self, g: int, values: np.ndarray) -> None:
        write_generation_inputs(
            self.template, self.layout, g, values, self.spec.parameters, self.spec.simulation
        )

    def _first_iteration(
        self, input_dir: str, output_dir: str, jar_path: str, timestamps: list[str]
    ) -> None:
        """Generation 0: Sobol draw over the prior box (same helper as the paper)."""
        g = self.current_iteration
        dir_postfix = f"iter_{g}"
        if generation_is_analysed(str(self.layout.iter_out(g))):
            logger.info("Skipping input generation for %s: already analysed", dir_postfix)
        elif inputs_complete(self.layout, g, self.n_samples):
            logger.info("Inputs for %s already exist; reusing them", dir_postfix)
        else:
            values = constrained_sobol(
                self.param_ranges,
                self.param_list,
                self.sobol_power,
                self.sobol_seed,
                self.order_constraints,
            )
            self._write_inputs(g, values)
        self._run_parallel_simulations(
            str(self.layout.iter_in(g)), str(self.layout.iter_out(g)), jar_path
        )

        # --- From here on identical to ABCSMCRF._first_iteration. ---
        statistics, valid_parameters = self._analyze_simulation_results(
            input_dir, output_dir, dir_postfix, timestamps
        )
        target_stats_array = np.array(
            [statistics[target_name] for target_name in self.target_names]
        ).T
        target_stats_array, valid_parameters, kept = self._remove_invalid_rows(
            target_stats_array, valid_parameters
        )
        self.particle_ids.append([self._pending_particle_ids[i] for i in kept])
        target_stats, self.scaler = self.normalize_statistics(target_stats_array)
        self.target_values = self.scaler.transform(
            np.array(self.target_values).reshape(-1, len(self.target_values))
        ).flatten()
        self.parameter_samples.append(np.array(valid_parameters))
        self.parameter_columns = list(valid_parameters.columns)
        self.statistics.append(target_stats)

    def _subsequent_iteration(
        self, input_dir: str, output_dir: str, jar_path: str, timestamps: list[str]
    ) -> None:
        """Generations >= 1: weighted resampling + paper kernel, rejecting outside the box
        and any UNIFORM pair with MIN > MAX.

        The draw loop is ABCSMCRF._subsequent_iteration's (same RNG calls in the same
        order), minus the paper's source-site special cases; only input writing differs.
        """
        prev_parameters = self.parameter_samples[-1]
        prev_weights = self.weights[-1]

        n_candidates = self.n_samples
        parameters = np.zeros((n_candidates, prev_parameters.shape[1]), dtype=float)
        parent_positions: list[int] = []
        updated_param_ranges = {
            name: bounds
            for name, bounds in self.param_ranges.items()
            if name in self.parameter_columns
        }
        max_attempts = 10_000 * n_candidates
        attempts = 0

        i = 0
        while i < n_candidates:
            idx = self.rng.choice(len(prev_parameters), p=prev_weights)
            theta_star = prev_parameters[idx]
            theta_candidate = self.perturbation_kernel(
                params=theta_star,
                param_columns=self.parameter_columns,
                param_ranges=self.param_ranges,
                iteration=self.current_iteration,
                max_iterations=self.n_iterations,
                config_params=self.config_params,
            )
            prior_density = self.prior_pdf(
                theta_candidate,
                self.parameter_columns,
                param_ranges=updated_param_ranges,
                config_params=self.config_params,
            )
            if prior_density > 0:
                parameters[i] = theta_candidate
                parent_positions.append(int(idx))
                i += 1
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    f"Generation {self.current_iteration}: only {i} of {n_candidates} "
                    f"proposals fell inside the parameter ranges after {attempts} attempts."
                )

        g = self.current_iteration
        dir_postfix = f"iter_{g}"
        iter_in = self.layout.iter_in(g)
        already_analysed = generation_is_analysed(str(self.layout.iter_out(g)))
        inputs_pre_existed = already_analysed or inputs_complete(self.layout, g, n_candidates)
        if already_analysed:
            logger.info("Skipping input generation for %s: already analysed", dir_postfix)
        elif inputs_pre_existed:
            logger.info("Inputs for %s already exist; reusing them", dir_postfix)
        else:
            self._write_inputs(g, parameters)
        prev_ids = self.particle_ids[-1] if self.particle_ids else []
        write_lineage(
            str(iter_in),
            child_ids=list(range(1, n_candidates + 1)),
            parent_ids=[prev_ids[p] if p < len(prev_ids) else -1 for p in parent_positions],
            inputs_pre_existed=inputs_pre_existed,
        )
        self._run_parallel_simulations(str(iter_in), str(self.layout.iter_out(g)), jar_path)

        # --- From here on identical to ABCSMCRF._subsequent_iteration. ---
        statistics, valid_parameters = self._analyze_simulation_results(
            input_dir, output_dir, dir_postfix, timestamps
        )
        target_stats = np.array([statistics[target_name] for target_name in self.target_names]).T
        target_stats, valid_parameters, kept = self._remove_invalid_rows(
            target_stats, valid_parameters
        )
        self.particle_ids.append([self._pending_particle_ids[i] for i in kept])
        target_stats = self.scaler.transform(target_stats)
        if len(valid_parameters) < n_candidates:
            logger.warning(
                "Only %d valid simulations out of %d attempts", len(valid_parameters), n_candidates
            )
        self.parameter_samples.append(np.array(valid_parameters))
        self.statistics.append(target_stats)

    # -- simulations ---------------------------------------------------------------------

    def _run_parallel_simulations(self, input_dir: str, output_dir: str, jar_path: str) -> None:
        """Run the generation's missing simulations (same skip/prune rules as the parent)."""
        g = self.current_iteration
        if generation_is_analysed(output_dir):
            return
        prune_incomplete_outputs(output_dir, input_dir)
        missing = [
            i for i in range(1, self.n_samples + 1) if not self.layout.sim_dir(g, i).is_dir()
        ]
        if not missing:
            logger.info("iter_%d: all %d simulations already exist", g, self.n_samples)
            return
        if self.prepare_only:
            raise PrepareOnlyStop(g, self.layout.iter_in(g), len(missing))
        logger.info(
            "iter_%d: running %d of %d ARCADE simulations with %d workers",
            g,
            len(missing),
            self.n_samples,
            self.n_cpu,
        )
        simulate_particles(
            Path(input_dir) / "inputs", Path(output_dir), Path(jar_path), missing, self.n_cpu
        )

    # -- metrics and parameters ------------------------------------------------------------

    def _analyze_simulation_results(
        self, input_dir: str, output_dir: str, dir_postfix: str, timestamps: list[str]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build (or reuse) ``final_metrics.csv`` and ``all_param_df.csv`` for a generation."""
        g = int(dir_postfix.split("_")[-1])
        final_metrics = self.layout.final_metrics(g)
        all_param = self.layout.all_param_df(g)
        if not final_metrics.exists():
            self._write_generation_tables(g)
        statistics = pd.read_csv(final_metrics)
        valid_parameters = pd.read_csv(all_param)
        if list(statistics[INPUT_FOLDER]) != list(valid_parameters[INPUT_FOLDER]):
            raise RuntimeError(f"{final_metrics} and {all_param} list different particles.")
        self._pending_particle_ids = particle_ids_from_folders(valid_parameters[INPUT_FOLDER])
        statistics = statistics.drop(columns=[INPUT_FOLDER])
        valid_parameters = valid_parameters[self.param_list]
        return statistics, valid_parameters

    def _write_generation_tables(self, g: int) -> None:
        n = self.n_samples
        names = [f"input_{i}" for i in range(1, n + 1)]
        log = pd.read_csv(self.layout.parameter_log(g))
        metrics = compute_metrics_table(
            [self.layout.sim_dir(g, i) for i in range(1, n + 1)],
            [self.layout.input_xml(g, i) for i in range(1, n + 1)],
            self.builtin_metrics,
            self.custom_file,
            self.custom_metrics,
            self.target_names,
            self.n_cpu,
        )
        stats = metrics[list(self.target_names)].to_numpy(dtype=float)
        n_valid = int(np.isfinite(stats).all(axis=1).sum())
        if n_valid == 0:
            raise RuntimeError(
                f"iter_{g}: none of the {n} simulations produced finite target metrics. "
                f"Check {self.layout.output_root / 'simulation_run.log'} and the ARCADE "
                f"outputs under {self.layout.iter_out(g) / 'inputs'}."
            )
        if n_valid < n:
            logger.warning("iter_%d: %d of %d particles have finite metrics", g, n_valid, n)
        params = log[self.param_list].copy()
        params[INPUT_FOLDER] = names
        # all_param_df first: final_metrics.csv marks the generation as analysed.
        write_csv_atomic(params, self.layout.all_param_df(g))
        write_csv_atomic(metrics, self.layout.final_metrics(g))
        logger.info("iter_%d: wrote %s", g, self.layout.final_metrics(g))
