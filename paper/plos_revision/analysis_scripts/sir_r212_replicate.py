"""Run and preserve independent SIR DED calibrations for reviewer R2.12.

The published worker assigned a seed to ``SIR_ABM_Simulator.rng``, while the
underlying SIR model actually draws from Python's ``random`` module and NumPy's
global RNG.  This isolated driver seeds both RNGs inside every worker, records
the particle seed schedule, and saves parameters, statistics, forest weights,
and histories for every generation.

The scientific comparison remains the published design: a 30x30 lattice, three
stochastic simulations per parameter evaluation, a 365-day observation window,
512 particles, and generations g0--g4.  Targets are read from the staged
published S3 artifact and are never regenerated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from inverse_design.rf.abc_smc_rf_sir import ABCSMCRF_SIR, SIR_ABM_Simulator


PARAMETERS = ["PI", "PR", "IIF", "ISF"]
STATISTICS = ["peak_I", "time_to_peak", "final_R", "area_I", "growth_rate"]
RANGES = {
    "PI": (0.001, 0.5),
    "PR": (0.0001, 0.1),
    "IIF": (0.01, 0.5),
    "ISF": (0.49, 0.99),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seeded_particle(params_and_seed, simulator_params):
    """Run one particle with the supplied seed controlling every RNG in use."""

    params, seed = params_and_seed
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    simulator = SIR_ABM_Simulator(**simulator_params)
    simulator.rng = np.random.RandomState(seed)
    try:
        stats, history = simulator.simulate(np.asarray(params, dtype=float))
        return stats, history, True
    except Exception as exc:  # retain the published driver's failure contract
        logging.exception("SIR particle failed for seed=%d params=%s", seed, params)
        return None, {"error": [repr(exc)]}, False


class RecordedABCSMCRFSIR(ABCSMCRF_SIR):
    """ABCSMCRF_SIR with explicit worker seeding and seed provenance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulation_seeds: list[np.ndarray] = []

    def simulate_particles_parallel(self, parameters, simulator_params):
        parameters = np.asarray(parameters, dtype=float)
        seeds = self.rng.randint(0, 2**31, size=len(parameters)).astype(np.int64)
        inputs = list(zip(parameters, seeds))
        worker = partial(seeded_particle, simulator_params=simulator_params)
        with mp.Pool(processes=self.n_processes) as pool:
            results = list(pool.imap(worker, inputs))

        valid_parameters = []
        valid_statistics = []
        valid_histories = []
        valid_seeds = []
        for params, seed, result in zip(parameters, seeds, results):
            stats, history, success = result
            if success and stats is not None:
                valid_parameters.append(params)
                valid_statistics.append(stats)
                valid_histories.append(history)
                valid_seeds.append(seed)

        if not valid_parameters:
            raise RuntimeError("No valid SIR particles were produced")
        self.simulation_seeds.append(np.asarray(valid_seeds, dtype=np.int64))
        return (
            np.asarray(valid_parameters, dtype=float),
            np.asarray(valid_statistics, dtype=float),
            valid_histories,
        )


def validate_generation(path: Path, expected_particles: int) -> dict:
    params = pd.read_csv(path / "params.csv")
    stats = pd.read_csv(path / "statistics.csv")
    weights = pd.read_csv(path / "weights.csv")["weight"].to_numpy(float)
    seeds = pd.read_csv(path / "simulation_seeds.csv")["seed"].to_numpy(np.int64)
    histories = sorted((path / "history").glob("history_*.csv"))

    sizes = {len(params), len(stats), len(weights), len(seeds), len(histories)}
    if sizes != {expected_particles}:
        raise ValueError(f"{path}: inconsistent output sizes {sorted(sizes)}")
    if list(params.columns) != PARAMETERS:
        raise ValueError(f"{path}: unexpected parameter columns")
    if list(stats.columns) != STATISTICS:
        raise ValueError(f"{path}: unexpected statistic columns")
    if not np.isfinite(params.to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite parameters")
    if not np.isfinite(stats.to_numpy(float)).all():
        raise ValueError(f"{path}: non-finite statistics")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError(f"{path}: invalid weights")
    if not np.isclose(weights.sum(), 1.0, atol=1e-10):
        raise ValueError(f"{path}: weights sum to {weights.sum()}")
    if len(set(seeds.tolist())) != expected_particles:
        raise ValueError(f"{path}: particle seeds are not unique")

    normalized = weights / weights.sum()
    return {
        "particles": expected_particles,
        "support": int((normalized > 0).sum()),
        "ess": float(1.0 / np.square(normalized).sum()),
        "params_sha256": sha256(path / "params.csv"),
        "statistics_sha256": sha256(path / "statistics.csv"),
        "weights_sha256": sha256(path / "weights.csv"),
        "seeds_sha256": sha256(path / "simulation_seeds.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--true-params", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=3)
    parser.add_argument("--max-time", type=float, default=365.0)
    parser.add_argument("--lattice-size", type=int, default=30)
    parser.add_argument("--trees", type=int, default=50)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    target_path = Path(args.target).resolve()
    true_path = Path(args.true_params).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output}")
    output.mkdir(parents=True)

    target = pd.read_csv(target_path)[STATISTICS].to_numpy(float).ravel()
    true_params = pd.read_csv(true_path)[PARAMETERS].to_numpy(float).ravel()
    if target.shape != (5,) or not np.isfinite(target).all():
        raise ValueError("published target must contain five finite statistics")
    if true_params.shape != (4,) or not np.isfinite(true_params).all():
        raise ValueError("published true-parameter file must contain four values")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    simulator = SIR_ABM_Simulator(
        lattice_size=args.lattice_size,
        n_simulations=args.simulations,
        max_time=args.max_time,
        true_params=dict(zip(PARAMETERS, true_params)),
    )
    engine = RecordedABCSMCRFSIR(
        n_particles=args.particles,
        n_iterations=args.generations,
        param_ranges=RANGES,
        n_processes=args.workers,
        output_dir=str(output),
        rf_type="DRF",
        n_trees=args.trees,
        min_samples_leaf=5,
        random_state=args.seed,
    )
    engine.fit(target_values=target, simulator=simulator)

    generation_records = []
    for generation in range(args.generations):
        generation_dir = output / f"iter_{generation}"
        pd.DataFrame(engine.parameter_samples[generation], columns=PARAMETERS).to_csv(
            generation_dir / "params.csv", index=False
        )
        pd.DataFrame(engine.statistics[generation], columns=STATISTICS).to_csv(
            generation_dir / "statistics.csv", index=False
        )
        pd.DataFrame({"weight": engine.weights[generation]}).to_csv(
            generation_dir / "weights.csv", index=False
        )
        pd.DataFrame(
            {
                "input_index": np.arange(len(engine.simulation_seeds[generation])),
                "seed": engine.simulation_seeds[generation],
            }
        ).to_csv(generation_dir / "simulation_seeds.csv", index=False)
        generation_records.append(validate_generation(generation_dir, args.particles))

    manifest = {
        "design": "R2.12 independent SIR DED calibration",
        "calibration_seed": args.seed,
        "particles": args.particles,
        "generations": args.generations,
        "simulations_per_parameter": args.simulations,
        "max_time": args.max_time,
        "lattice_size": args.lattice_size,
        "n_trees": args.trees,
        "min_samples_leaf": 5,
        "workers": args.workers,
        "target_path": str(target_path),
        "target_sha256": sha256(target_path),
        "true_params_path": str(true_path),
        "true_params_sha256": sha256(true_path),
        "worker_rng_fix": "random.seed(seed) and numpy.random.seed(seed) per particle",
        "generations_output": generation_records,
    }
    (output / "complete.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
