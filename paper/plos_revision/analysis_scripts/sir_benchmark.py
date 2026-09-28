"""Matched SIR parameter-estimation benchmark for reviewer R3.1/R4.1.

This benchmark is deliberately separate from the R4.6 parameter-reduction
comparison.  One inference evaluation is one parameter vector evaluated by
three stochastic SIR runs.  DED, NPE, SNPE-C, and matched-budget pyABC receive
5,120 inference evaluations.  The fixed-population pyABC arm instead keeps 512
accepted particles for up to ten populations and reports its actual cost, with
a wall-clock guard.

The script is safe to run one arm at a time.  It writes an atomic result file
for every completed arm and never overwrites an existing result unless
``--overwrite`` is explicit.  SNPE-C checkpoints after each completed round;
pyABC keeps its SQLite history in the arm output directory.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import random
import socket
import sys
import time
from ctypes import c_longlong
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SOURCE_ROOT = Path(os.environ.get("DED_INVERSE_DESIGN", "/home/pohaoc2/UW/bagherilab/inverse_design")) / "src"
DEFAULT_RESULTS = Path(os.environ.get("DED_INVERSE_DESIGN", "/home/pohaoc2/UW/bagherilab/inverse_design")) / "results/SIR/n3_t365_l30"
SETUP = dict(lattice_size=30, n_simulations=3, max_time=365)
PARAMS = ["PI", "PR", "IIF", "ISF"]
STATS = ["peak_I", "time_to_peak", "final_R", "area_I", "growth_rate"]
RANGES = {
    "PI": (0.001, 0.5),
    "PR": (0.0001, 0.1),
    "IIF": (0.01, 0.5),
    "ISF": (0.49, 0.99),
}
DED_BUDGET = 5120
DED_POP = 512
DED_GENS = 10
INFERENCE_SEED = 20260807
PREDICTIVE_SEED_START = 10_000_001
PYABC_DONE = "Done"


class BudgetExhausted(RuntimeError):
    pass


class Budget:
    """Hard inference-evaluation counter used by batched neural methods."""

    def __init__(self, limit, used=0):
        self.limit = int(limit)
        self.used = int(used)
        self.wall = 0.0

    def spend(self, k=1):
        k = int(k)
        if self.used + k > self.limit:
            raise BudgetExhausted(f"{self.used}+{k}>{self.limit}")
        start = self.used
        self.used += k
        return start


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def configure_paths(source_root, results):
    source_root = Path(source_root).resolve()
    results = Path(results).resolve()
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    return source_root, results


def target(results):
    return pd.read_csv(Path(results) / "target_values.csv")[STATS].to_numpy(float).ravel()


def true_params(results):
    return pd.read_csv(Path(results) / "true_params.csv")[PARAMS].to_numpy(float).ravel()


def prior_sample(rng, n):
    """Published prior: log-uniform rates and a uniform constrained fraction pair."""
    out = np.empty((int(n), 4))
    filled = 0
    while filled < len(out):
        m = int((len(out) - filled) * 1.6) + 16
        pi = np.exp(rng.uniform(*np.log(RANGES["PI"]), m))
        pr = np.exp(rng.uniform(*np.log(RANGES["PR"]), m))
        iif = rng.uniform(*RANGES["IIF"], m)
        isf = rng.uniform(*RANGES["ISF"], m)
        vals = np.column_stack([pi, pr, iif, isf])
        vals = vals[(iif + isf) <= 1.0]
        take = min(len(vals), len(out) - filled)
        out[filled : filled + take] = vals[:take]
        filled += take
    return out


def in_prior(theta):
    pi, pr, iif, isf = np.asarray(theta, float)
    return bool(
        RANGES["PI"][0] <= pi <= RANGES["PI"][1]
        and RANGES["PR"][0] <= pr <= RANGES["PR"][1]
        and RANGES["IIF"][0] <= iif <= RANGES["IIF"][1]
        and RANGES["ISF"][0] <= isf <= RANGES["ISF"][1]
        and iif + isf <= 1.0
    )


def prior_pdf(theta):
    if not in_prior(theta):
        return 0.0
    pi, pr, _iif, _isf = np.asarray(theta, float)
    rate_density = 1.0 / (
        pi
        * math.log(RANGES["PI"][1] / RANGES["PI"][0])
        * pr
        * math.log(RANGES["PR"][1] / RANGES["PR"][0])
    )
    a_max = RANGES["IIF"][1] - RANGES["IIF"][0]
    diagonal = 1.0 - RANGES["IIF"][0] - RANGES["ISF"][0]
    fraction_area = diagonal * a_max - 0.5 * a_max * a_max
    return rate_density / fraction_area


def physical_to_z(theta):
    theta = np.asarray(theta, float).copy()
    theta[..., 0] = np.log(theta[..., 0])
    theta[..., 1] = np.log(theta[..., 1])
    return theta


def z_to_physical(z):
    z = np.asarray(z, float).copy()
    z[..., 0] = np.exp(z[..., 0])
    z[..., 1] = np.exp(z[..., 1])
    return z


def _simulate_particle_seeded(theta_and_seed):
    """Seed both RNG libraries actually used by SIR_ABM for one evaluation."""
    from inverse_design.rf.abc_smc_rf_sir import SIR_ABM_Simulator

    theta, seed = theta_and_seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    simulator = SIR_ABM_Simulator(**SETUP)
    try:
        stats, history = simulator.simulate(np.asarray(theta, float))
        return stats, history, True
    except Exception:
        return None, None, False


def _simulate_particle_seeded_compat(theta_and_seed, simulator_params):
    """Signature-compatible seeded helper for the published DED engine."""
    from inverse_design.rf.abc_smc_rf_sir import SIR_ABM_Simulator

    theta, seed = theta_and_seed
    random.seed(int(seed))
    np.random.seed(int(seed))
    simulator = SIR_ABM_Simulator(**simulator_params)
    try:
        stats, history = simulator.simulate(np.asarray(theta, float))
        return stats, history, True
    except Exception:
        return None, None, False


def simulate_many(thetas, budget, workers, seed_start=None):
    """Evaluate a batch and charge exactly one inference evaluation per row."""
    thetas = np.asarray(thetas, float)
    offset = budget.spend(len(thetas))
    if seed_start is None:
        seed_start = offset + 1
    args = [(t, int(seed_start + i)) for i, t in enumerate(thetas)]
    t0 = time.perf_counter()
    with mp.Pool(int(workers)) as pool:
        res = pool.map(_simulate_particle_seeded, args)
    budget.wall += time.perf_counter() - t0
    out = np.full((len(thetas), len(STATS)), np.nan)
    for i, (stats, _hist, ok) in enumerate(res):
        if ok and stats is not None:
            out[i] = np.asarray(stats, float)
    return out


def score(theta_post, weights, workers, tgt, truth, n_pred=128):
    theta_post = np.asarray(theta_post, float)
    invalid = np.array([not in_prior(t) for t in theta_post])
    if invalid.any():
        raise ValueError(f"posterior contains {int(invalid.sum())} out-of-prior draws")
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    width = np.array([RANGES[p][1] - RANGES[p][0] for p in PARAMS])
    post_mean = (theta_post * weights[:, None]).sum(0)
    param_err = float(np.mean(np.abs(post_mean - truth) / width))

    rng = np.random.default_rng(0)
    n_pred = min(int(n_pred), len(theta_post))
    idx = rng.choice(len(theta_post), size=n_pred, p=weights, replace=True)
    pred_budget = Budget(n_pred)
    sims = simulate_many(
        theta_post[idx], pred_budget, workers, seed_start=PREDICTIVE_SEED_START
    )
    good = ~np.isnan(sims).any(1)
    rel = np.abs(sims[good] - tgt) / np.abs(tgt)
    return {
        "param_err": param_err,
        "stat_err": float(np.nanmean(rel)) if good.any() else float("nan"),
        "post_mean": {p: float(v) for p, v in zip(PARAMS, post_mean)},
        "n_post": int(len(theta_post)),
        "n_pred": int(n_pred),
        "n_pred_ok": int(good.sum()),
        "posterior_invalid": 0,
    }


def _strict_pyabc_work(
    simulate_one,
    out_queue,
    n_eval,
    n_acc,
    n,
    max_eval,
    all_accepted,
    sample_factory,
    deadline,
    worker_seed,
):
    """pyABC worker with atomic evaluation-token reservation and a time guard."""
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    sample = sample_factory()
    while True:
        with n_eval.get_lock():
            if n_acc.value >= n:
                break
            if all_accepted and n_eval.value >= n:
                break
            if max_eval is not None and n_eval.value >= max_eval:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            particle_id = n_eval.value
            n_eval.value += 1
        new_sim = simulate_one()
        sample.append(new_sim)
        if new_sim.accepted:
            with n_acc.get_lock():
                n_acc.value += 1
            out_queue.put((particle_id, sample))
            sample = sample_factory()
    out_queue.put(PYABC_DONE)


def make_strict_pyabc_sampler(pyabc, workers, hard_limit=None, deadline=None):
    from pyabc.sampler.multicorebase import get_if_worker_healthy

    class StrictSampler(pyabc.sampler.MulticoreEvalParallelSampler):
        def __init__(self):
            super().__init__(n_procs=int(workers), check_max_eval=True)
            self.hard_limit = None if hard_limit is None else int(hard_limit)
            self.deadline = deadline
            self.total_evaluations = 0
            self.hit_budget = False
            self.hit_deadline = False
            self.call_index = 0

        def sample_until_n_accepted(
            self, n, simulate_one, t, *, max_eval=np.inf, all_accepted=False, ana_vars=None
        ):
            del t, ana_vars
            remaining = None
            if self.hard_limit is not None:
                remaining = max(0, self.hard_limit - self.total_evaluations)
            if np.isfinite(max_eval):
                remaining = int(max_eval) if remaining is None else min(remaining, int(max_eval))

            n_eval = mp.Value(c_longlong, 0)
            n_acc = mp.Value(c_longlong, 0)
            out_queue = mp.Queue()
            common_args = (
                simulate_one,
                out_queue,
                n_eval,
                n_acc,
                int(n),
                remaining,
                bool(all_accepted),
                self._create_empty_sample,
                self.deadline,
            )
            processes = [
                mp.Process(
                    target=_strict_pyabc_work,
                    args=common_args
                    + (INFERENCE_SEED + 100_000 * self.call_index + worker_index,),
                )
                for worker_index in range(self.n_procs)
            ]
            self.call_index += 1
            for proc in processes:
                proc.start()

            id_results = []
            done = 0
            while done < len(processes):
                val = get_if_worker_healthy(processes, out_queue)
                if val == PYABC_DONE:
                    done += 1
                else:
                    id_results.append(val)
            for proc in processes:
                proc.join()

            id_results.sort(key=lambda item: item[0])
            id_results = id_results[: min(len(id_results), int(n))]
            self.nr_evaluations_ = int(n_eval.value)
            self.total_evaluations += self.nr_evaluations_
            if self.hard_limit is not None and self.total_evaluations >= self.hard_limit:
                self.hit_budget = True
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.hit_deadline = True

            sample = self._create_empty_sample()
            for _particle_id, result in id_results:
                sample += result
            if sample.n_accepted < n:
                sample.ok = False
            return sample

    return StrictSampler()


def make_pyabc_prior(pyabc):
    from pyabc.parameters import Parameter
    from pyabc.random_variables import DistributionBase

    class ConstrainedSIRPrior(DistributionBase):
        def rvs(self, *args, **kwargs):
            del args, kwargs
            # Each strict-sampler worker seeds NumPy independently before this call.
            vals = prior_sample(np.random, 1)[0]
            return Parameter(**dict(zip(PARAMS, vals)))

        def pdf(self, x):
            return prior_pdf([x[p] for p in PARAMS])

    return ConstrainedSIRPrior()


def run_pyabc(
    budget,
    workers,
    tgt,
    rng,
    arm_dir,
    mode="budget",
    smoke=False,
    max_hours=3.0,
    resume=False,
):
    import pyabc
    from inverse_design.rf.abc_smc_rf_sir import SIR_ABM_Simulator

    arm_dir = Path(arm_dir)
    pop = 8 if smoke else (DED_POP if mode == "cost" else 128)
    max_pops = 2 if smoke else DED_GENS
    guard_hours = min(max_hours, 0.10) if smoke else max_hours
    deadline = time.monotonic() + guard_hours * 3600 if mode == "cost" else None
    sampler = make_strict_pyabc_sampler(
        pyabc,
        workers,
        hard_limit=budget.limit if mode == "budget" else None,
        deadline=deadline,
    )
    simulator = SIR_ABM_Simulator(**SETUP)

    def model(par):
        theta = np.array([par[p] for p in PARAMS], float)
        if not in_prior(theta):
            return {"s": np.full(len(STATS), np.nan)}
        try:
            stats, _ = simulator.simulate(theta)
        except Exception:
            return {"s": np.full(len(STATS), np.nan)}
        return {"s": np.asarray(stats, float)}

    def distance(a, b):
        x, y = a["s"], b["s"]
        if np.isnan(x).any():
            return np.inf
        return float(np.mean(np.abs(x - y) / np.abs(y)))

    db_path = arm_dir / "pyabc.sqlite"
    db_uri = f"sqlite:///{db_path}"
    if db_path.exists() and not resume:
        raise FileExistsError(f"pyABC database already exists: {db_path}")
    abc = pyabc.ABCSMC(
        model,
        make_pyabc_prior(pyabc),
        distance,
        population_size=pop,
        sampler=sampler,
    )
    if db_path.exists():
        abc.load(db_uri, 1)
        sampler.total_evaluations = int(abc.history.total_nr_simulations)
        populations_remaining = max(0, max_pops - (int(abc.history.max_t) + 1))
        if populations_remaining == 0:
            raise RuntimeError("pyABC database already contains all planned populations")
    else:
        abc.new(db_uri, {"s": tgt})
        populations_remaining = max_pops
    try:
        abc.run(max_nr_populations=populations_remaining)
    finally:
        # Includes calibration and any incomplete final population, unlike
        # History.total_nr_simulations, which records completed populations only.
        budget.used = int(sampler.total_evaluations)
    gens = int(abc.history.max_t) + 1
    if gens <= 0:
        raise RuntimeError("pyABC stopped before completing a population")
    df, weights = abc.history.get_distribution()
    if sampler.hit_budget:
        stop_reason = "simulation_budget"
    elif sampler.hit_deadline:
        stop_reason = "wall_clock_guard"
    elif gens >= max_pops:
        stop_reason = "max_populations"
    else:
        stop_reason = "algorithm_termination"
    meta = {
        "population_size": int(pop),
        "generations_completed": int(gens),
        "stop_reason": stop_reason,
        "history_simulations": int(abc.history.total_nr_simulations),
        "wall_guard_hours": guard_hours if mode == "cost" else None,
        "database": str(db_path),
    }
    return df[PARAMS].to_numpy(float), np.asarray(weights, float), meta


def _make_sbi_prior(torch):
    from torch.distributions import Distribution, Independent, Uniform, constraints

    lo = torch.tensor(
        [math.log(RANGES["PI"][0]), math.log(RANGES["PR"][0]), RANGES["IIF"][0], RANGES["ISF"][0]],
        dtype=torch.float32,
    )
    hi = torch.tensor(
        [math.log(RANGES["PI"][1]), math.log(RANGES["PR"][1]), RANGES["IIF"][1], RANGES["ISF"][1]],
        dtype=torch.float32,
    )

    class ConstrainedZPrior(Distribution):
        arg_constraints = {}
        support = constraints.independent(constraints.real, 1)
        has_rsample = False

        def __init__(self):
            self.base = Independent(Uniform(lo, hi), 1)
            super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([4]), validate_args=False)

        def sample(self, sample_shape=torch.Size()):
            n = int(torch.Size(sample_shape).numel())
            chunks = []
            total = 0
            while total < n:
                draw = self.base.sample((max(32, 2 * (n - total)),))
                draw = draw[(draw[:, 2] + draw[:, 3]) <= 1.0]
                chunks.append(draw)
                total += len(draw)
            return torch.cat(chunks, dim=0)[:n].reshape(*sample_shape, 4)

        def log_prob(self, value):
            base_lp = self.base.log_prob(value)
            valid = (value[..., 2] + value[..., 3]) <= 1.0
            a_max = RANGES["IIF"][1] - RANGES["IIF"][0]
            diagonal = 1.0 - RANGES["IIF"][0] - RANGES["ISF"][0]
            valid_area = diagonal * a_max - 0.5 * a_max * a_max
            box_area = a_max * (RANGES["ISF"][1] - RANGES["ISF"][0])
            normalized_lp = base_lp - math.log(valid_area / box_area)
            return torch.where(valid, normalized_lp, torch.full_like(base_lp, -torch.inf))

    return ConstrainedZPrior()


def _restrict_proposal(torch, posterior):
    from torch.distributions import Distribution, constraints

    lo = torch.tensor(
        [
            math.log(RANGES["PI"][0]),
            math.log(RANGES["PR"][0]),
            RANGES["IIF"][0],
            RANGES["ISF"][0],
        ],
        dtype=torch.float32,
    )
    hi = torch.tensor(
        [
            math.log(RANGES["PI"][1]),
            math.log(RANGES["PR"][1]),
            RANGES["IIF"][1],
            RANGES["ISF"][1],
        ],
        dtype=torch.float32,
    )

    def valid(value):
        return (
            ((value >= lo) & (value <= hi)).all(dim=-1)
            & ((value[..., 2] + value[..., 3]) <= 1.0)
        )

    class SupportRestrictedProposal(Distribution):
        arg_constraints = {}
        support = constraints.independent(constraints.real, 1)
        has_rsample = False

        def __init__(self):
            super().__init__(batch_shape=torch.Size(), event_shape=torch.Size([4]), validate_args=False)

        def sample(self, sample_shape=torch.Size()):
            n = int(torch.Size(sample_shape).numel())
            chunks = []
            total = 0
            while total < n:
                draw = posterior.sample((max(32, 2 * (n - total)),), show_progress_bars=False)
                draw = draw[valid(draw)]
                chunks.append(draw)
                total += len(draw)
            return torch.cat(chunks, dim=0)[:n].reshape(*sample_shape, 4)

        def log_prob(self, value):
            lp = posterior.log_prob(value)
            return torch.where(valid(value), lp, torch.full_like(lp, -torch.inf))

    return SupportRestrictedProposal()


def _sbi_summary_writer(arm_dir, method):
    """Keep TensorBoard diagnostics with the selected benchmark arm."""
    from torch.utils.tensorboard import SummaryWriter

    log_dir = Path(arm_dir) / "tensorboard" / method
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def run_snpe(budget, workers, tgt, rng, arm_dir, smoke=False, resume=False):
    import torch
    from sbi.inference import NPE

    del rng
    torch.manual_seed(INFERENCE_SEED)
    rounds = 2 if smoke else DED_GENS
    if budget.limit % rounds:
        raise ValueError("SNPE budget must be divisible by the number of rounds")
    per_round = budget.limit // rounds
    prior = _make_sbi_prior(torch)
    x_o = torch.tensor(tgt, dtype=torch.float32)
    checkpoint = Path(arm_dir) / "snpe_checkpoint.pkl"
    if checkpoint.exists() and resume:
        import cloudpickle

        with checkpoint.open("rb") as fh:
            state = cloudpickle.load(fh)
        inference = state["inference"]
        sampling_proposal = state["sampling_proposal"]
        training_proposal = state["training_proposal"]
        start_round = int(state["rounds_completed"])
        budget.used = int(state["budget_used"])
        torch.set_rng_state(state["torch_rng_state"])
    else:
        if checkpoint.exists():
            raise FileExistsError(f"SNPE checkpoint already exists: {checkpoint}")
        inference = NPE(
            prior=prior,
            device="cpu",
            summary_writer=_sbi_summary_writer(arm_dir, "snpe"),
        )
        sampling_proposal = prior
        training_proposal = prior
        start_round = 0

    for round_index in range(start_round, rounds):
        z = sampling_proposal.sample((per_round,)).detach().cpu().numpy()
        theta = z_to_physical(z)
        x = simulate_many(theta, budget, workers)
        keep = ~np.isnan(x).any(1)
        inference.append_simulations(
            torch.tensor(z[keep], dtype=torch.float32),
            torch.tensor(x[keep], dtype=torch.float32),
            # Samples are rejection-restricted to the prior support.  The
            # density differs from this native NeuralPosterior by only a
            # round-specific normalization constant, which cancels in the
            # SNPE-C atomic ratios.
            proposal=training_proposal,
        )
        estimator = inference.train(show_train_summary=True)
        posterior = inference.build_posterior(estimator).set_default_x(x_o)
        training_proposal = posterior
        sampling_proposal = _restrict_proposal(torch, posterior)
        import cloudpickle

        tmp = checkpoint.with_suffix(f".tmp.{os.getpid()}")
        with tmp.open("wb") as fh:
            cloudpickle.dump(
                {
                    "rounds_completed": round_index + 1,
                    "budget_used": budget.used,
                    "inference": inference,
                    "sampling_proposal": sampling_proposal,
                    "training_proposal": training_proposal,
                    "torch_rng_state": torch.get_rng_state(),
                },
                fh,
            )
        os.replace(tmp, checkpoint)
        atomic_json(
            Path(arm_dir) / "progress.json",
            {
                "rounds_completed": round_index + 1,
                "rounds_planned": rounds,
                "budget_used": budget.used,
                "budget": budget.limit,
            },
        )

    z_post = sampling_proposal.sample((2048,)).detach().cpu().numpy()
    theta_post = z_to_physical(z_post)
    meta = {
        "population_size": None,
        "generations_completed": int(rounds),
        "stop_reason": "rounds_complete",
        "training_draws_per_round": int(per_round),
        "checkpoint": str(checkpoint),
    }
    return theta_post, np.ones(len(theta_post)), meta


def run_npe(budget, workers, tgt, rng, arm_dir, smoke=False):
    """Single-round NPE using the same transformed, constrained prior as SNPE-C."""
    import cloudpickle
    import torch
    from sbi.inference import NPE

    del rng, smoke
    torch.manual_seed(INFERENCE_SEED)
    prior = _make_sbi_prior(torch)
    z = prior.sample((budget.limit,)).detach().cpu().numpy()
    theta = z_to_physical(z)
    x = simulate_many(theta, budget, workers)
    keep = ~np.isnan(x).any(1)
    inference = NPE(
        prior=prior,
        device="cpu",
        summary_writer=_sbi_summary_writer(arm_dir, "npe"),
    )
    inference.append_simulations(
        torch.tensor(z[keep], dtype=torch.float32),
        torch.tensor(x[keep], dtype=torch.float32),
        proposal=prior,
    )
    estimator = inference.train(show_train_summary=True)
    posterior = inference.build_posterior(estimator).set_default_x(
        torch.tensor(tgt, dtype=torch.float32)
    )
    sampling_proposal = _restrict_proposal(torch, posterior)
    checkpoint = Path(arm_dir) / "npe_checkpoint.pkl"
    tmp = checkpoint.with_suffix(f".tmp.{os.getpid()}")
    with tmp.open("wb") as fh:
        cloudpickle.dump(
            {
                "budget_used": budget.used,
                "inference": inference,
                "sampling_proposal": sampling_proposal,
                "torch_rng_state": torch.get_rng_state(),
            },
            fh,
        )
    os.replace(tmp, checkpoint)
    z_post = sampling_proposal.sample((2048,)).detach().cpu().numpy()
    theta_post = z_to_physical(z_post)
    return theta_post, np.ones(len(theta_post)), {
        "population_size": None,
        "generations_completed": 1,
        "stop_reason": "rounds_complete",
        "training_draws_per_round": int(budget.limit),
        "checkpoint": str(checkpoint),
    }


def run_ded(budget, workers, tgt, rng, arm_dir, smoke=False):
    """Published DED engine with deterministic simulator seeding and isolated output."""
    del rng
    import inverse_design.rf.abc_smc_rf_sir as sir_module

    # The published helper assigned a wrapper RNG that SIR_ABM never consumes.
    # Patch only the module-level worker hook so both Python and NumPy RNGs use
    # the per-particle seeds already generated by the engine.
    sir_module.simulate_particle = _simulate_particle_seeded_compat
    pop = 8 if smoke else DED_POP
    generations = 2 if smoke else DED_GENS
    planned = pop * generations
    if planned > budget.limit:
        raise ValueError(f"DED plan {planned} exceeds budget {budget.limit}")
    engine_dir = Path(arm_dir) / "ded_engine"
    engine = sir_module.ABCSMCRF_SIR(
        n_particles=pop,
        n_iterations=generations,
        param_ranges=RANGES,
        n_processes=int(workers),
        output_dir=str(engine_dir),
        random_state=INFERENCE_SEED,
    )
    engine.fit(
        target_values=tgt,
        simulator=sir_module.SIR_ABM_Simulator(**SETUP),
    )
    budget.used = planned
    theta_post = np.asarray(engine.parameter_samples[-1], float)
    weights = np.asarray(engine.weights[-1], float)
    return theta_post, weights, {
        "population_size": int(pop),
        "generations_completed": int(generations),
        "stop_reason": "generations_complete",
        "engine_output": str(engine_dir),
    }


METHODS = {"ded", "npe", "snpe", "pyabc_budget", "pyabc_cost"}


def environment_record(source_root, results):
    import scipy

    record = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "source_root": str(source_root),
        "results": str(results),
        "target_sha256": sha256(Path(results) / "target_values.csv"),
        "true_params_sha256": sha256(Path(results) / "true_params.csv"),
        "simulator_sha256": sha256(
            Path(source_root) / "inverse_design/rf/abc_smc_rf_sir.py"
        ),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "benchmark_script_sha256": sha256(__file__),
    }
    for module_name in ("torch", "sbi", "pyabc"):
        try:
            module = __import__(module_name)
            record[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            record[module_name] = None
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    record["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return record


def run_one(name, args, tgt, truth, env):
    arm_dir = Path(args.out) / name
    result_path = arm_dir / ("smoke_result.json" if args.smoke else "result.json")
    if result_path.exists() and not args.overwrite:
        existing = json.loads(result_path.read_text())
        resumable = args.resume and existing.get("status") == "failed"
        if not resumable:
            raise FileExistsError(f"refusing to overwrite {result_path}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    budget_value = args.smoke_budget if args.smoke else args.budget
    budget = Budget(budget_value)
    rng = np.random.default_rng(INFERENCE_SEED)
    started = dt.datetime.now(dt.timezone.utc)
    t0 = time.perf_counter()
    atomic_json(
        arm_dir / "run_manifest.json",
        {
            "method": name,
            "status": "running",
            "smoke": args.smoke,
            "budget": budget_value,
            "workers": args.workers,
            "started_utc": started.isoformat(),
            "setup": SETUP,
            "ranges": RANGES,
            "constraint": "IIF + ISF <= 1",
            "rate_prior": "log-uniform",
            "environment": env,
        },
    )
    try:
        if name == "ded":
            theta, weights, meta = run_ded(
                budget, args.workers, tgt, rng, arm_dir, smoke=args.smoke
            )
        elif name == "npe":
            theta, weights, meta = run_npe(
                budget, args.workers, tgt, rng, arm_dir, smoke=args.smoke
            )
        elif name == "snpe":
            theta, weights, meta = run_snpe(
                budget,
                args.workers,
                tgt,
                rng,
                arm_dir,
                smoke=args.smoke,
                resume=args.resume,
            )
        elif name == "pyabc_budget":
            theta, weights, meta = run_pyabc(
                budget,
                args.workers,
                tgt,
                rng,
                arm_dir,
                mode="budget",
                smoke=args.smoke,
                resume=args.resume,
            )
        elif name == "pyabc_cost":
            theta, weights, meta = run_pyabc(
                budget,
                args.workers,
                tgt,
                rng,
                arm_dir,
                mode="cost",
                smoke=args.smoke,
                max_hours=args.pyabc_cost_hours,
                resume=args.resume,
            )
        else:
            raise KeyError(name)
        record = {
            "method": name,
            "status": "complete",
            "smoke": args.smoke,
            "budget": int(budget_value),
            "used": int(budget.used),
            "wall_s": round(time.perf_counter() - t0, 3),
            "simulation_batch_wall_s": round(budget.wall, 3),
            **meta,
            **score(theta, weights, args.workers, tgt, truth, n_pred=args.predictive_draws),
            "environment": env,
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    except Exception as exc:
        record = {
            "method": name,
            "status": "failed",
            "smoke": args.smoke,
            "budget": int(budget_value),
            "used": int(budget.used),
            "wall_s": round(time.perf_counter() - t0, 3),
            "error": repr(exc),
            "environment": env,
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        atomic_json(result_path, record)
        raise
    atomic_json(result_path, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=DED_BUDGET)
    parser.add_argument("--smoke-budget", type=int, default=64)
    parser.add_argument("--methods", default="ded,npe,snpe,pyabc_budget,pyabc_cost")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--predictive-draws", type=int, default=128)
    parser.add_argument("--pyabc-cost-hours", type=float, default=3.0)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = sorted(set(methods) - METHODS)
    if unknown:
        parser.error(f"unknown methods: {unknown}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    source_root, results = configure_paths(args.source_root, args.results)
    tgt, truth = target(results), true_params(results)
    env = environment_record(source_root, results)
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    for name in methods:
        print(f"--- {name} ---", flush=True)
        try:
            record = run_one(name, args, tgt, truth, env)
        except Exception as exc:
            print(f"FAILED {name}: {exc!r}", flush=True)
            raise
        records.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "environment"}), flush=True)
    atomic_json(args.out / ("smoke_summary.json" if args.smoke else "summary.json"), records)


if __name__ == "__main__":
    main()
