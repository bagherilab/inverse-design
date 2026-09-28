"""Extend published and replicate SIR posterior-predictive draws to I=0.

This is a forward diagnostic, not a new calibration target.  It keeps the
published 365-day fitting window intact and asks when independently simulated
trajectories from the g4 weighted clouds reach the absorbing infection-free
state.  A fixed cap is retained and non-extinction by the cap is reported rather
than silently treated as steady state.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import pandas as pd

from inverse_design.models.sir.sir import SIR_ABM
from inverse_design.rf.drf import DRF


PARAMETERS = ["PI", "PR", "IIF", "ISF"]
STATISTICS = ["peak_I", "time_to_peak", "final_R", "area_I", "growth_rate"]


def normalized_weights(path: Path) -> np.ndarray:
    weights = pd.read_csv(path)["weight"].to_numpy(float)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f"invalid weights: {path}")
    return weights / weights.sum()


def replay_published_weights(run: Path, target: np.ndarray, seed: int) -> np.ndarray:
    params = pd.read_csv(run / "iter_4" / "params.csv").to_numpy(float)
    stats = pd.read_csv(run / "iter_4" / "statistics.csv").to_numpy(float)
    model = DRF(
        n_trees=50,
        min_samples_leaf=5,
        n_try=None,
        random_state=seed,
        criterion="CART",
    )
    model.fit(params, stats)
    weights = np.asarray(model.predict_weights(target), dtype=float)
    return weights / weights.sum()


def trajectory_task(task):
    group, draw, params, seed, maximum_time, output = task
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = SIR_ABM(
        lattice_size=30,
        initial_infected_fraction=float(params[2]),
        initial_susceptible_fraction=float(params[3]),
        PI=float(params[0]),
        PR=float(params[1]),
        Pm=1.0,
    )
    model.run(max_time=maximum_time, record_interval=1.0)
    history = pd.DataFrame(model.history)
    destination = Path(output) / "histories" / group / f"draw_{draw:03d}.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(destination, index=False)
    endpoint = model.get_stats()
    return {
        "group": group,
        "draw": draw,
        "seed": seed,
        "extinct": int(endpoint["infected"] == 0),
        "end_time": float(endpoint["time"]),
        "susceptible_count": int(endpoint["susceptible"]),
        "infected_count": int(endpoint["infected"]),
        "recovered_count": int(endpoint["recovered"]),
        "history": str(destination),
    }


def select_group(name, run, weights, n_draws, rng, seed_base, maximum_time, output):
    params = pd.read_csv(run / "iter_4" / "params.csv")[PARAMETERS].to_numpy(float)
    support = int((weights > 0).sum())
    replace = support < n_draws
    indices = rng.choice(len(params), size=n_draws, replace=replace, p=weights)
    tasks = []
    selections = []
    for draw, index in enumerate(indices):
        seed = seed_base + draw
        tasks.append((name, draw, params[index], seed, maximum_time, str(output)))
        selections.append(
            {
                "group": name,
                "draw": draw,
                "input_index": int(index),
                "particle_weight": float(weights[index]),
                "simulation_seed": seed,
                **dict(zip(PARAMETERS, params[index])),
            }
        )
    return tasks, selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output")
    parser.add_argument("--draws", type=int, default=30)
    parser.add_argument("--max-time", type=float, default=5000.0)
    parser.add_argument("--workers", type=int, default=40)
    args = parser.parse_args()

    campaign = Path(args.campaign).resolve()
    output = Path(args.output).resolve() if args.output else campaign / "extinction"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    target = pd.read_csv(campaign / "published" / "target_values.csv")[STATISTICS].to_numpy(float).ravel()
    truth = pd.read_csv(campaign / "published" / "true_params.csv")[PARAMETERS].to_numpy(float).ravel()
    rng = np.random.default_rng(202608083)

    tasks = []
    selections = []
    published_run = campaign / "published" / "n512"
    published_weights = replay_published_weights(published_run, target, seed=202608080)
    group_tasks, group_selections = select_group(
        "published", published_run, published_weights, args.draws, rng,
        191001, args.max_time, output,
    )
    tasks.extend(group_tasks)
    selections.extend(group_selections)

    for replicate, seed_base in [(1, 192001), (2, 193001)]:
        run = campaign / "replicates" / f"replicate_{replicate}"
        weights = normalized_weights(run / "iter_4" / "weights.csv")
        group_tasks, group_selections = select_group(
            f"replicate_{replicate}", run, weights, args.draws, rng,
            seed_base, args.max_time, output,
        )
        tasks.extend(group_tasks)
        selections.extend(group_selections)

    for draw in range(args.draws):
        seed = 194001 + draw
        tasks.append(("reference", draw, truth, seed, args.max_time, str(output)))
        selections.append(
            {
                "group": "reference",
                "draw": draw,
                "input_index": -1,
                "particle_weight": 1.0 / args.draws,
                "simulation_seed": seed,
                **dict(zip(PARAMETERS, truth)),
            }
        )

    pd.DataFrame(selections).to_csv(output / "selected_parameters.csv", index=False)
    with mp.Pool(processes=args.workers) as pool:
        summaries = list(pool.imap_unordered(trajectory_task, tasks))
    summary = pd.DataFrame(summaries).sort_values(["group", "draw"])
    summary.to_csv(output / "extinction_summary.csv", index=False)

    grouped = []
    for group, frame in summary.groupby("group", sort=True):
        extinct_times = frame.loc[frame["extinct"] == 1, "end_time"]
        grouped.append(
            {
                "group": group,
                "draws": len(frame),
                "extinct_by_cap": int(frame["extinct"].sum()),
                "extinction_fraction": float(frame["extinct"].mean()),
                "median_extinction_time": (
                    float(extinct_times.median()) if len(extinct_times) else None
                ),
                "maximum_time": args.max_time,
            }
        )
    report = {
        "design": "R2.10/R2.12 forward diagnostic after fixed 365-day calibration",
        "calibration_window_days": 365,
        "absorbing_condition": "infected_count == 0",
        "draws_per_group": args.draws,
        "groups": grouped,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
