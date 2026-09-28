"""Persist the SMC bookkeeping that answers "is this peak real?".

Two quantities decide whether a mode in the posterior is structure or an
artifact of resampling, and neither survives the run as written:

  * the per-generation weights, from which the effective sample size follows.
    A generation of 5000 particles whose ESS is 50 is not 5000 independent
    samples, and any density estimated from it is 50 samples plus kernel
    smoothing.
  * the resampling lineage. Each particle after the prior draw is a perturbed
    copy of a parent chosen from the previous generation. If every particle in
    a mode descends from a handful of shared ancestors, the mode is one lucky
    draw spread out by the perturbation kernel rather than a region the
    posterior independently favours. The parent index is drawn, used, and
    discarded, so this is unrecoverable after the fact.

Both are written per generation and keyed by input index, so downstream
analysis can join them to the particles without re-deriving anything.

Lineage is written once, when a generation's inputs are actually generated.
Replaying the driver re-runs the sampling loop but throws its draws away
(the inputs already exist), so a replay must never be allowed to overwrite
the lineage that the real draw produced.
"""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import csv
import json
import logging
import re

import numpy as np

_INPUT_RE = re.compile(r"input_(\d+)")

LINEAGE_FILENAME = "lineage.csv"
WEIGHTS_FILENAME = "weights.csv"
ESS_FILENAME = "ess.json"


def particle_ids_from_folders(folders: Iterable) -> List[int]:
    """Input indices for the rows of a generation's parameter frame.

    ``all_param_df.csv`` carries an ``input_folder`` column that is dropped
    before fitting, but it is the only link between a row and the simulation
    it came from. Rows whose folder cannot be parsed get -1 rather than
    shifting every later row.
    """
    ids = []
    for folder in folders:
        match = _INPUT_RE.search(str(folder))
        ids.append(int(match.group(1)) if match else -1)
    return ids


def effective_sample_size(weights: Sequence[float]) -> float:
    """Kish ESS. Equals len(weights) for uniform weights, 1 for a point mass."""
    array = np.asarray(weights, dtype=float)
    total = array.sum()
    if total <= 0:
        return 0.0
    normalized = array / total
    denominator = np.square(normalized).sum()
    return float(1.0 / denominator) if denominator > 0 else 0.0


def write_weights(
    generation_dir: str, particle_ids: Sequence[int], weights: Sequence[float]
) -> Optional[Path]:
    """Write per-particle weights plus the generation's ESS.

    Unlike the lineage these are reproducible from a replay, so overwriting is
    harmless and keeps them consistent with the forest that produced them.
    """
    directory = Path(generation_dir)
    if not directory.is_dir():
        logging.warning("Cannot write weights: %s does not exist", directory)
        return None

    if len(particle_ids) != len(weights):
        logging.error(
            "Refusing to write weights: %d particle ids against %d weights",
            len(particle_ids),
            len(weights),
        )
        return None

    weights_path = directory / WEIGHTS_FILENAME
    with open(weights_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input_index", "weight"])
        writer.writerows(zip(particle_ids, weights))

    ess = effective_sample_size(weights)
    # Weights arrive as a numpy array, so every emptiness test here has to go
    # through len(): `if weights` raises on any array with more than one
    # element rather than telling you the array is non-empty.
    n_particles = len(weights)
    with open(directory / ESS_FILENAME, "w") as handle:
        json.dump(
            {
                "n_particles": n_particles,
                "ess": ess,
                "ess_fraction": ess / n_particles if n_particles else 0.0,
                "max_weight": float(np.max(weights)) if n_particles else 0.0,
            },
            handle,
            indent=2,
        )

    logging.info(
        "Generation %s: ESS %.1f of %d particles (%.1f%%)",
        directory.name,
        ess,
        len(weights),
        100.0 * ess / len(weights) if len(weights) else 0.0,
    )
    return weights_path


def write_lineage(
    generation_dir: str,
    child_ids: Sequence[int],
    parent_ids: Sequence[int],
    inputs_pre_existed: bool,
) -> Optional[Path]:
    """Record which previous-generation particle each new particle came from.

    Does nothing when a lineage is already on disk, or when the inputs predate
    lineage tracking: in both cases the draws made by this process were
    discarded, and writing them would replace a real lineage with a fictional
    one.
    """
    directory = Path(generation_dir)
    lineage_path = directory / LINEAGE_FILENAME

    if lineage_path.exists():
        logging.info("Lineage already recorded for %s; leaving it alone", directory)
        return lineage_path

    if inputs_pre_existed:
        # The sampling that produced these inputs happened in an earlier
        # process and was not recorded. Anything written now would describe
        # draws that never became simulations.
        logging.warning(
            "No lineage for %s: its inputs were generated before lineage "
            "tracking, and this run's draws were discarded",
            directory,
        )
        return None

    if not directory.is_dir():
        logging.warning("Cannot write lineage: %s does not exist", directory)
        return None

    if len(child_ids) != len(parent_ids):
        logging.error(
            "Refusing to write lineage: %d children against %d parents",
            len(child_ids),
            len(parent_ids),
        )
        return None

    with open(lineage_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input_index", "parent_input_index"])
        writer.writerows(zip(child_ids, parent_ids))

    distinct = len(set(parent_ids))
    logging.info(
        "Lineage for %s: %d particles from %d distinct parents",
        directory.name,
        len(child_ids),
        distinct,
    )
    return lineage_path


def count_input_files(generation_input_dir: str) -> int:
    """How many input XMLs a generation already has."""
    inputs = Path(generation_input_dir) / "inputs"
    if not inputs.is_dir():
        return 0
    return sum(1 for _ in inputs.glob("input_*.xml"))
