"""Helpers for running one ABC-SMC generation as several independent SLURM jobs.

The sharded workflow splits a generation's simulations across N jobs that each
own a disjoint slice of input indices, then rejoins at a barrier so the DRF fit
sees the complete generation. Two properties of the existing code make that
possible, and one makes it dangerous:

  * ``run_simulations`` accepts an explicit ``running_index`` list, so slices
    never overlap and never write to the same ``inputs/input_N/`` directory.
  * ``_run_parallel_simulations`` recomputes what is missing from disk on every
    call, so a shard that dies can simply be resubmitted.
  * but "missing" used to mean "the directory does not exist", and ARCADE
    creates that directory *before* it writes anything into it. A job killed
    mid-simulation therefore leaves partial output that the next run counts as
    finished, silently feeding truncated trajectories into the fit.

``sim_is_complete`` closes that hole by checking for the last file ARCADE
writes, derived from the input XML rather than hardcoded.
"""

from pathlib import Path
from typing import Iterable, List, Optional
import logging
import re
import shutil

# <series name="grid" start="0" end="9" ticks="10080" interval="720" ...>
_SERIES_RE = re.compile(r"<series\b[^>]*>")
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"')


def final_output_pattern(input_xml: Path) -> Optional[str]:
    """Glob matching the last file ARCADE writes for ``input_xml``.

    ARCADE walks seeds ``start..end`` and, within each seed, ticks ``0..ticks``
    at ``interval`` steps, writing ``<prefix>_<seed:04d>_<tick:06d>.CELLS.json``
    and ``...LOCATIONS.json``. The final artifact is therefore the last seed at
    the last tick. The prefix is left as a wildcard so callers do not have to
    reconstruct it from the set and series names.

    Returns None when the XML has no parsable series, which makes the caller
    fall back to the old existence check rather than delete real output.
    """
    try:
        text = Path(input_xml).read_text()
    except OSError:
        return None

    match = _SERIES_RE.search(text)
    if not match:
        return None

    attrs = dict(_ATTR_RE.findall(match.group(0)))
    try:
        end_seed = int(attrs["end"])
        ticks = int(attrs["ticks"])
    except (KeyError, ValueError):
        return None

    return f"*_{end_seed:04d}_{ticks:06d}.LOCATIONS.json"


def sim_is_complete(sim_dir: Path, input_xml: Path) -> bool:
    """True when ``sim_dir`` holds a finished simulation for ``input_xml``."""
    sim_dir = Path(sim_dir)
    if not sim_dir.is_dir():
        return False

    pattern = final_output_pattern(input_xml)
    if pattern is None:
        # Unparsable input: fall back to "directory exists", the pre-shard
        # behaviour. Better to re-use possibly-partial output than to delete
        # output we cannot judge.
        return True

    return any(sim_dir.glob(pattern))


def generation_is_analysed(output_dir: str) -> bool:
    """True once a generation has been reduced to metrics.

    final_metrics.csv is what _analyze_simulation_results reads instead of the
    raw output, so its presence means two things at once: the generation's
    results are settled, and its raw simulation output is expendable and may
    already have been deleted to reclaim inodes.
    """
    return (Path(output_dir) / "final_metrics.csv").exists()


def prune_incomplete_outputs(
    output_dir: str, input_dir: str, indices: Optional[Iterable[int]] = None
) -> List[int]:
    """Delete partial simulation directories so they get rerun.

    ``output_dir`` is the generation's output root (the one holding ``inputs/``)
    and ``input_dir`` the matching input root. When ``indices`` is given only
    those inputs are considered, which is what a shard uses to stay inside its
    own slice. Returns the indices that were removed.
    """
    sims_root = Path(output_dir) / "inputs"
    xml_root = Path(input_dir) / "inputs"
    if not sims_root.is_dir():
        return []
    if generation_is_analysed(output_dir):
        # The raw output may have been deleted on purpose to reclaim inodes:
        # the lightweight restore of the published N1024 run leaves 1024 empty
        # input_* directories per generation behind exactly this file. They are
        # "incomplete" by every test here, and deleting them would trigger a
        # re-run of simulations whose results are already summarised.
        logging.info(
            "Skipping completeness check for %s: already analysed", output_dir
        )
        return []
    if not xml_root.is_dir():
        # Without the XMLs every directory looks unjudgeable and nothing gets
        # pruned, which is exactly the silent failure this function exists to
        # prevent. Say so rather than reporting a clean sweep.
        logging.error(
            "Cannot check simulation completeness: no input XMLs at %s. "
            "Partial output under %s will be treated as finished.",
            xml_root,
            sims_root,
        )
        return []

    if indices is None:
        candidates = []
        for sim_dir in sims_root.glob("input_*"):
            match = re.search(r"input_(\d+)$", sim_dir.name)
            if match:
                candidates.append(int(match.group(1)))
    else:
        candidates = list(indices)

    removed = []
    for index in sorted(candidates):
        sim_dir = sims_root / f"input_{index}"
        if not sim_dir.is_dir():
            continue
        if sim_is_complete(sim_dir, xml_root / f"input_{index}.xml"):
            continue
        shutil.rmtree(sim_dir, ignore_errors=True)
        removed.append(index)

    if removed:
        logging.warning(
            "Removed %d partial simulation directories under %s: %s",
            len(removed),
            sims_root,
            _summarize(removed),
        )
    return removed


def _summarize(values: List[int], limit: int = 10) -> str:
    if len(values) <= limit:
        return ", ".join(str(v) for v in values)
    head = ", ".join(str(v) for v in values[:limit])
    return f"{head}, ... (+{len(values) - limit} more)"
