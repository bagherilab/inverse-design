"""Run one shard of a single ABC-SMC generation's simulations.

A shard owns a contiguous slice of input indices and nothing else: it does not
sample parameters, does not fit a forest, and does not decide which generation
to run. The prepare job lays down the inputs; N shards fill in the outputs; the
next prepare job sees a complete generation and moves on.

Because a shard only ever writes to ``inputs/input_N/`` for N inside its own
slice, shards never contend for the same path and a dead shard can be
resubmitted on its own.

Usage:
    python3 rf/shard_runner.py \
        --input-dir  .../abc_smc_..._n5000/iter_0 \
        --output-dir .../ABC_SMC_..._N5000/iter_0 \
        --jar-path   .../arcade_v3.jar \
        --shard-index 1 --n-shards 5 --n-particles 5000 --workers 32
"""

import argparse
import logging
import sys
from pathlib import Path

from inverse_design.examples.run_simulations import run_simulations
from inverse_design.rf.shard_support import prune_incomplete_outputs, sim_is_complete


def slice_for(shard_index: int, n_shards: int, n_particles: int) -> list[int]:
    """1-based indices this shard owns.

    The remainder is spread over the leading shards rather than dumped on the
    last one, so no shard runs materially longer than its peers and the barrier
    is not held open by a straggler.
    """
    if not 1 <= shard_index <= n_shards:
        raise ValueError(f"shard_index {shard_index} outside 1..{n_shards}")

    base, remainder = divmod(n_particles, n_shards)
    start = (shard_index - 1) * base + min(shard_index - 1, remainder)
    size = base + (1 if shard_index <= remainder else 0)
    return list(range(start + 1, start + size + 1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="generation input root")
    parser.add_argument("--output-dir", required=True, help="generation output root")
    parser.add_argument("--jar-path", required=True)
    parser.add_argument("--shard-index", type=int, required=True, help="1-based")
    parser.add_argument("--n-shards", type=int, required=True)
    parser.add_argument("--n-particles", type=int, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    indices = slice_for(args.shard_index, args.n_shards, args.n_particles)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    xml_root = input_dir / "inputs"
    missing_xml = [i for i in indices if not (xml_root / f"input_{i}.xml").exists()]
    if missing_xml:
        # The prepare job either did not run or did not finish. Failing loudly
        # keeps the dependency chain from advancing on a half-written
        # generation.
        print(
            f"ERROR: {len(missing_xml)} input XMLs missing from {xml_root} "
            f"(first: input_{missing_xml[0]}.xml)",
            file=sys.stderr,
        )
        return 1

    # Clean up after any earlier attempt at this shard before deciding what is
    # left to do; a partial directory otherwise reads as finished.
    prune_incomplete_outputs(str(output_dir), str(input_dir), indices)

    todo = [
        i
        for i in indices
        if not sim_is_complete(
            output_dir / "inputs" / f"input_{i}", xml_root / f"input_{i}.xml"
        )
    ]

    print(
        f"shard {args.shard_index}/{args.n_shards}: owns {len(indices)} inputs "
        f"({indices[0]}..{indices[-1]}), {len(todo)} still to run"
    )
    if not todo:
        print("nothing to do")
        return 0

    run_simulations(
        input_dir=str(xml_root),
        output_dir=str(output_dir),
        jar_path=args.jar_path,
        max_workers=args.workers,
        running_index=todo,
    )

    incomplete = [
        i
        for i in todo
        if not sim_is_complete(
            output_dir / "inputs" / f"input_{i}", xml_root / f"input_{i}.xml"
        )
    ]
    if incomplete:
        # Exiting non-zero holds the afterok dependency, so the next prepare job
        # does not fit a forest on a generation with holes in it.
        print(
            f"ERROR: {len(incomplete)} simulations did not finish "
            f"(first: input_{incomplete[0]})",
            file=sys.stderr,
        )
        return 1

    print(f"shard {args.shard_index}/{args.n_shards}: all {len(indices)} complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
