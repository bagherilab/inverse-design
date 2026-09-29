#!/bin/bash
# Reclaim one benchmark arm's settled generations while the arm is still running.
#
# The R4.6 grid is 21 arms x 5 generations x 512 particles x 301 files =
# 16.2M inodes if every arm keeps everything to the end. There is not that much
# headroom. But an arm never needs its own raw output after the generation has
# been reduced to metrics, so the peak is only what is in flight -- provided
# something deletes the rest, which is what this does.
#
# Usage:
#   prune_bench_arm.sh --run-root DIR --before ITER [--dry-run]
#
# Deletes *.json under iter_<g>/inputs/input_*/ for every g < ITER that holds
# both final_metrics.csv and all_param_df.csv. The input_* shells stay.
#
# WHY THIS IS SAFE, and it is checked in the code rather than assumed:
#
#   * `_analyze_simulation_results` (abc_smc_rf_arcade.py:381) recomputes metrics
#     only `if not path_exists(final_metrics)`. Once that CSV exists the raw
#     output is never read again, by this run or by any replay of it.
#   * `prune_incomplete_outputs` (shard_support.py:104) short-circuits on
#     `generation_is_analysed`, whose docstring says outright that the presence
#     of final_metrics.csv means the raw output "is expendable and may already
#     have been deleted to reclaim inodes". Without that guard an emptied
#     generation would read as incomplete and be re-simulated.
#   * the generation-skip check counts `input_*` directories, not their
#     contents, so leaving the shells keeps the chain's bookkeeping intact.
#
# WHAT IS GIVEN UP: recomputing final_metrics.csv from the ARCADE snapshots if a
# defect is found in the metric computation itself. That has happened before
# (2026-08-04, HANDOVER_0805.md:859). For the benchmark arms the trade is
# accepted -- their purpose is the MAE comparison, they are cheap to re-run at
# 512x5, and the alternative is not running R4.6 at all.

set -uo pipefail

RUN_ROOT=""
BEFORE=""
DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --before)   BEFORE="$2"; shift 2 ;;
    --dry-run)  DRY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$RUN_ROOT" && -n "$BEFORE" ]] || {
  echo "usage: $0 --run-root DIR --before ITER [--dry-run]" >&2; exit 2; }
[[ -d "$RUN_ROOT" ]] || { echo "prune_bench_arm: $RUN_ROOT absent, nothing to do"; exit 0; }
[[ "$BEFORE" =~ ^[0-9]+$ ]] || { echo "prune_bench_arm: --before must be an integer" >&2; exit 2; }

# Refuse to run against anything except the disposable benchmark arms and the
# explicitly migrated N512 pm50 reduced arms.  The N1024 parent and unrelated
# campaign chains remain outside this allowlist.
case "$(basename "$RUN_ROOT")" in
  bench_*|*_bench|*bench*|ABC_SMC_RF_N512_pm50_*_mean_only) ;;
  *) echo "REFUSING: $(basename "$RUN_ROOT") is not an allowed disposable arm" >&2; exit 2 ;;
esac

echo "prune_bench_arm  root=$RUN_ROOT  before=iter_$BEFORE  dry_run=$DRY"
total=0
for it in "$RUN_ROOT"/iter_*; do
  [[ -d "$it" ]] || continue
  g=$(basename "$it"); g=${g#iter_}
  [[ "$g" =~ ^[0-9]+$ ]] || continue
  if (( g >= BEFORE )); then
    echo "  iter_$g  (in flight or pending -- untouched)"
    continue
  fi
  # Both CSVs, not just final_metrics: all_param_df.csv is read on the same line
  # and a generation missing it would be re-derived from the XMLs it no longer has.
  if [[ ! -s "$it/final_metrics.csv" || ! -s "$it/all_param_df.csv" ]]; then
    echo "  iter_$g  NOT REDUCED (missing a CSV) -- untouched"
    continue
  fi
  n=$(find "$it/inputs" -mindepth 2 -type f -name '*.json' -print 2>/dev/null | wc -l)
  if (( n == 0 )); then
    echo "  iter_$g  already empty"
    continue
  fi
  if (( DRY == 1 )); then
    echo "  iter_$g  would delete $n"
  else
    find "$it/inputs" -mindepth 2 -type f -name '*.json' -delete
    left=$(find "$it/inputs" -mindepth 2 -type f -name '*.json' -print 2>/dev/null | wc -l)
    shells=$(find "$it/inputs" -maxdepth 1 -type d -name 'input_*' | wc -l)
    echo "  iter_$g  deleted $((n - left))  remaining=$left  shells=$shells"
  fi
  total=$((total + n))
done
echo "prune_bench_arm  total=$total  end=$(date -Is)"
exit 0
