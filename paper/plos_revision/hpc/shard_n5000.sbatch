#!/bin/bash
#SBATCH --job-name=ded_shard
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/cheme/chiu/logs/shard_%A_%a.out
#SBATCH --error=/gscratch/cheme/chiu/logs/shard_%A_%a.err

# One shard of one generation. Submitted as an array by prepare_n5000.sbatch,
# which passes the generation through DED_GEN_FILE.
#
# 32 cores and 128G per task deliberately match the single-node run that is
# known to work; scaling comes from running several of these on different
# nodes, never from widening one. A wider task previously failed outright and
# the cause was never established, so that configuration is left alone.
#
# Each task owns a disjoint slice of input indices, so tasks never write to the
# same path. A task that dies can be resubmitted on its own:
#   sbatch --array=<k> --export=ALL,DED_GEN_FILE=<file>,DED_N_SHARDS=5 shard_n5000.sbatch
# It picks up exactly the simulations still missing from its slice.

set -uo pipefail

ROOT=/gscratch/cheme/chiu
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
JAR=/app/src/inverse_design/models/arcade-logging-necrotic.jar

: "${DED_GEN_FILE:?DED_GEN_FILE not set (submit via prepare_n5000.sbatch)}"
: "${DED_N_SHARDS:=5}"
SHARD=${SLURM_ARRAY_TASK_ID:-1}

if [[ ! -f "$DED_GEN_FILE" ]]; then
  echo "ERROR: generation file $DED_GEN_FILE missing" >&2
  exit 1
fi

INPUT_DIR=$(python3 -c "import json;print(json.load(open('$DED_GEN_FILE'))['input_dir'])")
OUTPUT_DIR=$(python3 -c "import json;print(json.load(open('$DED_GEN_FILE'))['output_dir'])")
NPART=$(python3 -c "import json;print(json.load(open('$DED_GEN_FILE'))['n_particles'])")
ITER=$(python3 -c "import json;print(json.load(open('$DED_GEN_FILE'))['iteration'])")

echo "job=${SLURM_ARRAY_JOB_ID:-none} task=$SHARD/$DED_N_SHARDS host=$(hostname) start=$(date -Is)"
echo "iter=$ITER input=$INPUT_DIR output=$OUTPUT_DIR n_particles=$NPART"

export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# run_simulations logs to $DED_LOG_DIR/simulation_run.log with a plain
# FileHandler. Shards run concurrently, so they need separate directories or
# they interleave into one unreadable file.
SHARD_LOG_DIR=$ROOT/logs/sim_logs/iter_${ITER}_shard_${SHARD}
mkdir -p "$SHARD_LOG_DIR"

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  --env JAVA_OPTS="-Xmx4g" \
  --env DED_LOG_DIR="$SHARD_LOG_DIR" \
  "$SIF" \
  python3 rf/shard_runner.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --jar-path "$JAR" \
    --shard-index "$SHARD" \
    --n-shards "$DED_N_SHARDS" \
    --n-particles "$NPART" \
    --workers 32
status=$?

echo "exit=$status end=$(date -Is)"
exit $status
