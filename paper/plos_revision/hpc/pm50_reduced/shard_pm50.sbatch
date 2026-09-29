#!/bin/bash
#SBATCH --job-name=pm50_shard
#SBATCH --account=cheme
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/gscratch/cheme/chiu/bench_pm50_reduced/logs/shard_%A_%a.out
#SBATCH --error=/gscratch/cheme/chiu/bench_pm50_reduced/logs/shard_%A_%a.err

set -uo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN_ROOT=${DED_CAMPAIGN_ROOT:-/gscratch/cheme/chiu/bench_pm50_reduced}
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
JAR=/app/src/inverse_design/models/arcade-logging-necrotic.jar

: "${DED_GEN_FILE:?DED_GEN_FILE not set (submit via prepare_r46_pf.sbatch)}"
: "${DED_N_SHARDS:=5}"
SHARD=${SLURM_ARRAY_TASK_ID:-1}
WORKERS=${DED_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}

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
echo "workers=$WORKERS memory=${SLURM_MEM_PER_NODE:-unknown}MB"

export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

SHARD_LOG_DIR=$CAMPAIGN_ROOT/logs/sim_logs/${DED_CONFIG##*/}/iter_${ITER}_shard_${SHARD}
mkdir -p "$SHARD_LOG_DIR"

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$CAMPAIGN_ROOT/src/arcade_example.py":/app/src/inverse_design/rf/arcade_example.py \
  --bind "$ROOT/corrected/abc_smc_rf_arcade.py":/app/src/inverse_design/rf/abc_smc_rf_arcade.py \
  --bind "$ROOT/pm50/parameter_config.py":/app/src/inverse_design/config/parameter_config.py \
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
    --workers "$WORKERS"
status=$?

echo "exit=$status end=$(date -Is)"
exit $status
