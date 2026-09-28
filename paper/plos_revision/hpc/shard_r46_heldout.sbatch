#!/bin/bash
#SBATCH --job-name=r46_heldout_shard
#SBATCH --account=cheme
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/cheme/chiu/bench_pf/logs/heldout_shard_%A_%a.out
#SBATCH --error=/gscratch/cheme/chiu/bench_pf/logs/heldout_shard_%A_%a.err

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=${DED_CAMPAIGN_ROOT:-$ROOT/bench_pf}
SCRIPTS=$CAMPAIGN/scripts
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
TASK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

IFS=$'\t' read -r ARM SHARD INPUT_DIR OUTPUT_DIR < <(
  apptainer exec --cleanenv --bind "$ROOT":"$ROOT" "$SIF" \
    python3 "$SCRIPTS/r46_heldout.py" task-paths \
      --campaign-root "$CAMPAIGN" --task "$TASK" --format tsv
)

echo "task=$TASK arm=$ARM shard=$SHARD start=$(date -Is)"
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
SHARD_LOG_DIR=$CAMPAIGN/logs/heldout_sim_logs/$ARM/shard_$SHARD
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$SHARD_LOG_DIR"

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$ROOT/bench/arcade_example.py":/app/src/inverse_design/rf/arcade_example.py \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  --env JAVA_OPTS="-Xmx4g" \
  --env DED_LOG_DIR="$SHARD_LOG_DIR" \
  "$SIF" \
  python3 rf/shard_runner.py \
    --input-dir "$INPUT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --jar-path /app/src/inverse_design/models/arcade-logging-necrotic.jar \
    --shard-index "$SHARD" \
    --n-shards 5 \
    --n-particles 512 \
    --workers "${SLURM_CPUS_PER_TASK}"

echo "task=$TASK arm=$ARM shard=$SHARD end=$(date -Is)"
