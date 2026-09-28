#!/bin/bash
#SBATCH --job-name=sir_r212_cal
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/gscratch/cheme/chiu/sir_r212/logs/calibrate_%A_%a.out
#SBATCH --error=/gscratch/cheme/chiu/sir_r212/logs/calibrate_%A_%a.err

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=$ROOT/sir_r212
SIF=$ROOT/abc_simplified_model_v3.sif
SOURCE=/mmfs1/home/pohaoc2/inverse_design/src/inverse_design
TASK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SEEDS=(202608081 202608082)
if [[ "$TASK" -lt 1 || "$TASK" -gt 2 ]]; then
  echo "invalid array task: $TASK" >&2
  exit 2
fi
SEED=${SEEDS[$((TASK - 1))]}

if [[ "${SIR_SMOKE:-0}" == "1" ]]; then
  PARTICLES=16
  GENERATIONS=2
  SIMULATIONS=1
  MAX_TIME=20
  OUTPUT=$CAMPAIGN/smoke/replicate_$TASK
else
  PARTICLES=512
  GENERATIONS=5
  SIMULATIONS=3
  MAX_TIME=365
  OUTPUT=$CAMPAIGN/replicates/replicate_$TASK
fi

echo "job=${SLURM_JOB_ID:-none} task=$TASK seed=$SEED start=$(date -Is)"
echo "particles=$PARTICLES generations=$GENERATIONS simulations=$SIMULATIONS max_time=$MAX_TIME"

export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$CAMPAIGN/logs"

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src \
  --bind "$ROOT":"$ROOT" \
  --bind "$SOURCE":/app/src/inverse_design \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  --env PYTHONPATH=/app/src \
  --env PYTHONHASHSEED=0 \
  "$SIF" \
  python3 "$CAMPAIGN/scripts/sir_r212_replicate.py" \
    --target "$CAMPAIGN/published/target_values.csv" \
    --true-params "$CAMPAIGN/published/true_params.csv" \
    --output "$OUTPUT" \
    --seed "$SEED" \
    --particles "$PARTICLES" \
    --generations "$GENERATIONS" \
    --simulations "$SIMULATIONS" \
    --max-time "$MAX_TIME" \
    --workers "${SLURM_CPUS_PER_TASK}"

echo "job=${SLURM_JOB_ID:-none} task=$TASK end=$(date -Is)"
