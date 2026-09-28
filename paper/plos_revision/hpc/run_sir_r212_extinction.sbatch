#!/bin/bash
#SBATCH --job-name=sir_r212_ss
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/gscratch/cheme/chiu/sir_r212/logs/extinction_%j.out
#SBATCH --error=/gscratch/cheme/chiu/sir_r212/logs/extinction_%j.err

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=$ROOT/sir_r212
SIF=$ROOT/abc_simplified_model_v3.sif
SOURCE=/mmfs1/home/pohaoc2/inverse_design/src/inverse_design

echo "job=${SLURM_JOB_ID:-none} start=$(date -Is)"
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
  python3 "$CAMPAIGN/scripts/sir_r212_extinction.py" \
    --campaign "$CAMPAIGN" \
    --draws 30 \
    --max-time 5000 \
    --workers "${SLURM_CPUS_PER_TASK}"

echo "job=${SLURM_JOB_ID:-none} end=$(date -Is)"
