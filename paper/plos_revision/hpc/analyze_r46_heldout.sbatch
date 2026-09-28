#!/bin/bash
#SBATCH --job-name=r46_heldout_analyze
#SBATCH --account=cheme
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/gscratch/cheme/chiu/bench_pf/logs/heldout_analyze_%A_%a.out
#SBATCH --error=/gscratch/cheme/chiu/bench_pf/logs/heldout_analyze_%A_%a.err

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=${DED_CAMPAIGN_ROOT:-$ROOT/bench_pf}
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
ARM_INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  "$SIF" \
  python3 "$CAMPAIGN/scripts/r46_heldout.py" analyze-arm \
    --campaign-root "$CAMPAIGN" --arm-index "$ARM_INDEX"
