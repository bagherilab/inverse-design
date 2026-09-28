#!/bin/bash
#SBATCH --job-name=r46_heldout_report
#SBATCH --account=cheme
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/cheme/chiu/bench_pf/logs/heldout_report_%j.out
#SBATCH --error=/gscratch/cheme/chiu/bench_pf/logs/heldout_report_%j.err

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=${DED_CAMPAIGN_ROOT:-$ROOT/bench_pf}
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  "$SIF" \
  python3 "$CAMPAIGN/scripts/r46_heldout.py" summarize \
    --campaign-root "$CAMPAIGN"
