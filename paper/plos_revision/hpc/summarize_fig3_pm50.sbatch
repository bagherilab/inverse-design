#!/bin/bash
#SBATCH --job-name=fig3pm50_sum
#SBATCH --account=compute-cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=2:00:00

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=${DED_FIG3_CAMPAIGN:-$ROOT/fig3_pm50}
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  "$SIF" \
  python3 "$CAMPAIGN/scripts/summarize_fig3_pm50.py" --campaign "$CAMPAIGN"
