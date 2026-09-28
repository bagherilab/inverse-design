#!/bin/bash
#SBATCH --job-name=bwstab
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0:20:00
#SBATCH --output=/gscratch/cheme/chiu/sa_ess/logs/bwstab_%j.out
#SBATCH --error=/gscratch/cheme/chiu/sa_ess/logs/bwstab_%j.err

# Redraw the SI KDE covariance-factor sweep with the default detector settings
# (200x200 grid, 5x5 maximum filter, 5% relative-density threshold). Reads the
# cached generation 4 parameter cloud only -- nothing is simulated. It needs a
# batch job because numpy, scipy, pandas and matplotlib live in the container
# rather than on the login node, and the container cannot start on a login node
# (its slurm mounts are missing there).

set -uo pipefail

ROOT=/gscratch/cheme/chiu
SIF=$ROOT/abc_simplified_model_v3.sif
SCRIPT=${DED_BW_SCRIPT:-$ROOT/bandwidth_stability_peaks.py}
PARAMS=${DED_BW_PARAMS:-$HOME/inverse_design/data/N1024_breast_mean_only/iter_4/all_param_df.csv}
OUT=${DED_BW_OUT:-$ROOT/bandwidth_stability_out}

mkdir -p "$OUT"
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache

echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date -Is)"
echo "params=$PARAMS"

apptainer exec --cleanenv --writable-tmpfs \
  --bind "$ROOT":"$ROOT" \
  --bind "$HOME":"$HOME" \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  --env MPLCONFIGDIR=/tmp/mplconfig \
  "$SIF" \
  python3 "$SCRIPT" \
    --params "$PARAMS" \
    --figure "$OUT/bandwidth_stability_peaks.png" \
    --sweep "$OUT/bandwidth_stability_sweep.csv" \
    --provenance "$OUT/bandwidth_stability_provenance.json"

status=$?
echo "exit=$status end=$(date -Is)"
exit $status
