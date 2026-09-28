#!/bin/bash
#SBATCH --job-name=kdeboot
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/sa_ess/logs/kdeboot_%j.out
#SBATCH --error=/gscratch/cheme/chiu/sa_ess/logs/kdeboot_%j.err

# Bootstrap stability of the four peaks. Reads cached parameters and weights
# only -- nothing is simulated and nothing is written into a run directory. It
# needs a batch job because pandas, scipy and sklearn live in the container
# rather than on the login node, and because 400 KDE fits are not login-node
# work. Imports `weighted_kde.py`, which must sit beside it in $ROOT.

set -uo pipefail

ROOT=/gscratch/cheme/chiu
SIF=$ROOT/abc_simplified_model_v3.sif

export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache

apptainer exec --cleanenv --writable-tmpfs \
  --bind "$ROOT":"$ROOT" \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  "$SIF" \
  python3 "$ROOT/kde_bootstrap.py"

status=$?
echo "exit=$status end=$(date -Is)"
exit $status
