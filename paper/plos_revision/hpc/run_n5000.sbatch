#!/bin/bash
#SBATCH --job-name=ded_n5000
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=/gscratch/cheme/chiu/logs/n5000_%j.out
#SBATCH --error=/gscratch/cheme/chiu/logs/n5000_%j.err

# 5000 particles x 5 iterations x 10 seeds = 250,000 ARCADE simulations.
# n_iterations=5 gives iter_0 (prior draw) plus four update generations, matching
# the published run (g0 -> g4). n_iterations=4 would drop one update step.
# Answers R2.16 ("at least 5000 particles") and R2.14.
#
# Requires container v2: the published .sif predates both n_particles (which
# decouples the count from 2**sobol_power) and run_name.
#
# Flags that are not optional:
#   --writable-tmpfs  the code writes simulation_run.log inside /app, which is
#                     read-only in a .sif
#   --pwd             apptainer ignores the image WORKDIR
#   absolute paths    --config/--target are resolved against the container cwd

set -uo pipefail

ROOT=/gscratch/cheme/chiu
SIF=$ROOT/abc_simplified_model_v3.sif
CONFIG=$ROOT/config_combined_n5000_iter4.json
TARGET=$ROOT/ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/targets.json
RUN=ABC_SMC_RF_N5000_combined_grid_breast_only_mean

echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date -Is)"
echo "config=$CONFIG"
echo "target=$(cat "$TARGET" | tr -d '\n ')"

# Refuse to append to an existing run: this config has no resume story and a
# partial directory would be silently reused by the generation-skip logic.
if [[ -e "$ROOT/ARCADE_OUTPUT/$RUN" ]]; then
  echo "ERROR: $ROOT/ARCADE_OUTPUT/$RUN already exists; move it aside first" >&2
  exit 1
fi

# Inodes: 5000 particles x 10 seeds x 301 files x 5 iterations ~= 7.53M.
free=$(( 50000000 - $(hyakstorage 2>/dev/null | grep -oP '\d+(?= / 50000000 files)' | head -1) ))
echo "inodes free (approx): $free ; this run needs ~7530000"
if [[ -n "$free" && "$free" -lt 8500000 ]]; then
  echo "ERROR: insufficient inode headroom; let the prune job finish first" >&2
  exit 1
fi

# run_simulations.py used to log to a relative path, which under --pwd landed in
# the read-only image and thus in apptainer's small writable overlay -> ENOSPC.
# v3 honours DED_LOG_DIR, so the log goes to real storage and the overlay stays
# essentially empty.
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# -Xmx4g, not 16g: 40 concurrent JVMs at 16g would commit far past the node's 192 GB.
apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  --env JAVA_OPTS="-Xmx4g" \
  --env DED_LOG_DIR="$ROOT/logs" \
  "$SIF" \
  python3 rf/arcade_example.py --config "$CONFIG" --target "$TARGET"

status=$?
echo "exit=$status end=$(date -Is)"
for g in 0 1 2 3 4; do
  d="$ROOT/ARCADE_OUTPUT/$RUN/iter_$g"
  [[ -d "$d" ]] && echo "  iter_$g: $(find "$d/inputs" -maxdepth 1 -type d -name 'input_*' 2>/dev/null | wc -l) sims"
done
exit $status
