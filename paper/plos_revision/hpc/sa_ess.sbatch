#!/bin/bash
#SBATCH --job-name=sa_ess
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --array=1-25
#SBATCH --output=/gscratch/cheme/chiu/sa_ess/logs/sa_%A_%a.out
#SBATCH --error=/gscratch/cheme/chiu/sa_ess/logs/sa_%A_%a.err

# How effective sample size depends on the forest, holding the particles fixed.
#
# The published run's weights put 159-190 of ~1000 particles above zero and an
# ESS of 84-98, flat across five generations while the particle count rose from
# 836 to 1022. The mechanism is that trees split until a leaf holds
# min_samples_leaf samples and there is no depth limit, so leaves stay small
# however many particles exist, and the weighted support is bounded by
# n_trees x leaf size rather than by the particle count. This grid measures that
# relationship instead of asserting it.
#
# Each task re-runs the driver over the *same* cached particles with a different
# (n_trees, min_samples_leaf) and records the weights it produces. Nothing is
# simulated: every generation's final_metrics.csv is present, so the analysis is
# read from cache and the input tree is a symlink that the settled-generation
# guard keeps anything from writing to.
#
# What this can and cannot say:
#   - it CAN say how ESS responds to the forest hyperparameters, five times over
#     on five real particle sets;
#   - it CANNOT say what the sampler would have done under those hyperparameters.
#     Generation g+1's particles were drawn using the original weights, so the
#     five generations here are replicates, not a trajectory. A trajectory needs
#     a full re-run with new simulations.

set -uo pipefail

ROOT=/gscratch/cheme/chiu
SA=$ROOT/sa_ess
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
TARGET=$ROOT/ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/targets.json

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$SA/manifest.txt")
read -r NTREES NLEAF CONFIG <<< "$line"
echo "task=$SLURM_ARRAY_TASK_ID n_trees=$NTREES min_samples_leaf=$NLEAF host=$(hostname) start=$(date -Is)"

export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  --env DED_LOG_DIR="$SA/logs" \
  --env DED_ANALYSIS_WORKERS=1 \
  "$SIF" \
  python3 rf/arcade_example.py --config "$CONFIG" --target "$TARGET"

status=$?
echo "exit=$status end=$(date -Is)"
exit $status
