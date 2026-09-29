#!/bin/bash
#SBATCH --job-name=pm50_prune
#SBATCH --account=compute-cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=4:00:00

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=$ROOT/bench_pm50_reduced
for config in "$CAMPAIGN"/configs/pm50_*.json; do
  run_name=$(python3 -c "import json;print(json.load(open('$config'))['run_name'])")
  run_root=$ROOT/ARCADE_OUTPUT/$run_name
  bash "$CAMPAIGN/scripts/prune_bench_arm.sh" --run-root "$run_root" --before 4
done
