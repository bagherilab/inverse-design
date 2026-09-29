#!/bin/bash
#SBATCH --job-name=r46c_monitor
#SBATCH --account=compute-cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/gscratch/cheme/chiu/bench_mode_local_pm50_corrected/logs/monitor_%j.out
#SBATCH --error=/gscratch/cheme/chiu/bench_mode_local_pm50_corrected/logs/monitor_%j.err

set -euo pipefail

CAMPAIGN=/gscratch/cheme/chiu/bench_mode_local_pm50_corrected
python3 "$CAMPAIGN/scripts/pm50_health_monitor.py" \
  --campaign-root "$CAMPAIGN" \
  --interval-minutes 30
