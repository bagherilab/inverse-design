#!/bin/bash
#SBATCH --job-name=pm50_monitor
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --requeue
#SBATCH --output=/gscratch/cheme/chiu/bench_pm50_reduced/logs/monitor_%j.out
#SBATCH --error=/gscratch/cheme/chiu/bench_pm50_reduced/logs/monitor_%j.err

set -euo pipefail

CAMPAIGN=/gscratch/cheme/chiu/bench_pm50_reduced
python3 "$CAMPAIGN/scripts/pm50_health_monitor.py" \
  --campaign-root "$CAMPAIGN" \
  --interval-minutes 30
