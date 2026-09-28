#!/bin/bash
#SBATCH --job-name=r46_heldout_prep
#SBATCH --account=cheme
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/gscratch/cheme/chiu/bench_pf/logs/heldout_prepare_%j.out
#SBATCH --error=/gscratch/cheme/chiu/bench_pf/logs/heldout_prepare_%j.err

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=$ROOT/bench_pf
SCRIPTS=$CAMPAIGN/scripts
HELDOUT=$CAMPAIGN/heldout
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  "$SIF" \
  python3 "$SCRIPTS/r46_heldout.py" prepare --campaign-root "$CAMPAIGN"

if [[ -f "$HELDOUT/submission_record.json" ]]; then
  echo "Held-out evaluation is already submitted: $HELDOUT/submission_record.json"
  exit 0
fi

SIM_ID=$(sbatch --parsable \
  --array=0-104%15 \
  "$SCRIPTS/shard_r46_heldout.sbatch")
ANALYZE_ID=$(sbatch --parsable \
  --array=0-20%3 \
  --dependency=afterok:"$SIM_ID" \
  "$SCRIPTS/analyze_r46_heldout.sbatch")
REPORT_ID=$(sbatch --parsable \
  --dependency=afterok:"$ANALYZE_ID" \
  "$SCRIPTS/report_r46_heldout.sbatch")

python3 - "$HELDOUT/submission_record.json" "$SIM_ID" "$ANALYZE_ID" "$REPORT_ID" <<'PY'
import datetime as dt
import json
import sys

path, simulation, analysis, report = sys.argv[1:]
json.dump(
    {
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "simulation_array": simulation,
        "analysis_array": analysis,
        "report_job": report,
        "simulation_tasks": 105,
        "simulation_concurrency": 15,
        "analysis_tasks": 21,
        "analysis_concurrency": 3,
    },
    open(path, "w"),
    indent=2,
)
open(path, "a").write("\n")
PY

echo "HELDOUT_SUBMITTED simulations=$SIM_ID analysis=$ANALYZE_ID report=$REPORT_ID"