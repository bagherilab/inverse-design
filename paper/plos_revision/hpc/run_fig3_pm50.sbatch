#!/bin/bash
#SBATCH --job-name=fig3pm50
#SBATCH --account=compute-cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --array=1-6

set -euo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN=${DED_FIG3_CAMPAIGN:-$ROOT/fig3_pm50}
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
JAR=/app/src/inverse_design/models/arcade-logging-necrotic.jar
CASES=(mean mode cluster_1 cluster_2 cluster_3 cluster_4)
INDEX=${SLURM_ARRAY_TASK_ID:?array task id is required}
CASE=${CASES[$((INDEX - 1))]}
XML=$CAMPAIGN/inputs/input_${INDEX}.xml
OUTPUT=$CAMPAIGN/CLUSTER/$CASE

[[ -f "$XML" ]] || { echo "ERROR: missing $XML" >&2; exit 1; }
mkdir -p "$OUTPUT" "$CAMPAIGN/logs"

cells=$(find "$OUTPUT" -maxdepth 1 -type f -name '*.CELLS.json' | wc -l)
locations=$(find "$OUTPUT" -maxdepth 1 -type f -name '*.LOCATIONS.json' | wc -l)
if [[ "$cells" -eq 150 && "$locations" -eq 150 ]]; then
  echo "$CASE already complete (150 CELLS + 150 LOCATIONS); skipping"
  exit 0
fi

echo "job=${SLURM_ARRAY_JOB_ID}_${INDEX} case=$CASE start=$(date -Is)"
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  "$SIF" \
  java -Xmx4g -Djava.awt.headless=true -cp "$JAR" \
    arcade.core.ARCADE patch "$XML" "$OUTPUT"

cells=$(find "$OUTPUT" -maxdepth 1 -type f -name '*.CELLS.json' | wc -l)
locations=$(find "$OUTPUT" -maxdepth 1 -type f -name '*.LOCATIONS.json' | wc -l)
if [[ "$cells" -ne 150 || "$locations" -ne 150 ]]; then
  echo "ERROR: $CASE incomplete: CELLS=$cells LOCATIONS=$locations expected=150 each" >&2
  exit 1
fi
printf '%s\n' "job=${SLURM_ARRAY_JOB_ID}_${INDEX}" "case=$CASE" "finished=$(date -Is)" \
  "cells=$cells" "locations=$locations" > "$CAMPAIGN/logs/${CASE}.complete"
echo "$CASE complete end=$(date -Is)"
