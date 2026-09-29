#!/bin/bash
#SBATCH --job-name=r46c_prep
#SBATCH --account=compute-cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=8:00:00

set -uo pipefail

ROOT=/gscratch/cheme/chiu
CAMPAIGN_ROOT=${DED_CAMPAIGN_ROOT:-$ROOT/bench_mode_local_pm50_corrected}
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
SCRIPTS=$CAMPAIGN_ROOT/scripts
: "${DED_CONFIG:?DED_CONFIG not set}"
CONFIG=$DED_CONFIG
RUNTAG=$(basename "$CONFIG" .json)
TARGET=$ROOT/ARCADE_OUTPUT/ABC_SMC_RF_N1024_pm50/targets.json
N_SHARDS=${DED_N_SHARDS:-5}

echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date -Is) config=$CONFIG"
echo "step15=0 adaptive_kernel=0 template=10-seed"

MARKER=$CAMPAIGN_ROOT/logs/pending_${RUNTAG}_${SLURM_JOB_ID:-manual}.json
rm -f "$MARKER"
export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$CAMPAIGN_ROOT/src/arcade_example.py":/app/src/inverse_design/rf/arcade_example.py \
  --bind "$ROOT/corrected/abc_smc_rf_arcade.py":/app/src/inverse_design/rf/abc_smc_rf_arcade.py \
  --bind "$ROOT/pm50/parameter_config.py":/app/src/inverse_design/config/parameter_config.py \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  --env JAVA_OPTS="-Xmx4g" \
  --env DED_LOG_DIR="$CAMPAIGN_ROOT/logs" \
  --env DED_ANALYSIS_WORKERS=16 \
  --env DED_STEP15=0 \
  --env DED_ADAPTIVE_KERNEL=0 \
  --env DED_PREPARE_ONLY=1 \
  --env DED_PREPARE_MARKER="$MARKER" \
  "$SIF" \
  python3 rf/arcade_example.py --config "$CONFIG" --target "$TARGET"
status=$?
echo "driver exit=$status $(date -Is)"
[[ $status -ne 0 ]] && { echo "ERROR: driver failed" >&2; exit $status; }

RUN_NAME=$(python3 -c "import json;print(json.load(open('$CONFIG'))['run_name'])")
NITER=$(python3 -c "import json;print(json.load(open('$CONFIG'))['n_iterations'])")
RUN_ROOT=$ROOT/ARCADE_OUTPUT/$RUN_NAME
if [[ ! -f "$MARKER" ]]; then
  echo "No pending generation: chain finished."
  # NITER, not NITER-1: all five settled generations including g4 are reduced
  # to summaries, so the raw per-simulation JSON no longer consumes inodes.
  bash "$SCRIPTS/prune_bench_arm.sh" --run-root "$RUN_ROOT" --before "$NITER"
  exit 0
fi

ITER=$(python3 -c "import json;print(json.load(open('$MARKER'))['iteration'])")
OUTPUT_DIR=$(python3 -c "import json;print(json.load(open('$MARKER'))['output_dir'])")
PENDING=$(python3 -c "import json;print(json.load(open('$MARKER'))['pending'])")
NPART=$(python3 -c "import json;print(json.load(open('$MARKER'))['n_particles'])")
echo "prepared iter_$ITER: $PENDING of $NPART simulations pending"

bash "$SCRIPTS/prune_bench_arm.sh" --run-root "$(dirname "$OUTPUT_DIR")" --before "$ITER"
GEN_FILE=$CAMPAIGN_ROOT/logs/generation_${RUNTAG}_iter_${ITER}.json
cp "$MARKER" "$GEN_FILE"

if [[ "$ITER" -ge 4 ]]; then
  PART_ARGS=(--partition=compute --account=compute-cheme)
else
  PART_ARGS=(--partition=ckpt-all --account=ckpt-cheme --requeue)
fi

ARRAY_ID=$(sbatch --parsable "${PART_ARGS[@]}" \
  --output="$CAMPAIGN_ROOT/logs/shard_%A_%a.out" \
  --error="$CAMPAIGN_ROOT/logs/shard_%A_%a.err" \
  --array=1-"$N_SHARDS" \
  --export=ALL,DED_CAMPAIGN_ROOT="$CAMPAIGN_ROOT",DED_GEN_FILE="$GEN_FILE",DED_N_SHARDS="$N_SHARDS",DED_CONFIG="$CONFIG" \
  "$SCRIPTS/shard_pm50.sbatch")
[[ -z "$ARRAY_ID" ]] && { echo "ERROR: shard submission failed" >&2; exit 1; }
echo "submitted shard array $ARRAY_ID (1-$N_SHARDS) for iter_$ITER on ${PART_ARGS[0]}"

NEXT_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_ID" \
  --output="$CAMPAIGN_ROOT/logs/prep_%j.out" \
  --error="$CAMPAIGN_ROOT/logs/prep_%j.err" \
  --export=ALL,DED_CAMPAIGN_ROOT="$CAMPAIGN_ROOT",DED_CONFIG="$CONFIG",DED_N_SHARDS="$N_SHARDS" \
  "$SCRIPTS/prepare_pm50.sbatch")
echo "submitted next prepare $NEXT_ID (waits on $ARRAY_ID)"
echo "end=$(date -Is)"
