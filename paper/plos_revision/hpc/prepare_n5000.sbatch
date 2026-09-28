#!/bin/bash
#SBATCH --job-name=ded_prep
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gscratch/cheme/chiu/logs/prep_%j.out
#SBATCH --error=/gscratch/cheme/chiu/logs/prep_%j.err

# One link in the sharded ABC-SMC chain. Replays every finished generation from
# what is on disk, fits the forest, writes the next generation's input XMLs,
# then stops before running any simulation (DED_PREPARE_ONLY) and hands the
# work to a shard array. It re-submits itself behind that array, so the whole
# run advances without a long-lived driver holding a node for three days.
#
# The chain ends on its own: once every generation is complete the driver never
# reaches the prepare-only stop, runs the post-fit analysis and upload, and no
# further jobs are submitted.
#
# Resources are for the *last* invocation, which does the full analysis and S3
# upload; the intermediate ones finish in minutes.
#
# Submit the first link by hand:
#   sbatch analysis_scripts/prepare_n5000.sbatch

set -uo pipefail

ROOT=/gscratch/cheme/chiu
SIF=$ROOT/abc_simplified_model_v3.sif
SRC=$ROOT/src_override/inverse_design
CONFIG=$ROOT/config_combined_n5000_iter4.json
TARGET=$ROOT/ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2/targets.json
SCRIPTS=${DED_SCRIPT_DIR:-$ROOT/scripts}
N_SHARDS=${DED_N_SHARDS:-5}

echo "job=${SLURM_JOB_ID:-none} host=$(hostname) start=$(date -Is) n_shards=$N_SHARDS"

# The driver writes this only when it stops early with simulations pending. A
# leftover from the previous link would send the shards at the wrong
# generation, so clear it before the run rather than trusting it to be fresh.
MARKER=$ROOT/logs/pending_${SLURM_JOB_ID:-manual}.json
rm -f "$MARKER"

export APPTAINER_TMPDIR=$ROOT/apptainer_tmp
export APPTAINER_CACHEDIR=$ROOT/apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

# $SRC is the package extracted from the .sif with the sharding changes applied.
# Binding over /app/src/inverse_design avoids rebuilding the image, which the
# published run depends on staying byte-identical.
apptainer exec --cleanenv --writable-tmpfs --pwd /app/src/inverse_design \
  --bind "$ROOT":"$ROOT" \
  --bind "$SRC":/app/src/inverse_design \
  --bind "$ROOT/apptainer_tmp":/tmp \
  --bind "$ROOT/apptainer_cache":/var/tmp \
  --env JAVA_OPTS="-Xmx4g" \
  --env DED_LOG_DIR="$ROOT/logs" \
  --env DED_ANALYSIS_WORKERS=16 \
  --env DED_PREPARE_ONLY=1 \
  --env DED_PREPARE_MARKER="$MARKER" \
  "$SIF" \
  python3 rf/arcade_example.py --config "$CONFIG" --target "$TARGET"
status=$?

echo "driver exit=$status $(date -Is)"
if [[ $status -ne 0 ]]; then
  echo "ERROR: driver failed; chain stops here" >&2
  exit $status
fi

if [[ ! -f "$MARKER" ]]; then
  echo "No pending generation: every iteration is complete. Chain finished."
  exit 0
fi

ITER=$(python3 -c "import json;print(json.load(open('$MARKER'))['iteration'])")
INPUT_DIR=$(python3 -c "import json;print(json.load(open('$MARKER'))['input_dir'])")
OUTPUT_DIR=$(python3 -c "import json;print(json.load(open('$MARKER'))['output_dir'])")
NPART=$(python3 -c "import json;print(json.load(open('$MARKER'))['n_particles'])")
PENDING=$(python3 -c "import json;print(json.load(open('$MARKER'))['pending'])")
echo "prepared iter_$ITER: $PENDING of $NPART simulations pending"

# Pin the generation into a per-iteration file. The shards read this, not
# $MARKER, so the next link is free to clear its own marker whenever it starts.
GEN_FILE=$ROOT/logs/generation_iter_${ITER}.json
cp "$MARKER" "$GEN_FILE"

ARRAY_ID=$(sbatch --parsable \
  --array=1-"$N_SHARDS" \
  --export=ALL,DED_GEN_FILE="$GEN_FILE",DED_N_SHARDS="$N_SHARDS" \
  "$SCRIPTS/shard_n5000.sbatch")
if [[ -z "$ARRAY_ID" ]]; then
  echo "ERROR: shard array submission failed" >&2
  exit 1
fi
echo "submitted shard array $ARRAY_ID (1-$N_SHARDS) for iter_$ITER"

# afterok on an array waits for every task. If any shard fails the dependency is
# never satisfied and SLURM cancels this job, which is the intent: the forest
# must not be fit on a generation with holes in it.
NEXT_ID=$(sbatch --parsable --dependency=afterok:"$ARRAY_ID" "$SCRIPTS/prepare_n5000.sbatch")
echo "submitted next prepare $NEXT_ID (waits on $ARRAY_ID)"
echo "end=$(date -Is)"
