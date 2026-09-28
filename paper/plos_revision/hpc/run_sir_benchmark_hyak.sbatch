#!/bin/bash
#SBATCH --job-name=sir_r31_all5
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt
#SBATCH --qos=ckpt
#SBATCH --constraint=cascadelake
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/gscratch/cheme/chiu/sir_benchmark_r31_20260810/logs/%x_%j.out
#SBATCH --error=/gscratch/cheme/chiu/sir_benchmark_r31_20260810/logs/%x_%j.err

set -uo pipefail

CAMPAIGN=/gscratch/cheme/chiu/sir_benchmark_r31_20260810
PYTHON=$CAMPAIGN/venv/bin/python
SCRIPT=$CAMPAIGN/scripts/sir_benchmark.py
SIF=/gscratch/cheme/chiu/abc_simplified_model_v3.sif
SOURCE=$CAMPAIGN/source
RESULTS=/gscratch/cheme/chiu/SIR_OUTPUT/n3_t365_l30
MODE=${BENCH_MODE:-production}
OUT=$CAMPAIGN/$MODE
STATUS=$OUT/runner_status.log
MIN_ARM_REMAINING_S=$((4 * 3600 + 15 * 60))

mkdir -p "$OUT" "$CAMPAIGN/mplconfig"
export MPLCONFIGDIR=$CAMPAIGN/mplconfig
export PYTHONHASHSEED=0

printf 'runner_start=%s job=%s host=%s mode=%s\n' \
  "$(date -Is)" "${SLURM_JOB_ID:-none}" "$(hostname)" "$MODE" >> "$STATUS"

slurm_time_to_seconds() {
  local value=$1 days=0 rest hours=0 minutes=0 seconds=0
  if [[ "$value" == *-* ]]; then
    days=${value%%-*}
    rest=${value#*-}
  else
    rest=$value
  fi
  IFS=: read -r -a fields <<< "$rest"
  if (( ${#fields[@]} == 3 )); then
    hours=${fields[0]}
    minutes=${fields[1]}
    seconds=${fields[2]}
  elif (( ${#fields[@]} == 2 )); then
    minutes=${fields[0]}
    seconds=${fields[1]}
  else
    seconds=${fields[0]}
  fi
  printf '%s\n' $((10#$days * 86400 + 10#$hours * 3600 + 10#$minutes * 60 + 10#$seconds))
}

ensure_arm_window() {
  local method=$1 time_left remaining
  [[ "$MODE" == smoke || -z "${SLURM_JOB_ID:-}" ]] && return 0
  time_left=$(squeue -h -j "$SLURM_JOB_ID" -o '%L')
  [[ -z "$time_left" || "$time_left" == UNLIMITED ]] && return 0
  remaining=$(slurm_time_to_seconds "$time_left")
  if (( remaining < MIN_ARM_REMAINING_S )); then
    printf 'method=%s status=requeue_low_time remaining_s=%s time=%s\n' \
      "$method" "$remaining" "$(date -Is)" >> "$STATUS"
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
}

for method in ded npe snpe pyabc_budget pyabc_cost; do
  result=$OUT/$method/result.json
  extra=()
  if [[ "$MODE" == smoke ]]; then
    result=$OUT/$method/smoke_result.json
    extra=(--smoke --smoke-budget 64 --predictive-draws 8)
  else
    extra=(--predictive-draws 128)
  fi
  if [[ -f "$result" ]] && grep -q '"status": "complete"' "$result"; then
    printf 'method=%s status=skip_complete time=%s\n' \
      "$method" "$(date -Is)" >> "$STATUS"
    continue
  fi

  ensure_arm_window "$method"
  printf 'method=%s status=start time=%s\n' \
    "$method" "$(date -Is)" >> "$STATUS"
  apptainer exec --cleanenv \
    --bind /gscratch/cheme/chiu:/gscratch/cheme/chiu \
    --bind /mmfs1/home/pohaoc2:/mmfs1/home/pohaoc2 \
    --env MPLCONFIGDIR="$MPLCONFIGDIR" \
    --env PYTHONHASHSEED=0 \
    --env SLURM_JOB_ID="${SLURM_JOB_ID:-}" \
    --env SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}" \
    "$SIF" \
    "$PYTHON" "$SCRIPT" \
    --methods "$method" \
    --budget 5120 \
    --workers "${SLURM_CPUS_PER_TASK:-16}" \
    --resume \
    --pyabc-cost-hours 3 \
    --source-root "$SOURCE" \
    --results "$RESULTS" \
    --out "$OUT" \
    "${extra[@]}"
  code=$?
  printf 'method=%s status=exit code=%s time=%s\n' \
    "$method" "$code" "$(date -Is)" >> "$STATUS"
done

printf 'runner_end=%s\n' "$(date -Is)" >> "$STATUS"
