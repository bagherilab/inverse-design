#!/usr/bin/env bash
# Regenerate every figure under results/figures/.
# Run from the repository root.  Usage:
#   bash scripts/regenerate_all_figures.sh            # run all
#   bash scripts/regenerate_all_figures.sh -k         # keep going on failure
#   PY=python scripts/regenerate_all_figures.sh       # override interpreter

set -u
[[ "${1:-}" == "-k" ]] && KEEP_GOING=1 || KEEP_GOING=0
PY="${PY:-python3}"

STEPS=(
  "fit_exp          | $PY scripts/combine_fit_exp_figure.py"
  "equifinality     | $PY scripts/combine_equifinality_figure.py"
  "gliob            | $PY scripts/generate_gliob_fit_figures.py"
  "feasible         | $PY scripts/combine_feasible_metrics_ranges.py"
  "ed_convergence   | $PY scripts/make_ed_figure_generation_convergence.py"
  "simp_mi_table    | $PY scripts/combine_simplification_mi_table.py"
  "simp_metrics     | $PY scripts/combine_simplification_metrics.py"
  "simp_combined    | $PY scripts/combine_simplification_figure.py"
  "sir              | $PY scripts/generate_sir_figure.py"
  "model_simp_ex    | $PY scripts/generate_model_simplification_figure.py --peak 1 --save results/figures/simplified_model_example/simplified_model_example.png"
)

fail=0
for step in "${STEPS[@]}"; do
  name="${step%%|*}"; cmd="${step#*|}"
  name="${name// /}"; cmd="${cmd# }"
  printf '\n=== [%s] %s ===\n' "$name" "$cmd"
  if ! eval "$cmd"; then
    printf '!!! [%s] FAILED\n' "$name" >&2
    fail=$((fail+1))
    (( KEEP_GOING )) || exit 1
  fi
done

printf '\nDone. %d failure(s).\n' "$fail"
exit "$fail"
