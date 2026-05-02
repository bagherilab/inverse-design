# Figure generation lookup (`results/figures/`)

Run commands from the repository root (use `python3` or your project’s interpreter).

**Path placeholders**

- **`{REPO}`** — repository root (directory containing `scripts/`, `src/`).
- **ARCADE SMC outputs** — **`/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT`** (all `ABC_SMC_RF_*` run folders referenced below live here unless overridden).
- **SIR ABC outputs** — **`/home/pohaoc2/UW/bagherilab/SIR_OUTPUT`** (SIR SMC runs used by `make_ed_figure_generation_convergence.py`; same layout as `n3_t365_l30/n128` … `n1024` under that tree).

The 5-panel SIR figure script (`generate_sir_figure.py`) reads **in-repo** paths under `{REPO}/results/SIR/...` by default; if your data only exists under `SIR_OUTPUT`, point the script there or symlink/copy into `{REPO}/results/SIR/`.

Some scripts still use hardcoded absolute paths; change them if you relocate data.

One-page figure reports live under `docs/figure_reports/` and are linked from the
quick index below.

---

## `results/figures/fit_exp/` (breast ABC fit-to-experiment)

| Outputs | Script |
|--------|--------|
| `A_histograms.png`, `A_histogram_*.png`, `B_pca_with_peaks.png`, `C_metric_comparison.png`, `fit_exp_combined.png`, `panel_c_metrics/**`, `panel_c_metrics.zip` | `scripts/combine_fit_exp_figure.py` |
| `equifinality_combined.png` | `scripts/combine_equifinality_figure.py` (default `--output-dir` is `results/figures/fit_exp`) |

**Data sources (`combine_fit_exp_figure.py` defaults)**

| Role | Path |
|------|------|
| Panel A histograms (`--base-dir`) | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_breast` (`targets.json`, `iter_0` / `iter_4` / `final_metrics.csv`) |
| Panel B PCA (`--pca-base-dir`) | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2` (`iter_4/all_param_df.csv`) |
| Panel C cluster scenarios (`--cluster-dir`) | `{REPO}/CLUSTER` (subfolders `mean`, `mode`, `cluster_1` … with ARCADE cell outputs) |
| Panel D equifinality (`--summary-json`) | `{REPO}/out/pca_cluster_arcade_inputs_breast_only_mean_2/summary.json` |

**Data sources (`combine_equifinality_figure.py` defaults)**

| Role | Path |
|------|------|
| ABC run (`--base-dir`) | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2` |
| Cluster profiles (`--summary-json`) | `{REPO}/out/pca_cluster_arcade_inputs_breast_only_mean_2/summary.json` |
| Posterior re-simulation dirs | Derived under `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/` (see script: `mean` / `mode` / peak runs × `iter_<posterior>`) |

```bash
python3 scripts/combine_fit_exp_figure.py
python3 scripts/combine_equifinality_figure.py
```

---

## `results/figures/gliob/` (glioblastoma g0 vs g4 histograms)

| Outputs | Script |
|--------|--------|
| `A_histograms.png`, `A_histogram_*.png` | `scripts/generate_gliob_fit_figures.py` (wraps `combine_fit_exp_figure.py` with `--panel-a-only` and gliob defaults) |

**Data sources**

| Role | Path |
|------|------|
| ABC run (`--base-dir`) | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_glioblastoma` (`targets.json`, `iter_0` / `iter_4` / `final_metrics.csv`) |

```bash
python3 scripts/generate_gliob_fit_figures.py
```

---

## `results/figures/feasible/` (feasible ranges + single-metric fits)

| Outputs | Script |
|--------|--------|
| `A_feasible_metric_histograms.png`, `B_fit_doub_time_relationship.png`, `C_fit_symmetry_relationship.png`, `D_fit_act_ratio_relationship.png`, `feasible_metrics_ranges_combined.png` | `scripts/combine_feasible_metrics_ranges.py` |

**Data sources**

| Role | Path |
|------|------|
| Panel A three-metric histograms (default) | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_breast_only_mean` |
| Panel B (fit DT=32) | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_doub_32` |
| Panel C (fit S=0.75) | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_sym_075` |
| Panel D (fit A=0.70) | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_act_07` |

```bash
python3 scripts/combine_feasible_metrics_ranges.py
```

---

## `results/figures/error_iteration/`

| Outputs | Script |
|--------|--------|
| `ed_figure1_generation_convergence.png` | `scripts/make_ed_figure_generation_convergence.py` |

**Data sources (hardcoded in script — adjust paths if you relocate data)**

| Role | Path |
|------|------|
| SIR convergence | `/home/pohaoc2/UW/bagherilab/SIR_OUTPUT/n3_t365_l30/n512` (and `n128` / `n256` / `n1024` for error-vs-N) |
| ARCADE convergence | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_breast` (and N128 / N256 / N1024 breast runs for error-vs-N) |

```bash
python3 scripts/make_ed_figure_generation_convergence.py
```

---

## `results/figures/simplification/`

| Outputs | Script |
|--------|--------|
| `simplified_figure.png` (MI table + metric grid combined) | `scripts/combine_simplification_figure.py` |
| `simplified_metrics_grid.png` (metrics-only; also a building block for the combined figure) | `scripts/combine_simplification_metrics.py` |
| `simplified_mi_table.png` (MI-only; also a building block) | `scripts/combine_simplification_mi_table.py` |

**Data sources**

| Script | Key inputs (defaults) |
|--------|------------------------|
| `combine_simplification_mi_table.py` | `--n1024-dir` → `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2`; `--summary-json` → `{REPO}/out/pca_cluster_arcade_inputs_breast_only_mean_2/summary.json`; LR JSON under `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/<run>/<run>/iter_4` (see script for `linear_*` run names) |
| `combine_simplification_metrics.py` | `--cluster-dir` → `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast_only_mean_2`; optional `{REPO}/out/pca_cluster_arcade_inputs_breast_only_mean_2/metrics/<cluster>/final_metrics_seed.csv` |
| `combine_simplification_figure.py` | Same defaults as the two scripts above (imports their helpers) |

```bash
python3 scripts/combine_simplification_figure.py
python3 scripts/combine_simplification_metrics.py
python3 scripts/combine_simplification_mi_table.py
```

---

## SIR figure

| Outputs | Script |
|--------|--------|
| SIR summary PNG | `scripts/generate_sir_figure.py` — default save path is `results/figures/SIR/SIR.png`. |

**Data sources (`generate_sir_figure.py` — repo-relative constants)**

| Role | Path |
|------|------|
| Trajectory histories (`HIST_DIR`) | `{REPO}/results/SIR/with_history/n3_t365_l30/n512` |
| Posterior / metrics (`BASE_DIR`) | `{REPO}/results/SIR/n3_t365_l30/n512` |
| Targets | `{REPO}/results/SIR/n3_t365_l30/target_values.csv` |
| True parameters | `{REPO}/results/SIR/n3_t365_l30/true_params.csv` |

**Same runs on disk (`/home/pohaoc2/UW/bagherilab/SIR_OUTPUT`)**

Parallel layout for the `n3_t365_l30` study, e.g. **`/home/pohaoc2/UW/bagherilab/SIR_OUTPUT/n3_t365_l30/n512`** (and `n128` / `n256` / `n1024`). The ED convergence figure (`make_ed_figure_generation_convergence.py`) reads these paths directly; `generate_sir_figure.py` does not unless you change its constants.

```bash
python3 scripts/generate_sir_figure.py
```

---

## `results/figures/simplified_model_example/`

| Outputs | Script |
|--------|--------|
| `simplified_model_example.png` | Not tied to a fixed path in-repo. Use `scripts/generate_model_simplification_figure.py` with `--save` to write the LR + dendrogram example (e.g. under this folder). |

**Data sources (`generate_model_simplification_figure.py`)**

`ARCADE_ROOT` in code resolves to the parent of the repo’s parent; on this setup that is **`/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT`**.

| Peak `p` | Files |
|----------|--------|
| Linear regression JSON | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_linear_0.8_p{p}_mean_only/.../lr_predictions_r0.8_p{p}.json` |
| Linear posterior | `.../iter_4/all_param_df.csv` |
| Dendrogram redundancy JSON | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_dendrogram_1.0_p{p}_mean_only/.../redundancy_analysis_threshold_1.0_p{p}.json` |
| Base (full) posterior | `/home/pohaoc2/UW/bagherilab/ARCADE_OUTPUT/ABC_SMC_RF_N512_combined_grid_breast_only_mean/iter_4/all_param_df.csv` |

```bash
python3 scripts/generate_model_simplification_figure.py --peak 1 --save results/figures/simplified_model_example/simplified_model_example.png
```

---

## Quick index

| Location | Script(s) | Report |
|----------|-----------|--------|
| `fit_exp/` (except `equifinality_combined.png`) | `scripts/combine_fit_exp_figure.py` | `docs/figure_reports/fit_exp_combined.md` |
| `fit_exp/equifinality_combined.png` | `scripts/combine_equifinality_figure.py` | `docs/figure_reports/equifinality_combined.md` |
| `gliob/` | `scripts/generate_gliob_fit_figures.py` | `docs/figure_reports/gliob_histograms.md` |
| `feasible/` | `scripts/combine_feasible_metrics_ranges.py` | `docs/figure_reports/feasible_metrics_ranges.md` |
| `error_iteration/` | `scripts/make_ed_figure_generation_convergence.py` | `docs/figure_reports/ed_generation_convergence.md` |
| `simplification/simplified_figure.png` | `scripts/combine_simplification_figure.py` | `docs/figure_reports/simplification_combined.md` |
| `simplification/simplified_metrics_grid.png` | `scripts/combine_simplification_metrics.py` | `docs/figure_reports/simplification_metrics_grid.md` |
| `simplification/simplified_mi_table.png` | `scripts/combine_simplification_mi_table.py` | `docs/figure_reports/simplification_mi_table.md` |
| SIR PNG | `scripts/generate_sir_figure.py` | `docs/figure_reports/sir_summary.md` |
| `simplified_model_example/` | `scripts/generate_model_simplification_figure.py` (`--save`) | `docs/figure_reports/simplified_model_example.md` |
