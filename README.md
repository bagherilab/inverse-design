# A discovery of emergent drivers (DED) uncovers parameter hierarchies in spatiotemporal agent-based tumor models

**Po-Hao Chiu¹, Jacob I. Evarts², Jason Y. Cain¹, Neda Bagheri¹⋅²**
¹ Department of Chemical Engineering, University of Washington, Seattle, WA 98195, USA
² Department of Biology, University of Washington, Seattle, WA 98195, USA

[![CI](https://github.com/bagherilab/inverse-design/actions/workflows/ci.yml/badge.svg)](https://github.com/bagherilab/inverse-design/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/package-Poetry-60A5FA.svg)](https://python-poetry.org/)
[![Code style: Black](https://img.shields.io/badge/code%20style-Black-000000.svg)](https://github.com/psf/black)
[![Lint: Ruff](https://img.shields.io/badge/lint-Ruff-46A2F1.svg)](https://docs.astral.sh/ruff/)
[![Lint: Pylint](https://img.shields.io/badge/pylint-%E2%89%A58.0-yellowgreen.svg)](https://pylint.pycqa.org/)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-0A9EDC.svg)](https://docs.pytest.org/)
[![Coverage](https://codecov.io/gh/bagherilab/inverse-design/graph/badge.svg)](https://codecov.io/gh/bagherilab/inverse-design)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19963226.svg)](https://doi.org/10.5281/zenodo.19963226)

---

Approximate Bayesian Computation (ABC) for agent-based models. Estimates ABM parameters that drive desired emergent behaviors. Companion code for the paper.

## 0. What this project does

- **Inverse parameter estimation** for stochastic ABMs via rejection ABC and ABC-SMC with random forests (RF / DRF).
- **Forward models supported**: ARCADE (Java JAR cell agent simulator), SIR (Python ABM), BDM (birth-migration-death demo), Lotka-Volterra (benchmark).
- **Workflow**: prior → simulate → metric distance → posterior → analysis (MI, Sobol, redundancy) → publication figures.
- **Outputs**: posterior parameter distributions, sensitivity / mutual-information rankings, simplified models, paper figures.

## Table of contents

0. [What this project does](#0-what-this-project-does)
1. [Setup / Installation](#1-setup--installation)
2. [CLI: SIR models](#2-cli-sir-models)
3. [CLI: ARCADE input generation + simulation](#3-cli-arcade-input-generation--simulation)
4. [Analyze ARCADE and SIR results](#4-analyze-arcade-and-sir-results)
5. [Visualize paper figures](#5-visualize-paper-figures)
6. [License](#6-license)
7. [Reference](#7-reference)

### Path conventions

Replace these placeholders with paths on your machine. No path is hard-coded.

| Placeholder | Meaning | Example |
|---|---|---|
| `$ARCADE_INPUTS` | Generated XML input bundles | `/data/arcade_inputs` |
| `$ARCADE_OUTPUT` | ARCADE simulation outputs | `/data/arcade_output` |
| `$RESULTS` | Figures + analysis artifacts | `./results` |
| `$REPO` | Repo root | `./inverse_design` |

Set once per shell:

```bash
export ARCADE_INPUTS=/path/to/arcade_inputs
export ARCADE_OUTPUT=/path/to/arcade_output
export RESULTS=$PWD/results
```

---

## 1. Setup / Installation

Repo uses **Poetry** (`.venv/` in project root, see `poetry.toml`). Java 8+ required for ARCADE JARs.

```bash
git clone <repo-url> && cd inverse_design
poetry install --with dev
poetry shell
```

Sanity check:

```bash
poetry run python -c "import sys; print(sys.executable)"   # must point to .venv
poetry run pytest tests/ -v
```

Lint (matches CI):

```bash
poetry run black --check --line-length 100 src tests
poetry run ruff check src tests
poetry run pylint src/inverse_design --fail-under=8.0
```

### ARCADE JAR setup

ARCADE JARs are **not bundled** in this repo (too large for git). Download from Zenodo:

1. Go to <https://doi.org/10.5281/zenodo.19963226>.
2. Download the JAR(s) you need:
   - `arcade-logging-necrotic.jar` — w/ necrotic-cell logging
3. Place them at:
   ```
   $REPO/src/inverse_design/models/arcade-logging-necrotic.jar
   ```
4. Verify Java:
   ```bash
   java -version   # need Java 8+
   java -jar $REPO/src/inverse_design/models/arcade-logging-necrotic.jar --help
   ```

ARCADE source: <https://github.com/bagherilab/ARCADE>.

---

## 2. CLI: SIR models

### 2.1 SIR forward demo

Single SIR ABM run + S/I/R trajectory plot. No CLI args.

```bash
poetry run python -m inverse_design.models.sir.sir
```

**Expected output:**
- `sir_model.png` — S/I/R curves vs time (saved in CWD).
- stdout: peak infection day, final R count.

---

### 2.2 ABC-SMC-RF inference on SIR

Hydra-driven. Configs live at `src/inverse_design/conf/`. Override any field on the CLI with `key=value`.

**Tunable hyperparameters** (override via Hydra `key=value`):

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `abc.sobol_power` | int | 9 | `2^p` Sobol samples per iteration |
| `abc.epsilon` | float | 0.033 | Acceptance distance threshold |
| `abc.n_iterations` | int | 5–10 | SMC iterations |
| `abc.n_trees` | int | 50–500 | Trees per RF |
| `abc.min_samples_leaf` | int | 5 | Min leaf size |
| `abc.subsample_ratio` | float | 0.5 | Bootstrap subsample fraction |
| `abc.rf_type` | str | `DRF` | `RF` (regressor) or `DRF` (distributional) |
| `abc.criterion` | str | `CART` | RF split criterion |
| `abc.random_state` | int | 42 | Seed |
| `abc.parameter_ranges.<P>.{min,max}` | float | per-config | Override prior for parameter `P` |
| `abc.targets.<i>.{metric,value,weight}` | mixed | per-config | Override target metric `i` |

**Example — default run:**

```bash
poetry run python -m inverse_design.examples.example_with_model model=sir
```
Expected output:
- `$RESULTS/SIR/iter_*/all_param_df.csv` — accepted samples per SMC iteration.
- `$RESULTS/SIR/final_metrics.csv`, `final_parameters.csv`.
- stdout: per-iteration acceptance rate, distance threshold ε.

**Example — tighter tolerance + more trees:**

```bash
poetry run python -m inverse_design.examples.example_with_model \
    model=sir \
    abc.epsilon=0.01 \
    abc.n_trees=500 \
    abc.sobol_power=10
```
Expected output: same artifacts as above; finer posterior, ~5–10× longer runtime, narrower KDE in downstream figures.

**Example — override target infection peak:**

```bash
poetry run python -m inverse_design.examples.example_with_model \
    model=sir \
    abc.targets.0.value=0.4 \
    abc.targets.0.weight=2.0
```
Expected output: posterior shifts to parameter regimes producing 40% peak infection; weighted distance prioritizes that target.

---

## 3. CLI: ARCADE input generation + simulation

### 3.1 Generate ARCADE input bundles

Sobol-sampled parameter sets → ARCADE XML inputs.

**Args** (`inverse_design.examples.generate_combined_inputs`):

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `--profile` | `generic`\|`s3-coculture` | `generic` | Parameter profile |
| `--template-path` | path | — | XML template |
| `--output-dir` | path | — | Where bundle lands |
| `--sobol-power` | int | 8 | `2^p` samples |
| `--source-mode` | `grid`\|`point` | `grid` | Source perturbation geometry |
| `--radius` | float | 8 | Tumor radius |
| `--margin` | float | 2 | Padding → `radius_bound` |
| `--hex-size` | float | 30 | Hex side scale |
| `--y-interval` | int | 4 | Y spacing for source sampling |
| `--seed` | int | 42 | Sobol seed |
| `--params` | list | (all) | Subset of parameters to perturb |
| `--params-file` | path | — | JSON parameter spec / range overrides |
| `--chain-based` | flag | off | Chain-based combined generation |
| `--lr-summary-json` | path | — | Required w/ `--chain-based` |
| `--peak-name` | str | — | Required w/ `--chain-based` |
| `--list-params` | flag | off | Print available params + exit |
| `--continuous-samples-per-discrete-combo` | int | — | s3-coculture only |

**Example — default 256 samples:**

```bash
poetry run python -m inverse_design.examples.generate_combined_inputs \
    --profile generic \
    --template-path $REPO/src/inverse_design/sample_inputs/template.xml \
    --output-dir $ARCADE_INPUTS/run_default \
    --sobol-power 8
```
Expected output:
- `$ARCADE_INPUTS/run_default/iter_0/*.xml` — 256 ARCADE input files.
- `$ARCADE_INPUTS/run_default/iter_0/all_param_df.csv` — parameter table.
- `$ARCADE_INPUTS/run_default/config.json` — generation provenance.

**Example — high-resolution + parameter subset:**

```bash
poetry run python -m inverse_design.examples.generate_combined_inputs \
    --profile generic \
    --template-path $REPO/src/inverse_design/sample_inputs/template.xml \
    --output-dir $ARCADE_INPUTS/run_high_res \
    --sobol-power 10 \
    --params CELL_VOLUME_MU AFFINITY GLUCOSE_UPTAKE_RATE_MU \
    --seed 7
```
Expected output: 1024 inputs, only 3 parameters varied; remaining params fixed at defaults.

**Example — chain-based (uses LR predictor summary):**

```bash
poetry run python -m inverse_design.examples.generate_combined_inputs \
    --profile generic \
    --template-path $REPO/src/inverse_design/sample_inputs/template.xml \
    --output-dir $ARCADE_INPUTS/run_chain \
    --sobol-power 9 \
    --chain-based \
    --lr-summary-json $RESULTS/lr_summary_breast.json \
    --peak-name peak_2
```
Expected output: 512 inputs sampled along the LR-predicted parameter chain for `peak_2`; reduces effective dimensionality.

---

### 3.2 Run ARCADE simulations

ARCADE is a Java JAR. JARs ship under `src/inverse_design/models/`:
- `arcade-share_space.jar` — main
- `arcade-logging-necrotic.jar` — w/ necrotic logging
- `arcade-test-cycle-fix-affinity.jar` — affinity-fixed variant

**Direct invocation:**

```bash
ARCADE_JAR=$REPO/src/inverse_design/models/arcade-share_space.jar
java -jar "$ARCADE_JAR" patch \
    $ARCADE_INPUTS/run_default \
    $ARCADE_OUTPUT/run_default
```
Expected output:
- `$ARCADE_OUTPUT/run_default/<sample_id>_<seed>_<timestamp>.CELLS.json`
- `*.LOCATIONS.json`, `*.PARAMETERS.json` per seed
- stdout: tick progression, cell counts.

**Batch driver (Python):**

```bash
poetry run python -m inverse_design.examples.run_simulations \
    --input-dir $ARCADE_INPUTS/run_default \
    --output-dir $ARCADE_OUTPUT/run_default \
    --jar $REPO/src/inverse_design/models/arcade-share_space.jar
```
Expected output: same as direct, plus per-input launch log; can parallelize seeds.

---

### 3.3 ABC-SMC-RF inference on ARCADE

Main inference loop. JSON-config driven.

**Args** (`inverse_design.rf.arcade_example`):

| CLI arg | Type | Required | Meaning |
|---|---|---|---|
| `--config` | path | yes | JSON config (hyperparameters below) |
| `--target` | path | yes | JSON `{metric: value}` targets |

**Tunable hyperparameters** (in `--config` JSON):

| Key | Default | Meaning |
|---|---|---|
| `sobol_power` | 9 | `2^p` samples per iteration |
| `n_iterations` | 5 | SMC iterations |
| `rf_type` | `"DRF"` | `"RF"` or `"DRF"` |
| `n_trees` | 50 | Trees per forest |
| `min_samples_leaf` | 5 | Min leaf size |
| `subsample_ratio` | 0.5 | Bootstrap fraction |
| `criterion` | `"CART"` | RF split criterion |
| `random_state` | 42 | Seed |
| `simplify_model` | `true` | Enable redundancy-based simplification |
| `correlation_threshold` | 0.8 | LR redundancy cutoff |
| `n_group` | 3 | Grouping for parameter clustering |
| `n_min_sample` | 5 | Min accepted samples per iter |
| `radius` / `margin` / `hex_size` | 10 / 2 / 30 | Source geometry |
| `base_input_dir` | path | Where input bundles live |
| `base_output_dir` | path | Where ARCADE writes |
| `base_dir` | path | Run-specific working dir |
| `base_s3_dir` | str/null | Optional S3 prefix for upload |

**Example — minimal breast scenario:**

```bash
poetry run python -m inverse_design.rf.arcade_example \
    --config $REPO/configs/abc_smc_rf_n512_breast.json \
    --target $REPO/src/inverse_design/inputs/abc_smc_rf_n512_combined_grid_breast/targets.json
```
Expected output:
- `$ARCADE_OUTPUT/<run>/iter_k/all_param_df.csv` — parameters per iter `k`
- `iter_k/final_metrics.csv` — simulated metrics
- `iter_k/final_parameters.csv` — accepted posterior samples
- stdout: `Running ABC-SMC-RF iteration k/N`, acceptance counts

**Example — high-fidelity DRF:**

```json
// configs/high_fidelity.json
{
  "sobol_power": 10,
  "n_iterations": 8,
  "rf_type": "DRF",
  "n_trees": 500,
  "min_samples_leaf": 3,
  "simplify_model": true,
  "correlation_threshold": 0.75,
  "base_input_dir": "$ARCADE_INPUTS",
  "base_output_dir": "$ARCADE_OUTPUT",
  "base_dir": "$ARCADE_OUTPUT/breast_high_fidelity"
}
```

```bash
poetry run python -m inverse_design.rf.arcade_example \
    --config configs/high_fidelity.json \
    --target $REPO/src/inverse_design/inputs/abc_smc_rf_n1024_combined_grid_breast_mmd/targets.json
```
Expected output: 8 iterations × 1024 sims; tighter posterior; ~10–20× compute vs default.

**Example — disable simplification (full parameter space):**

```bash
poetry run python -m inverse_design.rf.arcade_example \
    --config configs/full_param.json \
    --target targets.json
```
With `"simplify_model": false`, retains all parameters; useful for sensitivity baseline.

**Pre-built input bundles** under `src/inverse_design/inputs/`:

| Bundle | Purpose |
|---|---|
| `abc_smc_rf_n1024_combined_grid_breast_mmd` | MMD-distance, breast |
| `abc_smc_rf_n512_combined_grid_simplified_breast_corr0.75` | Simplified, ρ ≥ 0.75 |
| `abc_smc_rf_n1024_combined_grid_glioblastoma` | Glioblastoma |
| `abc_smc_rf_n4_source` | Source perturbation tests |

**On-disk layout** centralized in `inverse_design.io.ArcadeRunLayout` (`src/inverse_design/io/arcade_layout.py`).

---

## 4. Analyze ARCADE and SIR results

### 4.1 Aggregate metrics across seeds

```bash
poetry run python -m inverse_design.analyze.save_aggregated_results \
    --base-dir $ARCADE_OUTPUT/run_default
```
Expected output: `final_metrics.csv` aggregating per-seed metrics; outliers stripped.

### 4.2 Mutual-information sensitivity

**Tunable** (edit constants at top of script): `TARGET_METRICS`, `DROP_COLS`, `DATA_DIR`.

```bash
poetry run python scripts/compute_mi_sensitivity.py
poetry run python scripts/make_ed_table_mi_sensitivity.py
```
Expected output:
- `mi_sensitivity.csv` — KSG MI per (parameter, metric)
- `$RESULTS/figures/simplification/ed_table_mi_sensitivity.{png,pdf}`

### 4.3 Sobol vs MI comparison

```bash
poetry run python scripts/compute_sobol_comparison.py
poetry run python scripts/make_ed_table_sobol_mi.py
```
Expected output:
- `sobol_mi_comparison.csv`
- `$RESULTS/figures/simplification/ed_table_sobol_mi.{png,pdf}`

### 4.4 Generation/convergence errors

```bash
poetry run python scripts/compute_generation_errors.py
poetry run python scripts/make_ed_figure_generation_convergence.py
```
Expected output: `$RESULTS/figures/error_iteration/convergence.{png,pdf}` showing error vs SMC iteration.

### 4.5 Build vis cache (optional speedup)

| Arg | Type | Default | Meaning |
|---|---|---|---|
| `--method` | str | — | LR method (`linear`, etc.) |
| `--peak-idx` | int | 2 | Zero-based peak index |
| `--cache-root` | path | `$RESULTS/vis_cache` | Override cache dir |

```bash
poetry run python scripts/build_vis_cache.py --method linear --peak-idx 2
```
Expected output: `$RESULTS/vis_cache/linear_p3.json`. Env: `VIS_USE_CACHE=0` disables; `VIS_CACHE_ROOT=...` overrides.

### 4.6 Calculate experimental data (glioblastoma)

```bash
poetry run python scripts/calculate_exp_data.py
```
Expected output: `exp_data.csv` with measured colony growth rate, doubling time per replicate.

### Key analysis modules

| Module | Role |
|---|---|
| `analyze/parameter_importance.py` | KSG/Gaussian MI + ranking heatmaps |
| `analyze/sensitivity_analysis.py` | SALib Sobol + RF importance + MI |
| `analyze/simplify_model.py` | SVD posterior redundancy → simplified model |
| `analyze/abc_analysis.py` | Errors vs samples / acceptance / #params |
| `analyze/mi_suggestions.py` | MI suitability diagnostics |
| `analyze/lr_predictor.py` | Linear correlation chains + validation |
| `analyze/abc_metrics_comparison.py` | Iteration / scenario comparison plots |
| `analyze/analyze_sir.py` | SIR sensitivity heatmaps + joint posteriors |

---

## 5. Visualize paper figures

All paper figures regenerate under `$RESULTS/figures/`.

**One-shot regenerate-all:**

```bash
bash scripts/regenerate_all_figures.sh
```

**Per-figure scripts:**

| Figure | Script | Output dir |
|---|---|---|
| SIR summary (5-panel) | `scripts/generate_sir_figure.py` | `$RESULTS/figures/SIR/` |
| Model simplification | `scripts/generate_model_simplification_figure.py` | `$RESULTS/figures/simplification/` |
| Simplification (isolated peak) | `scripts/combine_simplification_figure_isolated.py` | `$RESULTS/figures/simplification/` |
| Simplification metrics panel | `scripts/combine_simplification_metrics.py` | `$RESULTS/figures/simplification/` |
| Simplification MI table | `scripts/combine_simplification_mi_table.py` | `$RESULTS/figures/simplification/` |
| Glioblastoma fit | `scripts/generate_gliob_fit_figures.py` | `$RESULTS/figures/gliob/` |
| Fit vs experiment | `scripts/combine_fit_exp_figure.py` | `$RESULTS/figures/fit_exp/` |
| Equifinality | `scripts/combine_equifinality_figure.py` | `$RESULTS/figures/simplification/` |
| Feasible metric ranges | `scripts/combine_feasible_metrics_ranges.py` | `$RESULTS/figures/feasible/` |
| Generation convergence (ED) | `scripts/make_ed_figure_generation_convergence.py` | `$RESULTS/figures/error_iteration/` |
| Posterior overview / KDE / PCA | `scripts/vis_sandbox_main.py` | `$RESULTS/figures/` |

Common env knobs:
- `MPLCONFIGDIR` — matplotlib cache dir (default `/tmp/inverse_design_matplotlib`)
- `VIS_USE_CACHE`, `VIS_CACHE_ROOT` — vis cache control

---

## 6. License

MIT. See `LICENSE`.

ARCADE simulator licensed separately — see [bagherilab/ARCADE](https://github.com/bagherilab/ARCADE).

---

## 7. Reference

### Citation

If you find our work useful in your research or if you use parts of this code please consider citing our paper:

Chiu, P.-H., Evarts, J.I., Cain, J.Y., Bagheri, N. *A discovery of emergent drivers (DED) uncovers parameter hierarchies in spatiotemporal agent-based tumor models.* (Manuscript). Zenodo: <https://doi.org/10.5281/zenodo.19963226>

```bibtex
@article{chiu2026ded,
  title   = {A discovery of emergent drivers (DED) uncovers parameter hierarchies in spatiotemporal agent-based tumor models},
  author  = {Chiu, Po-Hao and Evarts, Jacob I. and Cain, Jason Y. and Bagheri, Neda},
  year    = {2026},
  doi     = {10.5281/zenodo.19963226},
  url     = {https://doi.org/10.5281/zenodo.19963226},
  note    = {Department of Chemical Engineering and Department of Biology, University of Washington, Seattle, WA 98195, USA}
}
```

Companion repos:
- ARCADE: <https://github.com/bagherilab/ARCADE>
- BDM demo: <https://github.com/pohaoc2/birth-migration-death>

Method references:

[^1]: Beaumont, M. A., Zhang, W., Balding, D. J., & Rannala, B. (2009). Approximate Bayesian computation in population genetics. *Genetics*, 182(2), 257–270.
[^2]: Raynal, L., et al. (2019). ABC random forests for Bayesian parameter inference. *Bioinformatics*, 35(10), 1720–1728.
[^3]: Sisson, S. A., Fan, Y., & Beaumont, M. (2018). *Handbook of approximate Bayesian computation.* CRC Press.
[^4]: Saltelli, A., et al. (2010). Variance-based sensitivity analysis of model output. *Computer Physics Communications*, 181(2), 259–270.
[^5]: Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). Estimating mutual information. *Phys. Rev. E*, 69(6), 066138.
