# Architecture notes (refactor log)

## Phase 0 — package naming

**Decision:** Keep the existing top-level package name `analyze/`. Add new subfolders under it (`analyze/core/`, `analyze/scenarios/`) rather than introducing a parallel `analysis/` tree, to avoid two synonymous roots.

## Phase 3 — `inverse_design.io`

- **`ArcadeRunLayout`** (`inverse_design.io.arcade_layout`): resolves standard ARCADE ABC-SMC iteration paths (`iter_k/final_metrics.csv`, `all_param_df.csv`, run-level `targets.json`).
- **First consumer:** `inverse_design.rf.abc_smc_rf_arcade.ABCSMCRF._analyze_simulation_results` uses `ArcadeRunLayout` for metrics/param CSV paths; iteration I/O roots use `Path(base) / dir_postfix` instead of string concatenation.

## Phase 4 — `analyze/scenarios/`

- **`registry.yaml`** + **`load_registry`**, **`combined_grid_n512_breast`** (`inverse_design.analyze.scenarios`): canonical N512 combined-grid output roots previously duplicated in `ABCMetricsComparison` and `parameter_importance` `__main__` blocks.
- **`vis_simplified.main`** still uses a different `ext_dendrogram_threshold_dirs` set; left local until aligned with YAML.

## Phase 5 — `analyze/core/pca_peaks.py`

- **`perform_pca_and_find_peaks`**, **`find_density_peaks`**: moved from `vis/sandbox.py`; `vis.sandbox` re-imports them for backward compatibility. Library code should import from **`inverse_design.analyze.core`** (or **`inverse_design.analyze.core.pca_peaks`**).

## Phase 6 — `plotting/theme.py`

- **`publication_rc_params`**, **`apply_publication_style`**: shared sans-serif / black-axes rc defaults (`tick_major_width` optional).
- **`apply_journal_style_nature_baseline`**: spine/tick/legend defaults for `vis_metrics_hyperparam`.
- **Theme migrations:** most `plt.rcParams.update` publication blocks → `apply_publication_style`; `parameter_importance`, `sensitivity_*`, `analyze_param_correlation`, `plot_dendrogram_pairwise`, `vis_experiment_data`, `vis_simulation_outcome.main`, `analyze_sir`, `models/sir.visualize_results`, inner helpers in `vis/sandbox`, `vis_simplified` (module level).

## Phase 6b — `plotting/colormap.py`

- **`DEFAULT_METRIC_COLORS`**, **`metric_color`**, **`merged_metric_colors`**: single source for ARCADE metric hexes (was duplicated in `ABCMetricsComparison` / `analyze_aggregated_results`; `vis_metrics_hyperparam` `nat_colors` builds from merge + legacy `peak_I` / `area_I` aliases).
- **`PEAK_COLORS`**, **`peak_color`**, **`PCA_CLUSTER_COLORS`**, **`pca_cluster_color`**: PCA peak assignment (used in `analyze/core/pca_peaks.py` and `vis.sandbox` / `vis_simplified` peak lists).
- **`LIGHT_PEAK_FILL_COLORS`**, **`light_peak_fill`**: soft fills for multi-peak bar panels in sandbox.
- **`sir_compartment_color`**, **`metric_colors_for_keys`**: SIR and ordered metric lists (`vis_simulation_outcome`).
