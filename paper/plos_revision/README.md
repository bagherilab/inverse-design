# PLOS Computational Biology revision: analysis scripts

Scripts that produce the numbers, figures, and tables of the revised manuscript
"Discovery of emergent drivers" (DED pipeline) and its Supporting Information.
The pipeline itself lives in `src/inverse_design/`; the main-figure scripts for
Fig 2, Fig 3, S5 Fig, and S7 Fig live in `scripts/` and
`src/inverse_design/figures/`.

## Layout expected by the scripts

The scripts were written for a workspace in which the manuscript directory and
this repository sit side by side. Recreate it as follows:

```bash
mkdir ded_workspace && cd ded_workspace
git clone https://github.com/bagherilab/inverse-design inverse_design
cp -r inverse_design/paper/plos_revision manuscript
# Extract the Zenodo bundle's analysis_outputs/ and source data into manuscript/
tar -xzf inverse_design_paper_v1.2.tar.gz
cp -r inverse_design_paper_v1.2/analysis_outputs manuscript/
cd manuscript
```

Scripts resolve `manuscript/` as their repository root and `../inverse_design`
as the pipeline source. Two environment variables override the original host
paths:

| Variable | Meaning | Default (original host) |
|---|---|---|
| `DED_HPC_ROOT` | Directory holding `ARCADE_OUTPUT/`, `pm50/`, `sa_ess/` on the cluster | `/gscratch/cheme/chiu` |
| `DED_INVERSE_DESIGN` | Fallback path to this repository when `../inverse_design` is absent | `/home/pohaoc2/UW/bagherilab/inverse_design` |

`hpc/` holds the Slurm drivers used on the University of Washington Hyak
cluster. Account names and paths in them are site-specific examples.

- `hpc/pm50_reduced/`: the reduced-configuration campaign behind Fig 4 and
  S10 Fig (`make_pm50_configs.py` writes the 28 chain configs in `configs/`).
  Generations 0-3 use `src/inverse_design/sample_inputs/sample_combined_v3_5seed.xml`
  (five seeds) and generation 4 the ten-seed template;
  `set_template_for_generation.py` makes the switch.
- `hpc/r46_corrected/`: the 12-arm `bench_pm50_corrected_ml_*` selector
  campaign (campaign, selection manifest, and per-arm configs).

## Manuscript item to script

| Item | Script (in `analysis_scripts/` unless noted) |
|---|---|
| Fig 2 | `scripts/generate_sir_figure.py` (repository root) |
| Fig 3 inputs, P1–P4 assignments | `export_pm50_fig3_inputs.py`, `freeze_pm50_canonical_assignments.py`, `summarize_fig3_pm50.py`; figure: `src/inverse_design/figures/combine_fit_exp_figure.py` |
| Fig 4, S10 Fig, S1 Data, reduced-configuration MAEs and rank correlations | `regenerate_simplification_matched.py` |
| S1 Fig (extended SIR endpoint) | `sir_r212_extinction.py`, then `reviewer_sir_common_kde.py --render-extinction-only` |
| S2 Fig, S3 Fig, S1 Table | `sir_r212_replicate.py`, `reviewer_sir_common_kde.py`, `sir_mode_target_errors.py` |
| S4 Fig | `regenerate_s3_empirical_stopping.py`; ARCADE values: `arcade_generation_errors.py`; per-generation MAE/ESS table on the cluster: `sweep_report.py` |
| S5 Fig | `scripts/combine_feasible_metrics_ranges.py` (repository root) |
| S6 Fig | `reviewer_seed_precision.py` |
| S7 Fig | `scripts/generate_gliob_fit_figures.py`, `scripts/generate_legacy_gliob_histograms.py` (repository root) |
| S8 Fig | `reviewer_arcade_timecourse.py` |
| S9 Fig, S2 Data | `regenerate_pm50_s2_s9.py` |
| S11 Fig | `bandwidth_stability_peaks.py` |
| S2 Table | `ess_vs_weighting.py`, `sa_report.py`, `sa_bias.py` |
| S3 Table | `sir_benchmark.py` |
| S6 Table | `family_geometry_pm50.py --gen 4` |
| S7 Table | `r46_mode_local_pm50.py`, `r46_heldout.py` |
| S10 Table | `smc_weight_replay_pm50.py` (set `DED_PM50_CONFIG` to the archived `pm50_parameter_config.py`) |
| S11 Table | `canonical_pm50_mi_k_stability.py` |
| S12 Table, bootstrap recovery | `kde_bandwidth_cv.py`, `kde_bootstrap.py` |
| S13 Table | `r214_lineage_geometry.py` |
| MI permutation tests (Results) | `canonical_pm50_mi_permutation.py` |

`reviewer_figure_style.py` holds the shared PLOS figure style.

## Data

Particle sets, summary metrics, target metrics, and per-figure source data are
archived on Zenodo (DOI to be added on release).
