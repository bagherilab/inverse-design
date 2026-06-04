# Full Parameter List — Supplementary Tables S1–S3

## Table S1 — CARCADE Model Parameters

### Diffusion & Environment
- DIFFUSIVITY_IL2 = 10 μm²/s
- CONCENTRATION_IL2 = 0 molecules/cm³

### Seeding / Spatial Constraints
- MAX_DAMAGE_SEED = 1e7 (unitless)
- MIN_RADIUS_SEED = 2 μm

### Cell Lifecycle
- DEATH_AGE_AVG_T = 6 weeks (60480 min)
- T_CELL_AGE_MIN = 0 min
- T_CELL_AGE_MAX = 1 week (10080 min)

### Cell Cycle & Timing
- SYNTHESIS_TIME_T = 360 min
- SYNTHESIS_TIME_T_RANGE = 0 min

### Cell Interaction (Binding)
- BOUND_TIME = 360 min
- BOUND_TIME_RANGE = 0 min

### Cell Size
- T_CELL_VOL_AVG = 175 μm³
- T_CELL_VOL_RANGE = 10 μm³

### Metabolism & Activation
- ACTIVE_ENERGY = 0.002 fmol ATP/μm³/cell/min
- FRAC_MASS_ACTIVE = 0.25
- META_PREF_IL2 = 0.1
- META_PREF_ACTIVE = 0.3

### Glucose Uptake
- GLUC_UPTAKE_RATE_IL2 = 0.56 fmol glucose/μm²/cell/min/M
- GLUC_UPTAKE_RATE_ACTIVE = 3.78 fmol glucose/μm²/cell/min/M

### Timing Delays
- META_SWITCH_DELAY = 60 min

### Signaling (IL-2 kinetics)
- IL2_RECEPTORS = 2e3 receptors/cell
- IL2_BINDING_ON_RATE_MIN = 0.038193 μm³/(molecule·min)
- IL2_BINDING_ON_RATE_MAX = 3.115 μm³/(molecule·min)
- IL2_BINDING_OFF_RATE = 0.015 min⁻¹
- K_CONVERT = 1e-3 s⁻¹
- K_REC = 1e-5 s⁻¹

### IL-2 Production
- IL2_PROD_RATE_IL2 = 16.62 molecules/cell/min
- IL2_PROD_RATE_ACTIVE = 293.27 molecules/cell/min

### Additional Timing
- IL2_SYNTHESIS_DELAY = 180 min
- GRANZ_SYNTHESIS_DELAY = 15 min


## Table S2 — Monoculture Parameter Ranges

- CAR T-cell dose: [250, 500, 1000]
- CD4+:CD8+ ratio: [100:0, 75:25, 50:50, 25:75, 0:100]
- CAR affinity (M): [1e-6, 1e-7, 1e-8, 1e-9]
- Cancer antigens (per cell): [100, 500, 1000, 5000, 10000]

Total combinations: 300
Total simulations: 3000


## Table S3 — Co-culture Parameter Ranges

- CAR T-cell dose: [250, 500, 1000]
- CD4+:CD8+ ratio: [100:0, 75:25, 50:50, 25:75, 0:100]
- CAR affinity (M): [1e-6, 1e-7, 1e-8, 1e-9]
- Cancer antigens (per cell): [100, 500, 1000, 5000, 10000]
- Healthy cell antigens: [0 (ideal), 100 (realistic)]

Total combinations: 600
Total simulations: 6000
