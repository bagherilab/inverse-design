"""Parameter space for S3 CAR T-cell co-culture input generation."""

PROFILE_NAME = "s3-coculture"

# Mixed discrete/continuous parameter specifications for the 11-D S3 design space.
# Discrete values are sampled by Sobol binning into the listed choices.
S3_COCULTURE_PARAM_SPECS = {
    "CAR_T_DOSE": {
        "type": "choice",
        "values": [250, 500, 1000],
        "description": "Total CAR T cells added at treatment time.",
    },
    "CD4_PERCENT": {
        "type": "choice",
        "values": [100, 75, 50, 25, 0],
        "description": "Percent of the CAR T dose assigned to CD4 cells.",
    },
    "CAR_AFFINITY": {
        "type": "choice",
        "values": [1e-6, 1e-7, 1e-8, 1e-9],
        "description": "CAR affinity applied to both CD4 and CD8 populations.",
    },
    "CANCER_ANTIGENS": {
        "type": "choice",
        "values": [100, 500, 1000, 5000, 10000],
        "description": "Average cancer-cell CAR antigen expression.",
    },
    "HEALTHY_ANTIGENS": {
        "type": "choice",
        "values": [0, 100],
        "description": "Average healthy-cell CAR antigen expression.",
    },
    "GLUCOSE_CONCENTRATION": {
        "type": "range",
        "bounds": (0.001, 0.01),
        "description": "Initial and source glucose concentration.",
    },
    "OXYGEN_CONCENTRATION": {
        "type": "range",
        "bounds": (0.0, 70.0),
        "description": "Initial and source oxygen concentration.",
    },
    "SYNTHESIS_TIME_T": {
        "type": "range",
        "bounds": (180.0, 720.0),
        "description": "Mean T-cell proliferation synthesis duration.",
    },
    "BOUND_TIME": {
        "type": "range",
        "bounds": (180.0, 720.0),
        "description": "Mean CAR T-cell target binding duration.",
    },
    "CANCER_METABOLIC_PREFERENCE_SCALE": {
        "type": "range",
        "bounds": (1.0, 3.0),
        "description": "Scale factor applied to cancer metabolic preference.",
    },
    "CANCER_MIGRATORY_THRESHOLD_SCALE": {
        "type": "range",
        "bounds": (0.1, 1.0),
        "description": "Scale factor applied to cancer migratory threshold.",
    },
}

S3_COCULTURE_PARAM_DEFAULTS = {
    "CAR_T_DOSE": 500,
    "CD4_PERCENT": 50,
    "CAR_AFFINITY": 1e-6,
    "CANCER_ANTIGENS": 1000,
    "HEALTHY_ANTIGENS": 100,
    "GLUCOSE_CONCENTRATION": 0.005,
    "OXYGEN_CONCENTRATION": 100.0,
    "SYNTHESIS_TIME_T": 360.0,
    "BOUND_TIME": 360.0,
    "CANCER_METABOLIC_PREFERENCE_SCALE": 1.5,
    "CANCER_MIGRATORY_THRESHOLD_SCALE": 0.5,
}

PARAMETER_LIST = list(S3_COCULTURE_PARAM_SPECS)
