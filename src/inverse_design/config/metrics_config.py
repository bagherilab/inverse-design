# Metrics configuration organized by cellular and population levels
"""
Glioblastoma reference:
[1] Oraiopoulou, M-E., et al. In vitro/in silico study on the role of doubling time heterogeneity among primary 
    glioblastoma cell lines. BioMed research international 2017.1 (2017): 8569328.
[2] Seyfoori, Amir, et al. "Self-filling microwell arrays (SFMAs) for tumor spheroid formation." 
    Lab on a Chip 18.22 (2018): 3516-3528.

Melanoma reference:

Breast Cancer reference:
[1] NCI-60 Breast Cancer Cell Line Panel
[2] Conger, Alan D., and Marvin C. Ziskin. "Growth of mammalian multicellular tumor spheroids." 
    Cancer Research 43.2 (1983): 556-560.
[3] Isert, Lorenz, et al. "An in vitro approach to model EMT in breast cancer." 
    International Journal of Molecular Sciences 24.9 (2023): 7757.

Colon reference:
[1] NCI-60 Breast Cancer Cell Line Panel
[2] Brú, Antonio, et al. The universal dynamics of tumor growth. Biophysical journal 85.5 (2003): 2948-2961.
[3] Hassani, Iman, et al. Establishment of a tissue‐engineered colon cancer model for comparative analysis of 
cancer cell lines. Journal of Biomedical Materials Research Part A 112.2 (2024): 231-249.
"""


CELLULAR_METRICS = {
    "vol": {
        "name": "Cell Volume",
        "keys": ["vol"],
        "stats": ["mean", "std"],
        "description": "Average cell volume",
        "spatial": False,
        "unit": "um^3",
    },
    "states": {
        "name": "Cell States",
        "keys": ["states"],
        "stats": ["distribution"],
        "description": "Distribution of cell states",
        "spatial": False,
        "unit": "none",
    },
    "age": {
        "name": "Cell Age",
        "keys": ["age"],
        "stats": ["median", "std"],
        "description": "Age distribution of cells",
        "spatial": False,
        "unit": "minutes",
    },
    "cycle_length": {
        "name": "Cell Cycle Length",
        "keys": ["cycle_length"],
        "stats": ["mean", "std"],
        "description": "Average cell cycle length",
        "spatial": False,
        "unit": "hours",
    },
}

POPULATION_METRICS = {
    "n_cells": {
        "name": "Population Size",
        "keys": ["n_cells"],
        "stats": ["median", "std"],
        "description": "Total number of cells",
        "spatial": False,
        "unit": "none",
    },
    "symmetry": {
        "name": "Colony Symmetry",
        "keys": ["sym"],
        "stats": ["median", "std"],
        "description": "Colony symmetry measure",
        "spatial": True,
        "unit": "none",
    },
    "shannon": {
        "name": "Cell state diversity",
        "keys": ["shannon"],
        "stats": ["value"],
        "description": "Shannon diversity index",
        "spatial": False,
        "unit": "none",
    },
    "colony_diameter": {
        "name": "Colony Diameter",
        "keys": ["col_dia"],
        "stats": ["median", "std"],
        "description": "Colony diameter measurement",
        "spatial": True,
        "unit": "um",
    },
    "act_ratio": {
        "name": "Cell Activity Ratio",
        "keys": ["act_ratio"],
        "stats": ["median", "std"],
        "description": "number of active cells over total number of cells",
        "spatial": False,
        "unit": "none",
    },
    "activity": {
        "name": "Cell Activity",
        "keys": ["activity"],
        "stats": ["median", "std"],
        "description": "number of active (PROLIFERATIVE, MIGRATORY) over inactive (NECROTIC, APOPTOTIC)",
        "spatial": False,
        "unit": "none",
    },
}

SUMMARY_METRICS = {
    "doub_time": {
        "name": "Population Doubling Time",
        "keys": ["doub_time"],
        "stats": ["median", "std"],
        "description": "Population doubling time",
        "spatial": False,
        "unit": "hours",
    },
    "colony_growth": {
        "name": "Colony Growth Properties",
        "keys": ["col_g_rate", "col_g_r_squared"],
        "stats": ["median", "std"],
        "description": "Colony growth measurements",
        "spatial": True,
        "unit": "um^2/hour",
    },
}

DEFAULT_METRICS = {
    "vol": {"mean": 3680.141, "std": 161.766},
    "cycle_length": {"mean": 24.063, "std": 0.881},
    "n_cells": {"mean": 122.0, "std": 17.131},
    "symmetry": {"mean": 0.913, "std": 0.03},
    "shannon": {"mean": 0.781, "std": 0.058},
    "colony_diameter": {"mean": 300.0, "std": 24.661},
    "act_ratio": {"mean": 0.486, "std": 0.038},
    "activity": {"mean": 1.000, "std": 0.000},
    "doub_time": {"mean": 40.743, "std": 2.298},
    "colony_growth": {"mean": 31.402, "std": 3.137},
}

EXP_METRICS = {
    "Breast Cancer": {
        "doubling_time": {  # 1
            "min": 22.5,
            "25%": 33.65,
            "median": 45.5,
            "75%": 53.85,
            "max": 62,
            "std": 13.79,
        },
        "symmetry": {"mean": 0.806, "std": 0.067},  # 3
        "colony_growth": {  # um/day #2
            "mean": 18.3,
        },
    },
    "Glioblastoma_U87MG": {
        "doubling_time": {"mean": 30.8, "std": 4.32},  # 1
        "volume": {  # 1
            "mean": 5203.72,
        },
        "colony_growth": {  # um/day #1
            "mean": 35.408,
        },
        "symmetry": {"mean": 0.91, "std": 0.11},  # 2
    },
    "Glioblastoma_GBP03": {
        "doubling_time": {"mean": 25.4, "std": 0.5},  # 1
        "volume": {  # 1
            "mean": 3591.36,
        },
        "colony_growth": {  # um/day #1
            "mean": 56.8815,
        },
    },
    "Colon": {
        "doubling_time": {"mean": 22.41, "std": 4.26},  # 1
        "symmetry": {"mean": 0.42, "std": 0.02},  # 3 (HT-29)
        "colony_growth": {  # um/day #2
            "mean": 46.32,
        },
    },
}
