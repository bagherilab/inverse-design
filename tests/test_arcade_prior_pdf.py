"""prior_pdf range-checks every free parameter (Step 8 of the sampler)."""

import numpy as np
import pytest

from inverse_design.rf.arcade_example import prior_pdf

CELLULAR = {"perturbed_config": "cellular", "point_based": False}
COMBINED = {"perturbed_config": "combined", "point_based": False}
RANGES = {"A_MU": (0.0, 2.0), "B_MU": (10.0, 20.0), "C_MU": (1.0, 5.0)}
COLUMNS = list(RANGES)


@pytest.mark.parametrize("config", [CELLULAR, COMBINED])
def test_inside_returns_product_density(config):
    density = prior_pdf(np.array([1.0, 15.0, 3.0]), COLUMNS, RANGES, config)
    assert density == pytest.approx(1.0 / (2.0 * 10.0 * 4.0))


@pytest.mark.parametrize("config", [CELLULAR, COMBINED])
@pytest.mark.parametrize("index", [0, 1, 2])
def test_any_parameter_out_of_range_is_rejected(config, index):
    # Before the fix, the cellular branch checked only the first parameter and the
    # combined branch none of these, so indices 1-2 (cellular) and 0-2 (combined) passed.
    values = np.array([1.0, 15.0, 3.0])
    values[index] = RANGES[COLUMNS[index]][1] + 1.0
    assert prior_pdf(values, COLUMNS, RANGES, config) == 0.0


def test_degenerate_range_is_fixed_and_not_checked():
    ranges = {"A_MU": (0.0, 2.0), "FIXED": (3.0, 3.0)}
    assert prior_pdf(np.array([1.0, 99.0]), list(ranges), ranges, CELLULAR) == 0.5


def test_source_spacing_strings_are_checked():
    ranges = {"X_SPACING": (2, 6), "GLUCOSE_CONCENTRATION": (0.5, 1.5), "A_MU": (0.0, 2.0)}
    cols = list(ranges)
    ok = np.array(["0:4", 1.0, 1.0], dtype=object)
    assert prior_pdf(ok, cols, ranges, COMBINED) == pytest.approx(1.0 / (1.0 * 2.0))
    wide = np.array(["0:9", 1.0, 1.0], dtype=object)
    assert prior_pdf(wide, cols, ranges, COMBINED) == 0.0
    bad_cell = np.array(["0:4", 1.0, 3.0], dtype=object)
    assert prior_pdf(bad_cell, cols, ranges, COMBINED) == 0.0


def test_mean_only_profile_perturbs_compression_tolerance():
    from inverse_design.config.parameter_config_mu_only import PARAM_RANGES

    assert PARAM_RANGES["COMPRESSION_TOLERANCE"] == (3, 10)
    free = [name for name, (lo, hi) in PARAM_RANGES.items() if lo < hi]
    # Column order of the published mean-only particles (Sobol dimension order).
    assert free[:5] == [
        "CELL_VOLUME_MU",
        "NECROTIC_FRACTION",
        "ACCURACY",
        "COMPRESSION_TOLERANCE",
        "SYNTHESIS_DURATION_MU",
    ]
