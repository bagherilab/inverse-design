import numpy as np
import pandas as pd

from inverse_design.analyze.sensitivity_analysis import perform_mi_analysis


def _make_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 3))
    y = x[:, 0] + 0.1 * rng.standard_normal(n)
    param_df = pd.DataFrame(x, columns=["p1", "p2", "p3"])
    metrics_df = pd.DataFrame({"metric": y})
    return param_df, metrics_df


def test_n_neighbors_param_accepted():
    param_df, metrics_df = _make_data()

    result_k3 = perform_mi_analysis(param_df, metrics_df, ["p1", "p2", "p3"], n_neighbors=3)
    result_k5 = perform_mi_analysis(param_df, metrics_df, ["p1", "p2", "p3"], n_neighbors=5)

    assert result_k3["metric"]["MI"].argmax() == 0
    assert result_k5["metric"]["MI"].argmax() == 0


def test_n_neighbors_default_is_3():
    param_df, metrics_df = _make_data()

    result_default = perform_mi_analysis(param_df, metrics_df, ["p1", "p2", "p3"])
    result_explicit = perform_mi_analysis(param_df, metrics_df, ["p1", "p2", "p3"], n_neighbors=3)

    np.testing.assert_array_equal(result_default["metric"]["MI"], result_explicit["metric"]["MI"])
