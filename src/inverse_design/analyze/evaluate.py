import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from inverse_design.metrics.metrics import MetricsFactory


def estimate_pdfs(params: List[Dict]) -> Dict:
    """
    Estimate PDFs for each parameter independently and jointly using multivariate normal
    Args:
        params: List of parameter dictionaries
    Returns:
        Dictionary mapping each parameter name to a dict containing:
            'pdf': KDE object for the parameter
            'mean': Mean value of the parameter
            'cov': Variance of the parameter
    """
    if not params:
        return {}

    # Extract parameter values into arrays
    param_names = list(params[0].keys())
    param_arrays = {param: np.array([p[param] for p in params]) for param in param_names}

    # Compute statistics for each parameter
    result = {}
    for param_name, values in param_arrays.items():
        try:
            kde = stats.gaussian_kde(values)
            result[param_name] = {"kde": kde, "mean": np.mean(values), "cov": np.var(values)}
        except Exception as e:
            print(f"Error computing KDE for {param_name}: {e}")
            result[param_name] = {"kde": None, "mean": np.mean(values), "cov": np.var(values)}

    return result


def evaluate_parameters(abc_instance, parameters: Dict[str, float]) -> Tuple[Dict, float]:
    """Helper function to evaluate a single parameter set
    Args:
        abc_instance: ABC instance for accessing config and metric calculations
        parameters: Dictionary of parameter names and values
    Returns:
        Tuple of (metrics, distance)
    """
    from inverse_design.models.bdm.bdm import BDM

    config = abc_instance.model_config.copy()
    for param_name, value in parameters.items():
        setattr(config.rates, param_name, value)
    model = BDM(config)
    model_output = model.step()
    metrics_calculator = MetricsFactory.create_metrics(
        abc_instance.model_type,
        model_output,
    )
    metrics = abc_instance.calculate_all_metrics(metrics_calculator=metrics_calculator)
    distance = abc_instance.calculate_distance(metrics)
    return {"metrics": metrics, "distance": distance}
