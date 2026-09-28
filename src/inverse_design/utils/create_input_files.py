import json
import os
import re
from itertools import product
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy.stats import qmc

from inverse_design.config.parameter_config_mu_only import (
    PARAM_RANGES,
    PARAMS_DEFAULTS,
    SOURCE_PARAM_RANGES,
)
from inverse_design.analyze.source_metrics import (
    calculate_capillary_density,
    calculate_distance_between_points,
    find_spacing_from_density,
)


def _format_param_value(value: float, precision: int, is_bounded_by_one: bool) -> str:
    """Format parameter value with bounds checking and precision."""
    if is_bounded_by_one:
        value = min(1, max(0, value))
    else:
        value = max(0, value)
    return f"{value:.{precision}f}"


def _get_param_path(base_name: str, subfolder: str | None) -> str:
    """Get full parameter path including subfolder if any."""
    return f"{subfolder}/{base_name}" if subfolder else base_name


# Define parameter formatting configurations
PARAM_CONFIGS = {
    # Format: (precision, is_bounded_by_one, subfolder)
    "CELL_VOLUME": (1, False, None),
    "APOPTOSIS_AGE": (1, False, None),
    "NECROTIC_FRACTION": (6, True, None),
    "ACCURACY": (6, True, None),
    "AFFINITY": (6, True, None),
    "COMPRESSION_TOLERANCE": (6, False, None),
    "SYNTHESIS_DURATION": (3, False, "proliferation"),
    "BASAL_ENERGY": (6, False, "metabolism"),
    "PROLIFERATION_ENERGY": (12, False, "metabolism"),
    "MIGRATION_ENERGY": (12, False, "metabolism"),
    "METABOLIC_PREFERENCE": (6, False, "metabolism"),
    "CONVERSION_FRACTION": (6, False, "metabolism"),
    "RATIO_GLUCOSE_PYRUVATE": (6, False, "metabolism"),
    "LACTATE_RATE": (6, False, "metabolism"),
    "AUTOPHAGY_RATE": (12, False, "metabolism"),
    "GLUCOSE_UPTAKE_RATE": (6, False, "metabolism"),
    "ATP_PRODUCTION_RATE": (6, False, "metabolism"),
    "MIGRATORY_THRESHOLD": (6, False, "signaling"),
}

VALID_CONFIG_TYPES = {"cellular", "source", "combined"}
SOURCE_LAYER_CONFIGS = (
    ("GLUCOSE", "GLUCOSE_CONCENTRATION"),
    ("OXYGEN", "OXYGEN_CONCENTRATION"),
)
SOURCE_SITE_PARAM_NAMES = (
    "X_SPACING",
    "Y_SPACING",
    "GLUCOSE_CONCENTRATION",
    "OXYGEN_CONCENTRATION",
)


_N_PARTICLES_OVERRIDE: int | None = None


def set_particle_count(n_particles: int | None) -> None:
    """Override the particle count, decoupling it from ``2**sobol_power``.

    Call once at start-up from the run configuration. ``None`` restores the
    default power-of-two behaviour, so existing configs are unaffected.

    A non-power-of-two count forfeits the balance properties that make a Sobol
    sequence low-discrepancy; SciPy warns about this. It only affects the
    generation-0 prior draw, since later generations sample from the posterior.
    """
    global _N_PARTICLES_OVERRIDE
    if n_particles is not None and n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")
    _N_PARTICLES_OVERRIDE = n_particles


def get_particle_count() -> int | None:
    """Return the active particle-count override, or None if unset."""
    return _N_PARTICLES_OVERRIDE


def _num_sobol_samples(sobol_power: int) -> int:
    if _N_PARTICLES_OVERRIDE is not None:
        return _N_PARTICLES_OVERRIDE
    return 2**sobol_power


def _base_param_name(param_name: str) -> str:
    if param_name.endswith("_MU"):
        return param_name[:-3]
    if param_name.endswith("_SIGMA"):
        return param_name[:-6]
    return param_name


def _is_cellular_param(param_name: str) -> bool:
    base_name = _base_param_name(param_name)
    return param_name in PARAM_RANGES or base_name in PARAM_RANGES


def _is_source_param(param_name: str) -> bool:
    return param_name in SOURCE_PARAM_RANGES


def _validate_config_type(config_type: str) -> None:
    if config_type not in VALID_CONFIG_TYPES:
        valid_types = ", ".join(f'"{config_name}"' for config_name in sorted(VALID_CONFIG_TYPES))
        raise ValueError(f"config_type must be one of {valid_types}")


def _ensure_inputs_dir(output_dir: str) -> Path:
    inputs_dir = Path(output_dir) / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    return inputs_dir


def _count_existing_input_files(output_dir: str) -> int:
    inputs_dir = Path(output_dir) / "inputs"
    if not inputs_dir.exists():
        return 0
    return sum(
        1 for path in inputs_dir.iterdir() if path.is_file() and path.name.startswith("input_")
    )


def _prepare_output_dir(output_dir: str, target_num_files: int) -> bool:
    output_path = Path(output_dir)
    num_files = _count_existing_input_files(output_dir)

    if output_path.exists():
        if num_files >= target_num_files:
            print(
                f"Input directory exists with sufficient files "
                f"({num_files} >= {target_num_files}), skipping generation"
            )
            return False

        print(
            f"Input directory exists but needs more files "
            f"({num_files} < {target_num_files}), generating new files"
        )

    _ensure_inputs_dir(output_dir)
    return True


def _subset_param_ranges(param_ranges: dict, param_names: list[str]) -> dict:
    selected_names = set(param_names)
    return {name: bounds for name, bounds in param_ranges.items() if name in selected_names}


def _split_combined_param_ranges(param_ranges: dict) -> tuple[dict, dict]:
    cellular_param_ranges = {
        name: bounds for name, bounds in param_ranges.items() if _is_cellular_param(name)
    }
    source_param_ranges = {
        name: bounds for name, bounds in param_ranges.items() if _is_source_param(name)
    }
    return cellular_param_ranges, source_param_ranges


def _draw_sobol_samples(
    dimensions: int,
    sobol_power: int,
    seed: int,
    *,
    scramble: bool = True,
    use_base2: bool = True,
) -> np.ndarray:
    if dimensions == 0:
        return np.empty((_num_sobol_samples(sobol_power), 0))

    sampler = qmc.Sobol(d=dimensions, scramble=scramble, seed=seed)
    # random_base2 can only emit 2**m points, so an override forces the general path.
    if use_base2 and _N_PARTICLES_OVERRIDE is None:
        return sampler.random_base2(m=sobol_power)
    return sampler.random(n=_num_sobol_samples(sobol_power))


def _scale_sobol_values(unit_samples: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    if min_val == max_val:
        return np.full(unit_samples.shape[0], min_val)
    return min_val + (max_val - min_val) * unit_samples


def _sample_sobol_parameters(
    param_ranges: dict,
    sobol_power: int,
    seed: int,
    *,
    scramble: bool = True,
    use_base2: bool = True,
) -> dict:
    if not param_ranges:
        return {}

    samples = _draw_sobol_samples(
        dimensions=len(param_ranges),
        sobol_power=sobol_power,
        seed=seed,
        scramble=scramble,
        use_base2=use_base2,
    )
    return {
        name: _scale_sobol_values(samples[:, index], min_val, max_val).tolist()
        for index, (name, (min_val, max_val)) in enumerate(param_ranges.items())
    }


def _scale_integer_samples(unit_samples: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    if min_val == max_val:
        return np.full(unit_samples.shape[0], min_val, dtype=int)

    return (
        np.round(qmc.scale(unit_samples, l_bounds=min_val, u_bounds=max_val)).flatten().astype(int)
    )


def _split_generated_params(all_params: dict) -> tuple[dict, dict]:
    cellular_params = {}
    source_params = {}

    for param_name, values in all_params.items():
        if _is_cellular_param(param_name):
            cellular_params[param_name] = values
        elif _is_source_param(param_name):
            source_params[param_name] = values

    return cellular_params, source_params


def _is_valid_predicted_sample(sample_row: pd.Series, param_ranges: dict) -> bool:
    for column_name, value in sample_row.items():
        if column_name.endswith("_r2") or pd.isna(value):
            continue
        if column_name in param_ranges and value < 0:
            return False
    return True


def _collect_valid_chain_samples(
    lr_summary: Dict,
    sampled_params: dict,
    validation_param_ranges: dict,
    resampling_param_ranges: dict,
    sobol_power: int,
    seed: int,
    *,
    max_resampling_attempts: int = 10000,
    verbose: bool = True,
) -> tuple[list[dict], int]:
    from inverse_design.analyze.lr_predictor import predict_from_json_models

    target_num_samples = _num_sobol_samples(sobol_power)
    valid_samples = []
    resampling_attempts = 0

    while len(valid_samples) < target_num_samples and resampling_attempts < max_resampling_attempts:
        sampled_df = pd.DataFrame(sampled_params)
        complete_samples_df = predict_from_json_models(lr_summary, sampled_df)

        for _, sample_row in complete_samples_df.iterrows():
            if _is_valid_predicted_sample(sample_row, validation_param_ranges):
                valid_samples.append(sample_row.to_dict())

        if verbose:
            print(f"Valid samples so far: {len(valid_samples)}/{target_num_samples}")

        if len(valid_samples) >= target_num_samples:
            break

        if not sampled_params:
            if verbose:
                print("No parameter ranges available for resampling!")
            break

        needed_samples = target_num_samples - len(valid_samples)
        if verbose:
            print(f"Need {needed_samples} more valid samples, resampling...")

        extra_samples_power = max(1, int(np.ceil(np.log2(needed_samples * 2))))
        sampled_params = _sample_sobol_parameters(
            param_ranges=resampling_param_ranges,
            sobol_power=extra_samples_power,
            seed=seed + resampling_attempts + 100,
        )
        resampling_attempts += 1

    return valid_samples[:target_num_samples], resampling_attempts


def _add_spacing_columns_from_density(
    complete_samples_df: pd.DataFrame,
    *,
    radius_bound: float,
    side_length: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x_spacing_values = []
    y_spacing_values = []

    for capillary_density in complete_samples_df["CAPILLARY_DENSITY"]:
        x_spacing, y_spacing = find_spacing_from_density(
            capillary_density,
            radius_bound,
            side_length,
            MICRON_to_MM=1e-3,
        )
        x_spacing += int(rng.integers(0, 5))
        y_spacing += int(rng.integers(0, 5))
        x_spacing = max(1, x_spacing)
        y_spacing = max(1, y_spacing)
        x_spacing, y_spacing = sorted((x_spacing, y_spacing))
        x_spacing_values.append(f"*:{x_spacing}")
        y_spacing_values.append(f"*:{y_spacing}")

    updated_samples_df = complete_samples_df.copy()
    updated_samples_df["X_SPACING"] = x_spacing_values
    updated_samples_df["Y_SPACING"] = y_spacing_values
    updated_samples_df["DISTANCE_TO_CENTER"] = np.nan
    return updated_samples_df


def save_parameter_ranges(param_ranges: dict, output_dir: str):
    """Save the parameter ranges used for generating inputs to a CSV file."""
    # Create DataFrame with min and max values
    ranges_df = pd.DataFrame(
        {
            "parameter": param_ranges.keys(),
            "min_value": [r[0] for r in param_ranges.values()],
            "max_value": [r[1] for r in param_ranges.values()],
        }
    )

    # Save to CSV
    ranges_df.to_csv(f"{output_dir}/parameter_ranges.csv", index=False)


def _create_parameter_log_entry(params: dict, file_number: int) -> dict:
    """Create a parameter log entry with bounds checking.

    Args:
        params: Dictionary of parameter values
        file_number: Number to use in the file name

    Returns:
        Dictionary containing the file name and formatted parameter values
    """
    # Create a dictionary for the log entry
    log_entry = {"file_name": f"input_{file_number}.xml"}

    # Add parameters with bounds checking using PARAM_CONFIGS
    for base_param, (precision, is_bounded, _) in PARAM_CONFIGS.items():
        # Handle direct parameters (those without MU/SIGMA)
        if base_param in params:
            value = _format_param_value(params[base_param], precision, is_bounded)
            log_entry[base_param] = float(value)

        # Handle MU/SIGMA parameters
        mu_key = f"{base_param}_MU"
        sigma_key = f"{base_param}_SIGMA"

        if mu_key in params:
            mu_value = _format_param_value(params[mu_key], precision, is_bounded)
            log_entry[mu_key] = float(mu_value)

        if sigma_key in params:
            sigma_value = _format_param_value(
                params[sigma_key], precision, False
            )  # sigma is never bounded by 1
            log_entry[sigma_key] = float(sigma_value)

    return log_entry


def generate_perturbed_parameters(
    sobol_power: int,
    param_ranges: dict,
    config_params: Dict[str, str],
    output_dir: str = "perturbed_inputs",
    seed: int = 42,
):
    """Generate perturbed parameter sets using Sobol sampling

    Args:
        sobol_power: Power of 2 for number of samples (n_samples = 2^sobol_power)
        param_ranges: Dictionary of parameter ranges {param_name: (min, max)}
        config_params: Dictionary of input configurations {config_type: {param_name: value}}
        output_dir: Directory to save generated XML files
        seed: Random seed for reproducibility
    """
    config_type = config_params["perturbed_config"]
    _validate_config_type(config_type)
    template_path = config_params["template_path"]
    point_based = config_params["point_based"]
    y_interval = config_params["y_interval"]
    radius_bound = config_params["radius_bound"]
    side_length = config_params["side_length"]
    target_num_files = _num_sobol_samples(sobol_power)

    if not _prepare_output_dir(output_dir, target_num_files):
        return

    # Save parameter ranges to CSV
    save_parameter_ranges(param_ranges, output_dir)

    if config_type == "cellular":
        params = _sample_sobol_parameters(param_ranges, sobol_power, seed)
        param_log = generate_cellular_perturbations(
            params=params,
            template_path=template_path,
            output_dir=output_dir,
        )

    elif config_type == "source":
        samples = generate_source_site_samples(
            param_ranges=param_ranges,
            sobol_power=sobol_power,
            point_based=point_based,
            y_interval=y_interval,
            radius_bound=radius_bound,
            side_length=side_length,
            seed=seed,
        )
        param_log = generate_source_site_perturbations(
            params=samples,
            template_path=template_path,
            output_dir=output_dir,
        )

    else:
        cellular_param_ranges, source_param_ranges = _split_combined_param_ranges(param_ranges)
        cellular_params = _sample_sobol_parameters(cellular_param_ranges, sobol_power, seed)
        source_samples = generate_source_site_samples(
            param_ranges=source_param_ranges,
            sobol_power=sobol_power,
            point_based=point_based,
            y_interval=y_interval,
            radius_bound=radius_bound,
            side_length=side_length,
            seed=seed,
        )
        param_log = generate_combined_perturbations(
            cellular_params=cellular_params,
            source_params=source_samples,
            template_path=template_path,
            output_dir=output_dir,
        )

    _save_param_log(param_log, output_dir)


def _save_param_log(param_log: list, output_dir: str):
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/parameter_log.csv", index=False)
    print(f"Saved parameter log to {output_dir}/parameter_log.csv")


_NORMAL_DISTRIBUTION_RE = re.compile(
    r"^\s*NORMAL\(MU=([^,]+),SIGMA=([^)]+)\)\s*$",
    re.IGNORECASE,
)


def _normalize_parameter_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, dict):
        spec_type = spec.get("type")
        if spec_type == "range":
            bounds = spec.get("bounds")
            if not isinstance(bounds, list | tuple) or len(bounds) != 2:
                raise ValueError("Range parameter specs must contain a 2-item 'bounds' field.")
            return {"type": "range", "bounds": (float(bounds[0]), float(bounds[1]))}
        if spec_type == "choice":
            values = spec.get("values")
            if not isinstance(values, list | tuple) or len(values) == 0:
                raise ValueError("Choice parameter specs must contain a non-empty 'values' field.")
            return {"type": "choice", "values": list(values)}
        raise ValueError(f"Unsupported parameter spec type: {spec_type}")

    if isinstance(spec, list | tuple) and len(spec) == 2:
        return {"type": "range", "bounds": (float(spec[0]), float(spec[1]))}

    raise ValueError(f"Unsupported parameter spec: {spec}")


def _sample_parameter_specs(
    param_specs: dict[str, Any],
    sobol_power: int,
    seed: int,
) -> dict[str, list[Any]]:
    if not param_specs:
        return {}

    samples = _draw_sobol_samples(
        dimensions=len(param_specs),
        sobol_power=sobol_power,
        seed=seed,
        scramble=True,
        use_base2=True,
    )
    sampled_params: dict[str, list[Any]] = {}

    for index, (param_name, raw_spec) in enumerate(param_specs.items()):
        spec = _normalize_parameter_spec(raw_spec)
        unit_samples = samples[:, index]

        if spec["type"] == "range":
            min_val, max_val = spec["bounds"]
            sampled_params[param_name] = _scale_sobol_values(
                unit_samples,
                min_val,
                max_val,
            ).tolist()
            continue

        values = spec["values"]
        choice_indices = np.floor(unit_samples * len(values)).astype(int)
        choice_indices = np.clip(choice_indices, 0, len(values) - 1)
        sampled_params[param_name] = [values[choice_index] for choice_index in choice_indices]

    return sampled_params


def _sample_parameter_specs_n(
    param_specs: dict[str, Any],
    num_samples: int,
    seed: int,
) -> dict[str, list[Any]]:
    if not param_specs:
        return {}
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")

    samples = _draw_sobol_samples(
        dimensions=len(param_specs),
        sobol_power=int(np.ceil(np.log2(max(1, num_samples)))),
        seed=seed,
        scramble=True,
        use_base2=False,
    )
    if samples.shape[0] > num_samples:
        samples = samples[:num_samples]

    sampled_params: dict[str, list[Any]] = {}
    for index, (param_name, raw_spec) in enumerate(param_specs.items()):
        spec = _normalize_parameter_spec(raw_spec)
        unit_samples = samples[:, index]

        if spec["type"] == "range":
            min_val, max_val = spec["bounds"]
            sampled_params[param_name] = _scale_sobol_values(
                unit_samples,
                min_val,
                max_val,
            ).tolist()
            continue

        values = spec["values"]
        choice_indices = np.floor(unit_samples * len(values)).astype(int)
        choice_indices = np.clip(choice_indices, 0, len(values) - 1)
        sampled_params[param_name] = [values[choice_index] for choice_index in choice_indices]

    return sampled_params


def _split_parameter_specs(
    param_specs: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    discrete_specs = {}
    continuous_specs = {}

    for param_name, raw_spec in param_specs.items():
        spec = _normalize_parameter_spec(raw_spec)
        if spec["type"] == "choice":
            discrete_specs[param_name] = spec
        else:
            continuous_specs[param_name] = spec

    return discrete_specs, continuous_specs


def _sample_s3_coculture_mixed_rows(
    param_specs: dict[str, Any],
    continuous_samples_per_discrete_combo: int,
    seed: int,
) -> list[dict[str, Any]]:
    if continuous_samples_per_discrete_combo < 1:
        raise ValueError("continuous_samples_per_discrete_combo must be at least 1.")

    discrete_specs, continuous_specs = _split_parameter_specs(param_specs)

    if discrete_specs:
        discrete_names = list(discrete_specs)
        discrete_value_lists = [discrete_specs[name]["values"] for name in discrete_names]
        discrete_rows = [
            dict(zip(discrete_names, discrete_values))
            for discrete_values in product(*discrete_value_lists)
        ]
    else:
        discrete_rows = [{}]

    sample_rows: list[dict[str, Any]] = []
    continuous_draws_per_combo = continuous_samples_per_discrete_combo if continuous_specs else 1

    for combo_index, discrete_row in enumerate(discrete_rows):
        continuous_samples = _sample_parameter_specs_n(
            continuous_specs,
            continuous_draws_per_combo,
            seed + combo_index,
        )

        for draw_index in range(continuous_draws_per_combo):
            sample_row = discrete_row.copy()
            for param_name, values in continuous_samples.items():
                sample_row[param_name] = values[draw_index]
            sample_rows.append(sample_row)

    return sample_rows


def save_parameter_specs(param_specs: dict[str, Any], output_dir: str) -> None:
    serialized_specs = {}
    for param_name, raw_spec in param_specs.items():
        spec = _normalize_parameter_spec(raw_spec)
        if spec["type"] == "range":
            serialized_specs[param_name] = {
                "type": "range",
                "bounds": list(spec["bounds"]),
            }
        else:
            serialized_specs[param_name] = {
                "type": "choice",
                "values": spec["values"],
            }

    with open(Path(output_dir) / "parameter_space.json", "w", encoding="utf-8") as handle:
        json.dump(serialized_specs, handle, indent=2)


def _format_xml_number(value: Any, precision: int = 6) -> str:
    if isinstance(value, str):
        return value
    numeric_value = float(value)
    if abs(numeric_value) < 1e-12:
        return "0"
    return f"{numeric_value:.{precision}g}"


def _find_population(
    root: ET.Element,
    *,
    pop_id: str | None = None,
    pop_class: str | None = None,
) -> ET.Element:
    population = None
    if pop_id is not None:
        population = root.find(f".//population[@id='{pop_id}']")
    if population is None and pop_class is not None:
        population = root.find(f".//population[@class='{pop_class}']")
    if population is None:
        raise ValueError(f"Could not find population id={pop_id!r} class={pop_class!r} in XML")
    return population


def _find_population_parameter(
    population: ET.Element,
    param_id: str,
    *,
    process: str | None = None,
) -> ET.Element | None:
    for param in population.findall("population.parameter"):
        if param.get("id") != param_id:
            continue
        if process is not None and param.get("process") != process:
            continue
        return param
    return None


def _get_or_create_population_parameter(
    population: ET.Element,
    param_id: str,
    *,
    process: str | None = None,
) -> ET.Element:
    param = _find_population_parameter(population, param_id, process=process)
    if param is not None:
        return param

    param = ET.SubElement(population, "population.parameter")
    param.set("id", param_id)
    if process is not None:
        param.set("process", process)
    return param


def _parse_normal_distribution(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    match = _NORMAL_DISTRIBUTION_RE.match(value)
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2))


def _set_distribution_mu(
    param: ET.Element,
    mu_value: float,
    *,
    precision: int = 6,
    zero_sigma_when_zero: bool = False,
) -> None:
    parsed = _parse_normal_distribution(param.get("value"))
    sigma = 0.0 if parsed is None else parsed[1]
    if zero_sigma_when_zero and abs(float(mu_value)) < 1e-12:
        sigma = 0.0
    param.set(
        "value",
        f"NORMAL(MU={_format_xml_number(mu_value, precision)},SIGMA={_format_xml_number(sigma, precision)})",
    )


def _set_distribution_or_direct_value(
    param: ET.Element,
    value: float,
    *,
    precision: int = 6,
    zero_sigma_when_zero: bool = False,
) -> None:
    if _parse_normal_distribution(param.get("value")) is None:
        param.set("value", _format_xml_number(value, precision))
        return

    _set_distribution_mu(
        param,
        value,
        precision=precision,
        zero_sigma_when_zero=zero_sigma_when_zero,
    )


def _set_layer_concentration(root: ET.Element, layer_id: str, value: float) -> None:
    layer = root.find(f".//layer[@id='{layer_id}']")
    if layer is None:
        raise ValueError(f"Could not find layer with id={layer_id!r} in XML")

    for param in layer.findall("layer.parameter"):
        if param.get("operation") == "generator" or param.get("id") == "INITIAL_CONCENTRATION":
            param.set("value", _format_xml_number(value))


def _remove_action_parameters(action: ET.Element, *param_ids: str) -> None:
    for param in list(action.findall("action.parameter")):
        if param.get("id") in param_ids:
            action.remove(param)


def _get_action_parameter(action: ET.Element, param_id: str) -> ET.Element:
    for param in action.findall("action.parameter"):
        if param.get("id") == param_id:
            return param
    return ET.SubElement(action, "action.parameter", {"id": param_id})


def _clear_action_registers(action: ET.Element) -> None:
    for register in list(action.findall("action.register")):
        action.remove(register)


def _configure_tcell_insert_actions(
    root: ET.Element,
    total_dose: int,
    cd4_percent: float,
) -> tuple[int, int]:
    actions = root.find(".//actions")
    if actions is None:
        raise ValueError("Could not find <actions> block in XML")

    base_action = root.find(".//action[@id='ADD_T_CELLS']")
    if base_action is None:
        raise ValueError("Could not find action with id='ADD_T_CELLS' in XML")

    cd4_count = int(round(total_dose * cd4_percent / 100.0))
    cd8_count = int(total_dose - cd4_count)

    cd4_action = base_action
    cd8_action = ET.fromstring(ET.tostring(base_action, encoding="unicode"))

    cd4_action.set("id", "ADD_T_CELLS_CD4")
    cd8_action.set("id", "ADD_T_CELLS_CD8")

    for action, pop_id, insert_number in (
        (cd4_action, "4", cd4_count),
        (cd8_action, "8", cd8_count),
    ):
        _remove_action_parameters(action, "DOSE")
        insert_number_param = _get_action_parameter(action, "INSERT_NUMBER")
        insert_number_param.set("value", _format_xml_number(insert_number))
        _clear_action_registers(action)
        ET.SubElement(action, "action.register", {"id": pop_id})

    actions.append(cd8_action)
    return cd4_count, cd8_count


def _build_s3_coculture_log_entry(
    params: dict[str, Any],
    file_number: int,
    cd4_count: int,
    cd8_count: int,
) -> dict[str, Any]:
    log_entry = {"file_name": f"input_{file_number}.xml"}
    log_entry.update(params)
    log_entry["CD8_PERCENT"] = 100 - float(params["CD4_PERCENT"])
    log_entry["CD4_CD8_RATIO"] = (
        f"{int(round(params['CD4_PERCENT']))}:{int(round(100 - params['CD4_PERCENT']))}"
    )
    log_entry["CD4_INSERT_NUMBER"] = cd4_count
    log_entry["CD8_INSERT_NUMBER"] = cd8_count
    return log_entry


def _write_s3_coculture_samples(
    sample_rows: list[dict[str, Any]],
    param_specs: dict[str, Any],
    template_path: str,
    output_dir: str,
) -> list[dict[str, Any]]:
    from inverse_design.analyze.immune_param_ranges import S3_COCULTURE_PARAM_DEFAULTS

    target_num_files = len(sample_rows)
    if not _prepare_output_dir(output_dir, target_num_files):
        return []

    inputs_dir = _ensure_inputs_dir(output_dir)
    save_parameter_specs(param_specs, output_dir)

    param_log = []
    for sample_index, sampled_row in enumerate(sample_rows):
        sample_params = S3_COCULTURE_PARAM_DEFAULTS.copy()
        sample_params.update(sampled_row)

        tree = ET.parse(template_path)
        root = tree.getroot()

        healthy_pop = _find_population(root, pop_id="healthy")
        cancer_pop = _find_population(root, pop_id="cancerous", pop_class="cancer")
        cd8_pop = _find_population(root, pop_id="8", pop_class="cart_cd8")
        cd4_pop = _find_population(root, pop_id="4", pop_class="cart_cd4")

        healthy_antigens = _get_or_create_population_parameter(healthy_pop, "CAR_ANTIGENS")
        _set_distribution_or_direct_value(
            healthy_antigens,
            sample_params["HEALTHY_ANTIGENS"],
            zero_sigma_when_zero=True,
        )

        cancer_antigens = _get_or_create_population_parameter(cancer_pop, "CAR_ANTIGENS")
        _set_distribution_or_direct_value(
            cancer_antigens,
            sample_params["CANCER_ANTIGENS"],
        )

        cancer_metabolic_pref = _get_or_create_population_parameter(
            cancer_pop,
            "METABOLIC_PREFERENCE",
            process="metabolism",
        )
        cancer_metabolic_pref.set(
            "scale",
            _format_xml_number(sample_params["CANCER_METABOLIC_PREFERENCE_SCALE"]),
        )

        cancer_migratory_threshold = _get_or_create_population_parameter(
            cancer_pop,
            "MIGRATORY_THRESHOLD",
            process="signaling",
        )
        cancer_migratory_threshold.set(
            "scale",
            _format_xml_number(sample_params["CANCER_MIGRATORY_THRESHOLD_SCALE"]),
        )

        for tcell_pop in (cd4_pop, cd8_pop):
            car_affinity = _get_or_create_population_parameter(tcell_pop, "CAR_AFFINITY")
            _set_distribution_or_direct_value(car_affinity, sample_params["CAR_AFFINITY"])

            bound_time = _get_or_create_population_parameter(tcell_pop, "BOUND_TIME")
            _set_distribution_mu(bound_time, sample_params["BOUND_TIME"])

            synthesis_duration = _get_or_create_population_parameter(
                tcell_pop,
                "proliferation/SYNTHESIS_DURATION",
            )
            _set_distribution_mu(synthesis_duration, sample_params["SYNTHESIS_TIME_T"])

        _set_layer_concentration(root, "GLUCOSE", sample_params["GLUCOSE_CONCENTRATION"])
        _set_layer_concentration(root, "OXYGEN", sample_params["OXYGEN_CONCENTRATION"])

        cd4_count, cd8_count = _configure_tcell_insert_actions(
            root,
            int(round(sample_params["CAR_T_DOSE"])),
            float(sample_params["CD4_PERCENT"]),
        )

        output_file = inputs_dir / f"input_{sample_index + 1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        param_log.append(
            _build_s3_coculture_log_entry(
                sample_params,
                sample_index + 1,
                cd4_count,
                cd8_count,
            )
        )

    _save_param_log(param_log, output_dir)
    print(f"Saved parameter space to {output_dir}/parameter_space.json")
    return param_log


def generate_s3_coculture_perturbations(
    sobol_power: int,
    param_specs: dict[str, Any],
    template_path: str,
    output_dir: str,
    seed: int = 42,
) -> list[dict[str, Any]]:
    sampled_params = _sample_parameter_specs(param_specs, sobol_power, seed)
    target_num_files = _num_sobol_samples(sobol_power)
    sample_rows = []
    for sample_index in range(target_num_files):
        sample_rows.append(
            {param_name: values[sample_index] for param_name, values in sampled_params.items()}
        )

    return _write_s3_coculture_samples(
        sample_rows=sample_rows,
        param_specs=param_specs,
        template_path=template_path,
        output_dir=output_dir,
    )


def generate_s3_coculture_mixed_perturbations(
    continuous_samples_per_discrete_combo: int,
    param_specs: dict[str, Any],
    template_path: str,
    output_dir: str,
    seed: int = 42,
) -> list[dict[str, Any]]:
    discrete_specs, continuous_specs = _split_parameter_specs(param_specs)
    discrete_combo_count = 1
    for spec in discrete_specs.values():
        discrete_combo_count *= len(spec["values"])

    continuous_draws_per_combo = continuous_samples_per_discrete_combo if continuous_specs else 1
    sample_rows = _sample_s3_coculture_mixed_rows(
        param_specs=param_specs,
        continuous_samples_per_discrete_combo=continuous_samples_per_discrete_combo,
        seed=seed,
    )
    print(
        "S3 mixed sampling: "
        f"{discrete_combo_count} discrete combination(s) x "
        f"{continuous_draws_per_combo} continuous sample(s) = "
        f"{len(sample_rows)} total XML inputs"
    )
    return _write_s3_coculture_samples(
        sample_rows=sample_rows,
        param_specs=param_specs,
        template_path=template_path,
        output_dir=output_dir,
    )


def generate_chain_based_perturbed_parameters(
    sobol_power: int,
    param_ranges: dict,
    config_params: Dict[str, str],
    lr_summary: Dict,
    peak_name: str,
    output_dir: str = "perturbed_inputs",
    seed: int = 42,
    verbose: bool = True,
):
    """Generate perturbed parameter sets using Sobol sampling with chain-based prediction

    Args:
        sobol_power: Power of 2 for number of samples (n_samples = 2^sobol_power)
        param_ranges: Dictionary of parameter ranges {param_name: (min, max)}
        config_params: Dictionary of input configurations {config_type: {param_name: value}}
        peak_name: Name of the peak for chain analysis
        output_dir: Directory to save generated XML files
        seed: Random seed for reproducibility
    """
    template_path = config_params["template_path"]
    target_num_files = _num_sobol_samples(sobol_power)

    if not _prepare_output_dir(output_dir, target_num_files):
        return

    # Save parameter ranges to CSV
    save_parameter_ranges(param_ranges, output_dir)

    independent_params = lr_summary["sampling_instructions"]["parameters_to_sample"]["independent"]
    chain_roots = lr_summary["sampling_instructions"]["parameters_to_sample"]["root"]
    derived_params = [
        param
        for param in param_ranges.keys()
        if param not in independent_params and param not in chain_roots
    ]

    if verbose:
        print(f"\n{'='*60}")
        print(f"CHAIN-BASED SOBOL SAMPLING FOR {peak_name}")
        print(f"{'='*60}")
        print(f"Independent parameters: {independent_params}")
        print(f"Chain root parameters: {chain_roots}")
        print(f"Derived parameters: {derived_params}")

    independent_param_ranges = _subset_param_ranges(param_ranges, independent_params)
    sampled_params = {}

    if independent_param_ranges:
        print(
            f"\nStep 1: Sampling {len(independent_param_ranges)} independent parameters using Sobol..."
        )
        sampled_params.update(_sample_sobol_parameters(independent_param_ranges, sobol_power, seed))
        print(f"Sampled independent parameters: {list(independent_param_ranges.keys())}")

    chain_root_ranges = _subset_param_ranges(param_ranges, chain_roots)
    if chain_root_ranges:
        print(f"\nStep 2: Sampling {len(chain_root_ranges)} chain root parameters using Sobol...")
        sampled_params.update(_sample_sobol_parameters(chain_root_ranges, sobol_power, seed + 1))
        print(f"Sampled chain root parameters: {list(chain_root_ranges.keys())}")

    print(f"\nStep 3: Predicting derived parameters using chain analysis...")

    sampled_input_ranges = {**independent_param_ranges, **chain_root_ranges}
    valid_samples, resampling_attempts = _collect_valid_chain_samples(
        lr_summary=lr_summary,
        sampled_params=sampled_params,
        validation_param_ranges=param_ranges,
        resampling_param_ranges=sampled_input_ranges,
        sobol_power=sobol_power,
        seed=seed,
        verbose=verbose,
    )

    if len(valid_samples) < target_num_files:
        print(
            f"Warning: Could only generate {len(valid_samples)} valid samples out of {target_num_files} requested"
        )
        print(f"Resampling attempts: {resampling_attempts}")

    if valid_samples:
        complete_samples_df = pd.DataFrame(valid_samples)
        print(f"Final valid samples: {len(complete_samples_df)}")
    else:
        print("Error: No valid samples generated!")
        return None

    complete_samples_df = _add_spacing_columns_from_density(
        complete_samples_df,
        radius_bound=config_params["radius_bound"],
        side_length=config_params["side_length"],
        seed=seed,
    )
    param_ranges = {k: v for k, v in param_ranges.items() if k in complete_samples_df.columns}
    complete_samples_df = complete_samples_df[list(param_ranges.keys())]
    all_params = {
        col: complete_samples_df[col].tolist()
        for col in complete_samples_df.columns
        if not col.endswith("_r2")
    }

    if verbose:
        print(f"\nStep 4: Categorizing parameters into cellular and source...")

    cellular_params, source_params = _split_generated_params(all_params)

    if verbose:
        print(f"Cellular parameters: {len(cellular_params)}")
        print(f"Source parameters: {len(source_params)}")

    if verbose:
        print(f"\nStep 5: Generating combined perturbations...")

    param_log = generate_combined_perturbations(
        cellular_params=cellular_params,
        source_params=source_params,
        template_path=template_path,
        output_dir=output_dir,
    )
    # Save parameter log
    _save_param_log(param_log, output_dir)

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Generated {target_num_files} parameter sets")
        print(f"Cellular parameters: {list(cellular_params.keys())}")
        print(f"Source parameters: {list(source_params.keys())}")
        print(f"Output directory: {output_dir}")

    return param_log


def generate_cellular_perturbations(
    params: dict,
    template_path: str,
    output_dir: str,
) -> list:
    """Generate input XML files for cellular parameter perturbations.

    Args:
        params: Dictionary of parameter values {param_name: [values]}
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files
    """
    inputs_dir = _ensure_inputs_dir(output_dir)

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare parameter logging
    param_log = []
    n_samples = len(next(iter(params.values())))

    # Generate files for each parameter set
    for i in range(n_samples):
        # Create parameter dictionary for this sample
        sample_params = {name: values[i] for name, values in params.items()}

        # Create log entry
        log_entry = _create_parameter_log_entry(sample_params, i + 1)
        param_log.append(log_entry)

        # Update XML parameters
        update_xml_parameters(root, sample_params)

        # Save modified XML
        output_file = inputs_dir / f"input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)
    return param_log


def generate_parameters_from_kde(
    parameter_pdfs: Dict,
    n_samples: int,
    output_dir: str = "kde_sampled_inputs",
    template_path: str = "sample_input_v3.xml",
    const_params_values: Dict[str, float] = None,
    seed: int = 42,
):
    """Generate parameter sets by sampling from kernel density estimates (KDE)

    Args:
        parameter_pdfs: Dictionary containing KDE objects for each parameter
        n_samples: Number of parameter sets to generate
        output_dir: Directory to save generated XML files
        template_path: Path to template XML file
        const_params_values: Dictionary of parameters to keep constant {param_name: value}
    """
    if not _prepare_output_dir(output_dir, n_samples):
        return
    inputs_dir = _ensure_inputs_dir(output_dir)

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare DataFrame for parameter logging
    param_log = []

    # Sample from each parameter's KDE
    kde_dict = {key: value["kde"] for key, value in parameter_pdfs.items()}
    param_names = parameter_pdfs.keys()

    # Generate samples for each parameter
    sampled_params = {
        param: kde.resample(n_samples, seed=seed)[0] for param, kde in kde_dict.items()
    }

    # Generate XML files for each parameter set
    for i in range(n_samples):
        # Get raw parameters
        raw_params = {param: sampled_params[param][i] for param in param_names}

        # Add constant parameters to raw_params
        if const_params_values:
            raw_params.update(const_params_values)

        # Create bounded parameters dictionary
        bounded_params = {"file_name": f"input_{i+1}.xml"}

        # Apply bounds to all parameters
        for base_param, (precision, is_bounded, _) in PARAM_CONFIGS.items():
            # Handle direct parameters (those without MU/SIGMA)
            if base_param in raw_params:
                value = float(_format_param_value(raw_params[base_param], precision, is_bounded))
                bounded_params[base_param] = value

            # Handle MU/SIGMA parameters
            mu_key = f"{base_param}_MU"
            sigma_key = f"{base_param}_SIGMA"

            if mu_key in raw_params:
                mu_value = float(_format_param_value(raw_params[mu_key], precision, is_bounded))
                bounded_params[mu_key] = mu_value

            if sigma_key in raw_params:
                sigma_value = float(
                    _format_param_value(raw_params[sigma_key], precision, False)
                )  # sigma is never bounded by 1
                bounded_params[sigma_key] = sigma_value

        update_xml_parameters(root, bounded_params)

        # Save modified XML
        output_file = inputs_dir / f"input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        # Log bounded parameters
        param_log.append(bounded_params)

    # Save parameters to CSV
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/kde_sampled_parameters_log.csv", index=False)

    print(f"Generated {n_samples} XML files and parameter log in {output_dir}/")


def generate_2param_perturbation(
    sensitivity_json: str,
    metric: str,
    template_path: str = "sample_input_v3.xml",
    perturbation_range: range = range(-50, 51, 10),
    output_dir: str = "inputs/STEM_CELL/",
):
    """Generate input files by perturbing the top 2 parameters based on MI scores.

    Args:
        sensitivity_json: Path to sensitivity analysis JSON file
        metric: Metric name to analyze ('symmetry', 'cycle_length', or 'vol_std')
        template_path: Path to template XML file
        perturbation_range: Range of perturbation percentages
    """
    # Load sensitivity analysis results
    with open(sensitivity_json, "r") as f:
        sensitivity_data = json.load(f)

    if metric not in sensitivity_data:
        raise ValueError(f"Metric {metric} not found in sensitivity analysis data")

    # Get parameter names and MI scores
    param_names = sensitivity_data[metric]["names"]
    mi_scores = sensitivity_data[metric]["MI"]

    # Find top 2 parameters
    top_2_indices = np.argsort(mi_scores)[-2:][::-1]  # Sort descending
    top_2_params = [param_names[i] for i in top_2_indices]

    print(f"Top 2 parameters for {metric}:")
    print(f"1. {top_2_params[0]} (MI = {mi_scores[top_2_indices[0]]:.4f})")
    print(f"2. {top_2_params[1]} (MI = {mi_scores[top_2_indices[1]]:.4f})")

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read template XML to get default values
    tree = ET.parse(template_path)
    root = tree.getroot()
    cancerous_pop = root.find(".//population[@id='cancerous']")

    # Get default values for the top 2 parameters
    default_values = {}

    # Check population parameters
    for param in cancerous_pop.findall("population.parameter"):
        param_id = param.get("id")
        full_param_id = param_id

        # Handle parameters in subfolders
        if (
            param_id.startswith("metabolism/")
            or param_id.startswith("signaling/")
            or param_id.startswith("proliferation/")
        ):
            full_param_id = param_id
            param_id = param_id.split("/")[1]

        # Case 1: Direct parameter (like CELL_VOLUME)
        base_param = param_id.split("/")[-1]
        if base_param in [p.replace("_MU", "").replace("_SIGMA", "") for p in top_2_params]:
            value_str = param.get("value")
            if "NORMAL" not in value_str:
                default_values[base_param] = {"type": "direct", "value": float(value_str)}
            else:
                # Store both MU and SIGMA values
                mu = float(value_str.split("MU=")[1].split(",")[0])
                sigma = float(value_str.split("SIGMA=")[1].split(")")[0])
                default_values[base_param] = {"type": "normal", "mu": mu, "sigma": sigma}

    # Check environment parameters (GLUCOSE and OXYGEN concentrations)
    for layer_id, param_name in [
        ("GLUCOSE", "GLUCOSE_CONCENTRATION"),
        ("OXYGEN", "OXYGEN_CONCENTRATION"),
    ]:
        if param_name in top_2_params:
            layer = root.find(f".//layer[@id='{layer_id}']")
            if layer is not None:
                for param in layer.findall("layer.parameter"):
                    if (
                        param.get("operation") == "generator"
                        or param.get("id") == "INITIAL_CONCENTRATION"
                        or param.get("id") == "CONCENTRATION"
                    ):
                        value_str = param.get("value")
                        default_values[param_name] = {"type": "direct", "value": float(value_str)}
                        break  # Found the parameter, no need to check others in this layer

    # Generate perturbed parameter combinations
    param_log = []
    file_counter = 1
    for p1_pct in perturbation_range:
        for p2_pct in perturbation_range:
            # Calculate perturbed values
            p1_base = top_2_params[0].replace("_MU", "").replace("_SIGMA", "")
            p2_base = top_2_params[1].replace("_MU", "").replace("_SIGMA", "")

            # Calculate perturbed values based on parameter type
            if top_2_params[0] in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
                # Environment parameter
                p1_value = default_values[top_2_params[0]]["value"] * (1 + p1_pct / 100)
            else:
                # Population parameter
                p1_property = top_2_params[0].split("_")[-1]  # MU or SIGMA
                if default_values[p1_base]["type"] == "direct":
                    p1_value = default_values[p1_base]["value"] * (1 + p1_pct / 100)
                else:
                    if p1_property == "MU":
                        p1_value = default_values[p1_base]["mu"] * (1 + p1_pct / 100)
                    else:  # SIGMA
                        p1_value = default_values[p1_base]["sigma"] * (1 + p1_pct / 100)

            if top_2_params[1] in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
                # Environment parameter
                p2_value = default_values[top_2_params[1]]["value"] * (1 + p2_pct / 100)
            else:
                # Population parameter
                p2_property = top_2_params[1].split("_")[-1]  # MU or SIGMA
                if default_values[p2_base]["type"] == "direct":
                    p2_value = default_values[p2_base]["value"] * (1 + p2_pct / 100)
                else:
                    if p2_property == "MU":
                        p2_value = default_values[p2_base]["mu"] * (1 + p2_pct / 100)
                    else:  # SIGMA
                        p2_value = default_values[p2_base]["sigma"] * (1 + p2_pct / 100)

            # Create new XML tree for this combination
            tree = ET.parse(template_path)
            root = tree.getroot()
            cancerous_pop = root.find(".//population[@id='cancerous']")

            # Update parameters
            for param in cancerous_pop.findall("population.parameter"):
                param_id = param.get("id")
                full_param_id = param_id

                # Handle parameters in subfolders
                if (
                    param_id.startswith("metabolism/")
                    or param_id.startswith("signaling/")
                    or param_id.startswith("proliferation/")
                ):
                    full_param_id = param_id
                    param_id = param_id.split("/")[1]

                base_param = full_param_id.split("/")[-1]
                if base_param == p1_base and top_2_params[0] not in [
                    "GLUCOSE_CONCENTRATION",
                    "OXYGEN_CONCENTRATION",
                ]:
                    param_info = default_values[base_param]
                    if param_info["type"] == "direct":
                        # Case 1: Direct parameter
                        new_value = param_info["value"] * (1 + p1_pct / 100)
                        param.set("value", f"{new_value:.6f}")
                    else:
                        # Case 2 & 3: NORMAL distribution
                        mu = param_info["mu"]
                        sigma = param_info["sigma"]
                        p1_property = top_2_params[0].split("_")[-1]  # MU or SIGMA
                        if p1_property == "MU":
                            mu = mu * (1 + p1_pct / 100)
                        else:  # SIGMA
                            sigma = sigma * (1 + p1_pct / 100)
                        param.set("value", f"NORMAL(MU={mu:.6f},SIGMA={sigma:.6f})")

                elif base_param == p2_base and top_2_params[1] not in [
                    "GLUCOSE_CONCENTRATION",
                    "OXYGEN_CONCENTRATION",
                ]:
                    param_info = default_values[base_param]
                    if param_info["type"] == "direct":
                        # Case 1: Direct parameter
                        new_value = param_info["value"] * (1 + p2_pct / 100)
                        param.set("value", f"{new_value:.6f}")
                    else:
                        # Case 2 & 3: NORMAL distribution
                        mu = param_info["mu"]
                        sigma = param_info["sigma"]
                        p2_property = top_2_params[1].split("_")[-1]  # MU or SIGMA
                        if p2_property == "MU":
                            mu = mu * (1 + p2_pct / 100)
                        else:  # SIGMA
                            sigma = sigma * (1 + p2_pct / 100)
                        param.set("value", f"NORMAL(MU={mu:.6f},SIGMA={sigma:.6f})")

            # Update environment parameters (GLUCOSE and OXYGEN concentrations)
            for layer_id, param_name in [
                ("GLUCOSE", "GLUCOSE_CONCENTRATION"),
                ("OXYGEN", "OXYGEN_CONCENTRATION"),
            ]:
                if param_name in top_2_params:
                    layer = root.find(f".//layer[@id='{layer_id}']")
                    if layer is not None:
                        for param in layer.findall("layer.parameter"):
                            if (
                                param.get("operation") == "generator"
                                or param.get("id") == "INITIAL_CONCENTRATION"
                                or param.get("id") == "CONCENTRATION"
                            ):
                                # Determine which parameter this is and apply the appropriate perturbation
                                if param_name == top_2_params[0]:
                                    new_value = default_values[param_name]["value"] * (
                                        1 + p1_pct / 100
                                    )
                                elif param_name == top_2_params[1]:
                                    new_value = default_values[param_name]["value"] * (
                                        1 + p2_pct / 100
                                    )
                                else:
                                    continue  # This shouldn't happen, but just in case

                                param.set("value", f"{new_value:.6f}")
                                # Don't break here - we want to update both INITIAL_CONCENTRATION and CONCENTRATION

            # Save modified XML
            if not os.path.exists(f"{output_dir}/inputs"):
                os.makedirs(f"{output_dir}/inputs")
            output_file = f"{output_dir}/inputs/input_{file_counter}.xml"
            tree.write(output_file, encoding="utf-8", xml_declaration=True)

            # Log parameters
            param_log.append(
                {
                    "file_name": f"input_{file_counter}.xml",
                    f"{top_2_params[0]}_perturbation": p1_pct,
                    f"{top_2_params[1]}_perturbation": p2_pct,
                    top_2_params[0]: p1_value,
                    top_2_params[1]: p2_value,
                }
            )

            file_counter += 1

    # Save parameter log
    df = pd.DataFrame(param_log)
    df.to_csv(f"{output_dir}/parameter_log.csv", index=False)

    # Calculate parameter ranges based on parameter type and property
    parameter_ranges = {}
    for param in top_2_params:
        if param in ["GLUCOSE_CONCENTRATION", "OXYGEN_CONCENTRATION"]:
            # Environment parameters are always direct
            param_info = default_values[param]
            base_value = param_info["value"]
            parameter_ranges[param] = (
                base_value * (1 + min(perturbation_range) / 100),
                base_value * (1 + max(perturbation_range) / 100),
            )
        else:
            # Population parameters may have _MU/_SIGMA variants
            base_param = param.replace("_MU", "").replace("_SIGMA", "")
            property_type = param.split("_")[-1]  # MU or SIGMA

            param_info = default_values[base_param]
            if param_info["type"] == "direct":
                base_value = param_info["value"]
                parameter_ranges[param] = (
                    base_value * (1 + min(perturbation_range) / 100),
                    base_value * (1 + max(perturbation_range) / 100),
                )
            else:  # normal distribution
                if property_type == "MU":
                    base_value = param_info["mu"]
                else:  # SIGMA
                    base_value = param_info["sigma"]
                parameter_ranges[param] = (
                    base_value * (1 + min(perturbation_range) / 100),
                    base_value * (1 + max(perturbation_range) / 100),
                )

    save_parameter_ranges(parameter_ranges, output_dir)

    print(f"Generated {len(param_log)} XML files and parameter log in {output_dir}/")


def generate_input_files(
    param_names: list[str],
    param_values: list[list[float]],
    abc_param_ranges: dict,
    config_params: Dict[str, str],
    output_dir: str = "inputs",
) -> None:
    """Generate input XML files from parameter values.

    Args:
        param_names: List of parameter names
        param_values: List of parameter value lists, where each inner list contains values
                     corresponding to param_names
        config_params: Dictionary of input configurations {config_type: {param_name: value}}
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files
    """
    config_type = config_params["perturbed_config"]
    _validate_config_type(config_type)
    template_path = config_params["template_path"]
    length = int(6 * config_params["radius_bound"] - 3)
    width = int(4 * config_params["radius_bound"] - 2)

    if not _prepare_output_dir(output_dir, len(param_values)):
        return

    sampled_param_ranges = {}
    for name, values in zip(param_names, zip(*param_values)):
        if name in ["X_SPACING", "Y_SPACING"]:
            values = [int(value.split(":")[1]) for value in values]
            sampled_param_ranges[name] = (min(values), max(values))
        else:
            sampled_param_ranges[name] = (min(values), max(values))
    save_parameter_ranges(sampled_param_ranges, output_dir)
    if config_type == "cellular":
        # Convert to dictionary format for cellular parameters
        params_dict = {
            name: [values[i] for values in param_values] for i, name in enumerate(param_names)
        }
        param_log = generate_cellular_perturbations(
            params=params_dict,
            template_path=template_path,
            output_dir=output_dir,
        )

    elif config_type == "source":  # source configuration
        params = {param_name: [] for param_name in SOURCE_SITE_PARAM_NAMES}
        for values in param_values:
            params_dict = dict(zip(param_names, values))
            for param_name in SOURCE_SITE_PARAM_NAMES:
                params[param_name].append(
                    params_dict.get(param_name, "" if "SPACING" in param_name else 0)
                )

        param_log = generate_source_site_perturbations(
            params=params,
            template_path=template_path,
            output_dir=output_dir,
        )

    else:  # combined configuration
        cellular_param_names = [name for name in param_names if not _is_source_param(name)]
        cellular_params = {name: [] for name in cellular_param_names}
        default_x_spacing = (
            f"{length // 2 - 1}:{length // 2 + 1}"
            if config_params["point_based"]
            else f"*:{abc_param_ranges['X_SPACING'][0]}"
        )
        default_y_spacing = (
            f"{int(abc_param_ranges['Y_SPACING'][0])}:{int(abc_param_ranges['Y_SPACING'][0]) + 1}"
            if config_params["point_based"]
            else f"*:{abc_param_ranges['Y_SPACING'][0]}"
        )
        source_params = {
            "X_SPACING": [],
            "Y_SPACING": [],
            "GLUCOSE_CONCENTRATION": [],
            "OXYGEN_CONCENTRATION": [],
            "CAPILLARY_DENSITY": [],
            "DISTANCE_TO_CENTER": [],
        }

        for values in param_values:
            params_dict = dict(zip(param_names, values))
            for name in cellular_param_names:
                if name in params_dict:
                    cellular_params[name].append(params_dict[name])

            source_params["X_SPACING"].append(params_dict.get("X_SPACING", default_x_spacing))
            source_params["Y_SPACING"].append(params_dict.get("Y_SPACING", default_y_spacing))
            source_params["GLUCOSE_CONCENTRATION"].append(
                params_dict.get(
                    "GLUCOSE_CONCENTRATION", abc_param_ranges["GLUCOSE_CONCENTRATION"][0]
                )
            )
            source_params["OXYGEN_CONCENTRATION"].append(
                params_dict.get("OXYGEN_CONCENTRATION", abc_param_ranges["OXYGEN_CONCENTRATION"][0])
            )
            if not config_params["point_based"]:
                if (
                    params_dict.get("X_SPACING", "") == ""
                    and params_dict.get("Y_SPACING", "") == ""
                ):
                    capillary_density = calculate_capillary_density(
                        radius_bound=config_params["radius_bound"],
                        length_spacing=abc_param_ranges["X_SPACING"][0],
                        width_spacing=abc_param_ranges["Y_SPACING"][0],
                        side_length=config_params["side_length"],
                    )
                else:
                    capillary_density = calculate_capillary_density(
                        radius_bound=config_params["radius_bound"],
                        length_spacing=int(
                            params_dict.get("X_SPACING", abc_param_ranges["X_SPACING"][0]).split(
                                ":"
                            )[1]
                        ),
                        width_spacing=int(
                            params_dict.get("Y_SPACING", abc_param_ranges["Y_SPACING"][0]).split(
                                ":"
                            )[1]
                        ),
                        side_length=config_params["side_length"],
                    )
                source_params["CAPILLARY_DENSITY"].append(capillary_density)
                source_params["DISTANCE_TO_CENTER"].append(np.nan)
            else:
                point_center = np.array([length // 2, width // 2, 0]).astype(int)
                y_spacing = int(params_dict.get("Y_SPACING", default_y_spacing).split(":")[1])
                source_site = np.array([length // 2, y_spacing, 0]).astype(int)
                distance_to_center = calculate_distance_between_points(
                    point_center,
                    source_site,
                    config_params["side_length"],
                    config_params["radius_bound"],
                )
                source_params["DISTANCE_TO_CENTER"].append(distance_to_center)
                capillary_density = calculate_capillary_density(
                    radius_bound=config_params["radius_bound"],
                    length_spacing=length,
                    width_spacing=width,
                    side_length=config_params["side_length"],
                )
                source_params["CAPILLARY_DENSITY"].append(capillary_density)
        param_log = generate_combined_perturbations(
            cellular_params=cellular_params,
            source_params=source_params,
            template_path=template_path,
            output_dir=output_dir,
        )
    _save_param_log(param_log, output_dir)


def generate_source_site_perturbations(
    params: dict,
    template_path: str,
    output_dir: str,
) -> list:
    """Generate input XML files for source site perturbations.

    Args:
        params: Dictionary containing:
            - x_spacing: List of X_SPACING values
            - y_spacing: List of Y_SPACING values
            - glucose: List of glucose concentrations
            - oxygen: List of oxygen concentrations
            - capillary_density: List of capillary densities
            - distance_to_center: List of distances
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files
    """
    # Generate XML files
    inputs_dir = _ensure_inputs_dir(output_dir)
    tree = ET.parse(template_path)
    root = tree.getroot()
    param_log = []
    n_samples = len(params["X_SPACING"])

    def get_param_value(param_name, index):
        """Get parameter value at index or return 0 if parameter doesn't exist"""
        if param_name not in params:
            return 0
        param_list = params[param_name]
        if index >= len(param_list):
            return 0
        return param_list[index]

    for i in range(n_samples):
        # Find and update source sites component
        sites_component = root.find(".//component[@id='SITES']")
        if sites_component is None:
            raise ValueError("Could not find component with id='SITES' in XML")

        # Update parameters
        sites_component.find("component.parameter[@id='X_SPACING']").set(
            "value", str(get_param_value("X_SPACING", i))
        )
        sites_component.find("component.parameter[@id='Y_SPACING']").set(
            "value", str(get_param_value("Y_SPACING", i))
        )

        # Update concentrations
        for layer_id, param_name in SOURCE_LAYER_CONFIGS:
            layer = root.find(f".//layer[@id='{layer_id}']")
            for param in layer.findall("layer.parameter"):
                if (
                    param.get("operation") == "generator"
                    or param.get("id") == "INITIAL_CONCENTRATION"
                ):
                    param.set("value", str(get_param_value(param_name, i)))

        # Save modified XML
        output_file = inputs_dir / f"input_{i+1}.xml"
        tree.write(output_file, encoding="utf-8", xml_declaration=True)

        # Log parameters
        param_log.append(
            {
                "file_name": f"input_{i+1}.xml",
                "X_SPACING": get_param_value("X_SPACING", i),
                "Y_SPACING": get_param_value("Y_SPACING", i),
                "GLUCOSE_CONCENTRATION": get_param_value("GLUCOSE_CONCENTRATION", i),
                "OXYGEN_CONCENTRATION": get_param_value("OXYGEN_CONCENTRATION", i),
                "CAPILLARY_DENSITY": get_param_value("CAPILLARY_DENSITY", i),
                "DISTANCE_TO_CENTER": get_param_value("DISTANCE_TO_CENTER", i),
            }
        )

    return param_log


def generate_source_site_samples(
    param_ranges: dict,
    sobol_power=10,
    point_based=True,
    y_interval=4,
    radius_bound=10,
    side_length=1,
    seed: int = 42,
):
    """
    Generate Sobol samples for x_spacings, y_spacings, glucose, and oxygen concentrations.

    Parameters:
    -----------
    sobol_power : int
        Power of 2 for number of samples (n_samples = 2^sobol_power)
    y_spacing_interval : bool
        If True, sample y_spacing from specific values
        If False, use continuous Sobol sampling
    y_spacing_values : list or None
        List of specific values to sample from when y_spacing_interval is True
        Default values are [2, 5, 8, 12, 15] if None

    Returns:
    --------
    dict : Dictionary containing the sampled parameters
    """
    n_samples = _num_sobol_samples(sobol_power)
    samples = _draw_sobol_samples(
        dimensions=len(param_ranges),
        sobol_power=sobol_power,
        seed=seed,
        scramble=False,
        use_base2=False,
    )

    length = int(6 * radius_bound - 3)
    width = int(4 * radius_bound - 2)
    scaled_samples = {}
    x_values = None
    for i, (param_name, (min_val, max_val)) in enumerate(param_ranges.items()):
        if point_based:
            if param_name == "X_SPACING":
                x_center = int(length / 2)
                scaled_samples["X_SPACING"] = [f"{x_center-1}:{x_center+1}" for _ in samples]
            elif param_name == "Y_SPACING":
                sample_2d = samples[:, i : i + 1]
                if 1 + (max_val - 1) * y_interval > width:
                    max_val = (width - 1) / y_interval + 1
                scaled_values = _scale_integer_samples(sample_2d, min_val, max_val)
                scaled_values = 1 + (scaled_values - 1) * y_interval
                scaled_samples["Y_SPACING"] = [f"{value}:{value+1}" for value in scaled_values]

        else:  # grid based source
            if param_name in ["X_SPACING", "Y_SPACING"]:
                sample_2d = samples[:, i : i + 1]
                scaled_values = _scale_integer_samples(sample_2d, min_val, max_val)
                if param_name == "X_SPACING":
                    x_values = scaled_values
                else:
                    if x_values is None:
                        raise ValueError("x_spacing must come before y_spacing in param_ranges")
                    # Ensure x_spacing ≤ y_spacing
                    x_final = np.minimum(x_values, scaled_values)
                    y_final = np.maximum(x_values, scaled_values)
                    scaled_samples["X_SPACING"] = [f"*:{i}" for i in x_final]
                    scaled_samples["Y_SPACING"] = [f"*:{i}" for i in y_final]
        if param_name not in ["X_SPACING", "Y_SPACING"]:
            sample_2d = samples[:, i : i + 1]
            scaled_samples[param_name] = _scale_sobol_values(
                sample_2d.flatten(), min_val, max_val
            ).tolist()
    if point_based:
        capillary_density = [
            calculate_capillary_density(
                radius_bound,
                length,
                width,
                side_length,
            )
            for i in range(n_samples)
        ]
        point_center = np.array([length // 2, width // 2, 0]).astype(int)
        distance_to_center = []
        for i in range(n_samples):
            source_site = np.array(
                [length // 2, scaled_samples["Y_SPACING"][i].split(":")[1], 0]
            ).astype(int)
            distance_to_center.append(
                calculate_distance_between_points(
                    point_center,
                    source_site,
                    side_length,
                    radius_bound,
                )
            )

    else:
        capillary_density = [
            calculate_capillary_density(
                radius_bound,
                int(scaled_samples["X_SPACING"][i].split(":")[1]),
                int(scaled_samples["Y_SPACING"][i].split(":")[1]),
                side_length,
            )
            for i in range(n_samples)
        ]
        distance_to_center = [np.nan] * n_samples
    scaled_samples["CAPILLARY_DENSITY"] = capillary_density
    scaled_samples["DISTANCE_TO_CENTER"] = distance_to_center

    return scaled_samples


def update_xml_parameters(root: ET.Element, params: dict) -> None:
    """Update XML parameters with provided values.

    Args:
        root: XML root element
        params: Dictionary of parameter values to update
    """
    # Find cancerous population element
    cancerous_pop = root.find(".//population[@id='cancerous']") or root.find(
        ".//population[@id='cancer']"
    )
    if cancerous_pop is None:
        raise ValueError("Could not find population with id='cancerous' or 'cancer' in XML")

    # Update parameters
    for param in cancerous_pop.findall("population.parameter"):
        param_id = param.get("id")
        # Handle all parameters
        for base_param, (precision, is_bounded, subfolder) in PARAM_CONFIGS.items():
            param_path = _get_param_path(base_param, subfolder)
            if param_id == param_path:
                mu_key = f"{base_param}_MU"
                sigma_key = f"{base_param}_SIGMA"

                # Handle direct parameters (those without MU/SIGMA)
                if base_param in params:
                    value = _format_param_value(params[base_param], precision, is_bounded)
                    param.set("value", value)

                # Handle normal distribution parameters
                elif mu_key in params and sigma_key in params:
                    mu = _format_param_value(params[mu_key], precision, is_bounded)
                    sigma = _format_param_value(
                        params[sigma_key], precision, False
                    )  # sigma is never bounded by 1
                    param.set("value", f"NORMAL(MU={mu},SIGMA={sigma})")
                # Only perturb mu parameters
                elif mu_key in params:
                    mu = _format_param_value(params[mu_key], precision, is_bounded)
                    sigma_default = _format_param_value(
                        PARAMS_DEFAULTS[sigma_key], precision, False
                    )
                    param.set("value", f"NORMAL(MU={mu},SIGMA={sigma_default})")
                # Only perturb sigma parameters
                elif sigma_key in params:
                    sigma = _format_param_value(params[sigma_key], precision, False)
                    mu_default = _format_param_value(PARAMS_DEFAULTS[mu_key], precision, is_bounded)
                    param.set("value", f"NORMAL(MU={mu_default},SIGMA={sigma})")
                # else:
                #    print(f"No parameter found for {param_id}")


def generate_combined_perturbations(
    cellular_params: dict,
    source_params: dict,
    template_path: str,
    output_dir: str,
) -> list:
    """Generate input XML files for combined cellular and source site perturbations.

    Args:
        cellular_params: Dictionary of cellular parameter values {param_name: [values]}
        source_params: Dictionary of source parameter values
        template_path: Path to template XML file
        output_dir: Directory to save generated XML files (can be local path or S3 URL)

    Returns:
        List of parameter log entries
    """
    inputs_dir = _ensure_inputs_dir(output_dir)

    # Read template XML
    tree = ET.parse(template_path)
    root = tree.getroot()

    # Prepare parameter logging
    param_log = []
    if cellular_params:
        n_samples = len(next(iter(cellular_params.values())))
    elif source_params:
        n_samples = len(next(iter(source_params.values())))
    else:
        return []

    # Verify that source_params has the same number of samples
    source_sample_counts = [
        len(values) for values in source_params.values() if isinstance(values, list)
    ]
    if source_sample_counts and source_sample_counts[0] != n_samples:
        raise ValueError(
            f"Cellular parameters have {n_samples} samples but source parameters have {source_sample_counts[0]} samples"
        )

    # Generate files for each parameter set
    for i in range(n_samples):
        # Create cellular parameter dictionary for this sample
        cellular_sample = {name: values[i] for name, values in cellular_params.items()}

        update_xml_parameters(root, cellular_sample)

        # Find and update source sites component
        sites_component = root.find(".//component[@id='SITES']")
        if sites_component is None:
            raise ValueError("Could not find component with id='SITES' in XML")

        # Update source parameters
        if "X_SPACING" in source_params:
            sites_component.find("component.parameter[@id='X_SPACING']").set(
                "value", str(source_params["X_SPACING"][i])
            )
        if "Y_SPACING" in source_params:
            sites_component.find("component.parameter[@id='Y_SPACING']").set(
                "value", str(source_params["Y_SPACING"][i])
            )

        # Update concentrations
        for layer_id, param_name in SOURCE_LAYER_CONFIGS:
            if param_name in source_params:
                layer = root.find(f".//layer[@id='{layer_id}']")
                for param in layer.findall("layer.parameter"):
                    if (
                        param.get("operation") == "generator"
                        or param.get("id") == "INITIAL_CONCENTRATION"
                    ):
                        param.set("value", str(source_params[param_name][i]))

        # Save modified XML locally first
        local_output_file = inputs_dir / f"input_{i+1}.xml"
        tree.write(local_output_file, encoding="utf-8", xml_declaration=True)

        # Create log entry with both cellular and source parameters
        log_entry = _create_parameter_log_entry(cellular_sample, i + 1)
        # Add source parameters to log entry
        for param_name in source_params:
            if i < len(source_params[param_name]):
                log_entry[param_name] = source_params[param_name][i]
        param_log.append(log_entry)

    return param_log


def main():

    radius = 8
    margin = 2
    hex_size = 30
    side_length = hex_size / np.sqrt(3)
    configs = [
        {
            "perturbed_config": "cellular",
            "template_path": "sample_inputs/sample_cellular_healthy.xml",
            "point_based": None,
            "y_interval": None,
            "radius_bound": None,
            "side_length": None,
        },
        {
            "perturbed_config": "source",
            "template_path": "sample_source_v3.xml",
            "point_based": False,
            "y_interval": 4,
            "radius_bound": radius + margin,
            "side_length": side_length,
        },
        {
            "perturbed_config": "combined",
            "template_path": "sample_combined_v3.xml",  # Template with both cellular and source parameters
            "point_based": False,
            "y_interval": 4,
            "radius_bound": radius + margin,
            "side_length": side_length,
        },
    ]
    if 1:
        output_dir = "inputs/cellular_healthy/"
        generate_perturbed_parameters(
            sobol_power=10,
            param_ranges=PARAM_RANGES,
            output_dir=output_dir,
            config_params=configs[0],
        )

    if 0:
        source_type = "point" if configs[1]["point_based"] else "grid"
        output_dir = f"inputs/STEM_CELL/density_source/low_low_oxygen/{source_type}"
        generate_perturbed_parameters(
            sobol_power=6,
            param_ranges=SOURCE_PARAM_RANGES,
            output_dir=output_dir,
            config_params=configs[1],
        )

    if 0:
        output_dir = "inputs/STEM_CELL/density_source/combined/grid"
        # Create a combined parameter ranges dictionary
        combined_ranges = {**PARAM_RANGES, **SOURCE_PARAM_RANGES}
        generate_perturbed_parameters(
            sobol_power=11,
            param_ranges=combined_ranges,
            output_dir=output_dir,
            config_params=configs[2],
        )

    if 0:
        metric = "doub_time"
        output_dir = f"inputs/sensitivity_analysis/{metric}"
        generate_2param_perturbation(
            sensitivity_json=f"../../../ARCADE_OUTPUT/ABC_SMC_RF_N1024_combined_grid_breast/iter_0/mi_analysis_{metric}.json",
            metric=metric,
            template_path="sample_inputs/sample_combined_v3.xml",
            perturbation_range=range(-100, 101, 10),
            output_dir=output_dir,
        )


if __name__ == "__main__":
    main()
