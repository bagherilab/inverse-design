import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np

GENERIC_PROFILE = "generic"
S3_COCULTURE_PROFILE = "s3-coculture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ARCADE input XML files from a parameter profile."
    )
    parser.add_argument(
        "--profile",
        choices=(GENERIC_PROFILE, S3_COCULTURE_PROFILE),
        default=GENERIC_PROFILE,
        help=("Parameter profile to use. " f"Default: {GENERIC_PROFILE}."),
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        help="Path to the XML template used to generate inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where generated inputs and logs will be written.",
    )
    parser.add_argument(
        "--sobol-power",
        type=int,
        default=8,
        help="Generate 2^sobol_power samples. Default: 8.",
    )
    parser.add_argument(
        "--source-mode",
        choices=("grid", "point"),
        default="grid",
        help="Source perturbation mode for the generic profile. Default: grid.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=8,
        help="Tumor radius used to derive source geometry for the generic profile. Default: 8.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=2,
        help="Margin added to radius to compute radius_bound for the generic profile. Default: 2.",
    )
    parser.add_argument(
        "--hex-size",
        type=float,
        default=30,
        help="Hex size used to derive side_length. Default: 30.",
    )
    parser.add_argument(
        "--y-interval",
        type=int,
        default=4,
        help="Y spacing interval for generic point/grid source sampling. Default: 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sobol seed. Default: 42.",
    )
    parser.add_argument(
        "--params",
        nargs="+",
        help=(
            "Optional subset of parameters to perturb. "
            "If omitted, the full parameter profile is used."
        ),
    )
    parser.add_argument(
        "--params-file",
        type=Path,
        help=(
            "Optional JSON file describing which parameters to perturb for the generic profile. "
            "Supports a JSON list of parameter names, "
            '"{\\"params\\": [...]}" or '
            '"{\\"ranges\\": {\\"PARAM\\": [min, max]}}", '
            'or a direct {"PARAM": [min, max]} mapping.'
        ),
    )
    parser.add_argument(
        "--chain-based",
        action="store_true",
        help="Use chain-based combined generation for the generic profile.",
    )
    parser.add_argument(
        "--lr-summary-json",
        type=Path,
        help="Path to lr_summary JSON file. Required with --chain-based.",
    )
    parser.add_argument(
        "--peak-name",
        type=str,
        help="Peak name label for chain-based generation. Required with --chain-based.",
    )
    parser.add_argument(
        "--list-params",
        action="store_true",
        help="Print available parameter names for the selected profile and exit.",
    )
    parser.add_argument(
        "--continuous-samples-per-discrete-combo",
        type=int,
        help=(
            "For the s3-coculture profile, enumerate all selected discrete parameter "
            "combinations and draw this many continuous Sobol samples for each one."
        ),
    )
    return parser.parse_args()


def load_profile_parameter_space(profile: str) -> dict[str, Any]:
    if profile == S3_COCULTURE_PROFILE:
        module = importlib.import_module("inverse_design.analyze.immune_param_ranges")
        return module.S3_COCULTURE_PARAM_SPECS.copy()

    module = importlib.import_module("inverse_design.config.parameter_config_mu_only")
    return {**module.PARAM_RANGES, **module.SOURCE_PARAM_RANGES}


def build_config(args: argparse.Namespace) -> dict:
    radius_bound = args.radius + args.margin
    side_length = args.hex_size / np.sqrt(3)
    return {
        "perturbed_config": "combined",
        "template_path": str(args.template_path),
        "point_based": args.source_mode == "point",
        "y_interval": args.y_interval,
        "radius_bound": radius_bound,
        "side_length": side_length,
    }


def validate_param_names(
    param_names: list[str],
    available_parameter_space: dict[str, Any],
) -> None:
    missing_params = [param for param in param_names if param not in available_parameter_space]
    if missing_params:
        available_names = ", ".join(sorted(available_parameter_space))
        missing_names = ", ".join(missing_params)
        raise ValueError(
            f"Unknown parameter(s): {missing_names}\n"
            f"Available parameter names:\n{available_names}"
        )


def normalize_explicit_ranges(
    raw_ranges: dict[str, Any],
    available_parameter_space: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    validate_param_names(list(raw_ranges), available_parameter_space)
    normalized_ranges = {}
    for param_name, bounds in raw_ranges.items():
        if not isinstance(bounds, list | tuple) or len(bounds) != 2:
            raise ValueError(f"Range for {param_name} must be a 2-item list like [min, max].")
        min_val, max_val = bounds
        if not isinstance(min_val, int | float) or not isinstance(max_val, int | float):
            raise ValueError(f"Range for {param_name} must contain only numeric values.")
        normalized_ranges[param_name] = (min_val, max_val)
    return normalized_ranges


def load_param_ranges_from_file(
    path: Path,
    available_parameter_space: dict[str, Any],
) -> dict[str, tuple[float, float]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, list):
        validate_param_names(payload, available_parameter_space)
        return {param: available_parameter_space[param] for param in payload}

    if not isinstance(payload, dict):
        raise ValueError(
            "--params-file must contain either a JSON list of parameter names " "or a JSON object."
        )

    if "params" in payload or "ranges" in payload:
        selected_ranges = {}
        if "params" in payload:
            if not isinstance(payload["params"], list):
                raise ValueError('"params" in --params-file must be a JSON list.')
            validate_param_names(payload["params"], available_parameter_space)
            selected_ranges.update(
                {param: available_parameter_space[param] for param in payload["params"]}
            )

        if "ranges" in payload:
            if not isinstance(payload["ranges"], dict):
                raise ValueError('"ranges" in --params-file must be a JSON object.')
            selected_ranges.update(
                normalize_explicit_ranges(payload["ranges"], available_parameter_space)
            )

        if selected_ranges:
            return selected_ranges

        raise ValueError(
            '--params-file object must contain a non-empty "params" list, '
            'a non-empty "ranges" object, or both.'
        )

    return normalize_explicit_ranges(payload, available_parameter_space)


def select_parameter_space(args: argparse.Namespace) -> dict[str, Any]:
    available_parameter_space = load_profile_parameter_space(args.profile)

    if args.params_file is not None:
        return load_param_ranges_from_file(args.params_file, available_parameter_space)

    if not args.params:
        return available_parameter_space

    validate_param_names(args.params, available_parameter_space)
    return {param: available_parameter_space[param] for param in args.params}


def load_lr_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_args(args: argparse.Namespace) -> None:
    if args.list_params:
        return

    if args.template_path is None:
        raise ValueError("--template-path is required unless --list-params is used.")
    if args.output_dir is None:
        raise ValueError("--output-dir is required unless --list-params is used.")
    if args.params is not None and args.params_file is not None:
        raise ValueError("Use either --params or --params-file, not both.")

    if args.profile == S3_COCULTURE_PROFILE:
        if args.chain_based:
            raise ValueError("--chain-based is not supported for the s3-coculture profile.")
        if args.params_file is not None:
            raise ValueError(
                "--params-file is not supported for the s3-coculture profile. "
                "Edit inverse_design/analyze/immune_param_ranges.py or use --params to select a subset."
            )
        if (
            args.continuous_samples_per_discrete_combo is not None
            and args.continuous_samples_per_discrete_combo < 1
        ):
            raise ValueError("--continuous-samples-per-discrete-combo must be at least 1.")
        return

    if args.continuous_samples_per_discrete_combo is not None:
        raise ValueError(
            "--continuous-samples-per-discrete-combo is only supported for the s3-coculture profile."
        )

    if args.chain_based:
        if args.lr_summary_json is None:
            raise ValueError("--lr-summary-json is required with --chain-based.")
        if args.peak_name is None:
            raise ValueError("--peak-name is required with --chain-based.")


def print_available_params(profile: str) -> None:
    parameter_space = load_profile_parameter_space(profile)
    print(f"Available parameters for profile '{profile}':")
    for param_name in sorted(parameter_space):
        print(param_name)


def main() -> None:
    args = parse_args()
    validate_args(args)

    if args.list_params:
        print_available_params(args.profile)
        return

    parameter_space = select_parameter_space(args)

    if args.profile == S3_COCULTURE_PROFILE:
        from inverse_design.utils.create_input_files import (
            generate_s3_coculture_mixed_perturbations,
            generate_s3_coculture_perturbations,
        )

        if args.continuous_samples_per_discrete_combo is not None:
            generate_s3_coculture_mixed_perturbations(
                continuous_samples_per_discrete_combo=args.continuous_samples_per_discrete_combo,
                param_specs=parameter_space,
                template_path=str(args.template_path),
                output_dir=str(args.output_dir),
                seed=args.seed,
            )
        else:
            generate_s3_coculture_perturbations(
                sobol_power=args.sobol_power,
                param_specs=parameter_space,
                template_path=str(args.template_path),
                output_dir=str(args.output_dir),
                seed=args.seed,
            )
        return

    config_params = build_config(args)

    if args.chain_based:
        from inverse_design.utils.create_input_files import (
            generate_chain_based_perturbed_parameters,
        )

        lr_summary = load_lr_summary(args.lr_summary_json)
        generate_chain_based_perturbed_parameters(
            sobol_power=args.sobol_power,
            param_ranges=parameter_space,
            config_params=config_params,
            lr_summary=lr_summary,
            peak_name=args.peak_name,
            output_dir=str(args.output_dir),
            seed=args.seed,
        )
    else:
        from inverse_design.utils.create_input_files import generate_perturbed_parameters

        generate_perturbed_parameters(
            sobol_power=args.sobol_power,
            param_ranges=parameter_space,
            config_params=config_params,
            output_dir=str(args.output_dir),
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
