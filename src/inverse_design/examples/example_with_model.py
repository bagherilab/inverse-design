import hydra
import os
import json
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import numpy as np

from inverse_design.abc.abc_with_model import ABCWithModel
from inverse_design.conf.config import BDMConfig, ABCConfig, ARCADEConfig
from inverse_design.analyze import evaluate
from inverse_design.vis.vis import plot_abc_results
from inverse_design.utils.utils import get_samples_data
from inverse_design.models.model_base import ModelRegistry
from inverse_design.common.enum import Target, Metric


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run_abc_with_model(cfg: DictConfig):
    """Example script demonstrating how to use the ABC inference"""

    model = ModelRegistry.get_model(cfg.abc.model_type)

    if cfg.abc.model_type == "BDM":
        model_config = BDMConfig.from_dictconfig(cfg.bdm)
    elif cfg.abc.model_type == "ARCADE":
        model_config = ARCADEConfig.from_dictconfig(cfg.arcade)
    else:
        # For custom models, get the appropriate config class from the model
        config_class = model.get_config_class()
        model_config = config_class.from_dictconfig(cfg.get(cfg.abc.model_type.lower()))

    abc_config = ABCConfig.from_dictconfig(cfg.abc)

    # Get targets from config or use model defaults
    if hasattr(cfg.abc, "targets") and cfg.abc.targets:
        targets = [
            Target(
                metric=Metric(target.metric), value=float(target.value), weight=float(target.weight)
            )
            for target in cfg.abc.targets
        ]
    else:
        # Use model defaults if no targets in config
        targets = model.get_default_targets()

    # Initialize ABC
    abc = ABCWithModel(model_config, abc_config, targets)
    try:
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"../results/abc_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Run inference
        param_metrics_distances_results = abc.run_inference()

        # Save all samples data
        all_samples_df = get_samples_data(
            param_metrics_distances_results=param_metrics_distances_results,
            model_type=abc_config.model_type,
            save_path=os.path.join(output_dir, "all_samples_metrics.csv"),
        )

        # Get accepted parameters using model-specific keys
        param_keys = model.get_parameter_keys()
        accepted_params = [
            {key: sample[key] for key in param_keys}
            for sample in param_metrics_distances_results
            if sample["accepted"]
        ]
        # Get best parameters
        best_sample = min(param_metrics_distances_results, key=lambda x: x["distance"])
        best_params = {key: best_sample[key] for key in param_keys}

        # Get best metrics using model-specific keys
        metric_keys = model.get_metric_keys()
        best_metrics = {
            display_key: best_sample[metric_key] for display_key, metric_key in metric_keys.items()
        }

        for param_name, value in best_params.items():
            print(f"{param_name}: {value:.3f}")

        config_dict = {
            "timestamp": timestamp,
            "targets": [
                {"metric": t.metric.value, "value": t.value, "weight": t.weight} for t in targets
            ],
            "best_parameters": best_params,
            "best_metrics": best_metrics,
            "model_config": OmegaConf.to_container(
                OmegaConf.create(model_config.__dict__), resolve=True
            ),
            "abc_config": OmegaConf.to_container(
                OmegaConf.create(abc_config.__dict__), resolve=True
            ),
        }

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

        # Calculate PDFs and compare sampling methods
        parameter_pdfs = evaluate.estimate_pdfs(accepted_params)
        sampling_results = evaluate.compare_sampling_methods(abc, parameter_pdfs, num_samples=1)

        # Save sampling results
        with open(os.path.join(output_dir, "sampling_results.json"), "w") as f:
            # Convert any numpy values to float and Metric enum to string for JSON serialization
            json_results = {}

            # Add prior information from config
            json_results["prior_info"] = {
                param_name: {"min": float(ranges.min), "max": float(ranges.max)}
                for param_name, ranges in abc_config.parameter_ranges.items()
            }

            # Add posterior information
            json_results["posterior_info"] = {
                "independent": {
                    param_name: {
                        "mean": float(kde.resample(1000).mean()),
                        "std": float(kde.resample(1000).std()),
                    }
                    for param_name, kde in parameter_pdfs["independent"].items()
                },
                "joint": {
                    "mean": [float(x) for x in parameter_pdfs["joint_mean"]],
                    "covariance": [[float(x) for x in row] for row in parameter_pdfs["joint_cov"]],
                    "parameter_names": parameter_pdfs["param_names"],
                },
            }

            # Add sampling method results
            for method, results in sampling_results.items():
                json_results[method] = {}
                for k, v in results.items():
                    if k == "metrics":
                        # Handle metrics dictionary separately to convert Metric enum keys
                        json_results[method][k] = {
                            metric.value if hasattr(metric, "value") else str(metric): float(value)
                            for metric, value in v.items()
                        }
                    else:
                        # Handle other values
                        json_results[method][k] = (
                            float(v) if isinstance(v, (np.float32, np.float64)) else v
                        )

            json.dump(json_results, f, indent=4)

        print("\nSampling Method Comparison:")
        for method, results in sampling_results.items():
            print(f"\n{method}:")
            if method == "independent_means":
                print(f"  Distance: {results['distance']:.4f}")
                print("  Metrics:")
                for metric_name, value in results["metrics"].items():
                    print(f"    {metric_name.value}: {value:.4f}")
            else:  # joint_sampling
                print(f"  Mean Distance: {results['distance_mean']:.4f}")
                print(f"  Distance Std: {results['distance_std']:.4f}")
                print("  Mean Metrics:")
                for metric_name, value in results["metrics"].items():
                    print(f"    {metric_name.value}: {value:.4f}")

        # Plot and save results
        all_metrics = [sample["cell_density"] for sample in param_metrics_distances_results]
        all_metrics += [sample["time_to_eq"] for sample in param_metrics_distances_results]
        accepted_metrics = [
            sample["cell_density"]
            for sample in param_metrics_distances_results
            if sample["accepted"]
        ]
        accepted_metrics += [
            sample["time_to_eq"] for sample in param_metrics_distances_results if sample["accepted"]
        ]
        print(parameter_pdfs)
        plot_abc_results(
            accepted_params=accepted_params,
            pdf_results=parameter_pdfs,
            all_metrics=all_metrics,
            accepted_metrics=accepted_metrics,
            targets=targets,
            abc_config=abc_config,
            model_config=model_config,
            save_dir=output_dir,
        )

        print(f"\nResults saved to: {output_dir}")

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    run_abc_with_model()
