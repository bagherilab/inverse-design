from typing import Optional, List, Dict, Union, Tuple, Callable, Any, Literal
from copy import deepcopy
from tqdm import tqdm
from pathlib import Path
import graphviz
import logging
import multiprocessing as mp
import concurrent.futures
import tempfile
import os
import re
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from inverse_design.rf.rf import RF
from inverse_design.rf.drf import DRF
from inverse_design.examples.run_simulations import run_simulations
from inverse_design.common.enum import Target
from inverse_design.rf.abc_smc_rf_base import ABCSMCRFBase
from inverse_design.rf.shard_support import (
    generation_is_analysed,
    prune_incomplete_outputs,
)
from inverse_design.rf.smc_provenance import (
    count_input_files,
    particle_ids_from_folders,
    write_lineage,
    write_weights,
)
from inverse_design.io import ArcadeRunLayout
from inverse_design.utils.s3_utils import (
    ensure_dir_exists,
    path_exists,
    read_csv,
    list_files,
    download_entire_s3_directory,
    is_s3_path,
)


class ABCSMCRF(ABCSMCRFBase):
    """
    Implementation of ABC Sequential Monte Carlo with Random Forests.

    This class implements Algorithms 5 and 6 from the paper, suitable for
    iteratively improving parameter inference using RF methods.
    """

    def __init__(
        self,
        n_iterations: int = 5,
        sobol_power: int = 2,
        n_particles: Optional[int] = None,
        rf_type: Literal["RF", "DRF"] = "DRF",
        n_trees: int = 500,
        min_samples_leaf: int = 5,
        param_ranges: Dict[str, Tuple[float, float]] = None,
        n_try: Optional[int] = None,
        random_state: Optional[int] = None,
        criterion: Literal["CART", "MMD"] = "CART",
        subsample_ratio: float = 0.5,
        n_cpu: int = 1,
        perturbation_kernel: Optional[Callable] = None,
        prior_pdf: Optional[Callable] = None,
        config_params: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Initialize the ABC-SMC-RF model.

        Parameters:
        -----------
        n_iterations : int, default=5
            Number of SMC iterations.
        sobol_power : int, default=2
            Power of the Sobol sequence.
        rf_type : {'RF', 'DRF'}, default='DRF'
            Type of random forest to use.
            - 'RF': uses ABC-RF (for univariate parameter inference)
            - 'DRF': uses ABC-DRF (for multivariate parameter inference)
        n_trees : int, default=500
            Number of trees in each random forest.
        min_samples_leaf : int, default=5
            Minimum number of samples required to be at a leaf node.
        n_try : int, optional
            Number of statistics to consider when looking for the best split.
        random_state : int, optional
            Controls the randomization in the random forests.
        criterion : {'CART', 'MMD'}, default='CART'
            The function to measure the quality of a split (for DRF only).
        perturbation_kernel : Callable, optional
            Function to perturb parameters between iterations.
            If None, parameters are perturbed by adding Gaussian noise.
        prior_pdf : Callable, optional
            Function to evaluate the prior density.
            If None, a uniform prior in [0, 1] for each parameter is assumed.
        config_params: Optional[Dict[str, Dict[str, str]]] = None
            The configuration to perturb.
        """
        super().__init__(
            n_iterations, rf_type, n_trees, min_samples_leaf, n_try, random_state, criterion
        )
        self.sobol_power = sobol_power
        # Particle count; falls back to 2**sobol_power when unset.
        self.n_particles = n_particles
        self.param_ranges = param_ranges
        self.subsample_ratio = subsample_ratio
        self.perturbation_kernel = (
            perturbation_kernel
            if perturbation_kernel is not None
            else self._default_perturbation_kernel
        )
        self.prior_pdf = prior_pdf if prior_pdf is not None else self._default_prior_pdf
        self.config_params = config_params
        self.scaler = None
        self.n_cpu = n_cpu
        self.source_param_list = [
            "X_SPACING",
            "Y_SPACING",
            "CAPILLARY_DENSITY",
            "DISTANCE_TO_CENTER",
            "OXYGEN_CONCENTRATION",
            "GLUCOSE_CONCENTRATION",
        ]
        self.param_list = [
            param for param in list(param_ranges.keys()) if param not in self.source_param_list
        ]

    def _default_perturbation_kernel(self, parameters: np.ndarray) -> np.ndarray:
        """
        Default perturbation kernel (Gaussian noise).

        Parameters:
        -----------
        parameters : np.ndarray of shape (n_parameters,)
            Parameters to perturb.

        Returns:
        --------
        perturbed : np.ndarray of shape (n_parameters,)
            Perturbed parameters.
        """
        # Scale the perturbation based on the iteration
        if not hasattr(self, "current_iteration"):
            scale = 0.1
        else:
            # Reduce scale as iterations progress
            scale = 0.1 / np.sqrt(1 + self.current_iteration)

        return parameters + scale * self.rng.normal(0, 1, size=parameters.shape)

    def _default_prior_pdf(self, parameters: np.ndarray) -> float:
        """
        Default prior PDF (uniform in [0, 1] for each parameter).

        Parameters:
        -----------
        parameters : np.ndarray of shape (n_parameters,)
            Parameters to evaluate.

        Returns:
        --------
        density : float
            Prior density at the given parameters.
        """
        # Check if all parameters are in [0, 1]
        if np.all((parameters >= 0) & (parameters <= 1)):
            return 1.0
        else:
            return 0.0

    def _run_parallel_simulations(
        self, input_dir: str, output_dir: str, jar_path: str
    ) -> List[str]:
        """Run ARCADE simulations in parallel."""
        from inverse_design.examples.run_simulations import run_simulations

        all_output_names = [f"input_{i}" for i in range(1, self.n_samples + 1)]
        if path_exists(output_dir):
            if not is_s3_path(output_dir):
                # A job killed mid-simulation leaves a directory that exists but
                # holds a truncated trajectory; without this it would be counted
                # as finished. See shard_support for why this matters.
                prune_incomplete_outputs(output_dir, input_dir)
            current_output_names = list_files(output_dir + "/inputs", "input_*")
            missing_output_indices = [
                int(f.split("_")[-1].split(".")[0])
                for f in all_output_names
                if f not in current_output_names
            ]
            if len(missing_output_indices) == 0:
                print(
                    f"ARCADE simulations for {output_dir} already exist, and have {len(current_output_names)} outputs (target = {self.n_samples}), skipping generation"
                )
                return
            else:
                print(
                    f"ARCADE simulations for {output_dir} exist but have {len(current_output_names)} outputs, expected {self.n_samples}"
                )
        else:
            ensure_dir_exists(output_dir)
            missing_output_indices = list(range(1, self.n_samples + 1))

        if os.environ.get("DED_PREPARE_ONLY"):
            # Sharded mode: this process exists only to replay finished
            # generations and lay down the next generation's inputs. The
            # simulations themselves belong to the shard jobs, so stop here
            # rather than running 5000 of them on one node.
            print(
                f"DED_PREPARE_ONLY: {output_dir} needs "
                f"{len(missing_output_indices)} simulations; leaving them to the "
                f"shard jobs and exiting."
            )
            logging.info(
                "DED_PREPARE_ONLY stop: iteration=%d pending=%d output_dir=%s",
                self.current_iteration,
                len(missing_output_indices),
                output_dir,
            )
            marker = os.environ.get("DED_PREPARE_MARKER")
            if marker:
                # The submitting script needs to know which generation to hand
                # to the shards; parsing it back out of the log would be
                # fragile.
                import json

                with open(marker, "w") as handle:
                    json.dump(
                        {
                            "iteration": self.current_iteration,
                            "input_dir": input_dir,
                            "output_dir": output_dir,
                            "pending": len(missing_output_indices),
                            "n_particles": self.n_samples,
                        },
                        handle,
                    )
            sys.stdout.flush()
            raise SystemExit(0)

        # Run simulations

        run_simulations(
            input_dir=input_dir + "/inputs",
            output_dir=output_dir,
            jar_path=jar_path,
            max_workers=self.n_cpu,
            running_index=missing_output_indices,
        )

    @property
    def n_samples(self) -> int:
        """Particles per generation: the override if given, else 2**sobol_power."""
        if self.n_particles is not None:
            return self.n_particles
        return 2**self.sobol_power

    def fit(
        self,
        target_names: List[str],
        target_values: List[float],
        input_dir: str,
        output_dir: str,
        jar_path: str,
        timestamps: List[str],
        peak_name: str,
    ) -> None:
        """
        Run the ABC-SMC-RF algorithm with ARCADE simulations.

        Parameters:
        -----------
        target_names : List[str]
            List of target names
        target_values : List[float]
            List of target values
        input_dir : str
            Directory containing input XML files
        output_dir : str
            Directory for simulation outputs
        jar_path : str
            Path to ARCADE jar file
        timestamps : List[str]
            List of timestamps to analyze (e.g., ["000000", "000720", ...])
        peak_name : str
            Name of the peak to use for clustering
        """
        self.target_names = target_names
        self.target_values = target_values
        self.n_statistics = len(target_names)
        self.peak_name = peak_name
        # Run iterations
        for t in range(self.n_iterations):
            self.current_iteration = t
            logging.info(f"Running ABC-SMC-RF iteration {t+1}/{self.n_iterations}")

            # First iteration: sample from prior
            if t == 0:
                self._first_iteration(input_dir, output_dir, jar_path, timestamps)
            # Subsequent iterations: sample from previous posterior and perturb
            else:
                self._subsequent_iteration(input_dir, output_dir, jar_path, timestamps)

            # Build random forest with current samples
            self._build_rf_model(t)
            # Compute weights for current samples
            self._compute_weights(t)
            # ESS decides how much of the nominal particle count is real; a
            # density estimated from a degenerate generation is a few particles
            # plus kernel smoothing. Written every replay since it follows
            # deterministically from the forest that just produced it.
            write_weights(
                str(Path(output_dir) / f"iter_{t}"),
                particle_ids=self.particle_ids[t] if t < len(self.particle_ids) else [],
                weights=self.weights[t],
            )

            logging.info(
                f"Iteration {t+1} completed with {len(self.parameter_samples[t])} particles"
            )

        return self

    def _remove_invalid_rows(
        self, statistics: np.ndarray, valid_parameters: pd.DataFrame
    ) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray]:
        """
        Remove rows containing NaN or infinite values.

        Also returns the positions that survived, so callers can carry the
        particles' input indices through the same filtering instead of
        re-deriving which simulations were dropped.
        """
        indices = np.where(np.isnan(statistics).any(axis=1) | np.isinf(statistics).any(axis=1))[0]
        kept = np.setdiff1d(np.arange(len(statistics)), indices)
        statistics = np.delete(statistics, indices, axis=0)
        valid_parameters = valid_parameters.drop(valid_parameters.index[indices])
        return statistics, valid_parameters, kept

    def normalize_statistics(self, statistics: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """
        Normalize statistics to be between 0 and 1.
        """
        scaler = StandardScaler()
        scaler.fit(statistics)
        return scaler.transform(statistics), scaler

    def _analyze_simulation_results(
        self, input_dir: str, output_dir: str, dir_postfix: str, timestamps: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze simulation results and extract statistics and parameters.

        Parameters:
        -----------
        output_dir : str
            Base output directory
        dir_postfix : str
            Directory suffix for current iteration
        timestamps : List[str]
            List of timestamps to analyze

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            DataFrames of statistics and valid parameters
        """
        from inverse_design.analyze.save_aggregated_results import SimulationMetrics

        out_layout = ArcadeRunLayout.from_root(output_dir)
        in_layout = ArcadeRunLayout.from_root(input_dir)
        out_iter = out_layout.iteration_subdir(dir_postfix)
        in_iter = in_layout.iteration_subdir(dir_postfix)
        sim_folders = sorted(
            [f for f in out_iter.glob("inputs/input_*")],
            key=lambda x: int(re.search(r"input_(\d+)", x.name).group(1)),
        )
        sim_folders = sim_folders[: self.n_samples]
        final_metrics = out_layout.final_metrics_csv(dir_postfix)
        all_param = out_layout.all_param_df_csv(dir_postfix)
        if not path_exists(str(final_metrics)):
            metrics_calculator = SimulationMetrics(str(out_iter), str(in_iter))
            metrics_calculator.analyze_all_simulations(timestamps, sim_folders)
            metrics_calculator.extract_and_save_parameters(
                sim_folders, self.param_list, self.source_param_list
            )
        statistics = read_csv(str(final_metrics))
        valid_parameters = read_csv(str(all_param))
        # input_folder is the only link between a row and the simulation it came
        # from, and it is dropped just below. Keep it so lineage and weights can
        # be written against real input indices rather than row positions.
        self._pending_particle_ids = particle_ids_from_folders(
            valid_parameters["input_folder"]
        )
        statistics.drop(columns=["input_folder", "states"], inplace=True)
        valid_parameters.drop(columns=["input_folder"], inplace=True)
        if len(valid_parameters.columns) > len(self.param_ranges.keys()):
            valid_parameters = valid_parameters[self.param_ranges.keys()]

        if self.config_params["point_based"]:
            valid_parameters.drop(columns=["CAPILLARY_DENSITY"], inplace=True)
        else:
            valid_parameters.drop(columns=["DISTANCE_TO_CENTER"], inplace=True)
        return statistics, valid_parameters

    def _first_iteration(
        self, input_dir: str, output_dir: str, jar_path: str, timestamps: List[str]
    ) -> None:
        """Execute the first iteration of ABC-SMC-RF (sampling from prior)."""
        # Generate input files using prior sampler
        from inverse_design.utils.create_input_files import generate_perturbed_parameters

        # Generate input XML files
        dir_postfix = f"iter_{self.current_iteration}"
        iter_in = str(Path(input_dir) / dir_postfix)
        iter_out = str(Path(output_dir) / dir_postfix)
        if generation_is_analysed(iter_out):
            # Replaying a settled generation. Generating inputs here would draw
            # a fresh Sobol set and write it over - or in place of - the one the
            # recorded results came from, which is only harmless for as long as
            # final_metrics.csv survives to keep anyone from re-deriving
            # parameters out of those XMLs.
            logging.info("Skipping input generation for %s: already analysed", iter_in)
        else:
            generate_perturbed_parameters(
                sobol_power=self.sobol_power,
                param_ranges=self.param_ranges,
                config_params=self.config_params,
                output_dir=iter_in,
            )
        # Run simulations in parallel
        self._run_parallel_simulations(iter_in, iter_out, jar_path)

        statistics, valid_parameters = self._analyze_simulation_results(
            input_dir, output_dir, dir_postfix, timestamps
        )
        # Debug: Check for infinite values
        target_stats_array = np.array(
            [statistics[target_name] for target_name in self.target_names]
        ).T
        target_stats_array, valid_parameters, kept = self._remove_invalid_rows(
            target_stats_array, valid_parameters
        )
        self.particle_ids.append([self._pending_particle_ids[i] for i in kept])
        target_stats, self.scaler = self.normalize_statistics(target_stats_array)
        self.target_values = self.scaler.transform(
            np.array(self.target_values).reshape(-1, len(self.target_values))
        ).flatten()
        self.parameter_samples.append(np.array(valid_parameters))
        self.parameter_columns = list(valid_parameters.columns)
        self.statistics.append(target_stats)

    def _subsequent_iteration(
        self, input_dir: str, output_dir: str, jar_path: str, timestamps: List[str]
    ) -> None:
        """
        Execute a subsequent iteration of ABC-SMC-RF (sampling from previous posterior).
        """
        from inverse_design.utils.create_input_files import generate_input_files
        from inverse_design.analyze.source_metrics import calculate_capillary_density

        prev_parameters = self.parameter_samples[-1]
        prev_weights = self.weights[-1]

        # Generate candidate parameters
        n_candidates = self.n_samples
        parameters = np.zeros((n_candidates, prev_parameters.shape[1]), dtype=object)
        # Parent of each accepted candidate, as a position in prev_parameters.
        # Rejected draws leave no trace, so this is appended to only on accept
        # and stays aligned with the rows of `parameters`.
        parent_positions: List[int] = []

        i = 0
        while i < n_candidates:
            # Sample from previous posterior
            # print(f"Attempting to sample from previous posterior {i+1}/{n_candidates}")
            idx = self.rng.choice(len(prev_parameters), p=prev_weights)
            theta_star = prev_parameters[idx]
            # Perturb the parameters
            # for idx in range(len(theta_star)):
            #    print(f"param_columns[{idx}]: {self.parameter_columns[idx]}, theta_star[{idx}]: {theta_star[idx]}")
            theta_candidate = self.perturbation_kernel(
                params=theta_star,
                param_columns=self.parameter_columns,
                param_ranges=self.param_ranges,
                iteration=self.current_iteration,
                max_iterations=self.n_iterations,
                config_params=self.config_params,
            )
            updated_param_ranges = {
                param_name: (min_val, max_val)
                for param_name, (min_val, max_val) in self.param_ranges.items()
                if param_name in self.parameter_columns
            }
            if 0:
                for param_name, (min_val, max_val) in updated_param_ranges.items():
                    if param_name in ["X_SPACING", "Y_SPACING"]:
                        if self.config_params["point_based"] and param_name == "Y_SPACING":
                            original_y_spacing = int(
                                theta_candidate[self.parameter_columns.index(param_name)].split(
                                    ":"
                                )[0]
                            )
                            if original_y_spacing < min_val or original_y_spacing > max_val:
                                print(
                                    f"Parameter {param_name} is outside the range {min_val} to {max_val}, got {theta_candidate[self.parameter_columns.index(param_name)]}"
                                )
                    else:
                        if (
                            theta_candidate[self.parameter_columns.index(param_name)] < min_val
                            or theta_candidate[self.parameter_columns.index(param_name)] > max_val
                        ):
                            print(
                                f"Parameter {param_name} is outside the range {min_val} to {max_val}, got {theta_candidate[self.parameter_columns.index(param_name)]}"
                            )
            prior_density = self.prior_pdf(
                theta_candidate,
                self.parameter_columns,
                param_ranges=updated_param_ranges,
                config_params=self.config_params,
            )
            if prior_density > 0:
                parameters[i] = theta_candidate
                parent_positions.append(int(idx))
                i += 1

        dir_postfix = f"iter_{self.current_iteration}"
        input_param_names = list(self.parameter_columns)
        if self.config_params["point_based"]:
            x_center = int((6 * self.config_params["radius_bound"] - 3) / 2)
            x_spacing_value = f"{x_center-1}:{x_center+1}"
            if "X_SPACING" in input_param_names:
                x_spacing_index = input_param_names.index("X_SPACING")
                for param in parameters:
                    param[x_spacing_index] = x_spacing_value
            else:
                added_params = np.zeros((n_candidates, parameters.shape[1] + 1), dtype=object)
                for i, param in enumerate(parameters):
                    added_params[i] = np.insert(param, 0, x_spacing_value)
                parameters = added_params
                input_param_names = ["X_SPACING", *input_param_names]

        if (
            "X_SPACING" in input_param_names
            and "Y_SPACING" in input_param_names
            and "CAPILLARY_DENSITY" in input_param_names
        ):
            x_spacing_index = input_param_names.index("X_SPACING")
            y_spacing_index = input_param_names.index("Y_SPACING")
            capillary_index = input_param_names.index("CAPILLARY_DENSITY")
            for param in parameters:
                capillary_density = calculate_capillary_density(
                    radius_bound=self.config_params["radius_bound"],
                    length_spacing=int(param[x_spacing_index].split(":")[1]),
                    width_spacing=int(param[y_spacing_index].split(":")[1]),
                    side_length=self.config_params["side_length"],
                )
                param[capillary_index] = capillary_density
        iter_in = str(Path(input_dir) / dir_postfix)
        iter_out = str(Path(output_dir) / dir_postfix)
        # Whether the inputs already exist decides whether the draws above
        # become simulations or get thrown away, and so whether the lineage
        # just computed describes this generation or a replay of it.
        already_analysed = generation_is_analysed(iter_out)
        inputs_pre_existed = already_analysed or count_input_files(iter_in) >= n_candidates
        if already_analysed:
            # See _first_iteration. The draws above belong to a replay and are
            # discarded; writing them would leave XMLs that disagree with the
            # parameters this generation's results were actually produced from.
            logging.info("Skipping input generation for %s: already analysed", iter_in)
        else:
            generate_input_files(
                param_names=input_param_names,
                param_values=parameters,
                abc_param_ranges=self.param_ranges,
                output_dir=iter_in,
                config_params=self.config_params,
            )
        # Parent positions index the previous generation's surviving particles;
        # translate them to that generation's input indices so the lineage joins
        # to simulations rather than to array offsets.
        prev_ids = self.particle_ids[-1] if self.particle_ids else []
        write_lineage(
            iter_in,
            child_ids=list(range(1, n_candidates + 1)),
            parent_ids=[
                prev_ids[p] if p < len(prev_ids) else -1 for p in parent_positions
            ],
            inputs_pre_existed=inputs_pre_existed,
        )
        self._run_parallel_simulations(iter_in, iter_out, jar_path)

        # Analyze results and get statistics and parameters
        statistics, valid_parameters = self._analyze_simulation_results(
            input_dir, output_dir, dir_postfix, timestamps
        )
        target_stats = np.array([statistics[target_name] for target_name in self.target_names]).T
        target_stats, valid_parameters, kept = self._remove_invalid_rows(
            target_stats, valid_parameters
        )
        self.particle_ids.append([self._pending_particle_ids[i] for i in kept])
        target_stats = self.scaler.transform(target_stats)
        if len(valid_parameters) < n_candidates:
            logging.warning(
                f"Only {len(valid_parameters)} valid simulations out of {n_candidates} attempts"
            )
        self.parameter_samples.append(np.array(valid_parameters))

        self.statistics.append(target_stats)

    def _compute_weights(self, t: int) -> None:
        """
        Compute weights for the current samples.

        Parameters:
        -----------
        t : int
            Current iteration index.
        """
        if self.rf_type == "RF":
            # Combine weights from separate RF models
            combined_weights = np.ones(self.n_samples)

            for p, model in enumerate(self.rf_models[t]):
                weights_p = model.predict_weights(self.observed_statistics)
                combined_weights *= weights_p

            # Renormalize
            if np.sum(combined_weights) > 0:
                combined_weights /= np.sum(combined_weights)

            self.weights.append(combined_weights)

        else:  # DRF
            model = self.rf_models[t][0]
            weights = model.predict_weights(self.target_values)
            self.weights.append(weights)

    def posterior_sample(self, n_samples: int = 1000) -> np.ndarray:
        """
        Generate samples from the final posterior distribution.

        Parameters:
        -----------
        n_samples : int, default=1000
            Number of samples to generate.

        Returns:
        --------
        samples : np.ndarray of shape (n_samples, n_parameters)
            Samples from the posterior distribution.
        """
        if not self.parameter_samples:
            raise ValueError("No samples available. Call fit() first.")

        final_parameters = self.parameter_samples[-1]
        final_weights = self.weights[-1]

        idx = self.rng.choice(len(final_parameters), size=n_samples, p=final_weights)
        return final_parameters[idx]
