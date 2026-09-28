import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional, Tuple, Dict, List
import logging
from scipy.stats import multivariate_normal
from inverse_design.rf.abc_smc_rf_base import ABCSMCRFBase
from inverse_design.rf.smc_provenance import write_lineage, write_weights
from inverse_design.models.sir.sir import SIR_ABM
import multiprocessing as mp
from functools import partial
import warnings
import pandas as pd
import os

warnings.filterwarnings("ignore")


# Global function for parallel processing (must be defined at module level)
def simulate_particle(params_and_seed, simulator_params):
    """
    Simulate a single particle (used for parallel processing).

    Parameters:
    -----------
    params_and_seed : tuple
        (parameters, random_seed)
    simulator_params : dict
        Parameters for simulator initialization
    """
    params, seed = params_and_seed

    # Create local simulator instance
    simulator = SIR_ABM_Simulator(**simulator_params)
    simulator.rng = np.random.RandomState(seed)

    try:
        stats, history = simulator.simulate(params)
        return stats, history, True  # stats, success
    except Exception as e:
        logging.debug(f"Simulation failed for params {params}: {e}")
        return None, None, False


class SIR_ABM_Simulator:
    """
    Wrapper class for SIR ABM simulation with summary statistics computation.
    """

    def __init__(
        self, lattice_size=30, n_simulations=10, max_time=365, true_params=None, fixed_Pm=1.0
    ):
        """
        Initialize the SIR simulator.

        Parameters:
        -----------
        lattice_size : int
            Size of the square lattice
        n_simulations : int
            Number of simulations to run for each parameter set
        max_time : float
            Maximum simulation time
        true_params : dict, optional
            True parameters for data generation
        fixed_Pm : float
            Fixed migration rate (not inferred)
        """
        self.lattice_size = lattice_size
        self.n_simulations = n_simulations
        self.max_time = max_time
        self.fixed_Pm = fixed_Pm
        self.true_params = true_params or {"PI": 0.3, "PR": 0.005, "IIF": 0.01, "ISF": 0.49}
        self.rng = np.random.RandomState()

    def get_params_dict(self):
        """Get parameters as dictionary for parallel processing."""
        return {
            "lattice_size": self.lattice_size,
            "n_simulations": self.n_simulations,
            "max_time": self.max_time,
            "fixed_Pm": self.fixed_Pm,
            "true_params": self.true_params,
        }

    def simulate(self, params: np.ndarray) -> np.ndarray:
        """
        Run SIR ABM simulation and compute summary statistics.

        Parameters:
        -----------
        params : np.ndarray
            Parameter vector [PI, PR, IIF, ISF]

        Returns:
        --------
        statistics : np.ndarray
            Summary statistics vector
        """
        PI, PR, IIF, ISF = params

        # Run multiple simulations
        all_statistics = []
        for _ in range(self.n_simulations):
            # Create and run model
            model = SIR_ABM(
                lattice_size=self.lattice_size,
                initial_infected_fraction=IIF,
                initial_susceptible_fraction=ISF,
                PI=PI,
                PR=PR,
                Pm=self.fixed_Pm,
            )
            model.run(max_time=self.max_time)

            # Compute statistics for this run
            stats = self._compute_statistics(model.history)
            all_statistics.append(stats)

        # Average over simulations
        return np.mean(all_statistics, axis=0), model.history

    def _compute_statistics(self, history: Dict) -> np.ndarray:
        """
        Compute summary statistics from simulation history.

        We use the following statistics:
        1. Peak infected fraction
        2. Time to peak infection
        3. Final recovered fraction
        4. Area under infected curve
        5. Initial growth rate (approximate)
        """
        times = np.array(history["time"])
        S = np.array(history["S"])
        I = np.array(history["I"])
        R = np.array(history["R"])

        statistics = []

        # 1. Peak infected fraction
        try:
            peak_I = np.max(I) if len(I) > 0 else 0
            statistics.append(peak_I)

            # 2. Time to peak infection
            if len(I) > 0:
                peak_idx = np.argmax(I)
                time_to_peak = times[peak_idx] if peak_idx < len(times) else times[-1]
            else:
                time_to_peak = 0
            statistics.append(time_to_peak)
        except:
            statistics.extend([np.nan, np.nan])

        # 3. Final recovered fraction
        final_R = R[-1] if len(R) > 0 else 0
        statistics.append(final_R)

        # 4. Area under infected curve (normalized by time)
        try:
            if len(times) > 1:
                area_I = np.trapz(I, times) / (times[-1] - times[0])
            else:
                area_I = 0
            statistics.append(area_I)
        except:
            statistics.append(np.nan)

        # 5. Initial growth rate
        try:
            n_initial = max(2, int(0.2 * len(I)))
            if n_initial > 2 and len(I) > n_initial:
                log_I_initial = np.log(I[:n_initial] + 1e-10)
                t_initial = times[:n_initial]
                if len(t_initial) > 1:
                    growth_rate = np.polyfit(t_initial, log_I_initial, 1)[0]
                else:
                    growth_rate = 0
            else:
                growth_rate = 0
            statistics.append(growth_rate)
        except:
            statistics.append(np.nan)

        return np.array(statistics)

    def generate_observed_data(self) -> Tuple[np.ndarray, Dict]:
        """Generate observed data using true parameters."""
        true_param_array = np.array(
            [
                self.true_params["PI"],
                self.true_params["PR"],
                self.true_params["IIF"],
                self.true_params["ISF"],
            ]
        )
        return self.simulate(true_param_array)


class ABCSMCRF_SIR(ABCSMCRFBase):
    """
    ABC-SMC-RF implementation specifically for SIR model parameter inference with parallel processing.
    """

    def __init__(
        self,
        n_particles: int = 1000,
        n_iterations: int = 5,
        param_ranges: Optional[Dict] = None,
        n_processes: Optional[int] = None,
        output_dir: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize ABC-SMC-RF for SIR model.

        Parameters:
        -----------
        n_particles : int
            Number of particles per iteration
        n_iterations : int
            Number of ABC-SMC iterations
        param_ranges : dict, optional
            Parameter ranges for PI and PR
        n_processes : int, optional
            Number of parallel processes. If None, uses cpu_count - 1
        """
        super().__init__(n_iterations=n_iterations, **kwargs)
        self.n_particles = n_particles
        self.param_ranges = param_ranges or {
            "PI": (0.001, 0.5),
            "PR": (0.0001, 0.1),
            "IIF": (0.01, 0.5),
            "ISF": (0.49, 0.99),
        }
        self.n_parameters = 4  # PI, PR, IIF, ISF
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)

        # Storage
        self.parameter_samples = []
        self.weights = []
        self.statistics = []
        self.rf_models = []
        self.output_dir = output_dir

    def prior_sampler(self, n_samples: int) -> np.ndarray:
        """Sample from prior distribution (uniform in log space for rates)."""
        samples = np.zeros((n_samples, self.n_parameters))

        # Sample PI and PR in log space
        log_PI_min = np.log(self.param_ranges["PI"][0])
        log_PI_max = np.log(self.param_ranges["PI"][1])
        samples[:, 0] = np.exp(self.rng.uniform(log_PI_min, log_PI_max, n_samples))

        log_PR_min = np.log(self.param_ranges["PR"][0])
        log_PR_max = np.log(self.param_ranges["PR"][1])
        samples[:, 1] = np.exp(self.rng.uniform(log_PR_min, log_PR_max, n_samples))

        # Vectorized rejection sampling for IIF and ISF
        n_remaining = n_samples
        idx = 0

        while n_remaining > 0:
            # Oversample to reduce iterations
            n_try = int(n_remaining * 1.5) + 10

            iif_candidates = self.rng.uniform(
                self.param_ranges["IIF"][0], self.param_ranges["IIF"][1], n_try
            )
            isf_candidates = self.rng.uniform(
                self.param_ranges["ISF"][0], self.param_ranges["ISF"][1], n_try
            )

            # Find valid samples (IIF + ISF <= 1)
            valid_mask = (iif_candidates + isf_candidates) <= 1
            n_valid = np.sum(valid_mask)

            if n_valid > 0:
                n_to_take = min(n_valid, n_remaining)
                samples[idx : idx + n_to_take, 2] = iif_candidates[valid_mask][:n_to_take]
                samples[idx : idx + n_to_take, 3] = isf_candidates[valid_mask][:n_to_take]

                idx += n_to_take
                n_remaining -= n_to_take

        return samples

    def prior_pdf(self, params: np.ndarray) -> float:
        """Evaluate prior density."""
        PI, PR, IIF, ISF = params

        # Check bounds
        if not (self.param_ranges["PI"][0] <= PI <= self.param_ranges["PI"][1]):
            return 0.0
        if not (self.param_ranges["PR"][0] <= PR <= self.param_ranges["PR"][1]):
            return 0.0
        if not (self.param_ranges["IIF"][0] <= IIF <= self.param_ranges["IIF"][1]):
            return 0.0
        if not (self.param_ranges["ISF"][0] <= ISF <= self.param_ranges["ISF"][1]):
            return 0.0
        if IIF + ISF > 1:
            return 0.0

        # Log-uniform density for PI and PR
        density_PI = 1.0 / (PI * np.log(self.param_ranges["PI"][1] / self.param_ranges["PI"][0]))
        density_PR = 1.0 / (PR * np.log(self.param_ranges["PR"][1] / self.param_ranges["PR"][0]))

        # Uniform density for IIF and ISF (accounting for constraint)
        # The valid region has area = 0.5 * (1 - IIF_min - ISF_min)^2
        density_IIF_ISF = 2.0 / (
            (1 - self.param_ranges["IIF"][0] - self.param_ranges["ISF"][0]) ** 2
        )

        return density_PI * density_PR * density_IIF_ISF

    def perturbation_kernel(self, params: np.ndarray, iteration: int) -> np.ndarray:
        """Perturb parameters using adaptive kernel."""
        scale = 0.2 / np.sqrt(1 + iteration)

        # Work in log space for PI and PR
        perturbed = params.copy()

        # Perturb PI and PR in log space
        log_PI = np.log(params[0])
        log_PR = np.log(params[1])

        perturbed[0] = np.exp(log_PI + self.rng.normal(0, scale))
        perturbed[1] = np.exp(log_PR + self.rng.normal(0, scale))

        # Perturb IIF and ISF in linear space
        perturbed[2] = params[2] + self.rng.normal(0, scale * 0.1)
        perturbed[3] = params[3] + self.rng.normal(0, scale * 0.1)

        # Ensure within bounds
        perturbed[0] = np.clip(perturbed[0], self.param_ranges["PI"][0], self.param_ranges["PI"][1])
        perturbed[1] = np.clip(perturbed[1], self.param_ranges["PR"][0], self.param_ranges["PR"][1])
        perturbed[2] = np.clip(
            perturbed[2], self.param_ranges["IIF"][0], self.param_ranges["IIF"][1]
        )
        perturbed[3] = np.clip(
            perturbed[3], self.param_ranges["ISF"][0], self.param_ranges["ISF"][1]
        )

        # Ensure IIF + ISF <= 1
        if perturbed[2] + perturbed[3] > 1:
            # Scale them down proportionally
            total = perturbed[2] + perturbed[3]
            perturbed[2] = perturbed[2] / total * 0.99
            perturbed[3] = perturbed[3] / total * 0.99

        return perturbed

    def simulate_particles_parallel(
        self, parameters: np.ndarray, simulator_params: dict
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple particles in parallel.

        Returns:
        --------
        valid_parameters : np.ndarray
            Parameters that produced valid simulations
        valid_statistics : np.ndarray
            Corresponding statistics
        valid_history : list
            Corresponding simulation histories
        valid_indices : list
            Position in `parameters` of each surviving row. Failed simulations
            are dropped silently otherwise, which leaves the caller unable to
            say which candidate a surviving particle came from -- and so unable
            to record its parent.
        """
        n_params = len(parameters)

        # Generate random seeds for reproducibility
        seeds = self.rng.randint(0, 2**31, size=n_params)

        # Prepare inputs for parallel processing
        inputs = [(parameters[i], seeds[i]) for i in range(n_params)]

        # Run simulations in parallel
        logging.info(f"  Simulating {n_params} particles using {self.n_processes} processes...")

        with mp.Pool(processes=self.n_processes) as pool:
            # Use partial to fix simulator_params
            simulate_func = partial(simulate_particle, simulator_params=simulator_params)

            # Run parallel simulations with progress updates
            results = []
            for i, result in enumerate(pool.imap(simulate_func, inputs)):
                results.append(result)
                if (i + 1) % max(1, n_params // 10) == 0:
                    logging.info(f"    Completed {i + 1}/{n_params} simulations")

        # Collect valid results
        valid_parameters = []
        valid_statistics = []
        valid_history = []
        valid_indices = []
        # `imap` yields in submission order, so results[i] belongs to
        # parameters[i] and the index is the candidate's position.
        for i, (stats, history, success) in enumerate(results):
            if success and stats is not None:
                valid_parameters.append(parameters[i])
                valid_statistics.append(stats)
                valid_history.append(history)
                valid_indices.append(i)

        if len(valid_parameters) == 0:
            raise RuntimeError("No valid simulations produced")

        return (
            np.array(valid_parameters),
            np.array(valid_statistics),
            valid_history,
            valid_indices,
        )

    def fit(self, target_values: np.ndarray, simulator: SIR_ABM_Simulator) -> None:
        """
        Run ABC-SMC-RF algorithm for SIR model with parallel processing.

        Parameters:
        -----------
        target_values : np.ndarray
            Target summary statistics
        simulator : SIR_ABM_Simulator
            SIR simulator object
        """
        self.target_values = target_values
        self.simulator = simulator
        self.n_statistics = target_values.shape[0]

        # Get simulator parameters for parallel processing
        simulator_params = simulator.get_params_dict()

        for t in range(self.n_iterations):
            logging.info(f"ABC-SMC-RF iteration {t+1}/{self.n_iterations}")
            if not os.path.exists(f"{self.output_dir}/iter_{t}"):
                os.makedirs(f"{self.output_dir}/iter_{t}")
                os.makedirs(f"{self.output_dir}/iter_{t}/history")
            if t == 0:
                # First iteration: sample from prior
                parameters = self.prior_sampler(self.n_particles)

                # Simulate statistics in parallel
                valid_params, valid_stats, valid_history, _ = (
                    self.simulate_particles_parallel(parameters, simulator_params)
                )
                for i, history in enumerate(valid_history):
                    pd.DataFrame(history).to_csv(
                        f"{self.output_dir}/iter_{t}/history/history_{i}.csv", index=False
                    )
                self.parameter_samples.append(valid_params)
                self.statistics.append(valid_stats)
                # The prior draw has no parents, so there is no lineage to write.
                parent_ids = None

            else:
                # Subsequent iterations
                prev_params = self.parameter_samples[-1]
                prev_weights = self.weights[-1]

                # Generate candidate parameters
                candidate_params = []
                # Parent of each accepted candidate, as a position in
                # prev_params. Appended only on acceptance so it stays aligned
                # with candidate_params; rejected draws leave no trace.
                candidate_parents = []
                attempts = 0
                max_attempts = self.n_particles * 10

                while len(candidate_params) < self.n_particles and attempts < max_attempts:
                    attempts += 1

                    # Sample from previous posterior
                    idx = self.rng.choice(len(prev_params), p=prev_weights)
                    theta_star = prev_params[idx]

                    # Perturb
                    theta_candidate = self.perturbation_kernel(theta_star, t)

                    # Check prior
                    if self.prior_pdf(theta_candidate) > 0:
                        candidate_params.append(theta_candidate)
                        candidate_parents.append(int(idx))

                if len(candidate_params) < self.n_particles:
                    logging.warning(f"Only generated {len(candidate_params)} candidates")

                candidate_params = np.array(candidate_params)

                # Simulate in parallel
                valid_params, valid_stats, valid_history, valid_indices = (
                    self.simulate_particles_parallel(candidate_params, simulator_params)
                )
                # Carry the parents through the same filter the particles went
                # through, and report them as 1-based ids to match the weights.
                parent_ids = [candidate_parents[i] + 1 for i in valid_indices]
                for i, history in enumerate(valid_history):
                    pd.DataFrame(history).to_csv(
                        f"{self.output_dir}/iter_{t}/history/history_{i}.csv", index=False
                    )
                if len(valid_params) < self.n_particles * 0.5:
                    logging.warning(
                        f"Only {len(valid_params)} valid simulations out of {self.n_particles}"
                    )

                self.parameter_samples.append(valid_params)
                self.statistics.append(valid_stats)

            # Build RF model and compute weights
            self._build_rf_model(t)
            self._compute_weights(t)

            # R2.12 asks for the effective sample size on the SIR figure, and
            # R2.16 for ancestor tracing. Both are in memory here and neither
            # survived the run, so persist them beside the generation.
            generation_dir = f"{self.output_dir}/iter_{t}"
            particle_ids = list(range(1, len(self.parameter_samples[-1]) + 1))
            write_weights(generation_dir, particle_ids, self.weights[-1])
            if parent_ids is not None:
                write_lineage(
                    generation_dir,
                    particle_ids,
                    parent_ids,
                    inputs_pre_existed=False,
                )

            # Log progress
            weights = self.weights[-1]
            params = self.parameter_samples[-1]
            mean_params = np.average(params, weights=weights, axis=0)
            logging.info(
                f"  Posterior mean: PI={mean_params[0]:.4f}, PR={mean_params[1]:.4f}, "
                f"IIF={mean_params[2]:.4f}, ISF={mean_params[3]:.4f}"
            )

        return self

    def posterior_sample(self, n_samples: int = 1000) -> np.ndarray:
        """Generate samples from posterior."""
        if not self.parameter_samples:
            raise ValueError("No samples available. Call fit() first.")

        final_params = self.parameter_samples[-1]
        final_weights = self.weights[-1]

        idx = self.rng.choice(len(final_params), size=n_samples, p=final_weights)
        return final_params[idx]


# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # True parameters
    true_params = {"PI": 0.3, "PR": 0.005, "IIF": 0.23, "ISF": 0.70}
    lattice_size = 30
    n_simulations = 3
    max_time = 365
    base_dir = f"../../../SIR_OUTPUT/n{n_simulations}_t{max_time}_l{lattice_size}"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    # Save true parameters to csv
    true_df = pd.DataFrame(true_params, columns=["PI", "PR", "IIF", "ISF"], index=["true_params"])
    true_df.to_csv(f"{base_dir}/true_params.csv", index=False)

    # Create simulator
    simulator = SIR_ABM_Simulator(
        lattice_size=lattice_size,
        n_simulations=n_simulations,
        max_time=max_time,
        true_params=true_params,
    )

    # Generate observed data
    target_names = ["peak_I", "time_to_peak", "final_R", "area_I", "growth_rate"]
    print("Generating observed data...")
    if not os.path.exists(f"{base_dir}/target_values.csv"):
        target_values, history = simulator.generate_observed_data()
        target_values = np.array(target_values)
        target_df = pd.DataFrame([target_values], columns=target_names)
        target_df.to_csv(f"{base_dir}/target_values.csv", index=False)
        pd.DataFrame(history).to_csv(f"{base_dir}/history.csv", index=False)
    else:
        target_values = pd.read_csv(f"{base_dir}/target_values.csv")
        target_values = target_values.values.flatten()

    print("Target values:")
    for name, value in zip(target_names, target_values):
        print(f"  {name}: {value:.4f}")

    # Run ABC-SMC-RF with parallel processing
    n_particles = 512
    n_iterations = 10
    abc = ABCSMCRF_SIR(
        n_particles=n_particles,  # Increased for demonstration
        n_iterations=n_iterations,
        rf_type="DRF",
        n_trees=50,
        n_processes=8,
        output_dir=f"{base_dir}/n{n_particles}",
    )

    print(f"\nRunning ABC-SMC-RF with {abc.n_processes} parallel processes...")
    abc.fit(target_values, simulator)

    # Save params and statistics for each iteration to two csv files
    param_names = ["PI", "PR", "IIF", "ISF"]

    for i in range(abc.n_iterations):
        param_df = pd.DataFrame(abc.parameter_samples[i], columns=param_names)
        stat_df = pd.DataFrame(abc.statistics[i], columns=target_names)
        if not os.path.exists(f"{base_dir}/n{n_particles}/iter_{i}"):
            os.makedirs(f"{base_dir}/n{n_particles}/iter_{i}")
        param_df.to_csv(f"{base_dir}/n{n_particles}/iter_{i}/params.csv", index=False)
        stat_df.to_csv(f"{base_dir}/n{n_particles}/iter_{i}/statistics.csv", index=False)

    # Get posterior samples
    posterior_samples = abc.posterior_sample(n_samples=1000)

    # Create visualization
    fig = plt.figure(figsize=(10, 6))

    # Parameter names and true values
    param_names = ["PI", "PR", "IIF", "ISF"]
    param_labels = [
        "PI (Infection rate)",
        "PR (Recovery rate)",
        "IIF (Initial Infected Fraction)",
        "ISF (Initial Susceptible Fraction)",
    ]
    true_values = [true_params["PI"], true_params["PR"], true_params["IIF"], true_params["ISF"]]

    # Print summary
    print("\nPosterior Summary:")
    for i, param_name in enumerate(param_names):
        mean_val = np.mean(posterior_samples[:, i])
        std_val = np.std(posterior_samples[:, i])
        print(f"{param_name}: mean={mean_val:.4f}, std={std_val:.4f}, true={true_values[i]:.4f}")

    print(f"\nTrue parameter values:")
    print(
        f"PI={true_params['PI']}, PR={true_params['PR']}, "
        f"IIF={true_params['IIF']}, ISF={true_params['ISF']}"
    )

    # Calculate the error between each statistic and the target value
    error = np.abs(abc.statistics[-1] - target_values) / target_values
    overall_mean_errors = np.mean(error, axis=1)
    print(f"Overall mean error: {overall_mean_errors}")

    # Print the best parameter which minimizes the overall mean error
    best_param_idx = np.argmin(overall_mean_errors)
    print(f"Best Error = {overall_mean_errors[best_param_idx]:.6f}")
    print(f"Best statistics: {abc.statistics[-1][best_param_idx]}")
    for i, param_name in enumerate(param_names):
        print(f"{param_name}: {abc.parameter_samples[-1][best_param_idx, i]:.8f}")
