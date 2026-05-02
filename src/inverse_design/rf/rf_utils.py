import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, List, Dict, Optional
import logging
import time
from scipy import stats


def plot_posterior_1d(
    true_value: float,
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    parameter_name: str = "θ",
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot the prior and posterior distributions for a 1D parameter.

    Parameters:
    -----------
    true_value : float
        True parameter value.
    prior_samples : np.ndarray of shape (n_samples,)
        Samples from the prior distribution.
    posterior_samples : np.ndarray of shape (n_samples,)
        Samples from the posterior distribution.
    parameter_name : str, default='θ'
        Name of the parameter.
    bins : int, default=30
        Number of bins in the histogram.
    figsize : tuple of int, default=(10, 6)
        Figure size.
    """
    plt.figure(figsize=figsize)

    # Plot prior
    plt.hist(prior_samples, bins=bins, alpha=0.5, label="Prior", density=True)

    # Plot posterior
    plt.hist(posterior_samples, bins=bins, alpha=0.5, label="Posterior", density=True)

    # Plot true value
    plt.axvline(true_value, color="red", linestyle="--", label=f"True {parameter_name}")

    plt.xlabel(parameter_name)
    plt.ylabel("Density")
    plt.legend()
    plt.title(f"Prior and Posterior Distributions for {parameter_name}")
    plt.tight_layout()
    plt.show()


def plot_posterior_2d(
    true_values: np.ndarray,
    prior_samples: np.ndarray,
    posterior_samples: np.ndarray,
    parameter_names: List[str] = ["θ₁", "θ₂"],
    bins: int = 30,
    figsize: Tuple[int, int] = (12, 10),
) -> None:
    """
    Plot the prior and posterior distributions for 2D parameters.

    Parameters:
    -----------
    true_values : np.ndarray of shape (2,)
        True parameter values.
    prior_samples : np.ndarray of shape (n_samples, 2)
        Samples from the prior distribution.
    posterior_samples : np.ndarray of shape (n_samples, 2)
        Samples from the posterior distribution.
    parameter_names : List[str], default=['θ₁', 'θ₂']
        Names of the parameters.
    bins : int, default=30
        Number of bins in the histograms.
    figsize : tuple of int, default=(12, 10)
        Figure size.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Marginal distribution for parameter 1
    axes[0, 0].hist(prior_samples[:, 0], bins=bins, alpha=0.5, label="Prior", density=True)
    axes[0, 0].hist(posterior_samples[:, 0], bins=bins, alpha=0.5, label="Posterior", density=True)
    axes[0, 0].axvline(
        true_values[0], color="red", linestyle="--", label=f"True {parameter_names[0]}"
    )
    axes[0, 0].set_xlabel(parameter_names[0])
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()

    # Joint distribution
    axes[0, 1].scatter(prior_samples[:, 0], prior_samples[:, 1], alpha=0.1, label="Prior", s=10)
    axes[0, 1].scatter(
        posterior_samples[:, 0], posterior_samples[:, 1], alpha=0.3, label="Posterior", s=10
    )
    axes[0, 1].scatter(true_values[0], true_values[1], color="red", marker="*", s=200, label="True")
    axes[0, 1].set_xlabel(parameter_names[0])
    axes[0, 1].set_ylabel(parameter_names[1])
    axes[0, 1].legend()

    # Marginal distribution for parameter 2
    axes[1, 1].hist(
        prior_samples[:, 1],
        bins=bins,
        alpha=0.5,
        label="Prior",
        density=True,
        orientation="horizontal",
    )
    axes[1, 1].hist(
        posterior_samples[:, 1],
        bins=bins,
        alpha=0.5,
        label="Posterior",
        density=True,
        orientation="horizontal",
    )
    axes[1, 1].axhline(
        true_values[1], color="red", linestyle="--", label=f"True {parameter_names[1]}"
    )
    axes[1, 1].set_ylabel(parameter_names[1])
    axes[1, 1].set_xlabel("Density")
    axes[1, 1].legend()

    # KDE plot of posterior
    try:
        from scipy.stats import gaussian_kde

        xx, yy = np.mgrid[
            np.min(posterior_samples[:, 0]) : np.max(posterior_samples[:, 0]) : 100j,
            np.min(posterior_samples[:, 1]) : np.max(posterior_samples[:, 1]) : 100j,
        ]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([posterior_samples[:, 0], posterior_samples[:, 1]])
        kernel = gaussian_kde(values)
        f = np.reshape(kernel(positions).T, xx.shape)

        axes[1, 0].contourf(xx, yy, f, cmap="viridis")
        axes[1, 0].scatter(
            true_values[0], true_values[1], color="red", marker="*", s=200, label="True"
        )
        axes[1, 0].set_xlabel(parameter_names[0])
        axes[1, 0].set_ylabel(parameter_names[1])
        axes[1, 0].legend()
    except:
        axes[1, 0].text(
            0.5,
            0.5,
            "KDE plot unavailable\n(requires scipy.stats)",
            ha="center",
            va="center",
            transform=axes[1, 0].transAxes,
        )

    plt.suptitle(
        f"Prior and Posterior Distributions for {parameter_names[0]} and {parameter_names[1]}"
    )
    plt.tight_layout()
    plt.show()


def calculate_posterior_statistics(
    posterior_samples: np.ndarray,
    true_values: np.ndarray,
    parameter_names: Optional[List[str]] = None,
) -> Dict:
    """
    Calculate summary statistics for the posterior distribution.

    Parameters:
    -----------
    posterior_samples : np.ndarray of shape (n_samples, n_parameters)
        Samples from the posterior distribution.
    true_values : np.ndarray of shape (n_parameters,)
        True parameter values.
    parameter_names : List[str], optional
        Names of the parameters.

    Returns:
    --------
    stats : Dict
        Dictionary with summary statistics.
    """
    n_parameters = posterior_samples.shape[1]

    if parameter_names is None:
        parameter_names = [f"θ{i+1}" for i in range(n_parameters)]

    stats = {}

    for i in range(n_parameters):
        param_stats = {}
        param_stats["mean"] = np.mean(posterior_samples[:, i])
        param_stats["median"] = np.median(posterior_samples[:, i])
        param_stats["std"] = np.std(posterior_samples[:, i])
        param_stats["95%_CI"] = np.percentile(posterior_samples[:, i], [2.5, 97.5])
        param_stats["true_value"] = true_values[i]
        param_stats["bias"] = param_stats["mean"] - true_values[i]
        param_stats["relative_bias"] = (
            param_stats["bias"] / true_values[i] if true_values[i] != 0 else float("inf")
        )
        param_stats["in_95%_CI"] = (
            param_stats["95%_CI"][0] <= true_values[i] <= param_stats["95%_CI"][1]
        )

        stats[parameter_names[i]] = param_stats

    return stats


def print_posterior_statistics(stats: Dict) -> None:
    """
    Print summary statistics for the posterior distribution.

    Parameters:
    -----------
    stats : Dict
        Dictionary with summary statistics.
    """
    for param_name, param_stats in stats.items():
        print(f"Statistics for {param_name}:")
        print(f"  True value: {param_stats['true_value']:.4f}")
        print(f"  Posterior mean: {param_stats['mean']:.4f}")
        print(f"  Posterior median: {param_stats['median']:.4f}")
        print(f"  Posterior std: {param_stats['std']:.4f}")
        print(f"  95% CI: [{param_stats['95%_CI'][0]:.4f}, {param_stats['95%_CI'][1]:.4f}]")
        print(f"  Bias: {param_stats['bias']:.4f}")
        print(f"  Relative bias: {param_stats['relative_bias']:.4f}")
        print(f"  True value in 95% CI: {param_stats['in_95%_CI']}")
        print()


def plot_iterations(
    true_values: np.ndarray,
    smc_rf_model,
    parameter_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    max_iterations: Optional[int] = None,
) -> None:
    """
    Plot the parameter distributions across iterations.

    Parameters:
    -----------
    true_values : np.ndarray
        True parameter values.
    smc_rf_model : ABCSMCRF
        Fitted ABC-SMC-RF model.
    parameter_names : List[str], optional
        Names of the parameters.
    figsize : tuple of int, default=(15, 10)
        Figure size.
    max_iterations : int, optional
        Maximum number of iterations to plot. If None, all iterations are plotted.
    """
    n_parameters = true_values.shape[0]

    if parameter_names is None:
        parameter_names = [f"θ{i+1}" for i in range(n_parameters)]

    n_iterations = len(smc_rf_model.parameter_samples)

    if max_iterations is not None:
        n_iterations = min(n_iterations, max_iterations)

    fig, axes = plt.subplots(n_parameters, n_iterations, figsize=figsize, squeeze=False)

    for p in range(n_parameters):
        for t in range(n_iterations):
            params, weights = smc_rf_model.get_iteration_results(t)

            # Plot weighted histogram
            ax = axes[p, t]
            ax.hist(params[:, p], bins=30, weights=weights, alpha=0.7, density=True)
            ax.axvline(true_values[p], color="red", linestyle="--")

            if p == 0:
                ax.set_title(f"Iteration {t+1}")

            if t == 0:
                ax.set_ylabel(parameter_names[p])

            if p == n_parameters - 1:
                ax.set_xlabel("Value")

    plt.suptitle("Parameter Distributions Across Iterations")
    plt.tight_layout()
    plt.show()


# Example: Simulate from a simple model with Gaussian likelihood
def example_gaussian_model():
    # True parameter values
    true_mu = 5.0
    true_sigma = 2.0
    true_params = np.array([true_mu, true_sigma])

    # Define the simulator function
    def simulator(params):
        mu, sigma = params
        # Generate data
        data = np.random.normal(mu, sigma, size=100)
        # Compute summary statistics (mean, std, median, IQR)
        stats = np.array(
            [
                np.mean(data),
                np.std(data),
                np.median(data),
                np.percentile(data, 75) - np.percentile(data, 25),
            ]
        )
        return stats

    # Generate observed data with true parameters
    np.random.seed(42)
    observed_stats = simulator(true_params)

    # Prior distributions
    def prior_sampler(n_samples):
        mu = np.random.uniform(0, 10, size=n_samples)
        sigma = np.random.uniform(0.1, 5, size=n_samples)
        return np.column_stack([mu, sigma])

    def prior_pdf(params):
        mu, sigma = params
        if 0 <= mu <= 10 and 0.1 <= sigma <= 5:
            return 1.0 / (10 * 4.9)  # Uniform density
        else:
            return 0.0

    def perturbation_kernel(params):
        mu, sigma = params
        # Perturb with adaptive scales
        mu_new = mu + np.random.normal(0, 0.5)
        sigma_new = sigma + np.random.normal(0, 0.2)
        return np.array([mu_new, sigma_new])

    # Sample from prior for comparison
    prior_samples = prior_sampler(1000)

    # Initialize and fit the ABC-SMC-RF model
    print("Fitting ABC-SMC-DRF model...")
    start_time = time.time()

    smc_rf = ABCSMCRF(  # noqa: F821
        n_iterations=5,
        n_particles=200,
        rf_type="DRF",
        n_trees=100,
        min_samples_leaf=5,
        random_state=42,
        prior_sampler=prior_sampler,
        perturbation_kernel=perturbation_kernel,
        prior_pdf=prior_pdf,
    )

    smc_rf.fit(observed_stats, simulator)

    print(f"Fitting completed in {time.time() - start_time:.2f} seconds")

    # Generate posterior samples
    posterior_samples = smc_rf.posterior_sample(1000)

    # Calculate and print statistics
    stats = calculate_posterior_statistics(
        posterior_samples, true_params, parameter_names=["μ", "σ"]
    )
    print_posterior_statistics(stats)

    # Plot results
    plot_posterior_2d(true_params, prior_samples, posterior_samples, parameter_names=["μ", "σ"])

    # Plot iterations
    plot_iterations(true_params, smc_rf, parameter_names=["μ", "σ"])

    # Variable importance
    importance = smc_rf.get_variable_importance()
    plt.figure(figsize=(10, 6))
    plt.bar(["Mean", "Std", "Median", "IQR"], importance)
    plt.title("Variable Importance")
    plt.ylabel("Importance")
    plt.show()

    print("Example completed successfully!")
