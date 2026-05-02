# ABC-RF: Approximate Bayesian Computation with Random Forests

This package implements Approximate Bayesian Computation (ABC) methods that use Random Forests (RF) for parameter inference, based on the paper "Approximate Bayesian Computation sequential Monte Carlo via random forests" by Dinh et al. (2024).

## Features

- **ABC-RF**: Implementation of the ABC Random Forest algorithm for univariate parameter inference
- **ABC-DRF**: Implementation of the ABC Distributional Random Forest algorithm for multivariate parameter inference
- **ABC-SMC-RF**: Sequential Monte Carlo framework that incorporates RF methods for iterative inference

## Installation

```bash
pip install abc-rf
```

## Quick Start

```python
import numpy as np
from abc_rf import ABCSMCRF

# Define your simulator function
def simulator(params):
    mu, sigma = params
    # Generate data
    data = np.random.normal(mu, sigma, size=100)
    # Return summary statistics
    return np.array([np.mean(data), np.std(data), np.median(data)])

# Define prior sampler and pdf
def prior_sampler(n_samples):
    return np.random.uniform(0, 10, size=(n_samples, 2))

def prior_pdf(params):
    if np.all((params >= 0) & (params <= 10)):
        return 0.01  # Uniform prior density
    else:
        return 0.0

# Generate observed data (in real applications this would be your actual data)
true_params = np.array([5.0, 2.0])
observed_stats = simulator(true_params)

# Initialize and run ABC-SMC-RF
smc_rf = ABCSMCRF(
    n_iterations=5, 
    n_particles=1000,
    rf_type='DRF',  # Use 'RF' for univariate inference
    prior_sampler=prior_sampler,
    prior_pdf=prior_pdf
)

# Fit the model
smc_rf.fit(observed_stats, simulator)

# Get posterior samples
posterior_samples = smc_rf.posterior_sample(1000)

# Analyze results
print(f"True parameters: {true_params}")
print(f"Posterior means: {np.mean(posterior_samples, axis=0)}")
print(f"Posterior standard deviations: {np.std(posterior_samples, axis=0)}")
```

## Class Reference

### BaseABCRF

Base class for ABC Random Forest implementations.

```python
BaseABCRF(n_trees=500, min_samples_leaf=5, n_try=None, random_state=None)
```

### ABCRF

Implementation of ABC Random Forest for single parameter inference.

```python
ABCRF(n_trees=500, min_samples_leaf=5, n_try=None, random_state=None)
```

### ABCDRF

Implementation of ABC Distributional Random Forest for multivariate parameter inference.

```python
ABCDRF(n_trees=500, min_samples_leaf=5, n_try=None, random_state=None, criterion='CART', subsample_ratio=0.5, n_fourier_features=50)
```

### ABCSMCRF

Implementation of ABC Sequential Monte Carlo with Random Forests.

```python
ABCSMCRF(n_iterations=5, n_particles=1000, rf_type='DRF', n_trees=500, min_samples_leaf=5, n_try=None, random_state=None, criterion='CART', prior_sampler=None, perturbation_kernel=None, prior_pdf=None)
```

## Key Methods

- `fit(parameters, statistics)`: Fit the RF model with parameter-statistic pairs
- `predict_weights(observed_statistics)`: Compute weights for particles
- `posterior_sample(n_samples)`: Generate samples from the posterior distribution
- `variable_importance()`: Get the importance of each summary statistic

## Citation

If you use this package, please cite:

```
Dinh, K. N., Xiang, Z., Liu, Z., & Tavaré, S. (2024). Approximate Bayesian Computation sequential Monte Carlo via random forests. arXiv:2406.15865v1 [stat.CO]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
