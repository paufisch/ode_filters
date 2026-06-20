# ODE Filters

[![PyPI](https://img.shields.io/pypi/v/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![Python](https://img.shields.io/pypi/pyversions/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![CI](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://paufisch.github.io/ode_filters/)
[![Coverage](https://codecov.io/gh/paufisch/ode_filters/branch/main/graph/badge.svg)](https://codecov.io/gh/paufisch/ode_filters)

A JAX-based implementation of probabilistic ODE solvers using Gaussian filtering and smoothing. This package provides tools for solving ordinary differential equations while quantifying uncertainty through Bayesian inference.

## Features

- **Pure JAX implementation** - Fully differentiable and JIT-compilable
- **Square-root filtering** - Numerically stable EKF and RTS smoothing
- **Flexible priors** - Integrated Wiener Process (IWP), Matern, and joint priors
- **First and second-order ODEs** - Native support for both ODE types
- **Constraint handling** - Conservation laws and time-varying measurements
- **State-parameter estimation** - Joint inference with hidden states
- **Black-box measurements** - Custom observation models with autodiff Jacobians
- **Transformed measurements** - Nonlinear state transformations with chain-rule Jacobians
- **Pluggable linearization** - EK0 / EK1 / IEKF corrections, selectable per solve
- **Adaptive step sizes** - jit/vmap/grad-safe adaptive solving, with optional fixed-point smoothing
- **Parameter estimation** - Differentiable marginal likelihood with an Optax-friendly `fit` API

## Installation

Install the latest release from PyPI:

```bash
pip install ode-filters
```

Or install from source with development dependencies:

```bash
git clone https://github.com/paufisch/ode_filters.git
cd ode_filters
pip install -e ".[dev]"
```

## Quick Example

```python
import jax.numpy as np
from ode_filters import (
    IWP,
    ODEInformation,
    gaussian_filter,
    rts_smoother,
    taylor_mode_initialization,
)

# Define ODE: dx/dt = -x (exponential decay)
def vf(x, *, t):
    return -x

x0 = np.array([1.0])
tspan = (0.0, 5.0)  # a tuple: hashable for use as a jax.jit static argument

# Set up the prior and the ODE-information measurement model
prior = IWP(q=2, d=1, Xi=0.5 * np.eye(1))
mu_0, Sigma_0_sqr = taylor_mode_initialization(vf, x0, q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)

# Filter on a fixed grid, then smooth
result = gaussian_filter(mu_0, Sigma_0_sqr, prior, measure, tspan, N=50)
m_smooth, P_smooth_sqr = rts_smoother(prior, result)

# result.m / result.P_sqr -> filtered means / square-root covariances at the grid
# result.log_likelihood   -> calibrated marginal log-likelihood
```

For adaptive step sizes, use `gaussian_filter_adaptive(mu_0, Sigma_0_sqr, prior,
measure, save_at=...)` — it is `jit` / `vmap` / `grad`-safe; pass `smoother=True`
to also get a fixed-point smoothing pass that `rts_smoother` consumes.

## Package Structure

```
ode_filters/
├── filters/          # EKF and RTS smoothing loops
├── inference/        # Square-root Gaussian algebra
├── measurement/      # ODE and observation models
└── priors/           # Gaussian Markov process priors
```

## Documentation

Full documentation is available at [paufisch.github.io/ode_filters](https://paufisch.github.io/ode_filters/).

## Development

Run the test suite:

```bash
uv run pytest --cov=ode_filters --cov-report=term-missing
```

Build documentation locally:

```bash
uv run mkdocs serve
```

## License

MIT License - see [LICENSE](LICENSE) for details.
