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
from ode_filters.filters import ekf1_sqr_loop, rts_sqr_smoother_loop
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization

# Define ODE: dx/dt = -x (exponential decay)
def vf(x, *, t):
    return -x

x0 = np.array([1.0])
tspan = [0, 5]

# Setup prior and measurement model
prior = IWP(q=2, d=1, Xi=0.5 * np.eye(1))
mu_0, Sigma_0_sqr = taylor_mode_initialization(vf, x0, q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)

# Run filter and smoother
m_seq, P_sqr, *_, G, d, P_back, _, _ = ekf1_sqr_loop(
    mu_0, Sigma_0_sqr, prior, measure, tspan, N=50
)
m_smooth, P_smooth_sqr = rts_sqr_smoother_loop(
    m_seq[-1], P_sqr[-1], G, d, P_back, N=50
)
```

## Package Structure

```
ode_filters/
├── filters/          # EKF and RTS smoothing loops
├── inference/        # Square-root Gaussian algebra
├── measurement/      # ODE and observation models
└── priors/           # Gaussian Markov process priors
```

## Next Steps

- Check out the [Quickstart Guide](examples/quickstart.ipynb) for a hands-on tutorial
- Browse the [API Reference](api/index.md) for detailed documentation
- Explore [Examples](examples/examples.ipynb) for more advanced use cases
