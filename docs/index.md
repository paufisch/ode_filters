# ODE Filters

[![PyPI](https://img.shields.io/pypi/v/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![Python](https://img.shields.io/pypi/pyversions/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![CI](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://paufisch.github.io/ode_filters/)
[![Coverage](https://codecov.io/gh/paufisch/ode_filters/branch/main/graph/badge.svg)](https://codecov.io/gh/paufisch/ode_filters)

**`ode_filters` is `solve_ivp`, but it returns a mean *and* a calibrated
uncertainty.** It is a pure-JAX library of *probabilistic* ODE solvers: instead of
a single trajectory, you get a Gaussian posterior over the solution, computed by
numerically stable square-root extended Kalman filtering and RTS smoothing.

```python
import jax.numpy as np
from ode_filters import IWP, ODEInformation, gaussian_filter, taylor_mode_initialization


def vf(x, *, t):                  # the ODE:  dx/dt = -x,  x(0) = 1
    return -x


prior = IWP(q=2, d=1)                                       # 1. a smoothness prior
mu_0, P0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)            # 2. the ODE as data

# 3. filter forward -> a Gaussian posterior at every grid point
result = gaussian_filter(mu_0, P0_sqr, prior, measure, (0.0, 5.0), N=50)

mean = result.m[:, 0]                                       # the solution estimate ...
P = np.einsum("nij,nik->njk", result.P_sqr, result.P_sqr)   # ... with full covariance
std = np.sqrt(P[:, 0, 0])                                   # ... so it comes with error bars

print(f"x(5) = {mean[-1]:.4f}  +/-  {2 * std[-1]:.0e}   (exact: {np.exp(-5.0):.4f})")
# x(5) = 0.0067  +/-  5e-06   (exact: 0.0067)
```

New here? Read **[What is a probabilistic ODE solver?](probabilistic-ode-solvers.md)**
for the intuition, then run the **[Quickstart](examples/quickstart.ipynb)**.

## Where to go next

| You want to… | Start here |
| --- | --- |
| understand the idea | [What is a probabilistic ODE solver?](probabilistic-ode-solvers.md) |
| run your first solve | [Quickstart](examples/quickstart.ipynb) |
| look up a symbol or convention | [Notation & conventions](notation.md) |
| pick a prior / order / correction | [How to choose](how-to-choose.md) |
| avoid common pitfalls | [Sharp bits & FAQ](sharp-bits.md) |
| browse the API | [API Reference](api/index.md) |

## Features

- **Pure JAX** — `jit` / `grad` / `vmap`-compatible on the scan-based paths.
- **Square-root filtering** — numerically stable EKF and RTS smoothing (Krämer–Hennig 2024).
- **Pluggable linearization** — EK0 / EK1 [corrections](corrections.md), selectable per solve.
- **Flexible priors** — IWP, Matern, and joint priors.
- **First- and second-order ODEs**, conservation laws, and time-varying measurements.
- **Diffusion calibration** and **adaptive step-size control**.

## Installation

```bash
pip install ode-filters
```

Or from source with development dependencies:

```bash
git clone https://github.com/paufisch/ode_filters.git
cd ode_filters
pip install -e ".[dev]"
```
