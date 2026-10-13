# ODE Filters

[![PyPI](https://img.shields.io/pypi/v/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![Python](https://img.shields.io/pypi/pyversions/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![CI](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://paufisch.github.io/ode_filters/)
[![Coverage](https://codecov.io/gh/paufisch/ode_filters/branch/main/graph/badge.svg)](https://codecov.io/gh/paufisch/ode_filters)

**`ode_filters` solves ordinary differential equations and estimates how accurate
its answer is.** A classical solver, such as SciPy's
[`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html),
returns a single approximate trajectory. `ode_filters` returns a Gaussian
distribution over the solution instead: a mean trajectory plus an uncertainty band
that estimates the numerical error, sized from the solver's own residuals. It is a
pure-JAX library built on numerically stable square-root Kalman filtering and
smoothing, so solves can be compiled, batched and differentiated.

```python
import jax
import jax.numpy as np

from ode_filters import IWP, ODEInformation, gaussian_filter, taylor_mode_initialization

jax.config.update("jax_enable_x64", True)


# The ODE dx/dt = -x with x(0) = 1
def vf(x, *, t):
    return -x


# 1. a smoothness prior, 2. the ODE as information about the solution
prior = IWP(q=2, d=1)
mu_0, P0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)

# 3. filter forward: a Gaussian posterior at every grid point
result = gaussian_filter(mu_0, P0_sqr, prior, measure, (0.0, 5.0), N=50)

# mean and standard deviation of x (the covariance is P = P_sqr.T @ P_sqr)
mean = result.m[:, 0]
P = np.einsum("nij,nik->njk", result.P_sqr, result.P_sqr)
std = np.sqrt(P[:, 0, 0])

print(f"x(5) = {mean[-1]:.4f} +/- {2 * std[-1]:.0e}  (exact: {np.exp(-5.0):.4f})")
# x(5) = 0.0067 +/- 5e-06  (exact: 0.0067)
```

New here? Read **[What is a probabilistic ODE solver?](probabilistic-ode-solvers.md)**
for the intuition, then run the **[Quickstart](examples/quickstart.ipynb)**.

## Where to go next

<div class="grid cards" markdown>

-   :material-lightbulb-on-outline:{ .lg .middle } **Understand the idea**

    ---

    What a probabilistic ODE solver computes, and what its uncertainty is good for.

    [:octicons-arrow-right-24: What is a probabilistic ODE solver?](probabilistic-ode-solvers.md)

-   :material-rocket-launch-outline:{ .lg .middle } **Run your first solve**

    ---

    Solve the logistic ODE end to end, smooth it, and plot the posterior.

    [:octicons-arrow-right-24: Quickstart](examples/quickstart.ipynb)

-   :material-tune-variant:{ .lg .middle } **Pick the components**

    ---

    Prior, order, linearization, step size and calibration, with sensible defaults.

    [:octicons-arrow-right-24: How to choose](how-to-choose.md)

-   :material-alert-outline:{ .lg .middle } **Avoid the pitfalls**

    ---

    Float64, `jit` and static arguments, and other things that bite.

    [:octicons-arrow-right-24: Sharp bits & FAQ](sharp-bits.md)

-   :material-sigma:{ .lg .middle } **Look up a symbol**

    ---

    The notation shared by the guides and the API, and the square-root convention.

    [:octicons-arrow-right-24: Notation & conventions](notation.md)

-   :material-api:{ .lg .middle } **Browse the API**

    ---

    Every public function and class, grouped by subpackage.

    [:octicons-arrow-right-24: API reference](api/index.md)

</div>

## Features

- **Pure JAX**: `jit`, `vmap` and `grad` through the fixed-grid and the adaptive
  solvers.
- **Square-root filtering**: numerically stable EKF and RTS smoothing
  (Krämer & Hennig 2024).
- **Pluggable linearization**: EK0, EK1, iterated (IEKF) and statistical
  [linearization](corrections.md), chosen per solve, plus an iterated
  posterior-linearization smoother.
- **Priors**: integrated Wiener (IWP), Matérn, integrated Ornstein–Uhlenbeck
  (probabilistic exponential integrators) and joint state / latent-force priors.
- **Model knowledge and data**: first- and second-order ODEs, conservation laws,
  noisy observations and hidden states.
- **Calibrated uncertainty**: online and post-hoc diffusion calibration and
  adaptive step-size control.
- **Parameter inference**: a differentiable marginal likelihood with helpers that
  plug into Optax, NumPyro and BlackJAX.

## Installation

```bash
pip install ode-filters
```

From source, with the development tools (tests, linting, type checking):

```bash
git clone https://github.com/paufisch/ode_filters.git
cd ode_filters
uv sync --group dev
```

With pip 25.1 or newer, `pip install -e . --group dev` does the same.
