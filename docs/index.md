![ODE Filters logo](assets/logo.svg#only-light){ width="240" }
![ODE Filters logo](assets/logo-dark.svg#only-dark){ width="240" }

# ODE Filters

[![PyPI](https://img.shields.io/pypi/v/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![Python](https://img.shields.io/pypi/pyversions/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![CI](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://paufisch.github.io/ode_filters/)
[![Coverage](https://codecov.io/gh/paufisch/ode_filters/branch/main/graph/badge.svg)](https://codecov.io/gh/paufisch/ode_filters)

**`ode_filters` fits ODE models to data, with the solver's numerical error
accounted for.** It is a probabilistic ODE solver: instead of a single
approximate trajectory, it returns a Gaussian distribution over the solution,
whose spread estimates the numerical error. It treats the ODE as *information*
about the solution, on the same footing as data, so one square-root Kalman
filter conditions on the ODE, on noisy measurements, on conservation laws, and
on an unknown input that drives the system. That gives a likelihood of the data
that accounts for the numerical error, differentiable in every parameter of the
model — in pure JAX, so fits can be compiled, batched and differentiated.

```python
import jax
import jax.numpy as np

from ode_filters import (
    IWP,
    InferenceProblem,
    Measurement,
    ODEInformation,
    PositiveReal,
    fit_scipy,
    marginal_loglik,
    prepare_observations,
    taylor_mode_initialization,
)

jax.config.update("jax_enable_x64", True)

# Noisy observations of the logistic ODE x' = r x (1 - x), x(0) = 0.1
ts = np.linspace(0.0, 10.0, 101)
x = 1.0 / (1.0 + 9.0 * np.exp(-0.8 * ts[1:]))
y = x + 0.02 * jax.random.normal(jax.random.PRNGKey(0), x.shape)

prior = IWP(q=2, d=1)
data = prepare_observations(
    [Measurement(np.eye(1), y[:, None], ts[1:], noise=0.02**2)], prior.E0, ts
)


# The ODE, as a function of the unknown rate r
def build(theta):
    def vf(x, *, t):
        return theta["r"] * x * (1.0 - x)

    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.1]), q=2)
    return m_0, P_0_sqr, ODEInformation(vf, prior.E0, prior.E1)


# Maximize the likelihood of the data, numerical error included
problem = InferenceProblem(build=build, prior=prior, ts=ts)
theta, _ = fit_scipy(
    lambda th: -marginal_loglik(th, data, model=problem), {"r": PositiveReal(0.3)}
)
print(f"r = {theta['r'].unwrap():.3f}  (true: 0.8)")
# r = 0.793  (true: 0.8)
```

New here? Read **[What is a probabilistic ODE solver?](probabilistic-ode-solvers.md)**
for the intuition, run the **[Quickstart](examples/quickstart.ipynb)** to solve an
ODE, then the **[Parameter estimation](examples/parameter-estimation.ipynb)**
tutorial to fit one.

## Where to go next

<div class="grid cards" markdown>

-   :material-chart-bell-curve:{ .lg .middle } **Fit a model to data**

    ---

    Estimate ODE parameters by maximum likelihood, check the fit, and sample the posterior.

    [:octicons-arrow-right-24: Parameter estimation](examples/parameter-estimation.ipynb)

-   :material-sine-wave:{ .lg .middle } **Recover a hidden input**

    ---

    Infer the unknown force that drives an oscillator, with error bars.

    [:octicons-arrow-right-24: Recovering a hidden input](examples/hidden-input.ipynb)

-   :material-lightbulb-on-outline:{ .lg .middle } **Understand the idea**

    ---

    What a probabilistic ODE solver computes, and what its uncertainty is good for.

    [:octicons-arrow-right-24: What is a probabilistic ODE solver?](probabilistic-ode-solvers.md)

-   :material-rocket-launch-outline:{ .lg .middle } **Solve an ODE**

    ---

    Solve the logistic ODE end to end, smooth it, and plot the posterior.

    [:octicons-arrow-right-24: Quickstart](examples/quickstart.ipynb)

-   :material-tune-variant:{ .lg .middle } **Pick the components**

    ---

    Prior, order, linearization, step size, calibration and the inference tools, with sensible defaults.

    [:octicons-arrow-right-24: How to choose](how-to-choose.md)

-   :material-scale-balance:{ .lg .middle } **Compare the libraries**

    ---

    What probdiffeq, ProbNumDiffEq.jl and rodeo offer, and where this library differs.

    [:octicons-arrow-right-24: How it compares](comparison.md)

-   :material-alert-outline:{ .lg .middle } **Avoid the pitfalls**

    ---

    Float64, `jit` and pytrees, data on the grid, and other things that bite.

    [:octicons-arrow-right-24: Sharp bits & FAQ](sharp-bits.md)

-   :material-api:{ .lg .middle } **Browse the API**

    ---

    Every public function and class, grouped by subpackage.

    [:octicons-arrow-right-24: API reference](api/index.md)

</div>

## Features

- **Inference from data**: a differentiable marginal likelihood of the data for
  any parameter of the ODE or the prior; `fit_scipy` (L-BFGS-B) and `fit`
  (Optax) for point estimates, NumPyro and BlackJAX for posteriors.
- **Hidden inputs**: joint state–input priors recover an unknown force that
  drives the system, with a consistent initialization and information criteria
  for what the data say about it.
- **Evidence for hyperparameters**: the sequential data evidence and the
  conditional evidence of the data given the ODE (Fenrir, Dalton).
- **Outputs**: smoothed marginals, joint posterior samples, dense output between
  the grid nodes, the predictive distribution of observations, and innovation
  diagnostics (NIS, whitened residuals) to check a fit.
- **The ODE as information**: first- and second-order ODEs, conservation laws,
  black-box and transformed operators, and data at any grid node, all in one
  filter.
- **Pluggable linearization**: EK0, EK1, iterated (IEKF) and statistical
  [linearization](corrections.md), plus the trajectory-level IEKS and IPLS.
- **Priors**: integrated Wiener (IWP), Matérn, integrated Ornstein–Uhlenbeck
  (probabilistic exponential integrators) and joint priors, plain or
  preconditioned.
- **Calibrated uncertainty**: online and post-hoc diffusion calibration, fixed
  grids (any spacing) and adaptive steps.
- **Pure JAX, square-root form**: `jit`, `vmap` and `grad` through every solver;
  numerically stable QR-based filtering and smoothing (Krämer & Hennig 2024).

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
