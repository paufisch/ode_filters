<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/paufisch/ode_filters/main/docs/assets/logo-dark.svg">
  <img alt="ODE Filters logo" src="https://raw.githubusercontent.com/paufisch/ode_filters/main/docs/assets/logo.svg" width="240">
</picture>

# ODE Filters

[![PyPI](https://img.shields.io/pypi/v/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![Python](https://img.shields.io/pypi/pyversions/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![CI](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://paufisch.github.io/ode_filters/)
[![Coverage](https://codecov.io/gh/paufisch/ode_filters/branch/main/graph/badge.svg)](https://codecov.io/gh/paufisch/ode_filters)

Fit ODE models to data, with the solver's numerical error accounted for. `ode_filters` is a probabilistic ODE solver in JAX: it returns a Gaussian distribution over the solution instead of a single trajectory, and it treats the ODE as information about the solution, on the same footing as data. One square-root Kalman filter conditions on the ODE, on noisy measurements, on conservation laws and on an unknown input that drives the system, which gives a likelihood of the data that is differentiable in every parameter of the model.

## Features

- **Inference from data** - a differentiable marginal likelihood for any parameter of the ODE or the prior; `fit_scipy` (L-BFGS-B), `fit` (Optax), NumPyro and BlackJAX
- **Hidden inputs** - joint state-input priors recover an unknown force driving the system, with information criteria for what the data say about it
- **Evidence for hyperparameters** - the sequential data evidence and the conditional evidence given the ODE (Fenrir, Dalton)
- **Outputs** - smoothed marginals, joint posterior samples, dense output, observation predictives, innovation diagnostics (NIS, whitened residuals)
- **The ODE as information** - first- and second-order ODEs, conservation laws, black-box and transformed operators, data at any grid node
- **Pluggable linearization** - EK0, EK1, IEKF and statistical linearization, plus the IEKS and IPLS smoothers
- **Priors** - integrated Wiener, Matern, integrated Ornstein-Uhlenbeck and joint priors, plain or preconditioned
- **Calibrated uncertainty** - online and post-hoc diffusion calibration; fixed grids of any spacing and adaptive steps
- **Pure JAX, square-root form** - `jit`, `vmap` and `grad` through every solver; numerically stable QR-based filtering and smoothing

## Installation

Install the latest release from PyPI:

```bash
pip install ode-filters
pip install "ode-filters[inference]"  # with Optax, SciPy, NumPyro and BlackJAX
```

Or install from source with the development tools (tests, linting, type checking):

```bash
git clone https://github.com/paufisch/ode_filters.git
cd ode_filters
uv sync --group dev
```

With pip 25.1 or newer, `pip install -e . --group dev` does the same.

## Quick example

Estimate the growth rate of the logistic ODE from noisy data:

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

Solving an ODE without data is the same filter without the data:

```python
from ode_filters import gaussian_filter, rts_smoother

vf = lambda x, *, t: 0.8 * x * (1.0 - x)
m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.1]), q=2)
result = gaussian_filter(m_0, P_0_sqr, prior, ODEInformation(vf, prior.E0, prior.E1), ts)
posterior = rts_smoother(result)  # posterior.m, posterior.std: mean and error estimate
```

Upgrading from 0.7? See the [migration guide](https://paufisch.github.io/ode_filters/migration/).

## Package Structure

```
ode_filters/
├── priors/           # Gauss-Markov priors (IWP, Matern, IOUP, joint)
├── information/      # ODE information, conservation laws, data observations
├── linearization/    # Taylor and statistical linearization, frozen surrogates
├── calibration/      # Diffusion (sigma^2) calibration policies and rescaling
├── filters/          # Filters, smoothers, the iterated smoother, step control
├── inference/        # Marginal likelihoods and parameter fitting
└── gaussian/         # Square-root Gaussian types and algebra
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
uv run --group docs mkdocs serve
```

## License

MIT License - see [LICENSE](LICENSE) for details.
