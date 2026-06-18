# Parameter Estimation

`ode_filters` exposes a differentiable marginal log-likelihood of observed data
under the ODE-constrained model, so ODE parameters can be recovered by
gradient-based optimization (Optax) or sampling (NumPyro / BlackJAX). There are
two layers: a pure function `marginal_loglik` (parameters as an argument) and an
ergonomic `ODEFilter` + `fit` object API on top.

!!! note "Fixed-grid only"
    Gradient-based inference runs on the fixed-grid `jax.lax.scan` loop, which is
    `jit` / `grad` / `vmap`-safe. The Python-loop and adaptive drivers are not
    reverse-differentiable; do not use them for inference.

## Object API: `ODEFilter` + `fit`

Bundle the vector field, prior, grid and parameters into an `ODEFilter` and fit
it with an Optax optimizer:

```python
import jax.numpy as np
import jax.random as jr
import optax
from ode_filters import ODEFilter, fit
from ode_filters.measurement import Measurement, prepare_observations
from ode_filters.priors import IWP

# Noisy observations of x(t) = 2 * exp(-0.8 t)
ts = np.linspace(0.0, 5.0, 201)
z = (2.0 * np.exp(-0.8 * ts[1:]) + 0.02 * jr.normal(jr.PRNGKey(0), (200,)))[:, None]
prior = IWP(q=2, d=1, Xi=np.eye(1))
data = prepare_observations(
    [Measurement(np.eye(1), z, ts[1:], noise=0.02**2)], prior.E0, ts
)


def vf(x, params, *, t):
    return -params[0] * x


def init_fn(params):
    return np.array([2.0, 0.0, 0.0]), np.diag(np.array([1e-2, 1.0, 1.0]))


model = ODEFilter(
    vf=vf, init_fn=init_fn, prior=prior, tspan=(0.0, 5.0), N=200,
    ode_params=np.array([0.4]),  # deliberately wrong starting guess
)
fitted, losses = fit(model, data, optax.adam(0.05), steps=300)
# fitted.ode_params[0] ~ 0.8
```

`ode_params` is the only trainable leaf; the vector field, prior and grid are
static. `init_fn(params) -> (mu_0, Sigma_0_sqr)` produces the initial Gaussian
(it may ignore `params` for a fixed initial condition). The runnable version is
in `examples/parameter_estimation.py`.

## Functional core: `marginal_loglik`

For NumPyro / BlackJAX or a custom optimizer, use the pure function directly. It
takes the parameter pytree `theta` and returns the data marginal log-likelihood;
`jax.grad` / `jax.vmap` over `theta` work out of the box:

```python
import jax
from ode_filters import InferenceProblem, marginal_loglik
from ode_filters.measurement import ODEInformation


def build(theta):
    vf = lambda x, *, t: -theta["lam"] * x  # noqa: E731
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return np.array([2.0, 0.0, 0.0]), np.diag(np.array([1e-2, 1.0, 1.0])), measure


problem = InferenceProblem(build=build, prior=prior, tspan=(0.0, 5.0), N=200)

ll = marginal_loglik({"lam": np.array(0.8)}, data, model=problem)
grad = jax.grad(lambda th: marginal_loglik(th, data, model=problem))(
    {"lam": np.array(0.8)}
)
```

`InferenceProblem.build(theta) -> (mu_0, Sigma_0_sqr, measure)` is the single
bridge from parameters to solver inputs; everything else is static configuration.
`model` and `data` are closed over (static); only `theta` is differentiated.

## Turn calibration off during inference

The likelihood is computed with `calibration="none"` by default. A dynamic
diffusion calibration absorbs model-data misfit into `sigma^2` and confounds the
likelihood over the parameters, so it should be off when fitting parameters
(probdiffeq gives the same guidance). Set the prior's structural scale `Xi`
deliberately instead.

## What the objective is

`marginal_loglik` returns the *observation* marginal log-likelihood -- the
evidence of the data under the ODE-constrained Gauss-Markov prior (the Fenrir
objective). Maximize it (minimize its negative) over the parameters.

## See also

- [Linearization Schemes](corrections.md)
- [Diffusion Calibration](calibration.md)
- API reference for `ode_filters.inference`.
