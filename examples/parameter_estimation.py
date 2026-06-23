"""Recover an ODE parameter from noisy data with ODEFilter + fit.

A worked example of the Layer-2 inference API: fit the decay rate ``lam`` in
``dx/dt = -lam * x`` to noisy observations of ``x(t)`` by maximizing the data
marginal log-likelihood with Optax.

Requires the inference extra: ``pip install ode-filters[inference]`` (or
``uv sync --group dev`` in a source checkout).

Run with: ``uv run python examples/parameter_estimation.py``
"""

from __future__ import annotations

import jax
import jax.numpy as np
import jax.random as jrandom
import optax

from ode_filters import ODEFilter, fit
from ode_filters.measurement import Measurement, prepare_observations
from ode_filters.priors import IWP

jax.config.update("jax_enable_x64", True)

X0 = 2.0
LAM_TRUE = 0.8
TSPAN = (0.0, 5.0)
N = 200
NOISE_STD = 0.02


def main() -> None:
    # --- Synthetic noisy observations of x(t) = X0 * exp(-lam_true * t) ---
    ts = np.linspace(TSPAN[0], TSPAN[1], N + 1)
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    z = (x_true + NOISE_STD * jrandom.normal(jrandom.PRNGKey(0), (N,))).reshape(-1, 1)

    prior = IWP(q=2, d=1, Xi=np.eye(1))
    data = prepare_observations(
        [Measurement(np.eye(1), z, ts[1:], noise=NOISE_STD**2)], prior.E0, ts
    )

    # --- A fittable first-order ODE solver: dx/dt = -lam * x ---
    def vf(x, params, *, t):
        return -params[0] * x

    def init_fn(params):
        # Fixed initial condition (x0 known, derivatives loosely prior'd).
        return np.array([X0, 0.0, 0.0]), np.diag(np.array([1e-2, 1.0, 1.0]))

    model = ODEFilter(
        vf=vf,
        init_fn=init_fn,
        prior=prior,
        tspan=TSPAN,
        N=N,
        ode_params=np.array([0.4]),  # deliberately wrong starting guess
    )

    # --- Fit by maximizing the data marginal log-likelihood ---
    fitted, losses = fit(model, data, optax.adam(0.05), steps=300)

    print(f"true lam      = {LAM_TRUE:.4f}")
    print(f"initial guess = {float(model.ode_params[0]):.4f}")
    print(f"recovered lam = {float(fitted.ode_params[0]):.4f}")
    print(f"neg-loglik    {float(losses[0]):.2f} -> {float(losses[-1]):.2f}")


if __name__ == "__main__":
    main()
