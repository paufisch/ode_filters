"""Tests for the ergonomic ODEFilter + fit wrapper (W5).

Same reference problem as ``test_marginal_loglik`` (noisy exponential decay,
unknown rate), now fit through the object API with an Optax optimizer.
"""

from __future__ import annotations

import jax.numpy as np
import jax.random as jrandom
import pytest

from ode_filters import ODEFilter, fit
from ode_filters.measurement import Measurement, prepare_observations
from ode_filters.priors import IWP

optax = pytest.importorskip("optax")

X0 = 2.0
LAM_TRUE = 0.8
TSPAN = (0.0, 5.0)
N = 200
NOISE_STD = 0.02


def _model_and_data(lam_init):
    ts = np.linspace(TSPAN[0], TSPAN[1], N + 1)
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    z = (x_true + NOISE_STD * jrandom.normal(jrandom.PRNGKey(0), (N,))).reshape(-1, 1)

    prior = IWP(q=2, d=1, Xi=np.eye(1))
    obs_model = prepare_observations(
        [Measurement(np.eye(1), z, ts[1:], noise=NOISE_STD**2)], prior.E0, ts
    )

    def vf(x, params, *, t):
        return -params[0] * x

    def init_fn(params):
        return np.array([X0, 0.0, 0.0]), np.diag(np.array([1e-2, 1.0, 1.0]))

    model = ODEFilter(
        vf=vf,
        init_fn=init_fn,
        prior=prior,
        tspan=TSPAN,
        N=N,
        ode_params=np.array([lam_init]),
    )
    return model, obs_model


def test_loglik_method_finite():
    model, data = _model_and_data(0.7)
    ll = model.loglik(data)
    assert ll.shape == ()
    assert np.isfinite(ll)


def test_fit_recovers_parameter():
    model, data = _model_and_data(lam_init=0.4)  # wrong start
    fitted, losses = fit(model, data, optax.adam(0.05), steps=300)

    assert abs(float(fitted.ode_params[0]) - LAM_TRUE) < 0.1
    assert float(losses[-1]) < float(losses[0])  # objective decreased
    assert losses.shape == (300,)


def test_fit_zero_steps_is_noop():
    model, data = _model_and_data(0.4)
    fitted, losses = fit(model, data, optax.adam(0.05), steps=0)
    assert np.allclose(fitted.ode_params, model.ode_params)
    assert losses.shape == (0,)


def test_fit_only_trains_ode_params():
    """vf / prior / grid are static; init is theta-independent here, so the only
    thing that changes is ode_params."""
    model, data = _model_and_data(0.4)
    fitted, _ = fit(model, data, optax.adam(0.05), steps=10)
    assert fitted.prior is model.prior  # static identity preserved
    assert fitted.N == model.N
    assert not np.allclose(fitted.ode_params, model.ode_params)  # params moved
