"""Tests for the unconstrained-parameter wrappers (slice 2)."""

from __future__ import annotations

import jax
import jax.numpy as np
import jax.random as jrandom
import pytest

from ode_filters import ODEFilter, PositiveReal, Real, fit, unwrap
from ode_filters.measurement import Measurement, prepare_observations
from ode_filters.priors import IWP

optax = pytest.importorskip("optax")


# --------------------------------------------------------------------------- #
# Wrapper round-trips and unwrap                                              #
# --------------------------------------------------------------------------- #


def test_positive_real_round_trip():
    for v in [1e-3, 0.5, 2.0, 100.0]:
        p = PositiveReal(v)
        assert np.allclose(p.unwrap(), v, rtol=1e-5)
        assert float(p.unwrap()) > 0.0


def test_real_is_identity():
    assert np.allclose(Real(-1.3).unwrap(), -1.3)


def test_unwrap_is_identity_on_plain_pytrees():
    theta = {"a": np.array(1.0), "b": np.arange(3.0)}
    out = unwrap(theta)
    assert np.allclose(out["a"], theta["a"])
    assert np.allclose(out["b"], theta["b"])


def test_unwrap_nested_mixed():
    theta = {"rate": PositiveReal(0.8), "shift": Real(2.0), "raw": np.array(3.0)}
    out = unwrap(theta)
    assert np.allclose(out["rate"], 0.8, rtol=1e-5)
    assert np.allclose(out["shift"], 2.0)
    assert np.allclose(out["raw"], 3.0)


def test_grad_round_trip_is_identity():
    # unwrap(PositiveReal(v)) == softplus(softplus_inv(v)) == v, so d/dv == 1.
    g = jax.grad(lambda v: PositiveReal(v).unwrap().sum())(np.array(2.0))
    assert np.isfinite(g)
    assert np.allclose(g, 1.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# End-to-end: fit a PositiveReal-wrapped rate                                 #
# --------------------------------------------------------------------------- #

X0, LAM_TRUE, TSPAN, N, NOISE = 2.0, 0.8, (0.0, 5.0), 200, 0.02


def _model_and_data(lam_init):
    ts = np.linspace(TSPAN[0], TSPAN[1], N + 1)
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    z = (x_true + NOISE * jrandom.normal(jrandom.PRNGKey(0), (N,))).reshape(-1, 1)
    prior = IWP(q=2, d=1, Xi=np.eye(1))
    data = prepare_observations(
        [Measurement(np.eye(1), z, ts[1:], noise=NOISE**2)], prior.E0, ts
    )

    def vf(x, params, *, t):
        return -params[0] * x

    def init_fn(params):
        return np.array([X0, 0.0, 0.0]), np.diag(np.array([1e-2, 1.0, 1.0]))

    # ode_params is a *positive* rate wrapped in a bijector.
    model = ODEFilter(
        vf=vf,
        init_fn=init_fn,
        prior=prior,
        tspan=TSPAN,
        N=N,
        ode_params=PositiveReal(np.array([lam_init])),
    )
    return model, data


def test_loglik_with_wrapped_param():
    model, data = _model_and_data(0.5)
    ll = model.loglik(data)
    assert np.isfinite(ll)


def test_fit_recovers_positive_param_in_unconstrained_space():
    model, data = _model_and_data(lam_init=0.3)
    fitted, losses = fit(model, data, optax.adam(0.05), steps=300)
    # The optimizer worked on the unconstrained leaf; unwrap to read the rate.
    lam_hat = float(unwrap(fitted.ode_params)[0])
    assert abs(lam_hat - LAM_TRUE) < 0.1, f"lam_hat={lam_hat}"
    assert lam_hat > 0.0  # positivity guaranteed by the bijection
    assert float(losses[-1]) < float(losses[0])
