"""Tests for the differentiable parameter-inference core (W4).

Reference problem: scalar exponential decay ``dx/dt = -lam * x`` with ``x0`` known
and ``lam`` the unknown parameter, observed noisily in ``x``. The marginal
log-likelihood of the data is differentiable / jittable / vmappable over ``lam``
and identifies the true rate.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import jax.random as jrandom

from ode_filters import InferenceProblem, marginal_loglik
from ode_filters.measurement import Measurement, ODEInformation, prepare_observations
from ode_filters.priors import IWP

X0 = 2.0
LAM_TRUE = 0.8
TSPAN = (0.0, 5.0)
N = 200
NOISE_STD = 0.02


def _problem_and_data():
    ts = np.linspace(TSPAN[0], TSPAN[1], N + 1)
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    z = (x_true + NOISE_STD * jrandom.normal(jrandom.PRNGKey(0), (N,))).reshape(-1, 1)

    prior = IWP(q=2, d=1, Xi=np.eye(1))
    e0, e1 = prior.E0, prior.E1
    measurement = Measurement(np.eye(1), z, ts[1:], noise=NOISE_STD**2)
    obs_model = prepare_observations([measurement], e0, ts)

    def build(theta):
        lam = theta["lam"]

        def vf(x, *, t):
            return -lam * x

        measure = ODEInformation(vf, e0, e1)
        # theta-independent diffuse init: x0 known, derivatives loosely prior'd.
        mu_0 = np.array([X0, 0.0, 0.0])
        sigma_0_sqr = np.diag(np.array([1e-2, 1.0, 1.0]))  # sqrt of the covariance
        return mu_0, sigma_0_sqr, measure

    problem = InferenceProblem(
        build=build, prior=prior, tspan=TSPAN, N=N, calibration="none"
    )
    return problem, obs_model


def _loglik(lam, problem, data):
    return marginal_loglik({"lam": lam}, data, model=problem)


def test_loglik_scalar_and_finite():
    problem, data = _problem_and_data()
    ll = marginal_loglik({"lam": np.array(LAM_TRUE)}, data, model=problem)
    assert ll.shape == ()
    assert np.isfinite(ll)


def test_loglik_jittable():
    problem, data = _problem_and_data()
    f = jax.jit(lambda lam: _loglik(lam, problem, data))
    assert np.isfinite(f(np.array(LAM_TRUE)))


def test_loglik_peaks_near_true():
    problem, data = _problem_and_data()
    at_true = _loglik(np.array(LAM_TRUE), problem, data)
    assert at_true > _loglik(np.array(LAM_TRUE + 0.3), problem, data)
    assert at_true > _loglik(np.array(LAM_TRUE - 0.3), problem, data)


def test_grad_matches_finite_difference():
    problem, data = _problem_and_data()
    f = jax.jit(lambda lam: _loglik(lam, problem, data))
    lam0 = np.array(0.6)
    g = jax.grad(f)(lam0)
    eps = 1e-4
    fd = (f(lam0 + eps) - f(lam0 - eps)) / (2 * eps)
    assert np.allclose(g, fd, rtol=1e-3, atol=1e-3)


def test_vmap_over_theta():
    problem, data = _problem_and_data()
    lams = np.linspace(0.4, 1.2, 9)
    lls = jax.vmap(lambda lam: _loglik(lam, problem, data))(lams)
    assert lls.shape == (9,)
    assert np.all(np.isfinite(lls))
    assert abs(lams[np.argmax(lls)] - LAM_TRUE) <= 0.15


def test_gradient_descent_recovers_parameter():
    """Adam on the negative log-likelihood recovers the true rate from a wrong start."""
    problem, data = _problem_and_data()
    neg_grad = jax.jit(jax.grad(lambda lam: -_loglik(lam, problem, data)))

    lam = np.array(0.4)  # wrong starting guess
    m, v = np.array(0.0), np.array(0.0)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for i in range(1, 401):
        grad = neg_grad(lam)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        mhat = m / (1 - b1**i)
        vhat = v / (1 - b2**i)
        lam = lam - lr * mhat / (np.sqrt(vhat) + eps)

    assert abs(float(lam) - LAM_TRUE) < 0.1, f"recovered lam={float(lam)}"
