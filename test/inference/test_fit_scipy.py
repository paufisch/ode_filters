"""Hyperparameter fitting with SciPy's quasi-Newton methods."""

from __future__ import annotations

import jax
import jax.numpy as np
import jax.random as jrandom
import pytest

from ode_filters import InferenceProblem, PositiveReal, fit_scipy, marginal_loglik
from ode_filters.information import Measurement, ODEInformation, prepare_observations
from ode_filters.priors import IWP

pytest.importorskip("scipy")

X0, LAM_TRUE, N, NOISE_STD = 2.0, 0.8, 200, 0.02


def _objective():
    ts = np.linspace(0.0, 5.0, N + 1)
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    z = (x_true + NOISE_STD * jrandom.normal(jrandom.PRNGKey(0), (N,))).reshape(-1, 1)
    prior = IWP(q=2, d=1)
    data = prepare_observations(
        [Measurement(np.eye(1), z, ts[1:], noise=NOISE_STD**2)], prior.E0, ts
    )

    def build(theta):
        def vf(x, *, t):
            return -theta["lam"] * x

        measure = ODEInformation(vf, prior.E0, prior.E1)
        return np.array([X0, 0.0, 0.0]), np.diag(np.array([1e-2, 1.0, 1.0])), measure

    problem = InferenceProblem(build=build, prior=prior, ts=ts, calibration="none")
    return lambda theta: -marginal_loglik(theta, data, model=problem)


def test_fit_scipy_finds_the_maximum_likelihood_estimate():
    objective = _objective()
    theta_hat, result = fit_scipy(objective, {"lam": PositiveReal(0.4)})
    assert result.success
    assert isinstance(theta_hat["lam"], PositiveReal)
    lam_hat = float(theta_hat["lam"].unwrap())
    assert abs(lam_hat - LAM_TRUE) < 0.05
    grad = jax.grad(lambda lam: objective({"lam": lam}))(np.asarray(lam_hat))
    assert abs(float(grad)) < 1e-2 * N


def test_fit_scipy_passes_options_through():
    objective = _objective()
    theta_hat, result = fit_scipy(
        objective, {"lam": np.array(0.4)}, bounds=[(0.1, 0.5)], method="L-BFGS-B"
    )
    assert float(theta_hat["lam"]) == pytest.approx(0.5)
    assert result.success
