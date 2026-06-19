"""Tests for the jit/vmap/grad-able save-at-grid adaptive solve."""

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    ODEInformation,
    prepare_observations,
    taylor_mode_initialization,
)
from ode_filters.filters.ode_filter_adaptive import ekf1_sqr_adaptive_solve
from ode_filters.filters.ode_filter_loop import ekf1_sqr_loop_dynamic_scan
from ode_filters.measurement.measurement_models import Measurement

SAVE_AT = np.linspace(0.0, 2.0, 5)


def _logistic_problem(x0_val=0.5, theta=1.0):
    def vf(x, *, t):
        return theta * x * (1.0 - x)

    x0 = np.array([x0_val])
    prior = IWP(q=2, d=1)
    mu_0, S0_sqr = taylor_mode_initialization(vf, x0, q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return prior, measure, mu_0, S0_sqr


def test_accuracy_vs_analytic_logistic():
    prior, measure, mu_0, S0_sqr = _logistic_problem()
    res = ekf1_sqr_adaptive_solve(
        mu_0, S0_sqr, prior, measure, SAVE_AT, atol=1e-8, rtol=1e-8
    )
    x_true = 1.0 / (1.0 + np.exp(-SAVE_AT))  # x0 = 0.5
    assert np.max(np.abs(res.m[:, 0] - x_true)) < 1e-6
    assert np.isfinite(res.log_likelihood)


def test_grid_shapes_and_initial_state():
    prior, measure, mu_0, S0_sqr = _logistic_problem()
    res = ekf1_sqr_adaptive_solve(mu_0, S0_sqr, prior, measure, SAVE_AT)
    assert res.t.shape == SAVE_AT.shape
    assert res.m.shape == (SAVE_AT.shape[0], mu_0.shape[0])
    assert res.P_sqr.shape == (SAVE_AT.shape[0], mu_0.shape[0], mu_0.shape[0])
    assert np.allclose(res.t, SAVE_AT)
    assert np.allclose(res.m[0], mu_0)  # save_at[0] returns the initial state
    assert np.allclose(res.P_sqr[0], S0_sqr)


def test_jit_matches_eager():
    prior, measure, mu_0, S0_sqr = _logistic_problem()
    args = (mu_0, S0_sqr, prior, measure, SAVE_AT)
    eager = ekf1_sqr_adaptive_solve(*args)
    jitted = jax.jit(ekf1_sqr_adaptive_solve, static_argnums=(2, 3))(*args)
    assert np.allclose(eager.m, jitted.m, atol=1e-10)
    assert np.allclose(eager.log_likelihood, jitted.log_likelihood, atol=1e-9)


def test_vmap_over_initial_conditions():
    prior = IWP(q=2, d=1)

    def vf(x, *, t):
        return x * (1.0 - x)

    measure = ODEInformation(vf, prior.E0, prior.E1)

    def solve(x0):
        mu_0, S0_sqr = taylor_mode_initialization(vf, x0, q=2)
        return ekf1_sqr_adaptive_solve(mu_0, S0_sqr, prior, measure, SAVE_AT).m[-1, 0]

    x0_batch = np.array([[0.3], [0.5], [0.7]])
    out = jax.vmap(solve)(x0_batch)
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
    assert np.all(np.diff(out) > 0)  # larger x0 -> larger logistic value at t=2


def test_reverse_grad_matches_finite_difference():
    prior = IWP(q=2, d=1)
    save_at = np.linspace(0.0, 2.0, 4)

    def endpoint(theta):
        def vf(x, *, t):
            return theta * x * (1.0 - x)

        x0 = np.array([0.5])
        mu_0, S0_sqr = taylor_mode_initialization(vf, x0, q=2)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        res = ekf1_sqr_adaptive_solve(
            mu_0, S0_sqr, prior, measure, save_at, atol=1e-7, rtol=1e-7
        )
        return res.m[-1, 0]

    theta = np.array(1.3)
    g = jax.grad(endpoint)(theta)
    eps = 1e-5
    fd = (endpoint(theta + eps) - endpoint(theta - eps)) / (2 * eps)
    assert np.isfinite(g)
    assert abs(float(g) - float(fd)) < 1e-5


@pytest.mark.parametrize("calibration", ["dynamic", "none", "diagonal_ekf0"])
def test_calibration_modes_run(calibration):
    prior, measure, mu_0, S0_sqr = _logistic_problem()
    res = ekf1_sqr_adaptive_solve(
        mu_0, S0_sqr, prior, measure, SAVE_AT, calibration=calibration
    )
    x_true = 1.0 / (1.0 + np.exp(-SAVE_AT))
    assert np.max(np.abs(res.m[:, 0] - x_true)) < 1e-2
    assert np.all(np.isfinite(res.P_sqr))


def test_obs_model_matches_fixed_grid_obs_scan():
    """Adaptive solve with observations agrees with the trusted fixed-grid
    dynamic obs-scan loop on the same grid (difference = adaptive vs fixed ODE
    discretisation between save points)."""
    prior, measure, mu_0, S0_sqr = _logistic_problem()
    save_at = np.linspace(0.0, 2.0, 9)  # save_at[4] == 1.0
    meas = Measurement(np.eye(1), np.array([[0.55]]), np.array([1.0]), noise=1e-3)
    obs = prepare_observations([meas], prior.E0, save_at)

    ad = ekf1_sqr_adaptive_solve(
        mu_0,
        S0_sqr,
        prior,
        measure,
        save_at,
        obs_model=obs,
        atol=1e-5,
        rtol=1e-5,
        calibration="none",
    )
    fx = ekf1_sqr_loop_dynamic_scan(
        mu_0, S0_sqr, prior, measure, (0.0, 2.0), 8, obs_model=obs, calibration="none"
    )
    assert np.max(np.abs(ad.m[:, 0] - fx[0][:, 0])) < 5e-3


def test_obs_model_assimilation_pulls_state():
    """A precise observation at a save time pulls the filtered state toward it."""
    prior, measure, mu_0, S0_sqr = _logistic_problem()
    save_at = np.linspace(0.0, 2.0, 5)  # save_at[2] == 1.0; free x(1.0) ~ 0.731
    free = ekf1_sqr_adaptive_solve(
        mu_0, S0_sqr, prior, measure, save_at, atol=1e-2, rtol=1e-2, calibration="none"
    )
    meas = Measurement(np.eye(1), np.array([[0.55]]), np.array([1.0]), noise=1e-8)
    obs = prepare_observations([meas], prior.E0, save_at)
    pulled = ekf1_sqr_adaptive_solve(
        mu_0,
        S0_sqr,
        prior,
        measure,
        save_at,
        obs_model=obs,
        atol=1e-2,
        rtol=1e-2,
        calibration="none",
    )
    assert pulled.m[2, 0] < free.m[2, 0]  # pulled down toward 0.55


def test_obs_model_grad_matches_finite_difference():
    prior = IWP(q=2, d=1)
    save_at = np.linspace(0.0, 2.0, 5)
    meas = Measurement(np.eye(1), np.array([[0.55]]), np.array([1.0]), noise=1e-3)
    obs = prepare_observations([meas], prior.E0, save_at)

    def loglik(theta):
        def vf(x, *, t):
            return theta * x * (1.0 - x)

        mu_0, S0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=2)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        res = ekf1_sqr_adaptive_solve(
            mu_0,
            S0_sqr,
            prior,
            measure,
            save_at,
            obs_model=obs,
            atol=1e-6,
            rtol=1e-6,
            calibration="none",
        )
        return res.log_likelihood

    theta = np.array(1.1)
    g = jax.grad(loglik)(theta)
    eps = 1e-5
    fd = (loglik(theta + eps) - loglik(theta - eps)) / (2 * eps)
    assert np.isfinite(g)
    assert abs(float(g) - float(fd)) < 1e-3 * (1.0 + abs(float(fd)))  # relative


def test_obs_model_wrong_length_raises():
    prior, measure, mu_0, S0_sqr = _logistic_problem()
    save_at = np.linspace(0.0, 2.0, 5)
    meas = Measurement(np.eye(1), np.array([[0.55]]), np.array([1.0]), noise=1e-3)
    # obs_model built against a DIFFERENT grid length -> mismatch
    obs_wrong = prepare_observations([meas], prior.E0, np.linspace(0.0, 2.0, 9))
    with pytest.raises(ValueError, match="save_at"):
        ekf1_sqr_adaptive_solve(
            mu_0, S0_sqr, prior, measure, save_at, obs_model=obs_wrong
        )


def test_time_gated_measurement_raises():
    class _MeasureWithObs:
        ode_dim = 1
        _constraints = (
            Measurement(np.eye(1), np.zeros((2, 1)), np.array([0.5, 1.0]), noise=1e-2),
        )

    prior = IWP(q=2, d=1)
    with pytest.raises(NotImplementedError, match="Conservation"):
        ekf1_sqr_adaptive_solve(
            np.zeros(3), np.eye(3), prior, _MeasureWithObs(), SAVE_AT
        )
