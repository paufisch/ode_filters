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


def _blowup_problem(x0_val=1.0):
    """dx/dt = x^2 has the closed-form solution x(t) = x0 / (1 - x0 t), which
    blows up at t = 1/x0. Integrating past that point cannot succeed."""

    def vf(x, *, t):
        return x * x

    x0 = np.array([x0_val])
    prior = IWP(q=2, d=1)
    mu_0, S0_sqr = taylor_mode_initialization(vf, x0, q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return prior, measure, mu_0, S0_sqr


def test_success_flag_true_on_normal_solve():
    prior, measure, mu_0, S0_sqr = _logistic_problem()
    res = ekf1_sqr_adaptive_solve(mu_0, S0_sqr, prior, measure, SAVE_AT)
    assert bool(res.success)
    assert np.all(np.isfinite(res.m))


def test_blowup_fails_gracefully_without_nan():
    """A finite-time blow-up must not silently emit NaN: the guard rejects the
    non-finite error, the solver shrinks h, and on exhaustion it reports
    success=False while leaving the output finite (the last accepted state)."""
    prior, measure, mu_0, S0_sqr = _blowup_problem(x0_val=1.0)
    save_at = np.linspace(0.0, 2.0, 5)  # blow-up at t=1.0 is inside the grid
    res = ekf1_sqr_adaptive_solve(mu_0, S0_sqr, prior, measure, save_at, max_steps=200)
    assert not bool(res.success)
    # Crucially, no NaN leaks into the returned arrays.
    assert np.all(np.isfinite(res.m))
    assert np.all(np.isfinite(res.P_sqr))


def test_blowup_failure_is_jittable():
    """The guard + success flag survive jit (no Python branching on traced err).

    ``max_steps`` bounds the checkpointed while-loop and so must be static.
    """
    prior, measure, mu_0, S0_sqr = _blowup_problem(x0_val=1.0)
    save_at = np.linspace(0.0, 2.0, 5)
    f = jax.jit(
        ekf1_sqr_adaptive_solve, static_argnums=(2, 3), static_argnames=("max_steps",)
    )
    res = f(mu_0, S0_sqr, prior, measure, save_at, max_steps=200)
    assert not bool(res.success)
    assert np.all(np.isfinite(res.m))


def test_gaussian_filter_adaptive_surfaces_success():
    from ode_filters import gaussian_filter_adaptive

    prior, measure, mu_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(mu_0, S0_sqr, prior, measure, SAVE_AT)
    assert res.success is not None
    assert bool(res.success)


# --------------------------------------------------------------------------- #
# Fixed-point smoother (smoother=True): adaptive + smoothing in the public API.
# --------------------------------------------------------------------------- #


def test_smoother_matches_fixed_grid_rts_one_step_per_interval():
    """Forced to one sub-step per save interval, the adaptive fixed-point smoother
    must reproduce the fixed-grid RTS smoother exactly (same steps, same math)."""
    from ode_filters import gaussian_filter, gaussian_filter_adaptive, rts_smoother

    prior, measure, mu_0, S0_sqr = _logistic_problem()
    save = np.linspace(0.0, 2.0, 9)  # 8 intervals of 0.25 == fixed grid N=8

    res_fixed = gaussian_filter(
        mu_0, S0_sqr, prior, measure, (0.0, 2.0), 8, calibration="none"
    )
    ms_fixed, Ps_fixed = rts_smoother(prior, res_fixed)

    # h_init == interval width + loose tolerances => exactly one accepted step
    # per interval, so the adaptive grid coincides with the fixed grid.
    res_ad = gaussian_filter_adaptive(
        mu_0,
        S0_sqr,
        prior,
        measure,
        save,
        smoother=True,
        calibration="none",
        h_init=0.25,
        atol=1e2,
        rtol=1e2,
    )
    assert bool(res_ad.success)
    ms_ad, Ps_ad = rts_smoother(prior, res_ad)

    def cov(P):
        return np.einsum("kij,kil->kjl", P, P)

    assert np.allclose(ms_ad, ms_fixed, atol=1e-8)
    assert np.allclose(cov(Ps_ad), cov(Ps_fixed), atol=1e-8)


def test_smoother_matches_analytic_when_substepping():
    """With genuine adaptive sub-stepping and tight tolerances, the smoothed mean
    matches the analytic logistic solution at the save grid."""
    from ode_filters import gaussian_filter_adaptive, rts_smoother

    prior, measure, mu_0, S0_sqr = _logistic_problem()  # x0=0.5
    save = np.linspace(0.0, 2.0, 6)
    res = gaussian_filter_adaptive(
        mu_0, S0_sqr, prior, measure, save, smoother=True, atol=1e-9, rtol=1e-9
    )
    assert bool(res.success)
    ms, _ = rts_smoother(prior, res)
    x_true = 1.0 / (1.0 + np.exp(-save))
    assert np.max(np.abs(ms[:, 0] - x_true)) < 1e-5


def test_smoother_last_equals_filtered_last():
    """RTS property: the last smoothed state equals the last filtered state."""
    from ode_filters import gaussian_filter_adaptive, rts_smoother

    prior, measure, mu_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(
        mu_0, S0_sqr, prior, measure, SAVE_AT, smoother=True, atol=1e-7, rtol=1e-7
    )
    ms, Ps = rts_smoother(prior, res)
    assert np.allclose(ms[-1], res.m[-1], atol=1e-8)
    assert np.allclose(Ps[-1].T @ Ps[-1], res.P_sqr[-1].T @ res.P_sqr[-1], atol=1e-8)


def test_smoother_reduces_uncertainty():
    """Smoothing must not increase the marginal variance at interior save points."""
    from ode_filters import gaussian_filter_adaptive, rts_smoother

    prior, measure, mu_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(
        mu_0,
        S0_sqr,
        prior,
        measure,
        SAVE_AT,
        smoother=True,
        calibration="none",
        atol=1e-7,
        rtol=1e-7,
    )
    _, Ps = rts_smoother(prior, res)
    k = SAVE_AT.shape[0] // 2
    var_filt = float(np.sum(res.P_sqr[k] ** 2))
    var_smooth = float(np.sum(Ps[k] ** 2))
    assert var_smooth <= var_filt + 1e-12


def test_smoother_filtered_means_match_filtering_only_run():
    """smoother=True must not change the filtered solution (only add a backward pass)."""
    from ode_filters import gaussian_filter_adaptive

    prior, measure, mu_0, S0_sqr = _logistic_problem()
    filt = gaussian_filter_adaptive(
        mu_0, S0_sqr, prior, measure, SAVE_AT, atol=1e-7, rtol=1e-7
    )
    smoo = gaussian_filter_adaptive(
        mu_0, S0_sqr, prior, measure, SAVE_AT, smoother=True, atol=1e-7, rtol=1e-7
    )
    assert np.allclose(filt.m, smoo.m, atol=1e-10)
    assert np.allclose(filt.P_sqr, smoo.P_sqr, atol=1e-10)
    assert filt.G_back is None and smoo.G_back is not None


def test_smoother_with_obs_raises():
    from ode_filters import gaussian_filter_adaptive

    prior, measure, mu_0, S0_sqr = _logistic_problem()
    meas = Measurement(
        np.eye(1), np.array([[0.55]]), np.array([SAVE_AT[2]]), noise=1e-3
    )
    obs = prepare_observations([meas], prior.E0, SAVE_AT)
    with pytest.raises(NotImplementedError, match="smoother"):
        gaussian_filter_adaptive(
            mu_0, S0_sqr, prior, measure, SAVE_AT, obs_model=obs, smoother=True
        )
