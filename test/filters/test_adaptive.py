"""Tests for ``ode_filters.filters.adaptive``: the adaptive solver.

Its results on the save grid, against fixed-grid runs and reference
solutions; calibration, data, smoothing and preconditioned priors on the
adaptive path; the recorded steps and the running-mean diffusion; jit, vmap
and grad. The step-size arithmetic is replayed in ``test_adaptive_steps.py``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    JointPrior,
    MaternPrior,
    ODEInformation,
    PController,
    PIController,
    PrecondIWP,
    PrecondMaternPrior,
    StepSizeController,
    gaussian_filter,
    gaussian_filter_adaptive,
    prepare_observations,
    rts_smoother,
    taylor_mode_initialization,
)
from ode_filters.calibration.sigma import quasi_mle_sigma_sqr_from_Q
from ode_filters.information import (
    Measurement,
    ODEInformationWithHidden,
)
from test.helpers import assert_grad_matches_fd

X0 = np.array([0.1])
SAVE = np.array([0.0, 5.0])


def logistic_vf(x, *, t):
    return x * (1 - x)


def logistic_analytic(t, x0=0.1):
    return 1.0 / (1.0 + (1.0 / x0 - 1.0) * np.exp(-t))


def _setup(q=2):
    prior = IWP(q=q, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(logistic_vf, X0, q=q)
    return prior, m_0, P_0_sqr, ODEInformation(logistic_vf, prior.E0, prior.E1)


def _solve(save_at=SAVE, record_steps=500, **kwargs):
    prior, m_0, P_0_sqr, measure = _setup()
    kwargs = {"atol": 1e-4, "rtol": 1e-2, **kwargs}
    res = gaussian_filter_adaptive(
        m_0, P_0_sqr, prior, measure, save_at, record_steps=record_steps, **kwargs
    )
    return prior, m_0, measure, res


def test_the_record_holds_every_accepted_step():
    _, _, _, res = _solve()
    steps = res.steps
    n = int(steps.count)
    assert bool(res.success)
    assert n == int(np.sum(res.n_accepted))
    assert float(steps.t[n - 1]) == pytest.approx(5.0, rel=1e-12)
    assert float(np.sum(steps.h[:n])) == pytest.approx(5.0, rel=1e-12)
    assert np.allclose(steps.m[n - 1], res.m[-1])
    assert bool(np.all(np.isnan(steps.t[n:])))


def test_a_short_record_keeps_the_first_steps_and_the_full_count():
    _, _, _, full = _solve()
    _, _, _, short = _solve(record_steps=3)
    assert int(short.steps.count) == int(full.steps.count)
    assert np.allclose(short.steps.t, full.steps.t[:3])


def test_tighter_tolerances_reduce_the_error_and_the_step_size():
    _, _, _, loose = _solve(atol=1e-2, rtol=1e-1)
    _, _, _, tight = _solve(atol=1e-7, rtol=1e-5)
    x_true = float(logistic_analytic(5.0))
    assert abs(float(tight.m[-1, 0]) - x_true) < abs(float(loose.m[-1, 0]) - x_true)
    mean_h = [float(np.mean(r.steps.h[: int(r.steps.count)])) for r in (tight, loose)]
    assert mean_h[0] < mean_h[1]


def test_p_controller_reaches_the_end():
    _, _, _, res = _solve(controller=PController(order=2), atol=1e-5, rtol=1e-3)
    assert bool(res.success) and int(res.steps.count) > 0


def _previous_means(m_0, steps, n):
    return np.concatenate([m_0[None], steps.m[: n - 1]], axis=0)


@pytest.mark.parametrize("calibration", ["dynamic", "diagonal"])
def test_each_steps_diffusion_is_the_quasi_mle_at_its_prediction(calibration):
    prior, m_0, measure, res = _solve(calibration=calibration)
    steps = res.steps
    n = int(steps.count)
    for i, m_prev in enumerate(_previous_means(m_0, steps, n)):
        h, t = float(steps.h[i]), float(steps.t[i])
        m_pred = prior.A(h) @ m_prev + prior.b(h)
        H, c = measure.linearize(m_pred, t=t)
        residual = H @ m_pred + c
        if calibration == "dynamic":
            expected = quasi_mle_sigma_sqr_from_Q(residual, H, prior.Q_sqr(h))
        else:
            expected = residual**2 / np.einsum("ij,jk,ik->i", H, prior.Q(h), H)
        assert np.allclose(steps.sigma_sqr[i], expected, rtol=1e-9)


def test_no_calibration_records_the_estimate_but_propagates_unscaled():
    _, _, _, dynamic = _solve(calibration="dynamic")
    _, _, _, none = _solve(calibration="none")
    n = int(none.steps.count)
    assert bool(np.all(none.steps.sigma_sqr[:n] > 0))
    sigmas = dynamic.steps.sigma_sqr[: int(dynamic.steps.count)]
    if float(np.max(sigmas)) < 1.0:
        cov_dyn = dynamic.P_sqr[-1].T @ dynamic.P_sqr[-1]
        cov_none = none.P_sqr[-1].T @ none.P_sqr[-1]
        assert float(np.trace(cov_dyn)) < float(np.trace(cov_none))


def test_running_mean_diffusion_in_the_error_rejects_fewer_steps():
    _, _, _, per_step = _solve(atol=1e-5, rtol=1e-3, sigma_in_error="per_step")
    _, _, _, running = _solve(atol=1e-5, rtol=1e-3, sigma_in_error="running_mean")
    assert bool(per_step.success) and bool(running.success)
    assert int(np.sum(running.n_rejected)) <= int(np.sum(per_step.n_rejected))


def test_diagonal_calibration_resolves_a_multiscale_problem():
    rate = 2.0
    x0 = np.asarray([1e-5, 1e-10])

    def vf(x, *, t):
        return rate * x * (1 - x)

    prior = IWP(q=3, d=2)
    m_0, P_0_sqr = taylor_mode_initialization(vf, x0, q=3)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    res = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.array([0.0, 15.0]),
        atol=1e-5,
        rtol=1e-3,
        calibration="diagonal",
        max_steps=20000,
    )
    x2_true = 1.0 / (1.0 + (1.0 / float(x0[1]) - 1.0) * np.exp(-rate * 15.0))
    assert bool(res.success)
    assert abs(float(res.m[-1, 1]) - float(x2_true)) < 1e-3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma_in_error": "ema"}, "sigma_in_error"),
        ({"record_steps": -1}, "record_steps"),
        ({"calibration": "bogus"}, "calibration must be"),
    ],
)
@pytest.mark.wrong_input_types
def test_invalid_arguments_raise(kwargs, match):
    prior, m_0, P_0_sqr, measure = _setup()
    with pytest.raises(ValueError, match=match):
        gaussian_filter_adaptive(m_0, P_0_sqr, prior, measure, SAVE, **kwargs)


SAVE_AT = np.linspace(0.0, 2.0, 5)


def _logistic_problem(x0_val=0.5, theta=1.0):
    def vf(x, *, t):
        return theta * x * (1.0 - x)

    x0 = np.array([x0_val])
    prior = IWP(q=2, d=1)
    m_0, S0_sqr = taylor_mode_initialization(vf, x0, q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return prior, measure, m_0, S0_sqr


def test_accuracy_vs_analytic_logistic():
    prior, measure, m_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, SAVE_AT, atol=1e-8, rtol=1e-8
    )
    x_true = 1.0 / (1.0 + np.exp(-SAVE_AT))  # x0 = 0.5
    assert np.max(np.abs(res.m[:, 0] - x_true)) < 1e-6
    assert np.isfinite(res.log_likelihood)


def test_grid_shapes_and_initial_state():
    prior, measure, m_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, SAVE_AT)
    assert res.t.shape == SAVE_AT.shape
    assert res.m.shape == (SAVE_AT.shape[0], m_0.shape[0])
    assert res.P_sqr.shape == (SAVE_AT.shape[0], m_0.shape[0], m_0.shape[0])
    assert np.allclose(res.t, SAVE_AT)
    assert np.allclose(res.m[0], m_0)  # save_at[0] returns the initial state
    assert np.allclose(res.P_sqr[0], S0_sqr)


def test_jit_matches_eager():
    prior, measure, m_0, S0_sqr = _logistic_problem()
    args = (m_0, S0_sqr, prior, measure, SAVE_AT)
    eager = gaussian_filter_adaptive(*args)
    jitted = eqx.filter_jit(gaussian_filter_adaptive)(*args)
    assert np.allclose(eager.m, jitted.m, atol=1e-10)
    assert np.allclose(eager.log_likelihood, jitted.log_likelihood, atol=1e-9)


def test_vmap_over_initial_conditions():
    prior = IWP(q=2, d=1)

    def vf(x, *, t):
        return x * (1.0 - x)

    measure = ODEInformation(vf, prior.E0, prior.E1)

    def solve(x0):
        m_0, S0_sqr = taylor_mode_initialization(vf, x0, q=2)
        return gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, SAVE_AT).m[-1, 0]

    x0_batch = np.array([[0.3], [0.5], [0.7]])
    out = jax.vmap(solve)(x0_batch)
    looped = np.array([jax.jit(solve)(x0) for x0 in x0_batch])
    assert np.allclose(out, looped, rtol=1e-10)
    # The logistic solution x0 e^t / (1 - x0 + x0 e^t) at t = 2.
    x0 = x0_batch[:, 0]
    exact = x0 * np.exp(2.0) / (1.0 - x0 + x0 * np.exp(2.0))
    assert np.allclose(out, exact, atol=1e-2)


def test_reverse_grad_matches_finite_difference():
    prior = IWP(q=2, d=1)
    save_at = np.linspace(0.0, 2.0, 4)

    def endpoint(theta):
        def vf(x, *, t):
            return theta * x * (1.0 - x)

        x0 = np.array([0.5])
        m_0, S0_sqr = taylor_mode_initialization(vf, x0, q=2)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        res = gaussian_filter_adaptive(
            m_0, S0_sqr, prior, measure, save_at, atol=1e-7, rtol=1e-7
        )
        return res.m[-1, 0]

    assert_grad_matches_fd(endpoint, np.array(1.3), rtol=0.0, atol=1e-5)


@pytest.mark.parametrize("calibration", ["dynamic", "none", "diagonal_ekf0"])
def test_calibration_modes_run(calibration):
    prior, measure, m_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, SAVE_AT, calibration=calibration
    )
    x_true = 1.0 / (1.0 + np.exp(-SAVE_AT))
    assert np.max(np.abs(res.m[:, 0] - x_true)) < 1e-2
    assert np.all(np.isfinite(res.P_sqr))


def test_obs_model_matches_fixed_grid_obs_scan():
    """Adaptive solve with observations agrees with the trusted fixed-grid
    dynamic obs-scan loop on the same grid (difference = adaptive vs fixed ODE
    discretisation between save points)."""
    prior, measure, m_0, S0_sqr = _logistic_problem()
    save_at = np.linspace(0.0, 2.0, 9)  # save_at[4] == 1.0
    meas = Measurement(np.eye(1), np.array([[0.55]]), np.array([1.0]), noise=1e-3)
    obs = prepare_observations([meas], prior.E0, save_at)

    ad = gaussian_filter_adaptive(
        m_0,
        S0_sqr,
        prior,
        measure,
        save_at,
        obs_model=obs,
        atol=1e-5,
        rtol=1e-5,
        calibration="none",
    )
    fx = gaussian_filter(
        m_0, S0_sqr, prior, measure, save_at, obs_model=obs, calibration="none"
    )
    assert np.max(np.abs(ad.m[:, 0] - fx.m[:, 0])) < 5e-3


def test_adaptive_surfaces_obs_innovation_matching_fixed_grid():
    """The adaptive obs-innovation sequence (mz_obs / Pz_obs_sqr) is surfaced and
    agrees with the trusted fixed-grid obs-scan loop on the same grid (difference
    = adaptive vs fixed ODE discretisation), and the ODE-defect innovation
    (mz / Pz_sqr) is surfaced with the right shape."""
    prior, measure, m_0, S0_sqr = _logistic_problem()
    save_at = np.linspace(0.0, 2.0, 9)  # save_at[4] == 1.0 (obs time)
    meas = Measurement(np.eye(1), np.array([[0.55]]), np.array([1.0]), noise=1e-3)
    obs = prepare_observations([meas], prior.E0, save_at)

    ad = gaussian_filter_adaptive(
        m_0,
        S0_sqr,
        prior,
        measure,
        save_at,
        obs_model=obs,
        atol=1e-5,
        rtol=1e-5,
        calibration="none",
    )
    fx = gaussian_filter(
        m_0, S0_sqr, prior, measure, save_at, obs_model=obs, calibration="none"
    )
    mz_obs_fx, Pz_obs_fx = fx.mz_obs, fx.Pz_obs_sqr
    assert ad.mz_obs.shape == mz_obs_fx.shape == (8, 1)
    assert ad.Pz_obs_sqr.shape == Pz_obs_fx.shape == (8, 1, 1)
    assert np.max(np.abs(ad.mz_obs - mz_obs_fx)) < 5e-3
    # ODE-defect innovation is surfaced too (meas_dim == 1: ODE rows only).
    assert ad.mz.shape == (8, 1)
    assert ad.Pz_sqr.shape == (8, 1, 1)
    assert np.all(np.isfinite(ad.mz)) and np.all(np.isfinite(ad.Pz_sqr))


def test_adaptive_no_obs_surfaces_ode_innovation_only():
    """Without an obs_model the ODE-defect innovation is surfaced and the
    observation channel stays None (no observations to innovate against)."""
    prior, measure, m_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, SAVE_AT)
    assert res.mz.shape == (SAVE_AT.shape[0] - 1, 1)
    assert res.Pz_sqr.shape == (SAVE_AT.shape[0] - 1, 1, 1)
    assert res.mz_obs is None and res.Pz_obs_sqr is None


def test_obs_model_assimilation_pulls_state():
    """A precise observation at a save time pulls the filtered state toward it."""
    prior, measure, m_0, S0_sqr = _logistic_problem()
    save_at = np.linspace(0.0, 2.0, 5)  # save_at[2] == 1.0; free x(1.0) ~ 0.731
    free = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, save_at, atol=1e-2, rtol=1e-2, calibration="none"
    )
    meas = Measurement(np.eye(1), np.array([[0.55]]), np.array([1.0]), noise=1e-8)
    obs = prepare_observations([meas], prior.E0, save_at)
    pulled = gaussian_filter_adaptive(
        m_0,
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

        m_0, S0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=2)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        res = gaussian_filter_adaptive(
            m_0,
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

    assert_grad_matches_fd(loglik, np.array(1.1), rtol=1e-3, atol=1e-3)


def test_obs_model_wrong_length_raises():
    prior, measure, m_0, S0_sqr = _logistic_problem()
    save_at = np.linspace(0.0, 2.0, 5)
    meas = Measurement(np.eye(1), np.array([[0.55]]), np.array([1.0]), noise=1e-3)
    # obs_model built against a DIFFERENT grid length -> mismatch
    obs_wrong = prepare_observations([meas], prior.E0, np.linspace(0.0, 2.0, 9))
    with pytest.raises(ValueError, match="steps but the grid has"):
        gaussian_filter_adaptive(
            m_0, S0_sqr, prior, measure, save_at, obs_model=obs_wrong
        )


def _blowup_problem(x0_val=1.0):
    """dx/dt = x^2 has the closed-form solution x(t) = x0 / (1 - x0 t), which
    blows up at t = 1/x0. Integrating past that point cannot succeed."""

    def vf(x, *, t):
        return x * x

    x0 = np.array([x0_val])
    prior = IWP(q=2, d=1)
    m_0, S0_sqr = taylor_mode_initialization(vf, x0, q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return prior, measure, m_0, S0_sqr


def test_success_flag_true_on_normal_solve():
    prior, measure, m_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, SAVE_AT)
    assert bool(res.success)
    assert np.all(np.isfinite(res.m))


def test_blowup_fails_gracefully_without_nan():
    """A finite-time blow-up must not silently emit NaN: the guard rejects the
    non-finite error, the solver shrinks h, and on exhaustion it reports
    success=False while leaving the output finite (the last accepted state)."""
    prior, measure, m_0, S0_sqr = _blowup_problem(x0_val=1.0)
    save_at = np.linspace(0.0, 2.0, 5)  # blow-up at t=1.0 is inside the grid
    res = gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, save_at, max_steps=200)
    assert not bool(res.success)
    # Crucially, no NaN leaks into the returned arrays.
    assert np.all(np.isfinite(res.m))
    assert np.all(np.isfinite(res.P_sqr))


def test_blowup_failure_is_jittable():
    """The guard + success flag survive jit (no Python branching on traced err).

    ``max_steps`` bounds the checkpointed while-loop and so must be static.
    """
    prior, measure, m_0, S0_sqr = _blowup_problem(x0_val=1.0)
    save_at = np.linspace(0.0, 2.0, 5)
    f = eqx.filter_jit(gaussian_filter_adaptive)
    res = f(m_0, S0_sqr, prior, measure, save_at, max_steps=200)
    assert not bool(res.success)
    assert np.all(np.isfinite(res.m))


def test_gaussian_filter_adaptive_surfaces_success():
    from ode_filters import gaussian_filter_adaptive

    prior, measure, m_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, SAVE_AT)
    assert res.success is not None
    assert bool(res.success)


# --------------------------------------------------------------------------- #
# Fixed-point smoother (smoother=True): adaptive + smoothing in the public API.
# --------------------------------------------------------------------------- #


def test_smoother_matches_fixed_grid_rts_one_step_per_interval():
    """Forced to one sub-step per save interval, the adaptive fixed-point smoother
    must reproduce the fixed-grid RTS smoother exactly (same steps, same math)."""
    from ode_filters import gaussian_filter, gaussian_filter_adaptive, rts_smoother

    prior, measure, m_0, S0_sqr = _logistic_problem()
    save = np.linspace(0.0, 2.0, 9)  # 8 intervals of 0.25 == fixed grid N=8

    res_fixed = gaussian_filter(
        m_0,
        S0_sqr,
        prior,
        measure,
        np.linspace(*(0.0, 2.0), 8 + 1),
        calibration="none",
    )
    ms_fixed, Ps_fixed = rts_smoother(res_fixed)

    # h_init == interval width + loose tolerances => exactly one accepted step
    # per interval, so the adaptive grid coincides with the fixed grid.
    res_ad = gaussian_filter_adaptive(
        m_0,
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
    ms_ad, Ps_ad = rts_smoother(res_ad)

    def cov(P):
        return np.einsum("kij,kil->kjl", P, P)

    assert np.allclose(ms_ad, ms_fixed, atol=1e-8)
    assert np.allclose(cov(Ps_ad), cov(Ps_fixed), atol=1e-8)


def test_smoother_matches_analytic_when_substepping():
    """With genuine adaptive sub-stepping and tight tolerances, the smoothed mean
    matches the analytic logistic solution at the save grid."""
    from ode_filters import gaussian_filter_adaptive, rts_smoother

    prior, measure, m_0, S0_sqr = _logistic_problem()  # x0=0.5
    save = np.linspace(0.0, 2.0, 6)
    res = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, save, smoother=True, atol=1e-9, rtol=1e-9
    )
    assert bool(res.success)
    ms, _ = rts_smoother(res)
    x_true = 1.0 / (1.0 + np.exp(-save))
    assert np.max(np.abs(ms[:, 0] - x_true)) < 1e-5


def test_smoother_last_equals_filtered_last():
    """RTS property: the last smoothed state equals the last filtered state."""
    from ode_filters import gaussian_filter_adaptive, rts_smoother

    prior, measure, m_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, SAVE_AT, smoother=True, atol=1e-7, rtol=1e-7
    )
    ms, Ps = rts_smoother(res)
    assert np.allclose(ms[-1], res.m[-1], atol=1e-8)
    assert np.allclose(Ps[-1].T @ Ps[-1], res.P_sqr[-1].T @ res.P_sqr[-1], atol=1e-8)


def test_smoother_reduces_uncertainty():
    """Smoothing must not increase the marginal variance at interior save points."""
    from ode_filters import gaussian_filter_adaptive, rts_smoother

    prior, measure, m_0, S0_sqr = _logistic_problem()
    res = gaussian_filter_adaptive(
        m_0,
        S0_sqr,
        prior,
        measure,
        SAVE_AT,
        smoother=True,
        calibration="none",
        atol=1e-7,
        rtol=1e-7,
    )
    _, Ps = rts_smoother(res)
    k = SAVE_AT.shape[0] // 2
    var_filt = float(np.sum(res.P_sqr[k] ** 2))
    var_smooth = float(np.sum(Ps[k] ** 2))
    assert var_smooth <= var_filt + 1e-12


def test_smoother_filtered_means_match_filtering_only_run():
    """smoother=True must not change the filtered solution (only add a backward pass)."""
    from ode_filters import gaussian_filter_adaptive

    prior, measure, m_0, S0_sqr = _logistic_problem()
    filt = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, SAVE_AT, atol=1e-7, rtol=1e-7
    )
    smoo = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, SAVE_AT, smoother=True, atol=1e-7, rtol=1e-7
    )
    assert np.allclose(filt.m, smoo.m, atol=1e-10)
    assert np.allclose(filt.P_sqr, smoo.P_sqr, atol=1e-10)
    assert filt.G_back is None and smoo.G_back is not None


def test_smoother_with_observations_matches_the_fixed_grid_smoother():
    """One step per save interval: the adaptive smoother with data is the
    fixed-grid smoother with data."""
    prior, measure, m_0, S0_sqr = _logistic_problem()
    meas = Measurement(
        np.eye(1), np.array([[0.55]]), np.array([SAVE_AT[2]]), noise=1e-3
    )
    obs = prepare_observations([meas], prior.E0, SAVE_AT)
    ad = gaussian_filter_adaptive(
        m_0,
        S0_sqr,
        prior,
        measure,
        SAVE_AT,
        obs_model=obs,
        smoother=True,
        calibration="none",
        h_init=1.0,
        atol=1e3,
        rtol=1e3,
    )
    assert np.all(ad.n_accepted == 1)
    fx = gaussian_filter(
        m_0, S0_sqr, prior, measure, SAVE_AT, obs_model=obs, calibration="none"
    )
    assert np.allclose(rts_smoother(ad).m, rts_smoother(fx).m, atol=1e-10)


# --------------------------------------------------------------------------- #
# Entry-point validation: the adaptive solvers accept exactly what the
# fixed-grid gaussian_filter accepts.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "prior",
    [
        PrecondIWP(q=2, d=1),
        PrecondMaternPrior(q=2, d=1, length_scale=1.0),
    ],
    ids=["PrecondIWP", "PrecondMaternPrior"],
)
def test_preconditioned_priors_match_their_plain_versions(prior):
    """Preconditioning is a change of variables in each step, adaptive or not."""
    plain = (
        IWP(q=2, d=1)
        if isinstance(prior, PrecondIWP)
        else MaternPrior(q=2, d=1, length_scale=1.0)
    )
    _, measure, m_0, S0_sqr = _logistic_problem()
    a = gaussian_filter_adaptive(m_0, S0_sqr, plain, measure, SAVE_AT, smoother=True)
    b = gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, SAVE_AT, smoother=True)
    assert np.allclose(b.m, a.m, rtol=1e-8, atol=1e-12)
    assert np.allclose(rts_smoother(b).m, rts_smoother(a).m, rtol=1e-8, atol=1e-12)
    assert np.all(b.n_accepted == a.n_accepted)


@pytest.mark.wrong_input_types
def test_adaptive_solve_rejects_unknown_calibration():
    """A misspelled mode used to fall through to "none" on the adaptive path."""
    prior, measure, m_0, S0_sqr = _logistic_problem()
    with pytest.raises(ValueError, match="calibration must be"):
        gaussian_filter_adaptive(
            m_0, S0_sqr, prior, measure, SAVE_AT, calibration="diagnoal"
        )


@pytest.mark.parametrize("calibration", ["diagonal", "diagonal_ekf0"])
def test_adaptive_solve_rejects_non_diagonal_xi_in_diagonal_modes(calibration):
    prior = IWP(q=2, d=2, Xi=np.array([[1.0, 0.5], [0.5, 1.0]]))
    measure = ODEInformation(lambda x, *, t: -x, prior.E0, prior.E1)
    with pytest.raises(ValueError, match="state block of Xi"):
        gaussian_filter_adaptive(
            np.ones(6),
            np.zeros((6, 6)),
            prior,
            measure,
            SAVE_AT,
            calibration=calibration,
        )


class _HalvingController:
    """Duck-typed controller: only ``propose``, never grows the step."""

    def propose(self, h, err, err_prev):
        return 0.5 * h


def test_adaptive_solve_calls_the_controllers_propose():
    """gaussian_filter_adaptive used to read private PI coefficients and run its
    own copy of the PI law, so a custom ``propose`` was silently ignored."""
    prior, measure, m_0, S0_sqr = _logistic_problem()
    controller = _HalvingController()
    assert isinstance(controller, StepSizeController)
    default = gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, SAVE_AT)
    halving = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, SAVE_AT, controller=controller, max_steps=32
    )
    assert bool(default.success)
    # Halving every step cannot reach the first save time in 32 steps.
    assert not bool(halving.success)


def test_adaptive_solve_with_explicit_default_controller_is_unchanged():
    """Passing the default controller explicitly gives the default result."""
    prior, measure, m_0, S0_sqr = _logistic_problem()
    default = gaussian_filter_adaptive(m_0, S0_sqr, prior, measure, SAVE_AT)
    explicit = gaussian_filter_adaptive(
        m_0, S0_sqr, prior, measure, SAVE_AT, controller=PIController(order=2)
    )
    assert np.array_equal(default.m, explicit.m)


def test_adaptive_solve_runs_joint_prior():
    """Plain joint priors are still supported on the adaptive path."""
    joint = JointPrior(IWP(q=2, d=1), IWP(q=1, d=1))

    def vf(x, u, *, t):
        return -x

    measure = ODEInformationWithHidden(vf, joint.E0_x, joint.E1, joint.E0_hidden)
    m_0 = np.array([1.0, -1.0, 1.0, 0.0, 0.0])
    res = gaussian_filter_adaptive(m_0, np.zeros((5, 5)), joint, measure, SAVE_AT)
    assert bool(res.success)
    assert np.allclose(res.m[:, 0], np.exp(-SAVE_AT), atol=1e-3)
