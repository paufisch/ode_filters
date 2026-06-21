"""Tests for the consolidated gaussian_filter / gaussian_filter_adaptive / rts_smoother API.

These verify the new entry points reproduce the existing (tested) loop internals
they wrap, that dispatch (preconditioned, observations) is correct, and that
jit/grad still work.
"""

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    IteratedTaylorCorrection,
    ODEInformation,
    PrecondIWP,
    TaylorCorrection,
    gaussian_filter,
    gaussian_filter_adaptive,
    prepare_observations,
    rts_smoother,
    taylor_mode_initialization,
)
from ode_filters.filters.ode_filter_adaptive import sqr_adaptive_solve
from ode_filters.filters.ode_filter_loop import (
    rts_sqr_smoother_loop,
    rts_sqr_smoother_loop_preconditioned,
    sqr_loop_dynamic_scan,
    sqr_loop_preconditioned_dynamic_scan,
)
from ode_filters.measurement.measurement_models import Measurement

TSPAN = (0.0, 2.0)
N = 40


def _logistic(prior_cls=IWP):
    def vf(x, *, t):
        return x * (1.0 - x)

    prior = prior_cls(q=2, d=1)
    mu_0, P0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return prior, measure, mu_0, P0_sqr


def test_plain_matches_dynamic_scan():
    prior, measure, mu_0, P0_sqr = _logistic()
    res = gaussian_filter(mu_0, P0_sqr, prior, measure, TSPAN, N, calibration="dynamic")
    ref = sqr_loop_dynamic_scan(
        mu_0, P0_sqr, prior, measure, TSPAN, N, calibration="dynamic"
    )
    assert np.allclose(res.m, ref[0])
    assert np.allclose(res.P_sqr, ref[1])
    assert np.allclose(res.log_likelihood, ref[-1])
    assert res.T is None  # plain prior
    assert res.log_likelihood_obs is None


def test_calibration_none_is_static():
    prior, measure, mu_0, P0_sqr = _logistic()
    res = gaussian_filter(mu_0, P0_sqr, prior, measure, TSPAN, N, calibration="none")
    x_true = 1.0 / (1.0 + np.exp(-np.linspace(*TSPAN, N + 1)))
    assert np.max(np.abs(res.m[:, 0] - x_true)) < 1e-2


def test_observations_dispatch_and_loglik():
    prior, measure, mu_0, P0_sqr = _logistic()
    ts = np.linspace(*TSPAN, N + 1)
    meas = Measurement(np.eye(1), np.array([[0.7]]), np.array([ts[20]]), noise=1e-3)
    obs = prepare_observations([meas], prior.E0, ts)
    res = gaussian_filter(
        mu_0, P0_sqr, prior, measure, TSPAN, N, calibration="none", obs_model=obs
    )
    ref = sqr_loop_dynamic_scan(
        mu_0, P0_sqr, prior, measure, TSPAN, N, calibration="none", obs_model=obs
    )
    assert np.allclose(res.m, ref[0])
    assert res.log_likelihood_obs is not None
    assert np.allclose(res.log_likelihood_obs, ref[-1])


def test_correction_ek0_passes_through():
    prior, measure, mu_0, P0_sqr = _logistic()
    res = gaussian_filter(
        mu_0, P0_sqr, prior, measure, TSPAN, N, correction=TaylorCorrection(order=0)
    )
    ref = sqr_loop_dynamic_scan(
        mu_0, P0_sqr, prior, measure, TSPAN, N, correction=TaylorCorrection(order=0)
    )
    assert np.allclose(res.m, ref[0])


def test_preconditioned_dispatch_matches_loop():
    prior, measure, mu_0, P0_sqr = _logistic(PrecondIWP)
    res = gaussian_filter(mu_0, P0_sqr, prior, measure, TSPAN, N, calibration="dynamic")
    ref = sqr_loop_preconditioned_dynamic_scan(
        mu_0, P0_sqr, prior, measure, TSPAN, N, calibration="dynamic"
    )
    assert np.allclose(res.m, ref[0])
    assert np.allclose(res.P_sqr, ref[1])
    assert res.T is not None  # preconditioned path populated the smoother inputs


def _cov_seq(P_sqr_seq):
    """Per-step covariance P = P_sqr.T @ P_sqr for a stack of square-root factors."""
    return np.einsum("kij,kil->kjl", P_sqr_seq, P_sqr_seq)


def test_preconditioned_correction_runs_for_ek0_and_iekf():
    """The Correction abstraction now reaches the preconditioned path."""
    prior, measure, mu_0, P0_sqr = _logistic(PrecondIWP)
    for corr in (TaylorCorrection(order=0), IteratedTaylorCorrection(max_iters=3)):
        res = gaussian_filter(mu_0, P0_sqr, prior, measure, TSPAN, N, correction=corr)
        assert np.all(np.isfinite(res.m))
        assert np.all(np.isfinite(res.P_sqr))


def test_preconditioned_ek0_matches_plain_ek0():
    """EK0 via the bar-space wrapper must equal plain-space EK0 (not just run).

    Preconditioning is a similarity transform, so with the same correction and
    calibration the plain and preconditioned recursions are mathematically
    identical -- this validates the _BarMeasure EK0 logic, not just that it runs.
    """
    ek0 = TaylorCorrection(order=0)
    pp, mp, mu0p, P0p = _logistic(IWP)
    res_plain = gaussian_filter(
        mu0p, P0p, pp, mp, TSPAN, N, calibration="none", correction=ek0
    )
    pc, mc, mu0c, P0c = _logistic(PrecondIWP)
    res_pre = gaussian_filter(
        mu0c, P0c, pc, mc, TSPAN, N, calibration="none", correction=ek0
    )
    assert np.allclose(res_plain.m, res_pre.m, atol=1e-7)
    assert np.allclose(_cov_seq(res_plain.P_sqr), _cov_seq(res_pre.P_sqr), atol=1e-7)


def test_preconditioned_ek0_differs_from_ek1():
    """EK0 and EK1 must give different results on a nonlinear ODE (precond path)."""
    prior, measure, mu_0, P0_sqr = _logistic(PrecondIWP)
    res0 = gaussian_filter(
        mu_0, P0_sqr, prior, measure, TSPAN, N, correction=TaylorCorrection(order=0)
    )
    res1 = gaussian_filter(
        mu_0, P0_sqr, prior, measure, TSPAN, N, correction=TaylorCorrection(order=1)
    )
    assert not np.allclose(res0.m, res1.m, atol=1e-7)


def test_adaptive_correction_ek0_and_iekf():
    """gaussian_filter_adaptive accepts a correction= (EK0/IEKF), no longer EK1-only."""
    prior, measure, mu_0, P0_sqr = _logistic()
    save_at = np.linspace(*TSPAN, 5)
    for corr in (TaylorCorrection(order=0), IteratedTaylorCorrection(max_iters=2)):
        res = gaussian_filter_adaptive(
            mu_0, P0_sqr, prior, measure, save_at, correction=corr
        )
        assert bool(res.success)
        assert np.all(np.isfinite(res.m))


def test_adaptive_iekf_max_iters_1_equals_ek1():
    """IEKF with max_iters=1 reduces to EK1 (the adaptive default)."""
    prior, measure, mu_0, P0_sqr = _logistic()
    save_at = np.linspace(*TSPAN, 5)
    ek1 = gaussian_filter_adaptive(mu_0, P0_sqr, prior, measure, save_at)
    iekf1 = gaussian_filter_adaptive(
        mu_0,
        P0_sqr,
        prior,
        measure,
        save_at,
        correction=IteratedTaylorCorrection(max_iters=1),
    )
    assert np.allclose(ek1.m, iekf1.m, atol=1e-10)


def test_preconditioned_with_obs_raises():
    prior, measure, mu_0, P0_sqr = _logistic(PrecondIWP)
    ts = np.linspace(*TSPAN, N + 1)
    meas = Measurement(np.eye(1), np.array([[0.7]]), np.array([ts[20]]), noise=1e-3)
    obs = prepare_observations([meas], prior.E0, ts)
    with pytest.raises(NotImplementedError, match=r"[Pp]reconditioned"):
        gaussian_filter(mu_0, P0_sqr, prior, measure, TSPAN, N, obs_model=obs)


def test_rts_smoother_plain_matches_manual():
    prior, measure, mu_0, P0_sqr = _logistic()
    res = gaussian_filter(mu_0, P0_sqr, prior, measure, TSPAN, N)
    m_s, P_s = rts_smoother(prior, res)
    m_ref, P_ref = rts_sqr_smoother_loop(
        res.m[-1], res.P_sqr[-1], res.G_back, res.d_back, res.P_back_sqr, N
    )
    assert np.allclose(m_s, m_ref)
    assert np.allclose(P_s, P_ref)
    assert np.allclose(m_s[-1], res.m[-1])  # last smoothed == last filtered


def test_rts_smoother_preconditioned_matches_manual():
    prior, measure, mu_0, P0_sqr = _logistic(PrecondIWP)
    res = gaussian_filter(mu_0, P0_sqr, prior, measure, TSPAN, N)
    m_s, P_s = rts_smoother(prior, res)
    m_ref, P_ref = rts_sqr_smoother_loop_preconditioned(
        res.m[-1],
        res.P_sqr[-1],
        res.m_bar[-1],
        res.P_bar_sqr[-1],
        res.G_back,
        res.d_back,
        res.P_back_sqr,
        N,
        res.T,
    )
    assert np.allclose(m_s, m_ref)
    assert np.allclose(P_s, P_ref)


def test_adaptive_matches_solve_and_has_no_backward_pass():
    prior, measure, mu_0, P0_sqr = _logistic()
    save_at = np.linspace(*TSPAN, 5)
    res = gaussian_filter_adaptive(
        mu_0, P0_sqr, prior, measure, save_at, atol=1e-6, rtol=1e-6
    )
    ref = sqr_adaptive_solve(
        mu_0, P0_sqr, prior, measure, save_at, atol=1e-6, rtol=1e-6
    )
    assert np.allclose(res.m, ref.m)
    assert np.allclose(res.log_likelihood, ref.log_likelihood)
    assert res.G_back is None  # filtering only


def test_rts_smoother_on_adaptive_result_raises():
    prior, measure, mu_0, P0_sqr = _logistic()
    res = gaussian_filter_adaptive(mu_0, P0_sqr, prior, measure, np.linspace(*TSPAN, 5))
    with pytest.raises(ValueError, match="backward pass"):
        rts_smoother(prior, res)


def test_gaussian_filter_jit_and_grad():
    prior = IWP(q=2, d=1)

    def endpoint(theta):
        def vf(x, *, t):
            return theta * x * (1.0 - x)

        mu_0, P0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=2)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        res = gaussian_filter(
            mu_0, P0_sqr, prior, measure, TSPAN, N, calibration="none"
        )
        return res.m[-1, 0]

    val = jax.jit(endpoint)(np.array(1.0))
    g = jax.grad(endpoint)(np.array(1.0))
    assert np.isfinite(val) and np.isfinite(g)
