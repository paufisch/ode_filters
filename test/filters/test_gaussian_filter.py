"""Tests for ``gaussian_filter`` and ``rts_smoother``: grids, corrections, preconditioning.

Values are pinned by the reference outputs and the dense oracle in
``test/integration``; these tests cover behaviour and configuration.
"""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    DynamicCalibration,
    IteratedTaylorCorrection,
    MaternPrior,
    ODEInformation,
    PrecondIWP,
    PrecondMaternPrior,
    TaylorCorrection,
    gaussian_filter,
    gaussian_filter_adaptive,
    prepare_observations,
    rts_smoother,
    taylor_mode_initialization,
)
from ode_filters.information import Measurement
from test.helpers import assert_close, cov

TSPAN = (0.0, 2.0)
N = 40
TS = np.linspace(*TSPAN, N + 1)


def _logistic(prior_cls=IWP, **kwargs):
    def vf(x, *, t):
        return x * (1.0 - x)

    prior = prior_cls(q=2, d=1, **kwargs)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=2)
    return prior, ODEInformation(vf, prior.E0, prior.E1), m_0, P_0_sqr


def test_calibration_none_solves_the_logistic_equation():
    prior, measure, m_0, P_0_sqr = _logistic()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="none")
    assert np.max(np.abs(res.m[:, 0] - 1.0 / (1.0 + np.exp(-TS)))) < 1e-2


def test_result_shapes_and_fields():
    prior, measure, m_0, P_0_sqr = _logistic()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    assert res.t.shape == (N + 1,)
    assert res.m.shape == (N + 1, 3) and res.P_sqr.shape == (N + 1, 3, 3)
    assert res.m_pred.shape == (N, 3) and res.G_back.shape == (N, 3, 3)
    assert res.mz.shape == (N, 1) and res.sigma_sqr.shape == (N,)
    assert res.success is None and res.n_accepted is None and res.steps is None
    assert_close(res.filtered.m, res.m)
    assert_close(res.backward.G, res.G_back)


@pytest.mark.parametrize("plain,precond", [(IWP, PrecondIWP)], ids=["iwp"])
@pytest.mark.parametrize(
    "correction",
    [TaylorCorrection(order=0), TaylorCorrection(), IteratedTaylorCorrection(3)],
    ids=["ek0", "ek1", "iekf"],
)
def test_preconditioning_does_not_change_the_result(plain, precond, correction):
    """Preconditioning is a change of variables: same model, same answer."""
    p0, m0, mu, P = _logistic(plain)
    p1, m1, _, _ = _logistic(precond)
    a = gaussian_filter(mu, P, p0, m0, TS, correction=correction)
    b = gaussian_filter(mu, P, p1, m1, TS, correction=correction)
    assert_close(b.m, a.m, rtol=1e-9)
    assert_close(cov(b.P_sqr), cov(a.P_sqr), rtol=1e-8)
    assert_close(b.log_likelihood, a.log_likelihood, rtol=1e-9)
    assert_close(cov(b.P_pred_sqr), cov(a.P_pred_sqr), rtol=1e-8)
    assert_close(b.G_back, a.G_back, rtol=1e-8)


def test_preconditioned_matern_matches_matern_with_observations():
    p0, m0, mu, P = _logistic(MaternPrior, length_scale=1.5)
    p1, m1, _, _ = _logistic(PrecondMaternPrior, length_scale=1.5)
    data = Measurement(np.eye(1), np.array([[0.7]]), np.array([TS[20]]), noise=1e-3)
    obs = prepare_observations([data], p0.E0, TS)
    a = gaussian_filter(mu, P, p0, m0, TS, obs_model=obs)
    b = gaussian_filter(mu, P, p1, m1, TS, obs_model=obs)
    assert_close(b.m, a.m, rtol=1e-9)
    assert_close(b.log_likelihood_obs, a.log_likelihood_obs, rtol=1e-9)
    assert_close(rts_smoother(b).m, rts_smoother(a).m, rtol=1e-9)


def test_ek0_differs_from_ek1():
    prior, measure, m_0, P_0_sqr = _logistic(PrecondIWP)
    ek0 = gaussian_filter(
        m_0, P_0_sqr, prior, measure, TS, correction=TaylorCorrection(0)
    )
    ek1 = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    assert not np.allclose(ek0.m, ek1.m, atol=1e-7)


def test_non_uniform_grid_matches_a_uniform_one_where_they_coincide():
    """A grid refined only in the middle agrees with the coarse grid's endpoint."""
    prior, measure, m_0, P_0_sqr = _logistic()
    coarse = np.linspace(*TSPAN, 401)
    refined = np.asarray(onp.union1d(onp.asarray(coarse), onp.linspace(0.9, 1.1, 30)))
    a = gaussian_filter(m_0, P_0_sqr, prior, measure, coarse, calibration="none")
    b = gaussian_filter(m_0, P_0_sqr, prior, measure, refined, calibration="none")
    assert abs(float(a.m[-1, 0] - b.m[-1, 0])) < 1e-6
    assert b.m.shape[0] == refined.shape[0]


def test_traced_grid_takes_the_per_step_path_and_agrees():
    prior, measure, m_0, P_0_sqr = _logistic()
    eager = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    traced = jax.jit(lambda ts: gaussian_filter(m_0, P_0_sqr, prior, measure, ts).m)(TS)
    assert_close(traced, eager.m, rtol=1e-10)


@pytest.mark.parametrize(
    "ts",
    [np.array(0.0), np.array([0.0]), np.array([0.0, 1.0, 0.5])],
    ids=["scalar", "single-node", "decreasing"],
)
def test_invalid_grids_raise(ts):
    prior, measure, m_0, P_0_sqr = _logistic()
    with pytest.raises(ValueError, match="ts must"):
        gaussian_filter(m_0, P_0_sqr, prior, measure, ts)


def test_obs_model_on_another_grid_raises():
    prior, measure, m_0, P_0_sqr = _logistic()
    data = Measurement(np.eye(1), np.array([[0.7]]), np.array([TS[2]]))
    obs = prepare_observations([data], prior.E0, TS[:10])
    with pytest.raises(ValueError, match="steps"):
        gaussian_filter(m_0, P_0_sqr, prior, measure, TS, obs_model=obs)


def test_calibration_policy_objects_and_strings_agree():
    prior, measure, m_0, P_0_sqr = _logistic()
    by_name = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="dynamic")
    by_object = gaussian_filter(
        m_0, P_0_sqr, prior, measure, TS, calibration=DynamicCalibration()
    )
    assert_close(by_object.m, by_name.m, rtol=0.0)


@pytest.mark.parametrize("calibration", ["bogus", 3])
def test_unknown_calibration_raises(calibration):
    prior, measure, m_0, P_0_sqr = _logistic()
    with pytest.raises((ValueError, TypeError), match="calibration must be"):
        gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration=calibration)


def test_rts_smoother_ends_on_the_filter_and_needs_a_backward_pass():
    prior, measure, m_0, P_0_sqr = _logistic()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    smoothed = rts_smoother(res)
    assert_close(smoothed.m[-1], res.m[-1])
    adaptive = gaussian_filter_adaptive(m_0, P_0_sqr, prior, measure, TS[::10])
    with pytest.raises(ValueError, match="backward conditionals"):
        rts_smoother(adaptive)


def test_jit_and_grad():
    prior = IWP(q=2, d=1)

    def endpoint(theta):
        def vf(x, *, t):
            return theta * x * (1.0 - x)

        m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=2)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="none")
        return res.m[-1, 0]

    assert np.isfinite(jax.jit(endpoint)(np.array(1.0)))
    assert np.isfinite(jax.grad(endpoint)(np.array(1.0)))
