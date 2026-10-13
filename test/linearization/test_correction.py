"""Tests for the corrections: EK0, EK1 and the IEKF.

A correction linearizes at the predicted mean (``linearize``) and updates the
predicted Gaussian (``correct``). Each is checked against the dense Kalman
update of the affine model it should produce.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    BlackBoxMeasurement,
    Conservation,
    Correction,
    IteratedTaylorCorrection,
    JointPrior,
    ODEInformation,
    ODEInformationWithHidden,
    TaylorCorrection,
    gaussian_filter,
    prepare_observations,
)
from ode_filters.gaussian import AffineModel, Gaussian, marginalize, update
from ode_filters.information import Measurement
from ode_filters.linearization import taylor_linearize
from ode_filters.priors import IWP, taylor_mode_initialization
from test.helpers import assert_close, cov

H_STEP = 0.1
T_EVAL = 0.3


def _predict(prior, m0, P0_sqr, h=H_STEP) -> Gaussian:
    return marginalize(prior.A(h), prior.b(h), prior.Q_sqr(h), Gaussian(m0, P0_sqr))


def _nonlinear():
    def vf(x, *, t):
        return -(x**2) + 0.5

    prior = IWP(q=2, d=1, Xi=0.5 * np.eye(1))
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
    return prior, ODEInformation(vf, prior.E0, prior.E1), m0, P0_sqr


def _dense_update(model: AffineModel, x: Gaussian):
    P = cov(x.P_sqr)
    r = model.H @ x.m + model.c
    S = model.H @ P @ model.H.T + cov(model.R_sqr)
    K = P @ model.H.T @ np.linalg.inv(S)
    return x.m - K @ r, P - K @ S @ K.T


# --------------------------------------------------------------------------- #
# EK1
# --------------------------------------------------------------------------- #


def test_ek1_linearizes_like_the_measurement_model():
    prior, measure, m0, P0_sqr = _nonlinear()
    x = _predict(prior, m0, P0_sqr)
    model = TaylorCorrection().linearize(measure, x.m, t=T_EVAL)
    H, c = measure.linearize(x.m, t=T_EVAL)
    assert_close(model.H, H)
    assert_close(model.c, c)
    assert_close(model.R_sqr, measure.get_noise(t=T_EVAL))


def test_ek1_update_is_the_kalman_update_of_its_linearization():
    prior, measure, m0, P0_sqr = _nonlinear()
    x = _predict(prior, m0, P0_sqr)
    correction = TaylorCorrection(order=1)
    model = correction.linearize(measure, x.m, t=T_EVAL)
    posterior, innovation = correction.correct(measure, x, model, t=T_EVAL)
    m, P = _dense_update(model, x)
    assert_close(posterior.m, m, rtol=1e-10)
    assert_close(cov(posterior.P_sqr), P, rtol=1e-8)
    assert_close(innovation.m, model.H @ x.m + model.c)


def test_ek1_with_conservation_rows():
    def vf(x, *, t):
        return np.array([-x[0] * x[1], x[0] * x[1] - 0.5 * x[1]])

    prior = IWP(q=2, d=2)
    cons = Conservation(np.ones((1, 2)), np.array([1.0]))
    measure = ODEInformation(vf, prior.E0, prior.E1, constraints=[cons])
    m0 = np.linspace(0.1, 0.6, 6)
    x = _predict(prior, m0, 0.2 * np.eye(6))
    model = TaylorCorrection().linearize(measure, x.m, t=T_EVAL)
    assert model.H.shape == (3, 6)
    posterior, innovation = TaylorCorrection().correct(measure, x, model, t=T_EVAL)
    assert innovation.m.shape == (3,)
    assert_close(posterior.m, _dense_update(model, x)[0], rtol=1e-10)


def test_hidden_state_model_composes_with_ek1():
    def vf(x, u, *, t):
        return -u * x

    joint = JointPrior(IWP(q=2, d=1), IWP(q=0, d=1, Xi=1e-10 * np.eye(1)))
    measure = ODEInformationWithHidden(vf, joint.E0_x, joint.E1, joint.E0_hidden)
    D = joint.E1.shape[1]
    x = _predict(joint, 0.5 * np.arange(1, D + 1, dtype=float), 0.3 * np.eye(D))
    model = TaylorCorrection().linearize(measure, x.m, t=T_EVAL)
    posterior, _ = TaylorCorrection().correct(measure, x, model, t=T_EVAL)
    assert_close(posterior.m, _dense_update(model, x)[0], rtol=1e-10)


# --------------------------------------------------------------------------- #
# EK0
# --------------------------------------------------------------------------- #


def test_ek0_jacobian_is_the_lhs_selector_and_the_residual_is_exact():
    prior, measure, m0, P0_sqr = _nonlinear()
    x = _predict(prior, m0, P0_sqr)
    model = TaylorCorrection(order=0).linearize(measure, x.m, t=T_EVAL)
    assert_close(model.H, measure.E_constraint)
    assert_close(model.H @ x.m + model.c, measure.g(x.m, t=T_EVAL))


def test_ek0_keeps_the_conservation_rows():
    def vf(x, *, t):
        return -x

    prior = IWP(q=2, d=2)
    cons = Conservation(np.array([[1.0, 2.0]]), np.array([1.0]))
    measure = ODEInformation(vf, prior.E0, prior.E1, constraints=[cons])
    model = taylor_linearize(measure, np.ones(6), t=0.0, order=0)
    assert_close(model.H[:2], prior.E1)
    assert_close(model.H[2:], cons.A @ prior.E0)


def test_ek0_never_differentiates_the_vector_field():
    @jax.custom_jvp
    def f(x):
        return -x

    @f.defjvp
    def _f_jvp(primals, tangents):
        raise AssertionError("EK0 must not differentiate the vector field")

    def vf(x, *, t):
        return f(x)

    prior = IWP(q=2, d=1)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    model = TaylorCorrection(order=0).linearize(measure, np.ones(3), t=0.0)
    assert_close(model.H, prior.E1)


def test_ek0_needs_an_ode_information_model():
    measure = BlackBoxMeasurement(lambda s, *, t: s[:1], state_dim=3, obs_dim=1)
    with pytest.raises(TypeError, match="EK0"):
        TaylorCorrection(order=0).linearize(measure, np.ones(3), t=0.0)


def test_ek0_differs_from_ek1_for_a_nonlinear_vector_field_only():
    prior, measure, m0, P0_sqr = _nonlinear()
    x = _predict(prior, m0, P0_sqr)

    def posterior_mean(order, measure):
        correction = TaylorCorrection(order=order)
        model = correction.linearize(measure, x.m, t=T_EVAL)
        return correction.correct(measure, x, model, t=T_EVAL)[0].m

    assert not np.allclose(posterior_mean(0, measure), posterior_mean(1, measure))

    def constant(x, *, t):
        return np.full_like(x, 0.7)

    flat = ODEInformation(constant, prior.E0, prior.E1)
    assert_close(posterior_mean(0, flat), posterior_mean(1, flat), rtol=1e-12)


def test_ek0_solves_linear_decay():
    def vf(x, *, t):
        return -x

    prior = IWP(q=2, d=1)
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    res = gaussian_filter(
        m0,
        P0_sqr,
        prior,
        measure,
        np.linspace(0.0, 2.0, 201),
        correction=TaylorCorrection(order=0),
    )
    assert np.isclose(res.m[-1, 0], np.exp(-2.0), atol=1e-2)


# --------------------------------------------------------------------------- #
# IEKF
# --------------------------------------------------------------------------- #


def test_iekf_with_one_iteration_is_ek1():
    prior, measure, m0, P0_sqr = _nonlinear()
    x = _predict(prior, m0, P0_sqr)
    model = TaylorCorrection().linearize(measure, x.m, t=T_EVAL)
    ek1 = TaylorCorrection().correct(measure, x, model, t=T_EVAL)
    iekf = IteratedTaylorCorrection(max_iters=1).correct(measure, x, model, t=T_EVAL)
    assert_close(iekf[0].m, ek1[0].m, rtol=1e-12)
    assert_close(cov(iekf[0].P_sqr), cov(ek1[0].P_sqr), rtol=1e-12)


def test_iekf_converges_to_a_fixed_point_of_the_relinearization():
    prior, measure, m0, P0_sqr = _nonlinear()
    x = _predict(prior, m0, P0_sqr)
    model = TaylorCorrection().linearize(measure, x.m, t=T_EVAL)
    a = IteratedTaylorCorrection(max_iters=10).correct(measure, x, model, t=T_EVAL)[0]
    b = IteratedTaylorCorrection(max_iters=20).correct(measure, x, model, t=T_EVAL)[0]
    assert_close(a.m, b.m, atol=1e-10)
    # One more update linearized at the converged mean must not move it.
    again = update(taylor_linearize(measure, a.m, t=T_EVAL), x)[0]
    assert_close(again.m, a.m, atol=1e-9)


def test_iekf_differs_from_ek1_for_a_nonlinear_vector_field():
    def vf(x, *, t):
        return -(x**2) + 0.5

    prior = IWP(q=2, d=1, Xi=0.5 * np.eye(1))
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([2.0]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    x = _predict(prior, m0, P0_sqr, h=1.0)
    model = TaylorCorrection().linearize(measure, x.m, t=0.0)
    ek1 = TaylorCorrection().correct(measure, x, model, t=0.0)[0]
    iekf = IteratedTaylorCorrection(max_iters=20).correct(measure, x, model, t=0.0)[0]
    assert not np.allclose(iekf.m, ek1.m, atol=1e-7)


# --------------------------------------------------------------------------- #
# Configuration, transformations, use in a filter
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_order", [-1, 2, 0.5])
def test_invalid_order_raises(bad_order):
    with pytest.raises(ValueError, match="order must be 0"):
        TaylorCorrection(order=bad_order)


def test_invalid_iteration_count_raises():
    with pytest.raises(ValueError, match="max_iters"):
        IteratedTaylorCorrection(max_iters=0)


@pytest.mark.parametrize(
    "correction",
    [TaylorCorrection(order=0), TaylorCorrection(), IteratedTaylorCorrection()],
    ids=["ek0", "ek1", "iekf"],
)
def test_corrections_are_jittable(correction):
    prior, measure, m0, P0_sqr = _nonlinear()
    x = _predict(prior, m0, P0_sqr)

    def run(x):
        model = correction.linearize(measure, x.m, t=T_EVAL)
        return correction.correct(measure, x, model, t=T_EVAL)[0].m

    assert_close(jax.jit(run)(x), run(x), rtol=1e-12)


def test_corrections_are_static_pytrees():
    assert isinstance(TaylorCorrection(order=0), Correction)
    assert jax.tree_util.tree_leaves(TaylorCorrection(order=1)) == []
    assert jax.tree_util.tree_leaves(IteratedTaylorCorrection()) == []


def test_base_correction_is_abstract():
    with pytest.raises(TypeError):
        Correction()


def test_every_correction_runs_with_observations():
    def vf(x, u, *, t):
        return -u * x

    joint = JointPrior(IWP(q=2, d=1), IWP(q=0, d=1, Xi=1e-10 * np.eye(1)))
    measure = ODEInformationWithHidden(vf, joint.E0_x, joint.E1, joint.E0_hidden)
    ts = np.linspace(0.0, 1.0, 41)
    data = Measurement(np.eye(1), np.ones((40, 1)), ts[1:], noise=1e-2)
    obs_model = prepare_observations([data], joint.E0_x, ts)
    D = joint.E1.shape[1]
    args = (np.zeros(D), 0.3 * np.eye(D), joint, measure, ts)

    default = gaussian_filter(*args, obs_model=obs_model)
    ek1 = gaussian_filter(*args, obs_model=obs_model, correction=TaylorCorrection())
    assert_close(default.m, ek1.m)
    for correction in (TaylorCorrection(order=0), IteratedTaylorCorrection(5)):
        res = gaussian_filter(*args, obs_model=obs_model, correction=correction)
        assert bool(np.all(np.isfinite(res.m)))
