"""Tests for the pluggable Correction abstraction (W1).

These prove two things:

1. ``TaylorCorrection(order=1)`` reproduces the historical EK1 update of
   :func:`ekf1_sqr_filter_step` to within 1e-12 (numerics-preserving relocation
   of the linearization out of the measurement model and into the Correction).
2. ``TaylorCorrection(order=0)`` is a genuine EK0 model: the ODE-defect rows of
   its effective Jacobian equal the selection matrix ``E_constraint`` (no
   vector-field Jacobian), and it agrees with EK1 exactly when the vector field
   is constant (zero Jacobian).

Plus orthogonality (any ``BaseODEInformation`` model composes with any
correction), input validation, and jit/pytree smoke checks.

Black-box / transformed measurement models are out of scope for this slice:
they do not expose ``ode_dim`` / ``E_constraint`` and are wired separately
(roadmap P1).
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters.filters import (
    Correction,
    CorrectionResult,
    TaylorCorrection,
    ekf1_sqr_filter_step,
)
from ode_filters.measurement import (
    ODEconservation,
    ODEInformation,
    ODEInformationWithHidden,
)
from ode_filters.priors import IWP, JointPrior, taylor_mode_initialization

H_STEP = 0.1
T_EVAL = 0.3


def _cov(sqr: np.ndarray) -> np.ndarray:
    """Reconstruct a covariance from its (gauge-dependent) square-root factor."""
    return sqr.T @ sqr


def _predict_and_reference(prior, measure, m0, P0_sqr, h=H_STEP, t=T_EVAL):
    """Run one historical EK1 step; return prediction + reference update."""
    A = prior.A(h)
    b = prior.b(h)
    Q_sqr = np.linalg.cholesky(prior.Q(h)).T
    (m_pred, P_pred_sqr), _, (m_z, P_z_sqr), (m_t, P_t_sqr) = ekf1_sqr_filter_step(
        A, b, Q_sqr, m0, P0_sqr, measure, t
    )
    return (m_pred, P_pred_sqr), (m_z, P_z_sqr), (m_t, P_t_sqr)


def _decay_ode():
    """Nonlinear scalar ODE so the vector-field Jacobian is nonzero."""

    def vf(x, *, t):
        return -(x**2) + 0.5

    q, d = 2, 1
    prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=q)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return prior, measure, m0, P0_sqr


# --------------------------------------------------------------------------- #
# 1. EK1 equivalence                                                          #
# --------------------------------------------------------------------------- #


def test_ek1_matches_historical_step():
    prior, measure, m0, P0_sqr = _decay_ode()
    (m_pred, P_pred_sqr), (m_z, P_z_sqr), (m_t, P_t_sqr) = _predict_and_reference(
        prior, measure, m0, P0_sqr
    )

    res = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)

    assert isinstance(res, CorrectionResult)
    assert np.allclose(res.m, m_t, atol=1e-12)
    assert np.allclose(_cov(res.P_sqr), _cov(P_t_sqr), atol=1e-12)
    assert np.allclose(res.mz, m_z, atol=1e-12)
    assert np.allclose(_cov(res.Pz_sqr), _cov(P_z_sqr), atol=1e-12)


def test_ek1_default_order_is_one():
    prior, measure, m0, P0_sqr = _decay_ode()
    (m_pred, P_pred_sqr), _, _ = _predict_and_reference(prior, measure, m0, P0_sqr)

    default = TaylorCorrection().correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    explicit = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    assert np.allclose(default.m, explicit.m, atol=1e-12)


def test_ek1_equivalence_with_conservation():
    """EK1 reproduces the stacked (ODE + conservation) update exactly."""

    def vf(x, *, t):
        # Rotation: conserves x0^2 + x1^2.
        return np.array([-x[1], x[0]])

    q, d = 1, 2
    prior = IWP(q=q, d=d, Xi=np.eye(d))
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([1.0, 0.0]), q=q)
    # Linear conservation A @ x = p (one row); equivalence holds regardless of
    # whether the constraint is physically conserved by vf.
    measure = ODEconservation(
        vf, prior.E0, prior.E1, np.array([[1.0, 1.0]]), np.array([1.0])
    )

    (m_pred, P_pred_sqr), (m_z, _), (m_t, P_t_sqr) = _predict_and_reference(
        prior, measure, m0, P0_sqr
    )
    res = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)

    assert res.mz.shape == m_z.shape  # ODE + conservation rows
    assert np.allclose(res.m, m_t, atol=1e-12)
    assert np.allclose(_cov(res.P_sqr), _cov(P_t_sqr), atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. EK0 behavior                                                             #
# --------------------------------------------------------------------------- #


def test_ek0_h_eff_is_selection_matrix():
    prior, measure, m0, P0_sqr = _decay_ode()
    (m_pred, P_pred_sqr), _, _ = _predict_and_reference(prior, measure, m0, P0_sqr)

    res0 = TaylorCorrection(order=0).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)

    # EK0 drops the vector-field Jacobian: H_eff == E_constraint (== E1 here).
    assert np.allclose(res0.H_eff, measure.E_constraint, atol=1e-12)
    assert np.allclose(res0.H_eff, prior.E1, atol=1e-12)


def test_ek0_differs_from_ek1_for_nonlinear_vf():
    prior, measure, m0, P0_sqr = _decay_ode()
    (m_pred, P_pred_sqr), _, _ = _predict_and_reference(prior, measure, m0, P0_sqr)

    res0 = TaylorCorrection(order=0).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    res1 = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    # The vector-field Jacobian is nonzero, so the two schemes must differ.
    assert not np.allclose(res0.m, res1.m, atol=1e-9)


def test_ek0_equals_ek1_for_constant_vf():
    """With a constant vector field the Jacobian is zero, so EK0 == EK1."""

    def vf(x, *, t):
        return np.array([2.0])  # constant -> zero Jacobian

    q, d = 2, 1
    prior = IWP(q=q, d=d, Xi=np.eye(d))
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([0.0]), q=q)
    measure = ODEInformation(vf, prior.E0, prior.E1)

    (m_pred, P_pred_sqr), _, _ = _predict_and_reference(prior, measure, m0, P0_sqr)
    res0 = TaylorCorrection(order=0).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    res1 = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)

    assert np.allclose(res0.m, res1.m, atol=1e-12)
    assert np.allclose(_cov(res0.P_sqr), _cov(res1.P_sqr), atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. Orthogonality: any model composes with any correction                    #
# --------------------------------------------------------------------------- #


def test_orthogonality_hidden_state_model():
    """A different measurement model (joint state-parameter) composes with EK1."""

    def vf(x, u, *, t):
        return -u * x

    prior_x = IWP(q=2, d=1, Xi=np.eye(1))
    prior_lam = IWP(q=0, d=1, Xi=1e-10 * np.eye(1))
    joint = JointPrior(prior_x, prior_lam)
    measure = ODEInformationWithHidden(
        vf=vf, E0=joint.E0_x, E1=joint.E1, E0_hidden=joint.E0_hidden
    )
    # Equivalence (Correction vs historical step) runs identical math, so any
    # valid joint-state Gaussian exercises it -- no physical init needed.
    state_dim = joint.E1.shape[1]
    m0 = 0.5 * np.arange(1, state_dim + 1, dtype=float)
    P0_sqr = 0.3 * np.eye(state_dim)

    (m_pred, P_pred_sqr), _, (m_t, _) = _predict_and_reference(
        joint, measure, m0, P0_sqr
    )
    res = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    assert np.allclose(res.m, m_t, atol=1e-12)
    assert res.mz_ode.shape[0] == measure.ode_dim


# --------------------------------------------------------------------------- #
# 4. Validation, jit, pytree                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_order", [-1, 2, 0.5])
def test_invalid_order_raises(bad_order):
    with pytest.raises(ValueError, match="order must be 0"):
        TaylorCorrection(order=bad_order)


def test_correct_is_jittable():
    prior, measure, m0, P0_sqr = _decay_ode()
    (m_pred, P_pred_sqr), _, _ = _predict_and_reference(prior, measure, m0, P0_sqr)

    correction = TaylorCorrection(order=1)

    @jax.jit
    def run(m, P):
        return correction.correct(measure, m, P, t=T_EVAL)

    res_jit = run(m_pred, P_pred_sqr)
    res_eager = correction.correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    assert np.allclose(res_jit.m, res_eager.m, atol=1e-12)


def test_correction_is_static_pytree():
    # order is a static field, so the module carries no array leaves and is
    # safe to capture as static config inside jit/scan.
    assert isinstance(TaylorCorrection(order=0), Correction)
    assert jax.tree_util.tree_leaves(TaylorCorrection(order=1)) == []


def test_base_correction_is_abstract():
    with pytest.raises(TypeError):
        Correction()
