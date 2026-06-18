"""Tests for the pluggable Correction abstraction (W1) and step delegation (W2).

W1 proves:

1. ``TaylorCorrection(order=1)`` reproduces the historical EK1 update -- computed
   here from the raw square-root primitives, so the anchor is independent of
   :func:`ekf1_sqr_filter_step` (which delegates to a Correction after W2).
2. ``TaylorCorrection(order=0)`` is a genuine EK0 model: the ODE-defect rows of
   its effective Jacobian equal the selection matrix ``E_constraint``, and it
   agrees with EK1 exactly when the vector field is constant.

W2 proves ``ekf1_sqr_filter_step`` delegates to a Correction without changing
its default (EK1) numerics, and that a non-default correction (EK0) is reachable
through the step.

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
    ekf1_sqr_loop_dynamic_scan,
)
from ode_filters.inference import sqr_inversion, sqr_marginalization
from ode_filters.measurement import (
    Measurement,
    ODEconservation,
    ODEInformation,
    ODEInformationWithHidden,
    prepare_observations,
)
from ode_filters.priors import IWP, JointPrior, taylor_mode_initialization

H_STEP = 0.1
T_EVAL = 0.3


def _cov(sqr: np.ndarray) -> np.ndarray:
    """Reconstruct a covariance from its (gauge-dependent) square-root factor."""
    return sqr.T @ sqr


def _step_inputs(prior, h=H_STEP):
    """The (A, b, Q_sqr) a filter step consumes for the prediction."""
    return prior.A(h), prior.b(h), np.linalg.cholesky(prior.Q(h)).T


def _predict(prior, m0, P0_sqr, h=H_STEP):
    """Square-root prediction (same operation as the step's prediction half)."""
    A, b, Q_sqr = _step_inputs(prior, h)
    return sqr_marginalization(A, b, Q_sqr, m0, P0_sqr)


def _reference_ek1_update(measure, m_pred, P_pred_sqr, t=T_EVAL):
    """Historical inline EK1 update via the raw square-root primitives.

    This is the canonical formula the Correction must reproduce; computing it
    here (not via the step) keeps the equivalence anchor independent of W2.
    """
    H, c = measure.linearize(m_pred, t=t)
    R_sqr = measure.get_noise(t=t)
    mz, Pz_sqr = sqr_marginalization(H, c, R_sqr, m_pred, P_pred_sqr)
    _, m_t, P_t_sqr = sqr_inversion(H, m_pred, P_pred_sqr, mz, Pz_sqr, R_sqr)
    return (mz, Pz_sqr), (m_t, P_t_sqr)


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
# 1. EK1 equivalence (W1)                                                     #
# --------------------------------------------------------------------------- #


def test_ek1_matches_raw_inline_update():
    prior, measure, m0, P0_sqr = _decay_ode()
    m_pred, P_pred_sqr = _predict(prior, m0, P0_sqr)
    (m_z, P_z_sqr), (m_t, P_t_sqr) = _reference_ek1_update(measure, m_pred, P_pred_sqr)

    res = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)

    assert isinstance(res, CorrectionResult)
    assert np.allclose(res.m, m_t, atol=1e-12)
    assert np.allclose(_cov(res.P_sqr), _cov(P_t_sqr), atol=1e-12)
    assert np.allclose(res.mz, m_z, atol=1e-12)
    assert np.allclose(_cov(res.Pz_sqr), _cov(P_z_sqr), atol=1e-12)


def test_ek1_default_order_is_one():
    prior, measure, m0, P0_sqr = _decay_ode()
    m_pred, P_pred_sqr = _predict(prior, m0, P0_sqr)

    default = TaylorCorrection().correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    explicit = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    assert np.allclose(default.m, explicit.m, atol=1e-12)


def test_ek1_equivalence_with_conservation():
    """EK1 reproduces the stacked (ODE + conservation) update exactly."""

    def vf(x, *, t):
        return np.array([-x[1], x[0]])

    q, d = 1, 2
    prior = IWP(q=q, d=d, Xi=np.eye(d))
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([1.0, 0.0]), q=q)
    # Linear conservation A @ x = p (one row); equivalence holds regardless of
    # whether the constraint is physically conserved by vf.
    measure = ODEconservation(
        vf, prior.E0, prior.E1, np.array([[1.0, 1.0]]), np.array([1.0])
    )

    m_pred, P_pred_sqr = _predict(prior, m0, P0_sqr)
    (m_z, _), (m_t, P_t_sqr) = _reference_ek1_update(measure, m_pred, P_pred_sqr)
    res = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)

    assert res.mz.shape == m_z.shape  # ODE + conservation rows
    assert np.allclose(res.m, m_t, atol=1e-12)
    assert np.allclose(_cov(res.P_sqr), _cov(P_t_sqr), atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. EK0 behavior (W1)                                                        #
# --------------------------------------------------------------------------- #


def test_ek0_h_eff_is_selection_matrix():
    prior, measure, m0, P0_sqr = _decay_ode()
    m_pred, P_pred_sqr = _predict(prior, m0, P0_sqr)

    res0 = TaylorCorrection(order=0).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)

    # EK0 drops the vector-field Jacobian: H_eff == E_constraint (== E1 here).
    assert np.allclose(res0.H_eff, measure.E_constraint, atol=1e-12)
    assert np.allclose(res0.H_eff, prior.E1, atol=1e-12)


def test_ek0_differs_from_ek1_for_nonlinear_vf():
    prior, measure, m0, P0_sqr = _decay_ode()
    m_pred, P_pred_sqr = _predict(prior, m0, P0_sqr)

    res0 = TaylorCorrection(order=0).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    res1 = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    assert not np.allclose(res0.m, res1.m, atol=1e-9)


def test_ek0_equals_ek1_for_constant_vf():
    """With a constant vector field the Jacobian is zero, so EK0 == EK1."""

    def vf(x, *, t):
        return np.array([2.0])

    q, d = 2, 1
    prior = IWP(q=q, d=d, Xi=np.eye(d))
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([0.0]), q=q)
    measure = ODEInformation(vf, prior.E0, prior.E1)

    m_pred, P_pred_sqr = _predict(prior, m0, P0_sqr)
    res0 = TaylorCorrection(order=0).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    res1 = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)

    assert np.allclose(res0.m, res1.m, atol=1e-12)
    assert np.allclose(_cov(res0.P_sqr), _cov(res1.P_sqr), atol=1e-12)


# --------------------------------------------------------------------------- #
# 3. Orthogonality: any model composes with any correction (W1)               #
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
    # Equivalence (Correction vs raw EK1) runs identical math, so any valid
    # joint-state Gaussian exercises it -- no physical init needed.
    state_dim = joint.E1.shape[1]
    m0 = 0.5 * np.arange(1, state_dim + 1, dtype=float)
    P0_sqr = 0.3 * np.eye(state_dim)

    m_pred, P_pred_sqr = _predict(joint, m0, P0_sqr)
    (_, _), (m_t, _) = _reference_ek1_update(measure, m_pred, P_pred_sqr)
    res = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    assert np.allclose(res.m, m_t, atol=1e-12)
    assert res.mz_ode.shape[0] == measure.ode_dim


# --------------------------------------------------------------------------- #
# 4. Validation, jit, pytree (W1)                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_order", [-1, 2, 0.5])
def test_invalid_order_raises(bad_order):
    with pytest.raises(ValueError, match="order must be 0"):
        TaylorCorrection(order=bad_order)


def test_correct_is_jittable():
    prior, measure, m0, P0_sqr = _decay_ode()
    m_pred, P_pred_sqr = _predict(prior, m0, P0_sqr)
    correction = TaylorCorrection(order=1)

    @jax.jit
    def run(m, P):
        return correction.correct(measure, m, P, t=T_EVAL)

    res_jit = run(m_pred, P_pred_sqr)
    res_eager = correction.correct(measure, m_pred, P_pred_sqr, t=T_EVAL)
    assert np.allclose(res_jit.m, res_eager.m, atol=1e-12)


def test_correction_is_static_pytree():
    assert isinstance(TaylorCorrection(order=0), Correction)
    assert jax.tree_util.tree_leaves(TaylorCorrection(order=1)) == []


def test_base_correction_is_abstract():
    with pytest.raises(TypeError):
        Correction()


# --------------------------------------------------------------------------- #
# 5. Step delegation (W2)                                                     #
# --------------------------------------------------------------------------- #


def test_step_default_matches_raw_ek1():
    """ekf1_sqr_filter_step (default correction) reproduces the raw EK1 update."""
    prior, measure, m0, P0_sqr = _decay_ode()
    A, b, Q_sqr = _step_inputs(prior)

    (m_pred, P_pred_sqr), _, (m_z, P_z_sqr), (m_t, P_t_sqr) = ekf1_sqr_filter_step(
        A, b, Q_sqr, m0, P0_sqr, measure, T_EVAL
    )
    (rmz, rPz), (rm_t, rP_t) = _reference_ek1_update(measure, m_pred, P_pred_sqr)

    assert np.allclose(m_t, rm_t, atol=1e-12)
    assert np.allclose(_cov(P_t_sqr), _cov(rP_t), atol=1e-12)
    assert np.allclose(m_z, rmz, atol=1e-12)
    assert np.allclose(_cov(P_z_sqr), _cov(rPz), atol=1e-12)


def test_step_explicit_ek1_equals_default():
    prior, measure, m0, P0_sqr = _decay_ode()
    A, b, Q_sqr = _step_inputs(prior)

    _, _, (mz_d, _), (mt_d, Pt_d) = ekf1_sqr_filter_step(
        A, b, Q_sqr, m0, P0_sqr, measure, T_EVAL
    )
    _, _, (mz_e, _), (mt_e, Pt_e) = ekf1_sqr_filter_step(
        A, b, Q_sqr, m0, P0_sqr, measure, T_EVAL, correction=TaylorCorrection(order=1)
    )
    assert np.allclose(mt_d, mt_e, atol=1e-12)
    assert np.allclose(_cov(Pt_d), _cov(Pt_e), atol=1e-12)
    assert np.allclose(mz_d, mz_e, atol=1e-12)


def test_step_ek0_changes_update():
    """Passing an EK0 correction through the step changes the (nonlinear) update."""
    prior, measure, m0, P0_sqr = _decay_ode()
    A, b, Q_sqr = _step_inputs(prior)

    (mp1, _), _, _, (mt1, _) = ekf1_sqr_filter_step(
        A, b, Q_sqr, m0, P0_sqr, measure, T_EVAL, correction=TaylorCorrection(order=1)
    )
    (mp0, _), _, _, (mt0, _) = ekf1_sqr_filter_step(
        A, b, Q_sqr, m0, P0_sqr, measure, T_EVAL, correction=TaylorCorrection(order=0)
    )
    assert np.allclose(mp0, mp1, atol=1e-12)  # prediction is correction-independent
    assert not np.allclose(mt0, mt1, atol=1e-9)  # update differs


def test_step_is_jittable_with_correction():
    prior, measure, m0, P0_sqr = _decay_ode()
    A, b, Q_sqr = _step_inputs(prior)
    correction = TaylorCorrection(order=0)

    @jax.jit
    def run(m, P):
        return ekf1_sqr_filter_step(
            A, b, Q_sqr, m, P, measure, T_EVAL, correction=correction
        )

    out = run(m0, P0_sqr)
    assert np.all(np.isfinite(out[3][0]))


# --------------------------------------------------------------------------- #
# 6. Loop wiring: ekf1_sqr_loop_dynamic_scan (W3)                             #
# --------------------------------------------------------------------------- #


def test_loop_explicit_ek1_equals_default():
    """Passing correction=EK1 reproduces the default loop bit-for-bit."""
    prior, measure, m0, P0_sqr = _decay_ode()
    tspan, n = (0.0, 2.0), 50

    default = ekf1_sqr_loop_dynamic_scan(m0, P0_sqr, prior, measure, tspan, n)
    explicit = ekf1_sqr_loop_dynamic_scan(
        m0, P0_sqr, prior, measure, tspan, n, correction=TaylorCorrection(order=1)
    )
    assert np.allclose(default[0], explicit[0], atol=1e-12)  # m_seq
    assert np.allclose(default[-1], explicit[-1], atol=1e-12)  # log-likelihood


def test_loop_ek0_differs_from_ek1_nonlinear():
    prior, measure, m0, P0_sqr = _decay_ode()  # nonlinear vf
    tspan, n = (0.0, 1.0), 50

    m_seq0 = ekf1_sqr_loop_dynamic_scan(
        m0, P0_sqr, prior, measure, tspan, n, correction=TaylorCorrection(order=0)
    )[0]
    m_seq1 = ekf1_sqr_loop_dynamic_scan(
        m0, P0_sqr, prior, measure, tspan, n, correction=TaylorCorrection(order=1)
    )[0]
    assert np.all(np.isfinite(m_seq0))
    assert not np.allclose(m_seq0, m_seq1, atol=1e-8)


def test_loop_ek0_solves_linear_decay():
    """EK0 through the loop produces a correct end-to-end ODE solution."""

    def vf(x, *, t):
        return -x

    prior = IWP(q=2, d=1, Xi=np.eye(1))
    m0, P0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)

    m_seq = ekf1_sqr_loop_dynamic_scan(
        m0,
        P0_sqr,
        prior,
        measure,
        (0.0, 2.0),
        200,
        correction=TaylorCorrection(order=0),
    )[0]
    x_final = (prior.E0 @ m_seq[-1])[0]
    assert np.isclose(x_final, np.exp(-2.0), atol=1e-2)


def test_loop_correction_with_obs_model_raises():
    """correction= is not yet supported alongside obs_model (clear error)."""

    def vf(x, u, *, t):
        return -u * x

    prior_x = IWP(q=2, d=1, Xi=np.eye(1))
    prior_lam = IWP(q=0, d=1, Xi=1e-10 * np.eye(1))
    joint = JointPrior(prior_x, prior_lam)
    measure = ODEInformationWithHidden(
        vf=vf, E0=joint.E0_x, E1=joint.E1, E0_hidden=joint.E0_hidden
    )
    n = 20
    ts = np.linspace(0.0, 1.0, n + 1)
    measurement = Measurement(np.eye(1), np.ones((n, 1)), ts[1:], noise=1e-2)
    obs_model = prepare_observations([measurement], joint.E0_x, ts)

    state_dim = joint.E1.shape[1]
    m0 = np.zeros(state_dim)
    P0_sqr = 0.3 * np.eye(state_dim)

    with pytest.raises(NotImplementedError, match="obs_model"):
        ekf1_sqr_loop_dynamic_scan(
            m0,
            P0_sqr,
            joint,
            measure,
            (0.0, 1.0),
            n,
            obs_model=obs_model,
            correction=TaylorCorrection(order=1),
        )
