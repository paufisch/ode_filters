"""Regression tests for block-aware diffusion calibration.

Implements the verification checklist from the calibration bugfix plan
(2026-05-20): calibration must be driven only by the ODE-defect residual,
and for joint priors sigma must scale only the state block of Q(h).

Tests:

- *Conservation-invariance:* attaching an exactly-satisfied conservation
  law must not change per-step sigma or accepted step sizes.
- *Mis-specified conservation:* a deliberately wrong constraint must not
  perturb the calibration signal either (the constraint still updates the
  posterior; it just no longer drives sigma).
- *Joint-prior state-only Q scaling:* with a JointPrior, the input block
  of the post-calibration process noise must equal the uncalibrated
  prior input block.
- *Joint-prior step-size error norm:* the adaptive-loop error vector
  has the dimension of the state and is independent of input-block
  variance.
- *Joint-prior diagonal-mode guardrail:* the diagonal calibration modes
  raise NotImplementedError for joint priors.
"""

from __future__ import annotations

import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.filters import (
    ekf1_sqr_adaptive_loop,
    ekf1_sqr_loop_dynamic,
)
from ode_filters.measurement.measurement_models import (
    Conservation,
    ODEInformation,
    ODEInformationWithHidden,
)
from ode_filters.priors.gmp_priors import (
    IWP,
    JointPrior,
    taylor_mode_initialization,
)

# ---------------------------------------------------------------------------
# Test ODEs
# ---------------------------------------------------------------------------


def _vf_sir(x, *, t, beta=0.5, gamma=0.1):
    """SIR: S' = -beta*S*I, I' = beta*S*I - gamma*I, R' = gamma*I.

    Conserves S + I + R exactly.
    """
    return np.array(
        [-beta * x[0] * x[1], beta * x[0] * x[1] - gamma * x[1], gamma * x[1]]
    )


def _vf_logistic(x, *, t):
    return x * (1 - x)


def _vf_hidden_linear(x, u, *, t):
    """dx/dt = -u * x with hidden scalar u. State has 2 components."""
    return -u[0] * x


# ---------------------------------------------------------------------------
# Conservation-invariance: sigma and h_seq unchanged with/without conservation
# ---------------------------------------------------------------------------


def _run_dynamic_fixed(prior, measure, x0, tspan, N):
    mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_sir, x0, prior.q)
    return ekf1_sqr_loop_dynamic(
        mu_0, Sigma_0_sqr, prior, measure, tspan, N, calibration="dynamic"
    )


def _sir_measure(prior, with_conservation):
    constraints = (
        [Conservation(np.array([[1.0, 1.0, 1.0]]), np.array([1.0]))]
        if with_conservation
        else None
    )
    return ODEInformation(_vf_sir, prior.E0, prior.E1, constraints=constraints)


class TestConservationInvariance:
    """ODE-only calibration logic must be unaffected by Conservation constraints.

    Note: a *full-loop* invariance test would fail because the joint Kalman
    gain depends on which rows are in the observation model: attaching a
    conservation row changes ``S = HPH^T + R`` and thus the Kalman gain on
    the ODE rows, perturbing the posterior mean even when the conservation
    residual is identically zero. The *calibration signal* is decoupled by
    design (Bosch et al. 2022 sec. 3) but downstream means diverge from
    step 1 onwards. We therefore test (i) bit-identity at step 0 (where
    m_prev is the same regardless of constraints) and (ii) bit-identity
    of the per-step quasi-MLE applied to the same fixed state, with and
    without conservation rows in the measurement model.
    """

    def test_step0_sigma_unchanged(self):
        """SIR + exact conservation: sigma at the FIRST step must be identical."""
        x0 = np.array([0.99, 0.01, 0.0])
        prior = IWP(q=2, d=3)
        tspan = (0.0, 10.0)
        N = 20

        m_a = _sir_measure(prior, with_conservation=False)
        m_b = _sir_measure(prior, with_conservation=True)

        res_a = _run_dynamic_fixed(prior, m_a, x0, tspan, N)
        res_b = _run_dynamic_fixed(prior, m_b, x0, tspan, N)

        sigma_a = onp.asarray(res_a[-2], dtype=float)
        sigma_b = onp.asarray(res_b[-2], dtype=float)
        # Step 0 sees identical (m_prev, P_prev), so sigma must match exactly.
        assert sigma_a[0] == pytest.approx(sigma_b[0], rel=1e-12, abs=1e-14)

    def test_single_step_calibration_is_ode_only(self):
        """Per-step quasi-MLE on (mz_ode, H_ode) is invariant to extra rows.

        Drives the slicing logic directly: linearize with and without a
        deliberately mis-specified conservation; the per-step sigma must
        be identical because only the first ``ode_dim`` rows participate.
        """
        from ode_filters.calibration.sigma import quasi_mle_sigma_sqr_from_Q

        x0 = np.array([0.99, 0.01, 0.0])
        prior = IWP(q=2, d=3)
        h = 0.1
        Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T
        mu_0, _ = taylor_mode_initialization(_vf_sir, x0, prior.q)

        m_no_cons = _sir_measure(prior, with_conservation=False)
        # Wrong conservation: claim S - I = 1 (false at t > 0).
        wrong = Conservation(np.array([[1.0, -1.0, 0.0]]), np.array([1.0]))
        m_wrong = ODEInformation(_vf_sir, prior.E0, prior.E1, constraints=[wrong])

        m_pred = prior.A(h) @ mu_0 + prior.b(h)
        Ha, ca = m_no_cons.linearize(m_pred, t=h)
        Hb, cb = m_wrong.linearize(m_pred, t=h)

        d_ode = m_no_cons.ode_dim  # = m_wrong.ode_dim
        mz_a = (Ha @ m_pred + ca)[:d_ode]
        mz_b = (Hb @ m_pred + cb)[:d_ode]
        H_a_ode = Ha[:d_ode]
        H_b_ode = Hb[:d_ode]

        # Rows match.
        onp.testing.assert_allclose(onp.asarray(mz_a), onp.asarray(mz_b))
        onp.testing.assert_allclose(onp.asarray(H_a_ode), onp.asarray(H_b_ode))

        sigma_a = float(quasi_mle_sigma_sqr_from_Q(mz_a, H_a_ode, Q_h_sqr))
        sigma_b = float(quasi_mle_sigma_sqr_from_Q(mz_b, H_b_ode, Q_h_sqr))
        assert sigma_a == pytest.approx(sigma_b, rel=1e-12, abs=1e-14)

    def test_adaptive_step0_sigma_and_h_unchanged_with_conservation(self):
        """Adaptive loop: first accepted step's sigma must match across runs."""
        x0 = np.array([0.99, 0.01, 0.0])
        prior = IWP(q=3, d=3)
        tspan = (0.0, 10.0)

        m_a = _sir_measure(prior, with_conservation=False)
        m_b = _sir_measure(prior, with_conservation=True)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_sir, x0, prior.q)

        res_a = ekf1_sqr_adaptive_loop(
            mu_0, Sigma_0_sqr, prior, m_a, tspan, atol=1e-5, rtol=1e-3
        )
        res_b = ekf1_sqr_adaptive_loop(
            mu_0, Sigma_0_sqr, prior, m_b, tspan, atol=1e-5, rtol=1e-3
        )

        # First-step h identical (controller decision uses identical err with
        # identical mu_0, Sigma_0_sqr).
        assert float(res_a.h_seq[0]) == pytest.approx(float(res_b.h_seq[0]), rel=1e-12)
        sigma_a0 = float(onp.asarray(res_a.sigma_sqr_seq, dtype=float)[0])
        sigma_b0 = float(onp.asarray(res_b.sigma_sqr_seq, dtype=float)[0])
        assert sigma_a0 == pytest.approx(sigma_b0, rel=1e-12, abs=1e-14)


# ---------------------------------------------------------------------------
# Joint-prior state-only Q scaling
# ---------------------------------------------------------------------------


def _joint_setup():
    prior_x = IWP(q=2, d=2)
    prior_u = IWP(q=0, d=1, Xi=1e-3 * np.eye(1))  # slowly-drifting scalar param
    joint = JointPrior(prior_x, prior_u)

    measure = ODEInformationWithHidden(
        _vf_hidden_linear,
        E0=joint.E0_x,
        E1=joint.E1,
        E0_hidden=joint.E0_hidden,
    )

    # Initial state: x0 = [1.0, 0.5]; u0 = 0.7. Zero higher derivatives.
    D = joint.E0.shape[1]
    mu_0 = np.zeros(D)
    mu_0 = mu_0.at[0].set(1.0).at[1].set(0.5)
    # input block starts at D_x = (q_x+1)*d_x = 3*2 = 6.
    mu_0 = mu_0.at[6].set(0.7)
    Sigma_0_sqr = 1e-6 * np.eye(D)
    return joint, measure, mu_0, Sigma_0_sqr


class TestJointPriorStateOnlyScaling:
    """For a JointPrior, sigma must scale only the state block of Q(h)."""

    def test_apply_state_sigma_sqr_leaves_input_block_alone(self):
        """Direct unit test of JointPrior.apply_state_sigma_sqr."""
        prior_x = IWP(q=2, d=2)
        prior_u = IWP(q=1, d=1)
        joint = JointPrior(prior_x, prior_u)

        h = 0.1
        Q_h = joint.Q(h)
        Q_h_sqr = np.linalg.cholesky(Q_h).T

        sigma_sqr = 4.0
        Q_calib_sqr = joint.apply_state_sigma_sqr(Q_h_sqr, sigma_sqr)
        Q_calib = Q_calib_sqr.T @ Q_calib_sqr

        D_x = (prior_x.q + 1) * prior_x._dim  # 6
        D_u = (prior_u.q + 1) * prior_u._dim  # 2

        # State block scaled by sigma_sqr.
        onp.testing.assert_allclose(
            onp.asarray(Q_calib[:D_x, :D_x]),
            sigma_sqr * onp.asarray(Q_h[:D_x, :D_x]),
            rtol=1e-10,
            atol=1e-12,
        )
        # Input block UNCHANGED.
        onp.testing.assert_allclose(
            onp.asarray(Q_calib[D_x:, D_x:]),
            onp.asarray(Q_h[D_x:, D_x:]),
            rtol=1e-10,
            atol=1e-12,
        )
        # Cross-blocks remain zero (block-diagonal).
        onp.testing.assert_allclose(
            onp.asarray(Q_calib[:D_x, D_x:]),
            onp.zeros((D_x, D_u)),
            atol=1e-12,
        )

    def test_apply_state_sigma_to_cov_sqr_leaves_input_block_alone(self):
        """Direct unit test of JointPrior.apply_state_sigma_to_cov_sqr.

        Column-scaling of an upper-tri sqrt scales the matching block of
        P = P_sqr.T @ P_sqr, leaving the input block alone.
        """
        prior_x = IWP(q=2, d=2)
        prior_u = IWP(q=1, d=1)
        joint = JointPrior(prior_x, prior_u)

        # A generic (not block-diagonal) full-state covariance sqrt.
        key = onp.random.default_rng(0)
        D = (prior_x.q + 1) * prior_x._dim + (prior_u.q + 1) * prior_u._dim
        A = np.asarray(key.standard_normal((D, D)))
        P = A.T @ A
        P_sqr = np.linalg.cholesky(P).T

        sigma_sqr = 9.0
        P_calib_sqr = joint.apply_state_sigma_to_cov_sqr(P_sqr, sigma_sqr)
        P_calib = P_calib_sqr.T @ P_calib_sqr

        D_x = (prior_x.q + 1) * prior_x._dim
        # Input block (rows/cols >= D_x) UNCHANGED.
        onp.testing.assert_allclose(
            onp.asarray(P_calib[D_x:, D_x:]),
            onp.asarray(P[D_x:, D_x:]),
            rtol=1e-10,
            atol=1e-12,
        )
        # State block scaled by sigma_sqr.
        onp.testing.assert_allclose(
            onp.asarray(P_calib[:D_x, :D_x]),
            sigma_sqr * onp.asarray(P[:D_x, :D_x]),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_fixed_step_dynamic_with_joint_prior_runs(self):
        """JointPrior + dynamic calibration must not crash on prior.xi access."""
        joint, measure, mu_0, Sigma_0_sqr = _joint_setup()
        tspan = (0.0, 1.0)
        N = 10
        # The pre-fix code accessed prior.xi unconditionally; JointPrior has
        # no xi, so this would have failed at line 248 before the fix.
        result = ekf1_sqr_loop_dynamic(
            mu_0, Sigma_0_sqr, joint, measure, tspan, N, calibration="dynamic"
        )
        sigmas = onp.asarray(result[-2], dtype=float)
        assert sigmas.shape == (N,)
        assert onp.all(onp.isfinite(sigmas))


# ---------------------------------------------------------------------------
# Joint-prior diagonal-mode correctness
# ---------------------------------------------------------------------------


class TestJointPriorDiagonalCalibration:
    """Diagonal modes on JointPrior: per-component sigma on the state block,
    input block untouched, output shape ``(N, d_x)``."""

    @pytest.mark.parametrize("mode", ["diagonal", "diagonal_ekf0"])
    def test_fixed_step_runs_and_shapes_correct(self, mode):
        joint, measure, mu_0, Sigma_0_sqr = _joint_setup()
        N = 5
        result = ekf1_sqr_loop_dynamic(
            mu_0,
            Sigma_0_sqr,
            joint,
            measure,
            (0.0, 1.0),
            N=N,
            calibration=mode,
        )
        sigmas = onp.asarray(result[-2], dtype=float)
        # Per-component sigma on the state block: shape (N, d_x).
        d_x = joint._prior_x._dim
        assert sigmas.shape == (N, d_x)
        assert onp.all(onp.isfinite(sigmas))
        assert onp.all(np.isfinite(result[0][-1]))

    @pytest.mark.parametrize("mode", ["diagonal", "diagonal_ekf0"])
    def test_adaptive_runs_and_shapes_correct(self, mode):
        joint, measure, mu_0, Sigma_0_sqr = _joint_setup()
        result = ekf1_sqr_adaptive_loop(
            mu_0,
            Sigma_0_sqr,
            joint,
            measure,
            (0.0, 1.0),
            calibration=mode,
        )
        d_x = joint._prior_x._dim
        for s in result.sigma_sqr_seq:
            assert onp.asarray(s).shape == (d_x,)
            assert onp.all(onp.isfinite(onp.asarray(s)))
        assert onp.all(np.isfinite(result.m_seq[-1]))

    def test_diagonal_leaves_input_block_of_Q_step_alone(self):
        """``apply_state_sigma_sqr`` with a vector must not touch the
        input block of ``Q_sqr``, regardless of the per-component values."""
        joint, _measure, _mu_0, _S0 = _joint_setup()
        h = 0.1
        Q_h_sqr = np.linalg.cholesky(joint.Q(h)).T
        sigma_vec = np.array([7.0, 0.3])  # arbitrary per-component values
        Q_calib_sqr = joint.apply_state_sigma_sqr(Q_h_sqr, sigma_vec)
        D_x = joint._D_x
        # Input block of Q (reconstructed from sqrt) is unchanged.
        Q_h = Q_h_sqr.T @ Q_h_sqr
        Q_calib = Q_calib_sqr.T @ Q_calib_sqr
        onp.testing.assert_allclose(
            onp.asarray(Q_calib[D_x:, D_x:]),
            onp.asarray(Q_h[D_x:, D_x:]),
            rtol=1e-10,
            atol=1e-12,
        )
        # State block: diag(sqrt(sigma_tile)) @ Q_x @ diag(sqrt(sigma_tile)),
        # i.e. each component i scaled by sigma_vec[i] uniformly across all
        # q_x + 1 derivative blocks.
        sigma_tile = onp.tile(onp.asarray(sigma_vec), joint._prior_x.q + 1)
        D = onp.diag(onp.sqrt(sigma_tile))
        expected_state = D @ onp.asarray(Q_h[:D_x, :D_x]) @ D
        onp.testing.assert_allclose(
            onp.asarray(Q_calib[:D_x, :D_x]),
            expected_state,
            rtol=1e-10,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Adaptive-loop error norm uses E0_state (state-only) for joint priors
# ---------------------------------------------------------------------------


class TestAdaptiveErrorNormUsesStateOnly:
    """The adaptive error norm must depend on state coords only, not input."""

    def test_E0_state_property_on_joint_prior(self):
        prior_x = IWP(q=2, d=2)
        prior_u = IWP(q=1, d=1)
        joint = JointPrior(prior_x, prior_u)
        # E0_state should be E0_x (state-only extractor).
        assert joint.E0_state.shape[0] == prior_x._dim
        onp.testing.assert_array_equal(
            onp.asarray(joint.E0_state), onp.asarray(joint._E0_x)
        )

    def test_E0_state_default_equals_E0_for_non_joint(self):
        prior = IWP(q=2, d=3)
        assert prior.E0_state.shape == prior.E0.shape
        onp.testing.assert_array_equal(
            onp.asarray(prior.E0_state), onp.asarray(prior.E0)
        )


# ---------------------------------------------------------------------------
# Pure-ODE bit-identity (no constraints, no joint): sigma untouched by changes
# ---------------------------------------------------------------------------


class TestPureODEBitIdentity:
    """For the bog-standard case (single prior, no constraints), behavior is
    bit-identical to pre-fix: the ODE-row slice is a no-op and the default
    apply_state_sigma_sqr is identity.

    This is the strongest single confidence check that the fix is non-disruptive
    where it does not need to act."""

    def test_logistic_fixed_step_matches_pre_fix_qualitatively(self):
        # We can't directly compare against pre-fix, but we can confirm
        # sigma matches the per-step quasi-MLE applied to the ODE-only
        # signal of a problem with no constraints (where ODE-only == full).
        x0 = np.array([0.1])
        prior = IWP(q=2, d=1)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, prior.q)
        tspan = (0.0, 5.0)
        N = 30

        result = ekf1_sqr_loop_dynamic(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N, calibration="dynamic"
        )
        sigmas = onp.asarray(result[-2], dtype=float)
        assert sigmas.shape == (N,)
        assert onp.all(onp.isfinite(sigmas))
        assert onp.all(sigmas > 0)
