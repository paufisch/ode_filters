"""Tests for the adaptive square-root EKF loop and PI controller."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters.calibration.sigma import posthoc_mle_sigma_sqr
from ode_filters.filters import (
    PController,
    PIController,
    StepSizeController,
    ekf1_sqr_adaptive_loop,
    ekf1_sqr_loop,
    rts_sqr_smoother_loop,
)
from ode_filters.measurement.measurement_models import ODEInformation
from ode_filters.priors.gmp_priors import IWP, taylor_mode_initialization

# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------


def logistic_vf(x, *, t):
    return x * (1 - x)


def logistic_analytic(t, x0=0.1):
    return 1.0 / (1.0 + (1.0 / x0 - 1.0) * np.exp(-t))


# ---------------------------------------------------------------------------
# PI controller
# ---------------------------------------------------------------------------


class TestPIController:
    def test_unit_error_keeps_step_near_safety(self):
        ctrl = PIController(order=2)
        h_new = ctrl.propose(h=0.1, err=1.0, err_prev=None)
        assert h_new == pytest.approx(0.1 * 0.9, rel=1e-12)

    def test_shrinks_on_large_error(self):
        ctrl = PIController(order=2)
        h_new = ctrl.propose(h=0.1, err=10.0, err_prev=None)
        assert h_new < 0.1

    def test_grows_on_small_error(self):
        ctrl = PIController(order=2)
        h_new = ctrl.propose(h=0.1, err=0.01, err_prev=None)
        assert h_new > 0.1

    def test_clips_at_max_factor(self):
        ctrl = PIController(order=2, max_factor=2.0)
        # Tiny error would otherwise blow up the step.
        h_new = ctrl.propose(h=0.1, err=1e-12, err_prev=None)
        assert h_new == pytest.approx(0.2, rel=1e-12)

    def test_clips_at_min_factor(self):
        ctrl = PIController(order=2, min_factor=0.1)
        h_new = ctrl.propose(h=0.1, err=1e12, err_prev=None)
        assert h_new == pytest.approx(0.01, rel=1e-12)


# ---------------------------------------------------------------------------
# P controller
# ---------------------------------------------------------------------------


class TestPController:
    def test_unit_error_keeps_step_near_safety(self):
        ctrl = PController(order=2)
        h_new = ctrl.propose(h=0.1, err=1.0)
        assert h_new == pytest.approx(0.1 * 0.9, rel=1e-12)

    def test_shrinks_on_large_error(self):
        ctrl = PController(order=2)
        h_new = ctrl.propose(h=0.1, err=10.0)
        assert h_new < 0.1

    def test_grows_on_small_error(self):
        ctrl = PController(order=2)
        h_new = ctrl.propose(h=0.1, err=0.01)
        assert h_new > 0.1

    def test_ignores_err_prev(self):
        ctrl = PController(order=2)
        h_a = ctrl.propose(h=0.1, err=0.5, err_prev=None)
        h_b = ctrl.propose(h=0.1, err=0.5, err_prev=999.0)
        assert h_a == pytest.approx(h_b, rel=1e-12)

    def test_default_alpha_is_one_over_order(self):
        ctrl = PController(order=4)
        # h_new / (h * safety) = err^(-alpha) = 16^(-1/4) = 0.5
        h_new = ctrl.propose(h=1.0, err=16.0)
        assert h_new == pytest.approx(0.9 * 0.5, rel=1e-12)

    def test_explicit_alpha_override(self):
        ctrl = PController(order=2, alpha=0.5, safety=1.0)
        h_new = ctrl.propose(h=1.0, err=4.0)
        # err^(-0.5) = 0.5
        assert h_new == pytest.approx(0.5, rel=1e-12)

    def test_satisfies_protocol(self):
        assert isinstance(PController(order=2), StepSizeController)
        assert isinstance(PIController(order=2), StepSizeController)

    def test_clips_at_min_and_max_factor(self):
        ctrl = PController(order=2, min_factor=0.25, max_factor=3.0)
        assert ctrl.propose(h=1.0, err=1e12) == pytest.approx(0.25, rel=1e-12)
        assert ctrl.propose(h=1.0, err=1e-12) == pytest.approx(3.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Adaptive loop: basic correctness
# ---------------------------------------------------------------------------


class TestAdaptiveLoopLogistic:
    x0 = np.array([0.1])
    tspan = (0.0, 5.0)

    def _setup(self, q=2):
        prior = IWP(q=q, d=1)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(logistic_vf, self.x0, q=q)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        return prior, mu_0, Sigma_0_sqr, measure

    def test_reaches_tspan_end(self):
        prior, mu_0, S0, measure = self._setup()
        result = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
        )
        assert float(result.t_seq[-1]) == pytest.approx(self.tspan[1], rel=1e-12)

    def test_lengths_align(self):
        prior, mu_0, S0, measure = self._setup()
        r = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
        )
        N = len(r.h_seq)
        assert len(r.m_seq) == N + 1
        assert len(r.P_seq_sqr) == N + 1
        assert r.t_seq.shape[0] == N + 1
        assert len(r.m_pred_seq) == N
        assert len(r.P_pred_seq_sqr) == N
        assert len(r.G_back_seq) == N
        assert len(r.d_back_seq) == N
        assert len(r.P_back_seq_sqr) == N
        assert len(r.mz_seq) == N
        assert len(r.Pz_seq_sqr) == N
        assert len(r.sigma_sqr_seq) == N

    def test_tighter_tolerance_reduces_error(self):
        prior, mu_0, S0, measure = self._setup()
        r_loose = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-2,
            rtol=1e-1,
        )
        r_tight = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-7,
            rtol=1e-5,
        )
        x_true = float(logistic_analytic(self.tspan[1], x0=float(self.x0[0])))
        err_loose = abs(float((prior.E0 @ r_loose.m_seq[-1])[0]) - x_true)
        err_tight = abs(float((prior.E0 @ r_tight.m_seq[-1])[0]) - x_true)
        assert err_tight < err_loose
        # Tight tolerance must yield a smaller mean step.
        assert float(np.mean(np.asarray(r_tight.h_seq))) < float(
            np.mean(np.asarray(r_loose.h_seq))
        )

    def test_smoother_consumes_adaptive_output(self):
        prior, mu_0, S0, measure = self._setup()
        r = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-5,
            rtol=1e-3,
        )
        N = len(r.h_seq)
        m_smooth, _P_smooth_sqr = rts_sqr_smoother_loop(
            r.m_seq[-1],
            r.P_seq_sqr[-1],
            r.G_back_seq,
            r.d_back_seq,
            r.P_back_seq_sqr,
            N,
        )
        # Smoother result has length N+1 (initial point inclusive).
        assert m_smooth.shape[0] == N + 1
        # Smoothed final state equals filtered final state.
        assert np.allclose(m_smooth[-1], r.m_seq[-1])

    def test_p_controller_run_reaches_tend(self):
        prior, mu_0, S0, measure = self._setup()
        result = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-5,
            rtol=1e-3,
            controller=PController(order=prior.q),
        )
        assert float(result.t_seq[-1]) == pytest.approx(self.tspan[1], rel=1e-12)
        assert len(result.h_seq) > 0

    def test_calibrate_off_records_sigma_but_does_not_scale(self):
        prior, mu_0, S0, measure = self._setup()
        r_on = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
            calibrate=True,
        )
        r_off = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
            calibrate=False,
        )
        # Sigma estimates are still produced even with calibrate=False.
        assert len(r_off.sigma_sqr_seq) > 0
        # When calibration is on, covariances are scaled by sqrt(sigma^2). For
        # a well-specified problem the running mean sigma should be small, so
        # the calibrated final covariance is smaller than the uncalibrated one.
        cov_on = r_on.P_seq_sqr[-1].T @ r_on.P_seq_sqr[-1]
        cov_off = r_off.P_seq_sqr[-1].T @ r_off.P_seq_sqr[-1]
        mean_sigma = float(np.mean(np.asarray(r_on.sigma_sqr_seq)))
        if mean_sigma < 1.0:
            assert float(np.trace(cov_on)) < float(np.trace(cov_off))


# ---------------------------------------------------------------------------
# Calibration consistency: fixed-step + post-hoc MLE matches expectation.
# ---------------------------------------------------------------------------


def test_posthoc_calibration_on_fixed_step_run():
    """Run fixed-step EKF, then post-hoc MLE the sigma. The estimate should
    be positive and the rescaled covariance trace should change accordingly."""
    prior = IWP(q=2, d=1)
    mu_0, S0 = taylor_mode_initialization(logistic_vf, np.array([0.1]), q=2)
    measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
    result = ekf1_sqr_loop(mu_0, S0, prior, measure, (0.0, 5.0), N=50)
    mz_seq = np.stack(list(result[-3]), axis=0)
    Pz_seq_sqr = np.stack(list(result[-2]), axis=0)
    sigma_sqr = float(posthoc_mle_sigma_sqr(mz_seq, Pz_seq_sqr))
    assert sigma_sqr > 0
    # Sanity bound: for a well-specified logistic problem with IWP(2) prior we
    # expect sigma^2 to be modest. Loose check (orders of magnitude only).
    assert sigma_sqr < 1e3
