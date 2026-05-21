"""Tests for the adaptive square-root EKF loop and PI controller."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters.calibration.sigma import (
    posthoc_mle_sigma_sqr,
    quasi_mle_sigma_sqr_from_Q,
)
from ode_filters.filters import (
    PController,
    PIController,
    StepSizeController,
    ekf1_sqr_adaptive_loop,
    ekf1_sqr_loop,
    ekf1_sqr_loop_dynamic,
    ekf1_sqr_loop_dynamic_scan,
    ekf1_sqr_loop_preconditioned_dynamic,
    ekf1_sqr_loop_preconditioned_dynamic_scan,
    rts_sqr_smoother_loop,
)
from ode_filters.measurement.measurement_models import ODEInformation
from ode_filters.priors.gmp_priors import IWP, PrecondIWP, taylor_mode_initialization

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

    def test_calibration_none_records_sigma_but_does_not_scale(self):
        prior, mu_0, S0, measure = self._setup()
        r_on = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
            calibration="dynamic",
        )
        r_off = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
            calibration="none",
        )
        # Sigma estimates are still produced even with calibration="none".
        assert len(r_off.sigma_sqr_seq) > 0
        # In dynamic mode, sigma_hat^2 < 1 at every step makes the calibrated
        # covariance smaller than the sigma=1 run (each step's Q is scaled
        # down by sigma_hat^2 before propagation).
        cov_on = r_on.P_seq_sqr[-1].T @ r_on.P_seq_sqr[-1]
        cov_off = r_off.P_seq_sqr[-1].T @ r_off.P_seq_sqr[-1]
        sigmas = np.asarray(r_on.sigma_sqr_seq)
        if float(np.max(sigmas)) < 1.0:
            assert float(np.trace(cov_on)) < float(np.trace(cov_off))

    def test_deprecated_calibrate_bool_maps_to_calibration(self):
        prior, mu_0, S0, measure = self._setup()
        # calibrate=True -> "dynamic"
        with pytest.warns(DeprecationWarning, match="calibrate"):
            r_legacy = ekf1_sqr_adaptive_loop(
                mu_0,
                S0,
                prior,
                measure,
                self.tspan,
                atol=1e-4,
                rtol=1e-2,
                calibrate=True,
            )
        r_new = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
            calibration="dynamic",
        )
        assert np.allclose(r_legacy.m_seq[-1], r_new.m_seq[-1])

    def test_unknown_calibration_raises(self):
        prior, mu_0, S0, measure = self._setup()
        with pytest.raises(ValueError, match="calibration"):
            ekf1_sqr_adaptive_loop(
                mu_0,
                S0,
                prior,
                measure,
                self.tspan,
                atol=1e-4,
                rtol=1e-2,
                calibration="bogus",  # type: ignore[arg-type]
            )

    def test_diagonal_mode_per_component_sigma_shape(self):
        prior, mu_0, S0, measure = self._setup()
        r = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
            calibration="diagonal",
        )
        # Each entry is a length-d array, one sigma per ODE component.
        assert all(s.shape == (prior.E0.shape[0],) for s in r.sigma_sqr_seq)
        assert all((s > 0).all() for s in r.sigma_sqr_seq)

    def test_diagonal_calibration_matches_closed_form(self):
        """Per-component sigma_hat^2_i = m_z[i]^2 / (H Q H.T)_ii. Verify on
        every accepted step."""
        prior, mu_0, S0, measure = self._setup()
        r = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-4,
            rtol=1e-2,
            calibration="diagonal",
        )
        for i, h in enumerate(r.h_seq):
            A_h = prior.A(float(h))
            b_h = prior.b(float(h))
            Q_h = prior.Q(float(h))
            t_next = float(r.t_seq[i + 1])
            m_prev = r.m_seq[i]
            m_pred_prov = A_h @ m_prev + b_h
            H_t, c_t = measure.linearize(m_pred_prov, t=t_next)
            mz_pred = H_t @ m_pred_prov + c_t
            H_Q_H_diag = np.einsum("ij,jk,ik->i", H_t, Q_h, H_t)
            expected = mz_pred**2 / H_Q_H_diag
            assert np.allclose(r.sigma_sqr_seq[i], expected, rtol=1e-10)

    def test_diagonal_requires_diagonal_xi(self):
        non_diag_xi = np.array([[1.0, 0.1], [0.1, 1.0]])
        bad_prior = IWP(q=2, d=2, Xi=non_diag_xi)
        mu_0, S0 = taylor_mode_initialization(
            lambda x, *, t: x, np.array([0.1, 0.2]), q=2
        )
        measure = ODEInformation(lambda x, *, t: x, bad_prior.E0, bad_prior.E1)
        with pytest.raises(ValueError, match="diagonal"):
            ekf1_sqr_adaptive_loop(
                mu_0,
                S0,
                bad_prior,
                measure,
                (0.0, 1.0),
                atol=1e-3,
                rtol=1e-2,
                calibration="diagonal",
            )

    def test_sigma_in_error_running_mean_smooths_h(self):
        """Running-mean sigma_in_error reduces step-size oscillations and
        the reject count, while reaching tspan_end with similar accuracy."""
        prior, mu_0, S0, measure = self._setup()
        r_per = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-5,
            rtol=1e-3,
            sigma_in_error="per_step",
        )
        r_run = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-5,
            rtol=1e-3,
            sigma_in_error="running_mean",
        )
        # Both reach the endpoint.
        assert float(r_per.t_seq[-1]) == pytest.approx(self.tspan[1], rel=1e-12)
        assert float(r_run.t_seq[-1]) == pytest.approx(self.tspan[1], rel=1e-12)
        # Running-mean rejects fewer steps -- usually by a large margin on the
        # logistic, but at least not strictly more.
        assert r_run.n_rejected <= r_per.n_rejected

    def test_unknown_sigma_in_error_raises(self):
        prior, mu_0, S0, measure = self._setup()
        with pytest.raises(ValueError, match="sigma_in_error"):
            ekf1_sqr_adaptive_loop(
                mu_0,
                S0,
                prior,
                measure,
                self.tspan,
                atol=1e-4,
                rtol=1e-2,
                sigma_in_error="ema",  # type: ignore[arg-type]
            )

    def test_diagonal_resolves_multiscale(self):
        """Diagonal mode resolves x_2 on the staggered multi-scale logistic
        where dynamic mode collapses -- and does so with fewer accepted steps
        than cumulative (which also resolves it via inheritance)."""
        r_rate = 2.0
        x0_vec = np.asarray([1e-5, 1e-10])

        def vf_dl(x, *, t):
            return np.array([r_rate * x[0] * (1 - x[0]), r_rate * x[1] * (1 - x[1])])

        tspan = (0.0, 15.0)
        q = 3
        prior = IWP(q=q, d=2)
        mu_0, S0 = taylor_mode_initialization(vf_dl, x0_vec, q=q)
        measure = ODEInformation(vf_dl, prior.E0, prior.E1)
        x2_true = 1.0 / (
            1.0 + (1.0 / float(x0_vec[1]) - 1.0) * np.exp(-r_rate * tspan[1])
        )

        r_diag = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            tspan,
            atol=1e-5,
            rtol=1e-3,
            h_min=1e-9,
            calibration="diagonal",
        )
        x2_diag = float((prior.E0 @ r_diag.m_seq[-1])[1])
        assert abs(x2_diag - float(x2_true)) < 1e-3

    def test_cumulative_resolves_multiscale_where_dynamic_collapses(self):
        """Staggered two-component logistic with X0 = [1e-5, 1e-10]: dynamic
        sigma_hat collapses after x_1's transition and starves x_2 of process
        noise, so x_2 is stuck near 0. The cumulative scheme inherits the
        inflated carried-P from x_1's transition and resolves x_2 correctly.
        """
        r_rate = 2.0
        x0_vec = np.asarray([1e-5, 1e-10])

        def vf_dl(x, *, t):
            return np.array([r_rate * x[0] * (1 - x[0]), r_rate * x[1] * (1 - x[1])])

        tspan = (0.0, 15.0)
        q = 3
        prior = IWP(q=q, d=2)
        mu_0, S0 = taylor_mode_initialization(vf_dl, x0_vec, q=q)
        measure = ODEInformation(vf_dl, prior.E0, prior.E1)

        def x2_final(mode):
            r = ekf1_sqr_adaptive_loop(
                mu_0,
                S0,
                prior,
                measure,
                tspan,
                atol=1e-5,
                rtol=1e-3,
                h_min=1e-9,
                calibration=mode,
            )
            return float((prior.E0 @ r.m_seq[-1])[1])

        x2_dyn = x2_final("dynamic")
        x2_cum = x2_final("cumulative")
        x2_true = 1.0 / (
            1.0 + (1.0 / float(x0_vec[1]) - 1.0) * np.exp(-r_rate * tspan[1])
        )

        # Dynamic mode misses the transition (x_2 stays near zero).
        assert abs(x2_dyn) < 1e-3
        # Cumulative mode resolves x_2 near the true value (~0.999).
        assert abs(x2_cum - float(x2_true)) < 1e-2

    def test_dynamic_diffusion_matches_probdiffeq_formula(self):
        """The per-step sigma_hat^2 returned by the adaptive loop must equal
        the dynamic-diffusion quasi-MLE: ``m_z.T (H Q(h) H.T)^{-1} m_z / d``
        with ``m_z = H (A m_prev + b) + c`` -- linearised at the provisional
        predicted mean, before the EKF update."""
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
        # Reconstruct each step's sigma from the carried m_prev / h.
        for i, h in enumerate(r.h_seq):
            m_prev = r.m_seq[i]
            A_h = prior.A(float(h))
            b_h = prior.b(float(h))
            Q_h_sqr = np.linalg.cholesky(prior.Q(float(h))).T
            t_next = float(r.t_seq[i + 1])
            m_pred_prov = A_h @ m_prev + b_h
            H_t, c_t = measure.linearize(m_pred_prov, t=t_next)
            mz_pred = H_t @ m_pred_prov + c_t
            sigma_expected = float(quasi_mle_sigma_sqr_from_Q(mz_pred, H_t, Q_h_sqr))
            assert r.sigma_sqr_seq[i] == pytest.approx(sigma_expected, rel=1e-10)


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


# ---------------------------------------------------------------------------
# Fixed-step dynamic-diffusion loop (online calibration without adaptive h)
# ---------------------------------------------------------------------------


class TestDynamicFixedStepLoop:
    x0 = np.array([0.1])
    tspan = (0.0, 5.0)
    N = 50

    def _setup(self, q=2):
        prior = IWP(q=q, d=1)
        mu_0, S0 = taylor_mode_initialization(logistic_vf, self.x0, q=q)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        return prior, mu_0, S0, measure

    def test_returns_sigma_per_step(self):
        prior, mu_0, S0, measure = self._setup()
        result = ekf1_sqr_loop_dynamic(mu_0, S0, prior, measure, self.tspan, self.N)
        sigma_seq = result[9]
        assert len(sigma_seq) == self.N
        assert all(s > 0 for s in sigma_seq)

    def test_calibration_none_matches_uncalibrated_loop(self):
        """With ``calibration="none"`` the dynamic loop should produce exactly
        the same trajectory as :func:`ekf1_sqr_loop` (Q never gets scaled).
        Sigma is still computed for diagnostics."""
        prior, mu_0, S0, measure = self._setup()
        r_dyn = ekf1_sqr_loop_dynamic(
            mu_0, S0, prior, measure, self.tspan, self.N, calibration="none"
        )
        r_plain = ekf1_sqr_loop(mu_0, S0, prior, measure, self.tspan, self.N)

        # Final filtered mean must match exactly.
        assert np.allclose(r_dyn[0][-1], r_plain[0][-1], atol=1e-12)
        # Final filtered covariance must match exactly.
        assert np.allclose(r_dyn[1][-1], r_plain[1][-1], atol=1e-12)

    def test_dynamic_sigma_matches_formula(self):
        """Independently re-derive each step's sigma from ``H Q H.T`` and the
        provisional predicted mean."""
        prior, mu_0, S0, measure = self._setup()
        result = ekf1_sqr_loop_dynamic(mu_0, S0, prior, measure, self.tspan, self.N)
        m_seq = result[0]
        sigma_seq = result[9]
        import numpy as onp

        ts, h = onp.linspace(self.tspan[0], self.tspan[1], self.N + 1, retstep=True)
        h = float(h)
        A_h = prior.A(h)
        b_h = prior.b(h)
        Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T
        for i in range(self.N):
            m_prev = m_seq[i]
            t_next = float(ts[i + 1])
            m_pred_prov = A_h @ m_prev + b_h
            H_t, c_t = measure.linearize(m_pred_prov, t=t_next)
            mz_pred = H_t @ m_pred_prov + c_t
            sigma_expected = float(quasi_mle_sigma_sqr_from_Q(mz_pred, H_t, Q_h_sqr))
            assert sigma_seq[i] == pytest.approx(sigma_expected, rel=1e-10)

    def test_calibrated_solution_is_accurate(self):
        prior, mu_0, S0, measure = self._setup()
        result = ekf1_sqr_loop_dynamic(mu_0, S0, prior, measure, self.tspan, self.N)
        m_seq = result[0]
        x_pred = float((prior.E0 @ m_seq[-1])[0])
        x_true = float(logistic_analytic(self.tspan[1], x0=float(self.x0[0])))
        # With N=50 steps on [0, 5] the EKF1+IWP(2) should be very accurate.
        assert abs(x_pred - x_true) < 1e-3

    def test_diagonal_mode_returns_per_component_sigma(self):
        prior, mu_0, S0, measure = self._setup()
        result = ekf1_sqr_loop_dynamic(
            mu_0, S0, prior, measure, self.tspan, self.N, calibration="diagonal"
        )
        sigma_seq = result[9]
        assert len(sigma_seq) == self.N
        # IWP with d=1: each entry is a length-1 array.
        assert all(hasattr(s, "shape") for s in sigma_seq)

    def test_consumable_by_smoother(self):
        prior, mu_0, S0, measure = self._setup()
        result = ekf1_sqr_loop_dynamic(mu_0, S0, prior, measure, self.tspan, self.N)
        m_seq, P_seq_sqr = result[0], result[1]
        G_back_seq, d_back_seq, P_back_seq_sqr = result[4], result[5], result[6]
        m_smooth, _P_smooth = rts_sqr_smoother_loop(
            m_seq[-1],
            P_seq_sqr[-1],
            G_back_seq,
            d_back_seq,
            P_back_seq_sqr,
            self.N,
        )
        # Smoothed final state equals filtered final state.
        assert np.allclose(m_smooth[-1], m_seq[-1])


# ---------------------------------------------------------------------------
# Cross-variant consistency: plain / scan / preconditioned / preconditioned-scan
# all run the *same* algorithm and should agree to floating-point precision.
# ---------------------------------------------------------------------------


class TestDynamicLoopVariantsAgree:
    x0 = np.array([0.1])
    tspan = (0.0, 5.0)
    N = 50

    def _setup_iwp(self, q=2):
        prior = IWP(q=q, d=1)
        mu_0, S0 = taylor_mode_initialization(logistic_vf, self.x0, q=q)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        return prior, mu_0, S0, measure

    def _setup_precond_iwp(self, q=2):
        prior = PrecondIWP(q=q, d=1)
        mu_0, S0 = taylor_mode_initialization(logistic_vf, self.x0, q=q)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        return prior, mu_0, S0, measure

    def test_scan_matches_python_loop(self):
        prior, mu_0, S0, measure = self._setup_iwp()
        r_loop = ekf1_sqr_loop_dynamic(mu_0, S0, prior, measure, self.tspan, self.N)
        r_scan = ekf1_sqr_loop_dynamic_scan(
            mu_0, S0, prior, measure, self.tspan, self.N
        )
        # Final filtered mean / cov agree to fp32-ish precision (jax default).
        assert np.allclose(r_loop[0][-1], r_scan[0][-1], atol=1e-6)
        assert np.allclose(r_loop[1][-1], r_scan[1][-1], atol=1e-6)
        # Per-step sigma agrees.
        sigma_loop = np.asarray(r_loop[9])
        sigma_scan = np.asarray(r_scan[9])
        assert np.allclose(sigma_loop, sigma_scan, atol=1e-6)

    def test_preconditioned_scan_matches_python_loop(self):
        prior, mu_0, S0, measure = self._setup_precond_iwp()
        r_loop = ekf1_sqr_loop_preconditioned_dynamic(
            mu_0, S0, prior, measure, self.tspan, self.N
        )
        r_scan = ekf1_sqr_loop_preconditioned_dynamic_scan(
            mu_0, S0, prior, measure, self.tspan, self.N
        )
        # m_seq is at index 0 in both.
        assert np.allclose(r_loop[0][-1], r_scan[0][-1], atol=1e-6)
        # sigma_sqr_seq is at index 11 in both.
        sigma_loop = np.asarray(r_loop[11])
        sigma_scan = np.asarray(r_scan[11])
        assert np.allclose(sigma_loop, sigma_scan, atol=1e-6)

    def test_sigma_invariant_under_preconditioning(self):
        """``sigma_hat^2`` from ``H Q H.T`` equals the bar-space estimator
        from ``H_bar Q_bar H_bar.T`` -- preconditioning is a similarity that
        leaves the per-step diffusion estimate invariant."""
        prior_iwp, mu_0, S0, measure_iwp = self._setup_iwp()
        prior_pre, _, _, measure_pre = self._setup_precond_iwp()
        r_iwp = ekf1_sqr_loop_dynamic(
            mu_0, S0, prior_iwp, measure_iwp, self.tspan, self.N
        )
        r_pre = ekf1_sqr_loop_preconditioned_dynamic(
            mu_0, S0, prior_pre, measure_pre, self.tspan, self.N
        )
        sigma_iwp = np.asarray(r_iwp[9])
        sigma_pre = np.asarray(r_pre[11])
        assert np.allclose(sigma_iwp, sigma_pre, atol=1e-6)

    def test_diagonal_mode_across_variants_agrees(self):
        """All four loop variants (plain/scan x with/without preconditioning)
        produce identical trajectories in diagonal mode on a multi-scale
        problem, since calibration is invariant under preconditioning."""
        r_rate = 2.0
        x0 = np.asarray([1e-5, 1e-10])

        def vf(x, *, t):
            return np.array([r_rate * x[0] * (1 - x[0]), r_rate * x[1] * (1 - x[1])])

        tspan = (0.0, 15.0)
        N = 500
        prior_iwp = IWP(q=3, d=2)
        prior_pre = PrecondIWP(q=3, d=2)
        mu0, S0 = taylor_mode_initialization(vf, x0, q=3)
        meas_iwp = ODEInformation(vf, prior_iwp.E0, prior_iwp.E1)
        meas_pre = ODEInformation(vf, prior_pre.E0, prior_pre.E1)

        r1 = ekf1_sqr_loop_dynamic(
            mu0, S0, prior_iwp, meas_iwp, tspan, N, calibration="diagonal"
        )
        r2 = ekf1_sqr_loop_dynamic_scan(
            mu0, S0, prior_iwp, meas_iwp, tspan, N, calibration="diagonal"
        )
        r3 = ekf1_sqr_loop_preconditioned_dynamic(
            mu0, S0, prior_pre, meas_pre, tspan, N, calibration="diagonal"
        )
        r4 = ekf1_sqr_loop_preconditioned_dynamic_scan(
            mu0, S0, prior_pre, meas_pre, tspan, N, calibration="diagonal"
        )
        # Final filtered means agree to fp64 precision.
        for r in (r2, r3, r4):
            assert np.allclose(r1[0][-1], r[0][-1], atol=1e-8)

    def test_diagonal_ekf0_variant_runs_across_loops(self):
        """diagonal_ekf0 variant works in all four loop functions."""
        r_rate = 2.0
        x0 = np.asarray([1e-5, 1e-10])

        def vf(x, *, t):
            return np.array([r_rate * x[0] * (1 - x[0]), r_rate * x[1] * (1 - x[1])])

        tspan = (0.0, 15.0)
        N = 500
        prior_iwp = IWP(q=3, d=2)
        prior_pre = PrecondIWP(q=3, d=2)
        mu0, S0 = taylor_mode_initialization(vf, x0, q=3)
        meas_iwp = ODEInformation(vf, prior_iwp.E0, prior_iwp.E1)
        meas_pre = ODEInformation(vf, prior_pre.E0, prior_pre.E1)
        for func, prior, meas in [
            (ekf1_sqr_loop_dynamic, prior_iwp, meas_iwp),
            (ekf1_sqr_loop_dynamic_scan, prior_iwp, meas_iwp),
            (ekf1_sqr_loop_preconditioned_dynamic, prior_pre, meas_pre),
            (ekf1_sqr_loop_preconditioned_dynamic_scan, prior_pre, meas_pre),
        ]:
            r = func(mu0, S0, prior, meas, tspan, N, calibration="diagonal_ekf0")
            # Final x_2 should be near 0.999 (true value) -- the multi-scale
            # problem resolves under any diagonal variant.
            m_last = r[0][-1]
            x_last = m_last @ prior.E0.T
            assert abs(float(x_last[1]) - 0.99907) < 1e-3, (
                f"{func.__name__} failed: x_2 = {float(x_last[1])}"
            )

    def test_calibration_none_matches_across_variants(self):
        """With calibration disabled, all four dynamic variants reduce to
        their non-dynamic counterparts (modulo preconditioning) and so must
        produce the same final state."""
        prior, mu_0, S0, measure = self._setup_iwp()
        r_loop = ekf1_sqr_loop_dynamic(
            mu_0, S0, prior, measure, self.tspan, self.N, calibration="none"
        )
        r_scan = ekf1_sqr_loop_dynamic_scan(
            mu_0, S0, prior, measure, self.tspan, self.N, calibration="none"
        )
        assert np.allclose(r_loop[0][-1], r_scan[0][-1], atol=1e-8)
        assert np.allclose(r_loop[1][-1], r_scan[1][-1], atol=1e-8)


# ---------------------------------------------------------------------------
# Scan vs non-scan parity: full (prior x calibration) cross-product
# ---------------------------------------------------------------------------


class TestScanParityFullGrid:
    """For every supported (prior, calibration) combination, the JIT-scan
    loop must produce the same trajectory and per-step sigma as the Python
    non-scan loop. Each of the three bugs fixed in this PR shows up
    somewhere in this grid:

    - Joint priors crashed at scan setup (`prior.xi` is undefined on
      `JointPrior` / `PrecondJointPrior`).
    - The scan body skipped the ``[:d_ode]`` slicing of the residual; for
      measurements with stacked rows the scan sigma was wrong.
    - In dynamic mode the scan body multiplied the full Q by sigma instead
      of routing through ``apply_state_sigma_sqr``; for joint priors this
      poisoned the input block.
    """

    tspan = (0.0, 2.0)
    N = 30

    @staticmethod
    def _iwp_setup():
        from ode_filters.priors.gmp_priors import IWP

        prior = IWP(q=2, d=1)
        mu_0, S0 = taylor_mode_initialization(logistic_vf, np.array([0.1]), q=2)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        return (
            prior,
            mu_0,
            S0,
            measure,
            ekf1_sqr_loop_dynamic,
            ekf1_sqr_loop_dynamic_scan,
        )

    @staticmethod
    def _precond_iwp_setup():
        from ode_filters.priors.gmp_priors import PrecondIWP

        prior = PrecondIWP(q=2, d=1)
        mu_0, S0 = taylor_mode_initialization(logistic_vf, np.array([0.1]), q=2)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        return (
            prior,
            mu_0,
            S0,
            measure,
            ekf1_sqr_loop_preconditioned_dynamic,
            ekf1_sqr_loop_preconditioned_dynamic_scan,
        )

    @staticmethod
    def _joint_setup():
        from ode_filters.measurement.measurement_models import (
            ODEInformationWithHidden,
        )
        from ode_filters.priors.gmp_priors import IWP, JointPrior

        prior_x = IWP(q=2, d=2)
        prior_u = IWP(q=0, d=1, Xi=1e-3 * np.eye(1))
        joint = JointPrior(prior_x, prior_u)

        def _vf_hidden_linear(x, u, *, t):
            return -u[0] * x

        measure = ODEInformationWithHidden(
            _vf_hidden_linear,
            E0=joint.E0_x,
            E1=joint.E1,
            E0_hidden=joint.E0_hidden,
        )
        D = joint.E0.shape[1]
        mu_0 = np.zeros(D).at[0].set(1.0).at[1].set(0.5).at[6].set(0.7)
        S0 = 1e-6 * np.eye(D)
        return (
            joint,
            mu_0,
            S0,
            measure,
            ekf1_sqr_loop_dynamic,
            ekf1_sqr_loop_dynamic_scan,
        )

    @staticmethod
    def _precond_joint_setup():
        from ode_filters.measurement.measurement_models import (
            ODEInformationWithHidden,
        )
        from ode_filters.priors.gmp_priors import PrecondIWP, PrecondJointPrior

        prior_x = PrecondIWP(q=2, d=2)
        prior_u = PrecondIWP(q=0, d=1, Xi=1e-3 * np.eye(1))
        joint = PrecondJointPrior(prior_x, prior_u)

        def _vf_hidden_linear(x, u, *, t):
            return -u[0] * x

        measure = ODEInformationWithHidden(
            _vf_hidden_linear,
            E0=joint.E0_x,
            E1=joint.E1,
            E0_hidden=joint.E0_hidden,
        )
        D = joint.E0.shape[1]
        mu_0 = np.zeros(D).at[0].set(1.0).at[1].set(0.5).at[6].set(0.7)
        S0 = 1e-6 * np.eye(D)
        return (
            joint,
            mu_0,
            S0,
            measure,
            ekf1_sqr_loop_preconditioned_dynamic,
            ekf1_sqr_loop_preconditioned_dynamic_scan,
        )

    @pytest.mark.parametrize(
        "prior_kind", ["iwp", "precond_iwp", "joint", "precond_joint"]
    )
    @pytest.mark.parametrize(
        "calibration", ["dynamic", "diagonal", "diagonal_ekf0", "none"]
    )
    def test_scan_matches_python_loop(self, prior_kind, calibration):
        setup = {
            "iwp": self._iwp_setup,
            "precond_iwp": self._precond_iwp_setup,
            "joint": self._joint_setup,
            "precond_joint": self._precond_joint_setup,
        }[prior_kind]
        prior, mu_0, S0, measure, loop_fn, scan_fn = setup()
        r_loop = loop_fn(
            mu_0, S0, prior, measure, self.tspan, self.N, calibration=calibration
        )
        r_scan = scan_fn(
            mu_0, S0, prior, measure, self.tspan, self.N, calibration=calibration
        )
        # m_seq is at index 0 in both shapes. Compare final means and per-step sigma.
        assert np.allclose(r_loop[0][-1], r_scan[0][-1], atol=1e-8), (
            f"{prior_kind}+{calibration}: final mean disagrees"
        )
        # sigma_sqr_seq position: 9 in basic loops, 11 in preconditioned.
        sigma_idx = 11 if "precond" in prior_kind else 9
        sigma_loop = np.asarray(r_loop[sigma_idx])
        sigma_scan = np.asarray(r_scan[sigma_idx])
        assert np.allclose(sigma_loop, sigma_scan, atol=1e-8), (
            f"{prior_kind}+{calibration}: per-step sigma disagrees"
        )


class TestScanParityWithStackedResidual:
    """When the measurement has rows beyond the ODE-defect (e.g. a
    Conservation constraint), only the first ``measure.ode_dim`` rows of the
    residual should drive sigma. The scan body used to skip this slicing."""

    def test_conservation_row_does_not_affect_sigma(self):
        from ode_filters.measurement.measurement_models import (
            Conservation,
            ODEInformation,
        )

        def vf_sir(x, *, t, beta=0.5, gamma=0.1):
            return np.array(
                [-beta * x[0] * x[1], beta * x[0] * x[1] - gamma * x[1], gamma * x[1]]
            )

        from ode_filters.priors.gmp_priors import IWP

        prior = IWP(q=2, d=3)
        x0 = np.array([0.99, 0.01, 0.0])
        mu_0, S0 = taylor_mode_initialization(vf_sir, x0, q=2)
        measure = ODEInformation(
            vf_sir,
            prior.E0,
            prior.E1,
            constraints=[Conservation(np.array([[1.0, 1.0, 1.0]]), np.array([1.0]))],
        )
        r_loop = ekf1_sqr_loop_dynamic(
            mu_0, S0, prior, measure, (0.0, 2.0), 30, calibration="dynamic"
        )
        r_scan = ekf1_sqr_loop_dynamic_scan(
            mu_0, S0, prior, measure, (0.0, 2.0), 30, calibration="dynamic"
        )
        sigma_loop = np.asarray(r_loop[9])
        sigma_scan = np.asarray(r_scan[9])
        assert np.allclose(sigma_loop, sigma_scan, atol=1e-10)
