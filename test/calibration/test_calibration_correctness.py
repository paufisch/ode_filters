"""Correctness tests for the adaptive + online-calibration EKF.

Targets behaviours that the original ``test_adaptive.py`` does not exercise:

- global convergence order of the dynamic-diffusion fixed-step loop;
- ``"diagonal"`` (EK1 H) vs ``"diagonal_ekf0"`` on a problem with
  off-diagonal Jacobian (Lotka-Volterra);
- block-diagonal equivalence: joint EKF with per-component diffusion on
  a decoupled RHS must match running independent scalar EKFs;
- the adaptive driver, locked to a single h, must reproduce
  the fixed-step :func:`gaussian_filter`;
- log-likelihood semantics across calibration modes;
- invariance under preconditioning: the preconditioned diagonal-mode
  loop's trajectory agrees with the non-preconditioned variant for both
  PrecondIWP and PrecondMaternPrior (a similarity transform leaves the
  per-component sigma estimator invariant).
"""

from __future__ import annotations

import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    PController,
    TaylorCorrection,
    gaussian_filter,
    gaussian_filter_adaptive,
)
from ode_filters.information import (
    ODEInformation,
    SecondOrderODEInformation,
)
from ode_filters.priors import (
    IWP,
    PrecondIWP,
    PrecondMaternPrior,
    taylor_mode_initialization,
)

# ---------------------------------------------------------------------------
# Test problems
# ---------------------------------------------------------------------------


def _adaptive(m_0, S0, prior, measure, tspan, **kwargs):
    """An adaptive solve over ``tspan`` recording its accepted steps."""
    kwargs.pop("h_min", None)  # the solver has no minimum step
    return gaussian_filter_adaptive(
        m_0,
        S0,
        prior,
        measure,
        np.array(tspan),
        record_steps=4096,
        max_steps=100_000,
        **kwargs,
    )


def _sigmas(r):
    return onp.asarray(r.steps.sigma_sqr[: int(r.steps.count)])


def logistic_vf(x, *, t):
    return x * (1 - x)


def logistic_analytic(t, x0=0.1):
    return 1.0 / (1.0 + (1.0 / x0 - 1.0) * np.exp(-t))


def lotka_volterra_vf(x, *, t):
    """Coupled Lotka-Volterra: dense Jacobian (off-diagonal entries nonzero)."""
    alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
    return np.array(
        [alpha * x[0] - beta * x[0] * x[1], delta * x[0] * x[1] - gamma * x[1]]
    )


# ---------------------------------------------------------------------------
# Convergence order (fixed-step dynamic loop)
# ---------------------------------------------------------------------------


class TestConvergenceOrderFixedStep:
    """Halving the step size should reduce global error by the expected power.

    For an EKF1 with IWP(q) prior, the global convergence order in h is q
    (Tronarp-Kersting-Hennig 2019; Bosch et al. 2021 §3). Halving h halves
    the error roughly by 2**q on a smooth problem.
    """

    @pytest.mark.parametrize("q", [2, 3])
    def test_global_error_decays_with_order_q(self, q):
        x0 = np.array([0.1])
        tspan = (0.0, 2.0)
        prior = IWP(q=q, d=1)
        m_0, S0 = taylor_mode_initialization(logistic_vf, x0, q=q)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        x_true = float(logistic_analytic(tspan[1], x0=float(x0[0])))

        Ns = [16, 32, 64, 128]
        errs = []
        for N in Ns:
            r = gaussian_filter(m_0, S0, prior, measure, np.linspace(*tspan, N + 1))
            x_pred = float((prior.E0 @ r.m[-1])[0])
            errs.append(abs(x_pred - x_true))

        # log-log slope of error vs h: expect ~ -q.
        hs = onp.array([(tspan[1] - tspan[0]) / N for N in Ns])
        log_h = onp.log(hs)
        log_e = onp.log(onp.maximum(onp.array(errs), 1e-16))
        slope = onp.polyfit(log_h, log_e, 1)[0]
        # Allow a generous band: smoothness, constant factors, and a small N
        # range mean we won't hit exactly q. q-0.6 is comfortably above
        # noise but well below the next order.
        assert slope >= q - 0.6, (
            f"q={q}: observed convergence slope {slope:.2f} below {q - 0.6:.2f}; "
            f"errs={errs}"
        )

    def test_adaptive_tolerance_drives_error(self):
        """Adaptive loop: error scales (roughly) with the prescribed
        tolerance. We don't fit a slope -- just verify that two-decade
        tolerance scans give two-decade error decreases."""
        x0 = np.array([0.1])
        tspan = (0.0, 5.0)
        prior = IWP(q=3, d=1)
        m_0, S0 = taylor_mode_initialization(logistic_vf, x0, q=3)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        x_true = float(logistic_analytic(tspan[1], x0=float(x0[0])))

        rtols = [1e-1, 1e-3, 1e-5]
        errs = []
        for rtol in rtols:
            r = _adaptive(m_0, S0, prior, measure, tspan, atol=rtol * 1e-2, rtol=rtol)
            errs.append(abs(float((prior.E0 @ r.m[-1])[0]) - x_true))
        # Two-orders-of-magnitude rtol reduction should reduce error by at
        # least one order of magnitude (PI controller has overheads).
        assert errs[1] < errs[0] * 0.1
        assert errs[2] < errs[1] * 0.1


# ---------------------------------------------------------------------------
# Off-diagonal-Jacobian regime: "diagonal" vs "diagonal_ekf0"
# ---------------------------------------------------------------------------


class TestDiagonalModesOffDiagonalJacobian:
    """Lotka-Volterra has a dense Jacobian: ``"diagonal"`` (EK1 H) loses
    its block-diagonal MLE interpretation. Both modes should still run,
    converge to the true trajectory at sufficient tolerance, and produce
    per-component sigma estimates that differ in a controlled way."""

    x0 = np.array([1.0, 0.5])
    tspan = (0.0, 3.0)

    def _setup(self, q=3):
        prior = IWP(q=q, d=2)
        m_0, S0 = taylor_mode_initialization(lotka_volterra_vf, self.x0, q=q)
        measure = ODEInformation(lotka_volterra_vf, prior.E0, prior.E1)
        return prior, m_0, S0, measure

    def _reference_endpoint(self):
        """Reference trajectory via a tight-tolerance run."""
        prior, m_0, S0, measure = self._setup(q=4)
        r = _adaptive(
            m_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-9,
            rtol=1e-8,
            h_min=1e-12,
        )
        return prior.E0 @ r.m[-1]

    def test_both_diagonal_modes_run_and_agree_with_reference(self):
        ref = self._reference_endpoint()
        prior, m_0, S0, measure = self._setup()
        for mode in ("diagonal", "diagonal_ekf0"):
            r = _adaptive(
                m_0,
                S0,
                prior,
                measure,
                self.tspan,
                atol=1e-7,
                rtol=1e-6,
                h_min=1e-10,
                calibration=mode,
            )
            x_final = prior.E0 @ r.m[-1]
            err = float(np.max(np.abs(x_final - ref)))
            assert err < 1e-3, f"mode={mode} max-component error {err:.2e}"

    def test_diagonal_disagrees_with_diagonal_ekf0_on_coupled_jacobian(self):
        """The diagonal-of-dense recipe (``"diagonal"``) and the
        exact-diagonal recipe (``"diagonal_ekf0"``) generally differ on
        problems with off-diagonal ``J_f``. This test documents that
        difference: at loose tolerance the per-step sigma estimates are
        not identical."""
        prior, m_0, S0, measure = self._setup()
        r1 = _adaptive(
            m_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-3,
            rtol=1e-2,
            calibration="diagonal",
        )
        r2 = _adaptive(
            m_0,
            S0,
            prior,
            measure,
            self.tspan,
            atol=1e-3,
            rtol=1e-2,
            calibration="diagonal_ekf0",
        )
        # Run-length / grid almost certainly differs, so just compare the
        # sigma magnitudes via mean ratio per component on each run.
        s1 = _sigmas(r1).mean(axis=0)
        s2 = _sigmas(r2).mean(axis=0)
        ratio = float(onp.max(onp.abs(s1 / s2 - 1.0)))
        # Lotka-Volterra has cross-coupling: expect at least a few percent
        # disagreement in the mean per-component sigma.
        assert ratio > 1e-3, (
            f"expected non-trivial disagreement on coupled-J problem, got "
            f"max relative diff {ratio:.2e}"
        )


# ---------------------------------------------------------------------------
# Block-diagonal cross-check: decoupled RHS reduces to independent EKFs
# ---------------------------------------------------------------------------


class TestBlockDiagonalEquivalence:
    """For a fully decoupled RHS ``f(x)_i = f_i(x_i)`` with ``xi`` diagonal,
    the joint EKF with ``"diagonal"`` calibration must produce the same
    per-component trajectory as running ``d`` independent scalar EKFs with
    ``"dynamic"`` calibration. The argument: ``J_f`` is block-diagonal in
    components, so the joint EK1 ``H_t = E_1 - J_f E_0`` only couples each
    component to its own derivatives; ``(H_t Q H_t.T)`` is block-diagonal
    and the per-component MLE in the joint filter coincides with the
    scalar MLE of each independent filter.

    Note: this equivalence holds for ``"diagonal"`` (uses the full EK1
    ``H_t`` in the denominator), not for ``"diagonal_ekf0"`` (uses
    ``E_1``, which drops the ``J_f`` contribution and therefore disagrees
    with the scalar EK1 even on decoupled problems)."""

    @staticmethod
    def _decoupled_vf(x, *, t):
        # Two independent logistics with the same growth rate.
        return np.array([x[0] * (1 - x[0]), x[1] * (1 - x[1])])

    def test_joint_diagonal_matches_independent_scalar(self):
        tspan = (0.0, 3.0)
        N = 60
        x0 = np.array([0.1, 0.3])
        q = 2

        # Joint d=2 EKF, "diagonal" (EK1 H) calibration.
        joint_prior = IWP(q=q, d=2)
        mu0_joint, S0_joint = taylor_mode_initialization(self._decoupled_vf, x0, q=q)
        joint_meas = ODEInformation(self._decoupled_vf, joint_prior.E0, joint_prior.E1)
        r_joint = gaussian_filter(
            mu0_joint,
            S0_joint,
            joint_prior,
            joint_meas,
            np.linspace(*tspan, N + 1),
            calibration="diagonal",
        )

        # Independent scalar EKFs.
        def vf_scalar(x, *, t):
            return x * (1 - x)

        x_final_indep = []
        sigma_indep = []
        for x0_i in x0:
            scalar_prior = IWP(q=q, d=1)
            mu0_s, S0_s = taylor_mode_initialization(vf_scalar, x0_i[None], q=q)
            scalar_meas = ODEInformation(vf_scalar, scalar_prior.E0, scalar_prior.E1)
            r_s = gaussian_filter(
                mu0_s,
                S0_s,
                scalar_prior,
                scalar_meas,
                np.linspace(*tspan, N + 1),
                calibration="dynamic",
            )
            x_final_indep.append(float((scalar_prior.E0 @ r_s.m[-1])[0]))
            sigma_indep.append(onp.asarray(r_s.sigma_sqr))

        x_final_joint = onp.asarray(joint_prior.E0 @ r_joint.m[-1])
        # Per-component endpoints agree.
        assert onp.allclose(x_final_joint, onp.asarray(x_final_indep), atol=1e-10)

        # Per-step per-component sigma matches the per-step sigma of each
        # independent scalar EKF -- exactly, by the algebra above.
        sigma_joint = onp.asarray(r_joint.sigma_sqr)  # shape (N, 2)
        assert onp.allclose(sigma_joint[:, 0], sigma_indep[0], atol=1e-10)
        assert onp.allclose(sigma_joint[:, 1], sigma_indep[1], atol=1e-10)


# ---------------------------------------------------------------------------
# Adaptive driver = fixed-step dynamic loop on a uniform grid
# ---------------------------------------------------------------------------


class TestAdaptiveEqualsFixedStep:
    """If the controller is locked (``min_factor=max_factor=1`` and
    ``safety=1``) the adaptive loop must run on a uniform grid identical
    to the fixed-step :func:`gaussian_filter`. The two trajectories should
    agree to floating-point precision."""

    def test_locked_controller_reproduces_fixed_step(self):
        prior = IWP(q=2, d=1)
        m_0, S0 = taylor_mode_initialization(logistic_vf, np.array([0.1]), q=2)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        # Pick a span/N pair where h is exactly representable in fp64 so
        # the adaptive loop terminates cleanly at the last step rather than
        # taking a sub-epsilon trailing iteration.
        tspan = (0.0, 1.0)
        N = 16
        h = (tspan[1] - tspan[0]) / N  # 1/16 -- exact in fp64

        # Lock the controller to always propose h * 1 = h.
        controller = PController(
            order=prior.q, safety=1.0, alpha=1.0, min_factor=1.0, max_factor=1.0
        )

        r_adapt = _adaptive(
            m_0,
            S0,
            prior,
            measure,
            tspan,
            atol=1.0,  # Loose: every step accepted.
            rtol=1.0,
            h_init=h,
            controller=controller,
            sigma_in_error="per_step",
        )
        r_fixed = gaussian_filter(m_0, S0, prior, measure, np.linspace(*tspan, N + 1))

        assert int(r_adapt.steps.count) == N
        assert np.allclose(r_adapt.m[-1], r_fixed.m[-1], atol=1e-10)
        assert np.allclose(r_adapt.P_sqr[-1], r_fixed.P_sqr[-1], atol=1e-10)
        sigma_adapt = _sigmas(r_adapt)[:, 0]
        sigma_fixed = onp.asarray(r_fixed.sigma_sqr)
        assert onp.allclose(sigma_adapt, sigma_fixed, atol=1e-10)


# ---------------------------------------------------------------------------
# Log-likelihood semantics
# ---------------------------------------------------------------------------


class TestLogLikelihoodSemantics:
    """The log-likelihood field in the dynamic loop is the *calibrated*
    per-step log N(0; mz_n, Pz_n). With ``calibration="none"`` this equals
    the uncalibrated loop's log-likelihood; with ``calibration="dynamic"``
    each step's Pz_sqr is post-calibration."""

    def _setup(self):
        prior = IWP(q=2, d=1)
        m_0, S0 = taylor_mode_initialization(logistic_vf, np.array([0.1]), q=2)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        return prior, m_0, S0, measure

    def test_dynamic_log_likelihood_is_sum_of_per_step_calibrated(self):
        """Re-derive log_likelihood independently from the stored
        (mz, Pz_sqr) sequence and verify it matches the loop's report."""
        prior, m_0, S0, measure = self._setup()
        tspan, N = (0.0, 5.0), 40
        r = gaussian_filter(m_0, S0, prior, measure, np.linspace(*tspan, N + 1))
        mz_seq, Pz_seq_sqr, ll = r.mz, r.Pz_sqr, r.log_likelihood

        ll_recomputed = 0.0
        for mz, Pz_sqr in zip(mz_seq, Pz_seq_sqr, strict=True):
            log_det = 2.0 * float(np.sum(np.log(np.abs(np.diag(Pz_sqr)))))
            v = np.linalg.solve(Pz_sqr.T, mz)
            maha = float(v @ v)
            obs_dim = mz.shape[0]
            ll_recomputed += -0.5 * (
                obs_dim * float(np.log(2 * np.pi)) + log_det + maha
            )
        assert float(ll) == pytest.approx(ll_recomputed, rel=1e-10)


# ---------------------------------------------------------------------------
# Boundary-condition regression: integration-endpoint roundoff
# ---------------------------------------------------------------------------


class TestAdaptiveBoundaryRoundoff:
    """Float accumulation of ``t + h`` must not leave a sub-ulp trailing step:
    the solver lands on each save time within a span-scaled tolerance."""

    def _setup(self):
        prior = IWP(q=2, d=1)
        m_0, S0 = taylor_mode_initialization(logistic_vf, np.array([0.1]), q=2)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        return prior, m_0, S0, measure

    def test_non_fp_exact_h_terminates_cleanly(self):
        """h = 0.1 is not exactly representable; ten 0.1-steps land on 1.0."""
        prior, m_0, S0, measure = self._setup()
        controller = PController(
            order=prior.q, safety=1.0, alpha=1.0, min_factor=1.0, max_factor=1.0
        )
        r = gaussian_filter_adaptive(
            m_0,
            S0,
            prior,
            measure,
            np.linspace(0.0, 1.0, 11),
            h_init=0.1,
            atol=1.0,
            rtol=1.0,
            controller=controller,
        )
        assert bool(r.success)
        assert onp.all(onp.asarray(r.n_accepted) == 1)

    def test_an_unreachable_tolerance_fails_without_raising(self):
        """The jittable solver cannot raise: it reports the failure."""
        prior, m_0, S0, measure = self._setup()
        r = gaussian_filter_adaptive(
            m_0,
            S0,
            prior,
            measure,
            np.array([0.0, 1.0]),
            atol=1e-30,
            rtol=1e-30,
            max_steps=20,
        )
        assert not bool(r.success)


# ---------------------------------------------------------------------------
# Preconditioned diagonal calibration: refuse non-PrecondIWP priors
# ---------------------------------------------------------------------------


class TestPreconditionedDiagonalMatchesNonPreconditioned:
    """Preconditioning is a similarity transform: it does not change the
    per-component diffusion estimate. The diagonal-mode trajectory of the
    preconditioned loop must therefore agree with the non-preconditioned
    loop on the same prior family. This is the structural invariant that
    used to require ``_Q_bar`` on every preconditioned prior; the current
    implementation works through ``apply_state_sigma_sqr`` on a bar-space
    ``Q_sqr`` and so handles every preconditioned prior whose ``Q(h)``
    factors as ``Q_bar ⊗ xi``.
    """

    def test_precond_matern_diagonal_matches_matern(self):
        from ode_filters.priors import MaternPrior

        prior_p = PrecondMaternPrior(q=2, d=1, length_scale=1.0)
        prior_n = MaternPrior(q=2, d=1, length_scale=1.0)
        m_0, S0 = taylor_mode_initialization(logistic_vf, np.array([0.1]), q=2)
        m_p = ODEInformation(logistic_vf, prior_p.E0, prior_p.E1)
        m_n = ODEInformation(logistic_vf, prior_n.E0, prior_n.E1)

        r_p = gaussian_filter(
            m_0,
            S0,
            prior_p,
            m_p,
            np.linspace(*(0.0, 1.0), 10 + 1),
            calibration="diagonal_ekf0",
        )
        r_n = gaussian_filter(
            m_0,
            S0,
            prior_n,
            m_n,
            np.linspace(*(0.0, 1.0), 10 + 1),
            calibration="diagonal_ekf0",
        )
        # Final filtered mean agrees up to fp64.
        assert np.allclose(r_p.m[-1], r_n.m[-1], atol=1e-10)


@pytest.mark.parametrize("prior_cls", [IWP, PrecondIWP])
def test_diagonal_ekf0_uses_the_measurement_selector_for_second_order_odes(
    prior_cls,
):
    """For x'' = f the EK0 Jacobian of the ODE rows is E2, not the prior's E1.

    ``"diagonal_ekf0"`` used to build its denominator from ``prior.E1``, which
    inflated sigma^2 by orders of magnitude for second-order ODEs. Oracle: with
    an EK0 correction the reported ``mz`` is the ODE residual at the prediction,
    so ``sigma^2_n = mz_n^2 / diag(E2 Q(h) E2.T)``. Preconditioning leaves this
    unchanged, so the plain-space ``Q`` is the oracle for both priors.
    """
    prior = prior_cls(q=3, d=1)
    measure = SecondOrderODEInformation(
        lambda x, v, *, t: -x, prior.E0, prior.E1, prior.E2
    )
    m_0 = np.array([1.0, 0.0, -1.0, 0.0])
    tspan, N = (0.0, 3.0), 60
    res = gaussian_filter(
        m_0,
        np.zeros((4, 4)),
        prior,
        measure,
        np.linspace(*tspan, N + 1),
        correction=TaylorCorrection(order=0),
        calibration="diagonal_ekf0",
    )
    h = (tspan[1] - tspan[0]) / N
    E2Q_sqr = prior.E2 @ IWP(q=3, d=1).Q_sqr(h).T
    denom = np.sum(E2Q_sqr**2, axis=1)  # diag(E2 Q E2.T)
    expected = res.mz[:, :1] ** 2 / denom[None, :]
    assert np.allclose(res.sigma_sqr, expected, rtol=1e-6, atol=1e-300)
