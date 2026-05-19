"""Tests for ode_filters.calibration.sigma."""

from __future__ import annotations

import jax.numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ode_filters.calibration.sigma import (
    aggregate_sigma_sqr,
    posthoc_mle_sigma_sqr,
    quasi_mle_sigma_sqr,
    quasi_mle_sigma_sqr_from_Q,
)


def _random_sqr(d: int, seed: int) -> np.ndarray:
    """Build a well-conditioned upper-triangular square-root covariance."""
    import numpy as onp

    rng = onp.random.default_rng(seed)
    A = rng.standard_normal((d, d))
    Sigma = A @ A.T + d * onp.eye(d)
    L = onp.linalg.cholesky(Sigma)
    # Library convention: P = P_sqr.T @ P_sqr  with P_sqr upper-triangular.
    P_sqr = L.T
    return np.asarray(P_sqr)


class TestQuasiMLE:
    def test_matches_closed_form_scalar(self):
        # For d=1: sigma^2 = m^2 / S.
        m_z = np.array([2.0])
        P_z_sqr = np.array([[3.0]])  # S = 9.0
        expected = (2.0**2) / 9.0
        got = float(quasi_mle_sigma_sqr(m_z, P_z_sqr))
        assert got == pytest.approx(expected, rel=1e-12)

    def test_matches_dense_inverse(self):
        d = 4
        P_sqr = _random_sqr(d, seed=0)
        m_z = np.asarray([0.4, -1.1, 0.2, 0.9])
        S = P_sqr.T @ P_sqr
        expected = float(m_z @ np.linalg.solve(S, m_z)) / d
        got = float(quasi_mle_sigma_sqr(m_z, P_sqr))
        assert got == pytest.approx(expected, rel=1e-10)

    @given(scale=st.floats(min_value=0.1, max_value=10.0))
    @settings(max_examples=20, deadline=None)
    def test_invariant_under_joint_rescaling(self, scale):
        """Scaling both m_z and the covariance scale appropriately leaves
        sigma_hat invariant up to the variance factor: if S -> c^2 * S then
        sigma_hat -> sigma_hat / c^2."""
        d = 3
        P_sqr = _random_sqr(d, seed=1)
        m_z = np.asarray([0.3, -0.2, 0.7])
        sigma_base = float(quasi_mle_sigma_sqr(m_z, P_sqr))
        # If P_sqr -> c * P_sqr then S -> c^2 * S, so sigma_hat -> sigma_hat / c^2.
        sigma_scaled = float(quasi_mle_sigma_sqr(m_z, scale * P_sqr))
        assert sigma_scaled == pytest.approx(sigma_base / (scale**2), rel=1e-9)

    def test_nonnegative(self):
        d = 3
        P_sqr = _random_sqr(d, seed=2)
        m_z = np.asarray([0.0, 0.0, 0.0])
        assert float(quasi_mle_sigma_sqr(m_z, P_sqr)) == pytest.approx(0.0)


class TestNoiseExclusion:
    """The quasi-MLE assumes P_z_sqr excludes measurement noise R; verify the
    bias direction and the adaptive loop's exclusion fix."""

    def test_added_R_biases_sigma_downward(self):
        """Adding R to the residual covariance lowers sigma_hat -- the classic
        confounding of diffusion scale and measurement noise."""
        from ode_filters.inference.sqr_gaussian_inference import sqr_marginalization

        H = np.asarray([[1.0, -0.5]])
        m_pred = np.asarray([0.3, 0.7])
        P_pred_sqr = _random_sqr(2, seed=42)
        c = np.zeros(1)

        # Noise-free residual sqrt covariance.
        _, Pz_clean = sqr_marginalization(H, c, np.zeros((1, 1)), m_pred, P_pred_sqr)
        # R > 0 added.
        _, Pz_with_R = sqr_marginalization(
            H, c, np.asarray([[0.5]]), m_pred, P_pred_sqr
        )
        mz = H @ m_pred + c

        sigma_clean = float(quasi_mle_sigma_sqr(mz, Pz_clean))
        sigma_with_R = float(quasi_mle_sigma_sqr(mz, Pz_with_R))
        assert sigma_with_R < sigma_clean, (
            "Including R in the residual covariance should bias the per-step "
            "quasi-MLE downward."
        )

    def test_adaptive_step_body_excludes_R_from_calibration(self):
        """A single adaptive step body call: the per-step sigma_sqr should
        equal the dynamic-diffusion quasi-MLE from ``H Q(h) H.T`` regardless
        of the R the measurement model exposes (R only enters the update,
        never the calibration)."""
        from ode_filters.filters.ode_filter_adaptive import _make_step_body
        from ode_filters.measurement.measurement_models import ODEInformation
        from ode_filters.priors.gmp_priors import IWP, taylor_mode_initialization

        def vf(x, *, t):
            return x * (1 - x)

        x0 = np.asarray([0.1])
        prior = IWP(q=2, d=1)
        mu_0, S0 = taylor_mode_initialization(vf, x0, q=2)
        h, t = 0.1, 0.1

        m_zero = ODEInformation(vf, prior.E0, prior.E1)
        body_zero = _make_step_body(prior, m_zero, atol=1e-4, rtol=1e-2)
        out_zero = body_zero(h, t, mu_0, S0)
        # Step body always returns sigma as a length-d array (broadcast from
        # scalar in non-diagonal modes). Index 0 for d=1 logistic.
        sigma_zero = float(out_zero[9][0])

        m_with_R = ODEInformation(vf, prior.E0, prior.E1)
        m_with_R.R = 0.5  # large R to make any leakage obvious
        body_R = _make_step_body(prior, m_with_R, atol=1e-4, rtol=1e-2)
        out_R = body_R(h, t, mu_0, S0)
        sigma_R = float(out_R[9][0])

        # R=0 and R>0 calibration values agree: R never enters the
        # dynamic-diffusion estimator.
        assert sigma_R == pytest.approx(sigma_zero, rel=1e-10)

        # Independent re-derivation via the dynamic-diffusion estimator.
        # Linearise at the provisional predicted mean A m_prev + b (the
        # quantity actually used inside the step body).
        A_h = prior.A(h)
        b_h = prior.b(h)
        Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T
        m_pred_prov = A_h @ mu_0 + b_h
        H_t, c_t = m_zero.linearize(m_pred_prov, t=t)
        mz_pred = H_t @ m_pred_prov + c_t
        sigma_recompute = float(quasi_mle_sigma_sqr_from_Q(mz_pred, H_t, Q_h_sqr))
        assert sigma_zero == pytest.approx(sigma_recompute, rel=1e-10)

        # Sanity: a naive estimator that *includes* R in P_z differs.
        full_mz = out_R[5]
        full_Pz = out_R[6]
        sigma_naive_with_R = float(quasi_mle_sigma_sqr(full_mz, full_Pz))
        assert sigma_naive_with_R != pytest.approx(sigma_zero, rel=1e-3)


class TestPosthocMLE:
    def test_single_step_matches_quasi_mle(self):
        d = 3
        P_sqr = _random_sqr(d, seed=3)
        m_z = np.asarray([0.5, 0.1, -0.4])
        per_step = float(quasi_mle_sigma_sqr(m_z, P_sqr))
        posthoc = float(posthoc_mle_sigma_sqr(m_z[None, :], P_sqr[None, :, :]))
        assert posthoc == pytest.approx(per_step, rel=1e-12)

    def test_mean_of_per_step_estimates(self):
        d = 2
        P_sqrs = np.stack([_random_sqr(d, seed=k) for k in (4, 5, 6)], axis=0)
        mzs = np.asarray([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]])
        per_step = np.asarray(
            [float(quasi_mle_sigma_sqr(mzs[i], P_sqrs[i])) for i in range(3)]
        )
        posthoc = float(posthoc_mle_sigma_sqr(mzs, P_sqrs))
        assert posthoc == pytest.approx(float(np.mean(per_step)), rel=1e-12)


class TestAggregate:
    def test_mean_matches_numpy(self):
        seq = np.asarray([0.1, 0.2, 0.4, 0.3])
        assert float(aggregate_sigma_sqr(seq, kind="mean")) == pytest.approx(0.25)

    def test_last(self):
        seq = np.asarray([0.1, 0.2, 0.4, 0.3])
        assert float(aggregate_sigma_sqr(seq, kind="last")) == pytest.approx(0.3)

    def test_running_is_cumulative_mean(self):
        seq = np.asarray([1.0, 3.0, 5.0, 7.0])
        running = aggregate_sigma_sqr(seq, kind="running")
        expected = np.asarray([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(running, expected)

    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregator kind"):
            aggregate_sigma_sqr(np.asarray([1.0]), kind="bogus")  # type: ignore[arg-type]
