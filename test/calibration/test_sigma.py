"""Tests for ode_filters.calibration.sigma."""

from __future__ import annotations

import jax.numpy as np
import pytest
from hypothesis import given
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
        mz = np.array([2.0])
        Pz_sqr = np.array([[3.0]])  # S = 9.0
        expected = (2.0**2) / 9.0
        got = float(quasi_mle_sigma_sqr(mz, Pz_sqr))
        assert got == pytest.approx(expected, rel=1e-12)

    def test_matches_dense_inverse(self):
        d = 4
        P_sqr = _random_sqr(d, seed=0)
        mz = np.asarray([0.4, -1.1, 0.2, 0.9])
        S = P_sqr.T @ P_sqr
        expected = float(mz @ np.linalg.solve(S, mz)) / d
        got = float(quasi_mle_sigma_sqr(mz, P_sqr))
        assert got == pytest.approx(expected, rel=1e-10)

    @given(scale=st.floats(min_value=0.1, max_value=10.0))
    def test_invariant_under_joint_rescaling(self, scale):
        """Scaling both mz and the covariance scale appropriately leaves
        sigma_hat invariant up to the variance factor: if S -> c^2 * S then
        sigma_hat -> sigma_hat / c^2."""
        d = 3
        P_sqr = _random_sqr(d, seed=1)
        mz = np.asarray([0.3, -0.2, 0.7])
        sigma_base = float(quasi_mle_sigma_sqr(mz, P_sqr))
        # If P_sqr -> c * P_sqr then S -> c^2 * S, so sigma_hat -> sigma_hat / c^2.
        sigma_scaled = float(quasi_mle_sigma_sqr(mz, scale * P_sqr))
        assert sigma_scaled == pytest.approx(sigma_base / (scale**2), rel=1e-9)

    def test_nonnegative(self):
        d = 3
        P_sqr = _random_sqr(d, seed=2)
        mz = np.asarray([0.0, 0.0, 0.0])
        assert float(quasi_mle_sigma_sqr(mz, P_sqr)) == pytest.approx(0.0)


class TestNoiseExclusion:
    """The quasi-MLE assumes Pz_sqr excludes measurement noise R; verify the
    bias direction and the adaptive loop's exclusion fix."""

    def test_added_R_biases_sigma_downward(self):
        """Adding R to the residual covariance lowers sigma_hat -- the classic
        confounding of diffusion scale and measurement noise."""
        from ode_filters.gaussian import Gaussian, marginalize

        H = np.asarray([[1.0, -0.5]])
        m_pred = np.asarray([0.3, 0.7])
        P_pred_sqr = _random_sqr(2, seed=42)
        c = np.zeros(1)

        # Noise-free residual sqrt covariance.
        Pz_clean = marginalize(
            H, c, np.zeros((1, 1)), Gaussian(m_pred, P_pred_sqr)
        ).P_sqr
        # R > 0 added.
        Pz_with_R = marginalize(
            H, c, np.asarray([[0.5]]), Gaussian(m_pred, P_pred_sqr)
        ).P_sqr
        mz = H @ m_pred + c

        sigma_clean = float(quasi_mle_sigma_sqr(mz, Pz_clean))
        sigma_with_R = float(quasi_mle_sigma_sqr(mz, Pz_with_R))
        assert sigma_with_R < sigma_clean, (
            "Including R in the residual covariance should bias the per-step "
            "quasi-MLE downward."
        )

    def test_measurement_noise_never_enters_the_calibration(self):
        """The per-step estimate is the quasi-MLE from ``H Q(h) H.T``, whatever
        noise the measurement model carries (it only enters the update)."""
        from ode_filters import gaussian_filter
        from ode_filters.information import ODEInformation
        from ode_filters.priors import IWP, taylor_mode_initialization

        def vf(x, *, t):
            return x * (1 - x)

        prior = IWP(q=2, d=1)
        m_0, P_0_sqr = taylor_mode_initialization(vf, np.asarray([0.1]), q=2)
        ts = np.asarray([0.0, 0.1])
        clean = ODEInformation(vf, prior.E0, prior.E1)
        noisy = ODEInformation(vf, prior.E0, prior.E1)
        noisy = noisy.with_noise(0.5)  # large, so any leakage would show
        sigma_clean = gaussian_filter(m_0, P_0_sqr, prior, clean, ts).sigma_sqr[0]
        sigma_noisy = gaussian_filter(m_0, P_0_sqr, prior, noisy, ts).sigma_sqr[0]
        assert float(sigma_noisy) == pytest.approx(float(sigma_clean), rel=1e-10)

        # Independent re-derivation at the predicted mean.
        h = 0.1
        m_pred = prior.A(h) @ m_0 + prior.b(h)
        H, c = clean.linearize(m_pred, t=h)
        expected = quasi_mle_sigma_sqr_from_Q(H @ m_pred + c, H, prior.Q_sqr(h))
        assert float(sigma_clean) == pytest.approx(float(expected), rel=1e-10)


class TestPosthocMLE:
    def test_single_step_matches_quasi_mle(self):
        d = 3
        P_sqr = _random_sqr(d, seed=3)
        mz = np.asarray([0.5, 0.1, -0.4])
        per_step = float(quasi_mle_sigma_sqr(mz, P_sqr))
        posthoc = float(posthoc_mle_sigma_sqr(mz[None, :], P_sqr[None, :, :]))
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

    @pytest.mark.wrong_input_types
    def test_unknown_kind_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregator kind"):
            aggregate_sigma_sqr(np.asarray([1.0]), kind="bogus")  # type: ignore[arg-type]
