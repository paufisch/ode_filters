"""Tests for log marginal likelihood returned by the Gaussian filter."""

import jax
import jax.numpy as np
import pytest

from ode_filters import gaussian_filter
from ode_filters.measurement.measurement_models import ODEInformation
from ode_filters.priors.gmp_priors import IWP, PrecondIWP, taylor_mode_initialization


def _vf(x, *, t):
    """Simple logistic ODE: dx/dt = x(1-x)."""
    return x * (1 - x)


def _manual_log_likelihood(mz_seq, Pz_seq_sqr):
    """Compute log marginal likelihood from innovation sequences."""
    ll = 0.0
    for mz, Pz_sqr in zip(mz_seq, Pz_seq_sqr, strict=True):
        Pz = Pz_sqr.T @ Pz_sqr
        log_det = 2.0 * np.sum(np.log(np.abs(np.diag(Pz_sqr))))
        maha = mz @ np.linalg.solve(Pz, mz)
        obs_dim = mz.shape[0]
        ll += -0.5 * (obs_dim * np.log(2 * np.pi) + log_det + maha)
    return ll


# ---------------------------------------------------------------------------
# Test: returned LML matches manual computation
# ---------------------------------------------------------------------------


class TestLogLikelihoodMatchesManual:
    """Verify that the accumulated LML matches a manual recomputation."""

    x0 = np.array([0.1])
    tspan = (0.0, 1.0)
    N = 10
    q = 2
    d = 1

    def test_ekf1_sqr_loop(self):
        prior = IWP(q=self.q, d=self.d)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf, self.x0, self.q)
        measure = ODEInformation(_vf, prior.E0, prior.E1)

        result = gaussian_filter(
            mu_0, Sigma_0_sqr, prior, measure, self.tspan, self.N, calibration="none"
        )

        mz_seq = result.mz
        Pz_seq_sqr = result.Pz_sqr
        log_likelihood = result.log_likelihood

        expected = _manual_log_likelihood(mz_seq, Pz_seq_sqr)
        assert float(log_likelihood) == pytest.approx(float(expected), rel=1e-10)

    def test_ekf1_sqr_loop_preconditioned(self):
        prior = PrecondIWP(q=self.q, d=self.d)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf, self.x0, self.q)
        measure = ODEInformation(_vf, prior.E0, prior.E1)

        result = gaussian_filter(
            mu_0, Sigma_0_sqr, prior, measure, self.tspan, self.N, calibration="none"
        )

        mz_seq = result.mz
        Pz_seq_sqr = result.Pz_sqr
        log_likelihood = result.log_likelihood

        expected = _manual_log_likelihood(mz_seq, Pz_seq_sqr)
        assert float(log_likelihood) == pytest.approx(float(expected), rel=1e-10)


# ---------------------------------------------------------------------------
# Test: LML is differentiable w.r.t. inputs flowing through Q_h_sqr
# ---------------------------------------------------------------------------


class TestLogLikelihoodDifferentiable:
    """Verify that jax.grad through the LML does not error and is finite."""

    x0 = np.array([0.1])
    tspan = (0.0, 1.0)
    N = 5
    q = 2
    d = 1

    def test_ekf1_sqr_loop_grad(self):
        prior = IWP(q=self.q, d=self.d)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf, self.x0, self.q)
        measure = ODEInformation(_vf, prior.E0, prior.E1)

        def lml_fn(Sigma_0_sqr_):
            result = gaussian_filter(
                mu_0,
                Sigma_0_sqr_,
                prior,
                measure,
                self.tspan,
                self.N,
                calibration="none",
            )
            return result.log_likelihood

        grad_val = jax.grad(lml_fn)(Sigma_0_sqr)
        assert np.all(np.isfinite(grad_val))

    def test_ekf1_sqr_loop_preconditioned_grad(self):
        prior = PrecondIWP(q=self.q, d=self.d)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf, self.x0, self.q)
        measure = ODEInformation(_vf, prior.E0, prior.E1)

        def lml_fn(Sigma_0_sqr_):
            result = gaussian_filter(
                mu_0,
                Sigma_0_sqr_,
                prior,
                measure,
                self.tspan,
                self.N,
                calibration="none",
            )
            return result.log_likelihood

        grad_val = jax.grad(lml_fn)(Sigma_0_sqr)
        assert np.all(np.isfinite(grad_val))
