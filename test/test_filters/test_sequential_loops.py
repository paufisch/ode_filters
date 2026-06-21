"""Tests for sequential observation assimilation via ``gaussian_filter``.

These exercise the public ``gaussian_filter(obs_model=...)`` path, which folds
in external observations on top of the ODE-information update. (Cross-checks of
the historical for-loop/scan/sequential ``sqr_loop*`` variants against each
other were dropped when those variants were consolidated into
``gaussian_filter``; the remaining behavioural coverage lives here.)
"""

import jax.numpy as np

from ode_filters import gaussian_filter, prepare_observations
from ode_filters.measurement.measurement_models import (
    Measurement,
    ODEInformation,
)
from ode_filters.priors.gmp_priors import IWP, taylor_mode_initialization


def _vf_logistic(x, *, t):
    """Simple logistic ODE: dx/dt = x(1-x)."""
    return x * (1 - x)


# ---------------------------------------------------------------------------
# Test: observation assimilation
# ---------------------------------------------------------------------------


class TestSeqWithObservations:
    """Verify gaussian_filter assimilates external observations correctly."""

    def test_sequential_with_observations(self):
        """Filter with observations produces finite results and nonzero ll_obs."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        ts = np.linspace(tspan[0], tspan[1], N + 1)
        z_t = ts[5:15]
        z = np.array([[0.1 + 0.05 * i] for i in range(10)])
        A = np.array([[1.0]])
        observation = Measurement(A, z, z_t, noise=0.01)
        obs_model = prepare_observations([observation], prior.E0, ts)

        result = gaussian_filter(
            mu_0,
            Sigma_0_sqr,
            prior,
            measure,
            tspan,
            N,
            calibration="none",
            obs_model=obs_model,
        )

        m_seq = np.array(result.m)
        ll_ode = float(result.log_likelihood)
        ll_obs = float(result.log_likelihood_obs)

        assert m_seq.shape == (N + 1, q * d + d)
        assert np.all(np.isfinite(m_seq))
        assert np.isfinite(ll_ode)
        assert np.isfinite(ll_obs)
        # With observations, ll_obs should be nonzero
        assert ll_obs != 0.0

    def test_observations_none_equals_no_observations(self):
        """Passing obs_model=None gives same result as omitting it."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        result_default = gaussian_filter(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N, calibration="none"
        )
        result_none = gaussian_filter(
            mu_0,
            Sigma_0_sqr,
            prior,
            measure,
            tspan,
            N,
            calibration="none",
            obs_model=None,
        )

        m_default = np.array(result_default.m)
        m_none = np.array(result_none.m)
        assert np.allclose(m_default, m_none, atol=1e-12)

    def test_observations_improve_solution(self):
        """Observations near true solution should pull estimates closer."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        # True solution at some grid points
        ts = np.linspace(tspan[0], tspan[1], N + 1)
        z_t = ts[5:15]
        z_true = 1.0 / (1.0 + 9.0 * np.exp(-z_t))
        z = z_true[:, None]
        A = np.array([[1.0]])
        observation = Measurement(A, z, z_t, noise=0.001)
        obs_model = prepare_observations([observation], prior.E0, ts)

        result_no_obs = gaussian_filter(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N, calibration="none"
        )
        result_obs = gaussian_filter(
            mu_0,
            Sigma_0_sqr,
            prior,
            measure,
            tspan,
            N,
            calibration="none",
            obs_model=obs_model,
        )

        E0 = prior.E0
        m_no_obs = np.array(result_no_obs.m)
        m_obs = np.array(result_obs.m)

        # Compare at observed time steps (indices 5-14)
        y_true = 1.0 / (1.0 + 9.0 * np.exp(-ts))
        err_no_obs = np.mean(np.abs((E0 @ m_no_obs[5:15].T)[0] - y_true[5:15]))
        err_obs = np.mean(np.abs((E0 @ m_obs[5:15].T)[0] - y_true[5:15]))

        assert err_obs < err_no_obs
