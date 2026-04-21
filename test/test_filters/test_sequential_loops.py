"""Tests for sequential update filter and smoother loops."""

import jax.numpy as np
import pytest

from ode_filters.filters.ode_filter_loop import (
    ekf1_sqr_loop,
    ekf1_sqr_loop_preconditioned,
    ekf1_sqr_loop_preconditioned_sequential,
    ekf1_sqr_loop_sequential,
)
from ode_filters.measurement.measurement_models import (
    Conservation,
    Measurement,
    ODEInformation,
)
from ode_filters.priors.gmp_priors import IWP, PrecondIWP, taylor_mode_initialization


def _vf_logistic(x, *, t):
    """Simple logistic ODE: dx/dt = x(1-x)."""
    return x * (1 - x)


def _vf_sir(x, *, t, beta=0.5, gamma=0.1):
    """SIR epidemic model."""
    return np.array(
        [
            -beta * x[0] * x[1],
            beta * x[0] * x[1] - gamma * x[1],
            gamma * x[1],
        ]
    )


# ---------------------------------------------------------------------------
# Test: without measurements, sequential equals joint
# ---------------------------------------------------------------------------


class TestSeqEqualsJointWithoutMeasurements:
    """When no Measurement constraints exist, sequential and joint are identical."""

    def test_no_measurements_standard(self):
        """Sequential matches joint without measurements (standard)."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        result_joint = ekf1_sqr_loop(mu_0, Sigma_0_sqr, prior, measure, tspan, N)
        result_seq = ekf1_sqr_loop_sequential(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N
        )

        m_joint = np.array(result_joint[0])
        m_seq = np.array(result_seq[0])

        assert np.allclose(m_seq, m_joint, atol=1e-10)
        # log_likelihood_ode is at index -2, log_likelihood_obs at index -1
        ll_ode = float(result_seq[-2])
        ll_obs = float(result_seq[-1])
        assert ll_ode == pytest.approx(float(result_joint[-1]), rel=1e-10)
        assert ll_obs == 0.0

    def test_conservation_only_standard(self):
        """Sequential matches joint with conservation but no measurements."""
        x0 = np.array([0.99, 0.01, 0.0])
        tspan = (0.0, 50.0)
        N = 50
        q = 2
        d = 3

        prior = IWP(q=q, d=d, Xi=1.0 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_sir, x0, q)

        A_cons = np.array([[1.0, 1.0, 1.0]])
        p_cons = np.array([1.0])
        conservation = Conservation(A_cons, p_cons)
        measure = ODEInformation(
            _vf_sir, prior.E0, prior.E1, constraints=[conservation]
        )

        result_joint = ekf1_sqr_loop(mu_0, Sigma_0_sqr, prior, measure, tspan, N)
        result_seq = ekf1_sqr_loop_sequential(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N
        )

        m_joint = np.array(result_joint[0])
        m_seq = np.array(result_seq[0])

        assert np.allclose(m_seq, m_joint, atol=1e-10)


# ---------------------------------------------------------------------------
# Test: sequential with observations
# ---------------------------------------------------------------------------


class TestSeqWithObservations:
    """Verify sequential loops work correctly with external observations."""

    def test_sequential_with_observations(self):
        """Sequential loop with observations produces finite results."""
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

        result = ekf1_sqr_loop_sequential(
            mu_0,
            Sigma_0_sqr,
            prior,
            measure,
            tspan,
            N,
            observations=[observation],
        )

        m_seq = np.array(result[0])
        ll_ode = float(result[-2])
        ll_obs = float(result[-1])

        assert m_seq.shape == (N + 1, q * d + d)
        assert np.all(np.isfinite(m_seq))
        assert np.isfinite(ll_ode)
        assert np.isfinite(ll_obs)
        # With observations, ll_obs should be nonzero
        assert ll_obs != 0.0

    def test_observations_none_equals_no_observations(self):
        """Passing observations=None gives same result as omitting it."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        result_default = ekf1_sqr_loop_sequential(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N
        )
        result_none = ekf1_sqr_loop_sequential(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N, observations=None
        )

        m_default = np.array(result_default[0])
        m_none = np.array(result_none[0])
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

        result_no_obs = ekf1_sqr_loop_sequential(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N
        )
        result_obs = ekf1_sqr_loop_sequential(
            mu_0,
            Sigma_0_sqr,
            prior,
            measure,
            tspan,
            N,
            observations=[observation],
        )

        E0 = prior.E0
        m_no_obs = np.array(result_no_obs[0])
        m_obs = np.array(result_obs[0])

        # Compare at observed time steps (indices 5-14)
        y_true = 1.0 / (1.0 + 9.0 * np.exp(-ts))
        err_no_obs = np.mean(np.abs((E0 @ m_no_obs[5:15].T)[0] - y_true[5:15]))
        err_obs = np.mean(np.abs((E0 @ m_obs[5:15].T)[0] - y_true[5:15]))

        assert err_obs < err_no_obs


# ---------------------------------------------------------------------------
# Test: preconditioned sequential
# ---------------------------------------------------------------------------


class TestPrecondSequential:
    """Verify preconditioned sequential loops."""

    def test_precond_sequential_equals_precond_joint_no_measurements(self):
        """Precond sequential matches precond joint without measurements."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = PrecondIWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        result_joint = ekf1_sqr_loop_preconditioned(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N
        )
        result_seq = ekf1_sqr_loop_preconditioned_sequential(
            mu_0, Sigma_0_sqr, prior, measure, tspan, N
        )

        m_joint = np.array(result_joint[0])
        m_seq = np.array(result_seq[0])

        assert np.allclose(m_seq, m_joint, atol=1e-10)
        ll_ode = float(result_seq[-2])
        assert ll_ode == pytest.approx(float(result_joint[-1]), rel=1e-10)

    def test_precond_sequential_with_observations(self):
        """Preconditioned sequential with observations produces finite results."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = PrecondIWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        ts = np.linspace(tspan[0], tspan[1], N + 1)
        z_t = ts[5:15]
        z = np.array([[0.1 + 0.05 * i] for i in range(10)])
        A = np.array([[1.0]])
        observation = Measurement(A, z, z_t, noise=0.01)

        result = ekf1_sqr_loop_preconditioned_sequential(
            mu_0,
            Sigma_0_sqr,
            prior,
            measure,
            tspan,
            N,
            observations=[observation],
        )

        m_seq = np.array(result[0])
        ll_ode = float(result[-2])
        ll_obs = float(result[-1])

        assert m_seq.shape == (N + 1, q * d + d)
        assert np.all(np.isfinite(m_seq))
        assert np.isfinite(ll_ode)
        assert np.isfinite(ll_obs)
        assert ll_obs != 0.0
