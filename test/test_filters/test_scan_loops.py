"""Tests for jax.lax.scan-based filter and smoother loops."""

import jax
import jax.numpy as np
import pytest

from ode_filters.filters.ode_filter_loop import (
    ekf1_sqr_loop,
    ekf1_sqr_loop_scan,
    rts_sqr_smoother_loop,
    rts_sqr_smoother_loop_scan,
)
from ode_filters.measurement.measurement_models import (
    Conservation,
    Measurement,
    ODEInformation,
    ScanData,
)
from ode_filters.priors.gmp_priors import IWP, taylor_mode_initialization


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
# Test: scan loop matches original loop (no measurements)
# ---------------------------------------------------------------------------


class TestScanLoopMatchesOriginal:
    """Verify that scan-based loops produce identical results to original loops."""

    def test_ekf1_sqr_loop_scan_no_measurements(self):
        """Test scan loop without measurements matches original."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        result_loop = ekf1_sqr_loop(mu_0, Sigma_0_sqr, prior, measure, tspan, N)
        result_scan = ekf1_sqr_loop_scan(mu_0, Sigma_0_sqr, prior, measure, tspan, N)

        m_loop = np.array(result_loop[0])
        m_scan = result_scan.m_seq

        assert m_scan.shape == m_loop.shape
        assert np.allclose(m_scan, m_loop, atol=1e-6)
        assert float(result_scan.log_likelihood) == pytest.approx(
            float(result_loop[-1]), rel=1e-6
        )

    def test_ekf1_sqr_loop_scan_with_measurements(self):
        """Test scan loop with sparse measurements matches original."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)

        ts = np.linspace(tspan[0], tspan[1], N + 1)
        z_t = ts[5:15]
        z = np.array([[0.1 + 0.05 * i] for i in range(10)])
        A = np.array([[1.0]])
        measurement = Measurement(A, z, z_t, noise=0.01)
        measure = ODEInformation(
            _vf_logistic, prior.E0, prior.E1, constraints=[measurement]
        )

        result_loop = ekf1_sqr_loop(mu_0, Sigma_0_sqr, prior, measure, tspan, N)
        result_scan = ekf1_sqr_loop_scan(mu_0, Sigma_0_sqr, prior, measure, tspan, N)

        m_loop = np.array(result_loop[0])
        m_scan = result_scan.m_seq

        assert m_scan.shape == m_loop.shape
        assert np.allclose(m_scan, m_loop, atol=1e-6)
        assert float(result_scan.log_likelihood) == pytest.approx(
            float(result_loop[-1]), rel=1e-6
        )

    def test_ekf1_sqr_loop_scan_with_conservation(self):
        """Test scan loop with conservation constraint matches original."""
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

        result_loop = ekf1_sqr_loop(mu_0, Sigma_0_sqr, prior, measure, tspan, N)
        result_scan = ekf1_sqr_loop_scan(mu_0, Sigma_0_sqr, prior, measure, tspan, N)

        m_loop = np.array(result_loop[0])
        m_scan = result_scan.m_seq

        assert m_scan.shape == m_loop.shape
        assert np.allclose(m_scan, m_loop, atol=1e-6)
        assert float(result_scan.log_likelihood) == pytest.approx(
            float(result_loop[-1]), rel=1e-5
        )

    def test_rts_smoother_scan_matches_original(self):
        """Test scan-based smoother matches original."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 20
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        result_filter = ekf1_sqr_loop_scan(mu_0, Sigma_0_sqr, prior, measure, tspan, N)

        # Original smoother
        m_smooth_loop, P_smooth_loop = rts_sqr_smoother_loop(
            result_filter.m_seq[-1],
            result_filter.P_seq_sqr[-1],
            result_filter.G_back_seq,
            result_filter.d_back_seq,
            result_filter.P_back_seq_sqr,
            N,
        )

        # Scan smoother
        m_smooth_scan, P_smooth_scan = rts_sqr_smoother_loop_scan(
            result_filter.m_seq[-1],
            result_filter.P_seq_sqr[-1],
            result_filter.G_back_seq,
            result_filter.d_back_seq,
            result_filter.P_back_seq_sqr,
        )

        assert m_smooth_scan.shape == m_smooth_loop.shape
        assert np.allclose(m_smooth_scan, m_smooth_loop, atol=1e-6)
        assert np.allclose(P_smooth_scan, P_smooth_loop, atol=1e-6)


# ---------------------------------------------------------------------------
# Test: ScanData structure
# ---------------------------------------------------------------------------


class TestScanDataStructure:
    """Test the ScanData structure and prepare_scan_data method."""

    def test_scan_data_shapes_no_measurements(self):
        """Test ScanData shapes when no measurements are present."""
        q = 2
        d = 1
        N = 10

        prior = IWP(q=q, d=d)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        ts = np.linspace(0.0, 1.0, N + 1)
        scan_data = measure.prepare_scan_data(ts)

        assert isinstance(scan_data, ScanData)
        assert scan_data.H_meas.shape == (0, prior.E0.shape[1])
        assert scan_data.H_cons.shape == (0, prior.E0.shape[1])
        assert scan_data.c_meas.shape == (N, 0)
        assert scan_data.R_meas_sqr.shape == (N, 0, 0)
        assert scan_data.R_fixed_sqr.shape == (d, d)
        assert scan_data.obs_mask.shape == (N, d)
        assert scan_data.max_obs_dim == d
        assert scan_data.fixed_dim == d
        assert scan_data.ts.shape == (N,)

    def test_scan_data_shapes_with_measurements(self):
        """Test ScanData shapes when measurements are present."""
        q = 2
        d = 1
        N = 10

        prior = IWP(q=q, d=d)
        ts = np.linspace(0.0, 1.0, N + 1)

        z_t = ts[3:7]
        z = np.array([[0.1], [0.2], [0.3], [0.4]])
        A = np.array([[1.0]])
        measurement = Measurement(A, z, z_t, noise=0.01)
        measure = ODEInformation(
            _vf_logistic, prior.E0, prior.E1, constraints=[measurement]
        )

        scan_data = measure.prepare_scan_data(ts)

        assert scan_data.H_meas.shape == (1, prior.E0.shape[1])
        assert scan_data.c_meas.shape == (N, 1)
        assert scan_data.R_meas_sqr.shape == (N, 1, 1)
        assert scan_data.obs_mask.shape == (N, d + 1)
        assert scan_data.max_obs_dim == d + 1
        assert scan_data.fixed_dim == d

    def test_scan_data_obs_mask_correctness(self):
        """Test that obs_mask correctly identifies active observations."""
        q = 2
        d = 1
        N = 10

        prior = IWP(q=q, d=d)
        ts = np.linspace(0.0, 1.0, N + 1)

        z_t = ts[3:7]
        z = np.array([[0.1], [0.2], [0.3], [0.4]])
        A = np.array([[1.0]])
        measurement = Measurement(A, z, z_t, noise=0.01)
        measure = ODEInformation(
            _vf_logistic, prior.E0, prior.E1, constraints=[measurement]
        )

        scan_data = measure.prepare_scan_data(ts)

        # ODE observation is always active (first row)
        assert np.all(scan_data.obs_mask[:, 0])

        # Measurement is active at steps 2-5 (indices for ts[3:7])
        expected_meas_active = np.array(
            [False, False, True, True, True, True, False, False, False, False]
        )
        assert np.array_equal(scan_data.obs_mask[:, 1], expected_meas_active)


# ---------------------------------------------------------------------------
# Test: JIT compatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Test that scan loops can be JIT-compiled."""

    def test_filter_scan_body_jittable(self):
        """Test that the filter scan body can be JIT-compiled."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 10
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)

        ts = np.linspace(tspan[0], tspan[1], N + 1)
        z_t = ts[3:7]
        z = np.array([[0.1], [0.2], [0.3], [0.4]])
        A = np.array([[1.0]])
        measurement = Measurement(A, z, z_t, noise=0.01)
        measure = ODEInformation(
            _vf_logistic, prior.E0, prior.E1, constraints=[measurement]
        )

        scan_data = measure.prepare_scan_data(ts)
        h = float(ts[1] - ts[0])
        A_h = prior.A(h)
        b_h = prior.b(h)
        Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T

        @jax.jit
        def jitted_scan(mu_0, Sigma_0_sqr, A_h, b_h, Q_h_sqr, scan_data):
            from ode_filters.filters.ode_filter_step import ekf1_sqr_filter_step_scan

            def scan_body(carry, step_idx):
                m, P_sqr, log_lik = carry
                (
                    (_m_pred, _P_pred_sqr),
                    (_G_back, _d_back, _P_back_sqr),
                    (_mz, _Pz_sqr),
                    (m_new, P_new_sqr),
                ) = ekf1_sqr_filter_step_scan(
                    A_h, b_h, Q_h_sqr, m, P_sqr, measure, step_idx, scan_data
                )
                new_carry = (m_new, P_new_sqr, log_lik)
                return new_carry, m_new

            init_carry = (mu_0, Sigma_0_sqr, 0.0)
            final, m_seq = jax.lax.scan(scan_body, init_carry, np.arange(N))
            return final[0], m_seq

        m_final, m_seq = jitted_scan(mu_0, Sigma_0_sqr, A_h, b_h, Q_h_sqr, scan_data)

        assert m_final.shape == mu_0.shape
        assert m_seq.shape == (N, mu_0.shape[0])

    def test_smoother_scan_jittable(self):
        """Test that the smoother scan can be JIT-compiled."""
        x0 = np.array([0.01])
        tspan = (0.0, 5.0)
        N = 10
        q = 2
        d = 1

        prior = IWP(q=q, d=d, Xi=0.5 * np.eye(d))
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        result = ekf1_sqr_loop_scan(mu_0, Sigma_0_sqr, prior, measure, tspan, N)

        @jax.jit
        def jitted_smoother(m_N, P_N_sqr, G_back_seq, d_back_seq, P_back_seq_sqr):
            return rts_sqr_smoother_loop_scan(
                m_N, P_N_sqr, G_back_seq, d_back_seq, P_back_seq_sqr
            )

        m_smooth, P_smooth_sqr = jitted_smoother(
            result.m_seq[-1],
            result.P_seq_sqr[-1],
            result.G_back_seq,
            result.d_back_seq,
            result.P_back_seq_sqr,
        )

        assert m_smooth.shape == (N + 1, mu_0.shape[0])
        assert P_smooth_sqr.shape == (N + 1, mu_0.shape[0], mu_0.shape[0])


# ---------------------------------------------------------------------------
# Test: Differentiability
# ---------------------------------------------------------------------------


class TestDifferentiability:
    """Test that scan loops are differentiable."""

    def test_log_likelihood_gradient(self):
        """Test that gradient through log-likelihood is finite."""
        x0 = np.array([0.1])
        tspan = (0.0, 1.0)
        N = 5
        q = 2
        d = 1

        prior = IWP(q=q, d=d)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        def lml_fn(Sigma_0_sqr_):
            result = ekf1_sqr_loop_scan(mu_0, Sigma_0_sqr_, prior, measure, tspan, N)
            return result.log_likelihood

        grad_val = jax.grad(lml_fn)(Sigma_0_sqr)
        assert np.all(np.isfinite(grad_val))

    def test_final_state_gradient(self):
        """Test that gradient through final state is finite."""
        x0 = np.array([0.1])
        tspan = (0.0, 1.0)
        N = 5
        q = 2
        d = 1

        prior = IWP(q=q, d=d)
        mu_0, Sigma_0_sqr = taylor_mode_initialization(_vf_logistic, x0, q)
        measure = ODEInformation(_vf_logistic, prior.E0, prior.E1)

        def final_state_fn(mu_0_):
            result = ekf1_sqr_loop_scan(mu_0_, Sigma_0_sqr, prior, measure, tspan, N)
            return np.sum(result.m_seq[-1])

        grad_val = jax.grad(final_state_fn)(mu_0)
        assert np.all(np.isfinite(grad_val))
