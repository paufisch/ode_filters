"""Tests verifying ODE solver accuracy against known solutions and references.

Convergence orders are checked in ``test_convergence_order.py``, the size of
the posterior uncertainty in ``test_uncertainty_calibration.py``.
"""

import jax
import jax.numpy as np
import pytest
from scipy.integrate import solve_ivp

from ode_filters import gaussian_filter, rts_smoother
from ode_filters.information import ODEInformation
from ode_filters.priors import IWP, PrecondIWP, taylor_mode_initialization

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solve_standard(vf, x0, *, tspan, N, q=3, Xi_scale=0.5):
    """Run standard EKF filter + RTS smoother, return solution arrays.

    Returns:
        ts: Time grid, shape [N+1].
        m_filter: Filtered solution (E0 @ state), shape [N+1, d].
        P_filter_sqr: Square-root factor F of the filtered marginal covariance
            of E0 @ state (cov = F.T @ F), shape [N+1, D, d].
        m_smooth: Smoothed solution (E0 @ state), shape [N+1, d].
        P_smooth_sqr: Same as P_filter_sqr for the smoothed marginal.
    """
    d = x0.shape[0]
    prior = IWP(q=q, d=d, Xi=Xi_scale * np.eye(d))
    m_0, P_0_sqr = taylor_mode_initialization(vf, x0, q)
    measure = ODEInformation(vf, prior.E0, prior.E1)

    result = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*tspan, N + 1),
        calibration="none",
    )

    m_seq = np.array(result.m)
    P_seq_sqr_arr = np.array(result.P_sqr)

    m_smooth, P_smooth_sqr = rts_smoother(result)

    E0 = prior.E0
    ts = np.linspace(tspan[0], tspan[1], N + 1)
    m_filter = np.einsum("ij,nj->ni", E0, m_seq)
    m_smooth_x = np.einsum("ij,nj->ni", E0, m_smooth)

    # P = P_sqr.T @ P_sqr, so E0 P E0.T = (P_sqr @ E0.T).T @ (P_sqr @ E0.T):
    # the marginal factor is P_sqr @ E0.T (right-multiplied), not E0 @ P_sqr.
    P_filter_sqr = P_seq_sqr_arr @ E0.T
    P_smooth_sqr_x = P_smooth_sqr @ E0.T

    return ts, m_filter, P_filter_sqr, m_smooth_x, P_smooth_sqr_x


def _solve_preconditioned(vf, x0, *, tspan, N, q=3, Xi_scale=0.5):
    """Run preconditioned EKF filter + RTS smoother, return solution arrays.

    Returns same structure as _solve_standard.
    """
    d = x0.shape[0]
    prior = PrecondIWP(q=q, d=d, Xi=Xi_scale * np.eye(d))
    m_0, P_0_sqr = taylor_mode_initialization(vf, x0, q)
    measure = ODEInformation(vf, prior.E0, prior.E1)

    result = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*tspan, N + 1),
        calibration="none",
    )

    m_seq = np.array(result.m)
    P_seq_sqr_arr = np.array(result.P_sqr)

    m_smooth, P_smooth_sqr = rts_smoother(result)

    E0 = prior.E0
    ts = np.linspace(tspan[0], tspan[1], N + 1)
    m_filter = np.einsum("ij,nj->ni", E0, m_seq)
    m_smooth_x = np.einsum("ij,nj->ni", E0, m_smooth)

    # P = P_sqr.T @ P_sqr, so E0 P E0.T = (P_sqr @ E0.T).T @ (P_sqr @ E0.T):
    # the marginal factor is P_sqr @ E0.T (right-multiplied), not E0 @ P_sqr.
    P_filter_sqr = P_seq_sqr_arr @ E0.T
    P_smooth_sqr_x = P_smooth_sqr @ E0.T

    return ts, m_filter, P_filter_sqr, m_smooth_x, P_smooth_sqr_x


SOLVERS = [
    pytest.param(_solve_standard, id="standard"),
    pytest.param(_solve_preconditioned, id="preconditioned"),
]


# ---------------------------------------------------------------------------
# ODE problems
# ---------------------------------------------------------------------------


def _vf_exp_decay(x, *, t):
    """Exponential decay: dx/dt = -x."""
    return -x


def _vf_logistic(x, *, t):
    """Logistic ODE: dx/dt = x(1-x)."""
    return x * (1 - x)


def _vf_linear_2d(x, *, t):
    """2D linear system: dx/dt = A @ x."""
    A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
    return A @ x


def _vf_lotka_volterra(x, *, t, a=1.5, b=1.0, c=3.0, d=1.0):
    """Lotka-Volterra predator-prey system."""
    return np.array([a * x[0] - b * x[0] * x[1], -c * x[1] + d * x[0] * x[1]])


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
# Category 1: Analytical Solutions
# ---------------------------------------------------------------------------


class TestAnalyticalSolutions:
    """Test solver accuracy against problems with known closed-form solutions."""

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_exponential_decay(self, solver):
        """Solve y' = -y, y(0) = 1. Exact: y(t) = exp(-t)."""
        x0 = np.array([1.0])
        tspan = (0.0, 5.0)
        N = 100

        ts, _, _, m_smooth, _ = solver(_vf_exp_decay, x0, tspan=tspan, N=N, q=3)
        y_exact = np.exp(-ts)[:, None]

        max_err = float(np.max(np.abs(m_smooth - y_exact)))
        # Measured 5.7e-10; the bound leaves ~200x headroom.
        assert max_err < 1e-7, f"Exponential decay max error: {max_err}"

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_logistic(self, solver):
        """Solve y' = y(1-y), y(0) = 0.1. Exact: y(t) = 1/(1 + 9*exp(-t))."""
        x0 = np.array([0.1])
        tspan = (0.0, 5.0)
        N = 100

        ts, _, _, m_smooth, _ = solver(_vf_logistic, x0, tspan=tspan, N=N, q=3)
        y_exact = (1.0 / (1.0 + 9.0 * np.exp(-ts)))[:, None]

        max_err = float(np.max(np.abs(m_smooth - y_exact)))
        # Measured 1.4e-9; the bound leaves ~70x headroom.
        assert max_err < 1e-7, f"Logistic max error: {max_err}"

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_linear_system(self, solver):
        """Solve y' = Ay with A = [[-0.5, 1], [-1, -0.5]].

        Exact: y(t) = expm(A*t) @ y0.
        """
        x0 = np.array([1.0, 0.0])
        tspan = (0.0, 5.0)
        N = 150

        ts, _, _, m_smooth, _ = solver(_vf_linear_2d, x0, tspan=tspan, N=N, q=3)

        A = np.array([[-0.5, 1.0], [-1.0, -0.5]])
        y_exact = jax.vmap(lambda t: jax.scipy.linalg.expm(A * t) @ x0)(ts)

        max_err = float(np.max(np.abs(m_smooth - y_exact)))
        # Measured 1.8e-9; the bound leaves ~55x headroom.
        assert max_err < 1e-7, f"Linear system max error: {max_err}"


# ---------------------------------------------------------------------------
# Category 2: Reference Solver Comparison
# ---------------------------------------------------------------------------


def _scipy_reference(vf_scipy, x0_np, tspan, t_eval):
    """Compute a high-accuracy reference solution using scipy DOP853."""
    sol = solve_ivp(
        vf_scipy,
        tspan,
        x0_np,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-12,
        atol=1e-12,
    )
    return sol.y.T  # shape [len(t_eval), d]


class TestReferenceSolverComparison:
    """Compare solver output against scipy DOP853 at tight tolerances."""

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_lotka_volterra_vs_scipy(self, solver):
        """Lotka-Volterra: compare smoothed mean against scipy."""
        x0 = np.array([10.0, 5.0])
        tspan = (0.0, 10.0)
        N = 500

        ts, _, _, m_smooth, _ = solver(_vf_lotka_volterra, x0, tspan=tspan, N=N, q=3)

        def vf_scipy(t, x):
            a, b, c, d = 1.5, 1.0, 3.0, 1.0
            return [a * x[0] - b * x[0] * x[1], -c * x[1] + d * x[0] * x[1]]

        y_ref = _scipy_reference(
            vf_scipy, [10.0, 5.0], (0.0, 10.0), list(float(t) for t in ts)
        )

        rel_err = float(np.max(np.abs(m_smooth - y_ref) / (np.abs(y_ref) + 1e-10)))
        # Measured 2.2e-2 (the coarse grid limits accuracy on this oscillator).
        assert rel_err < 5e-2, f"Lotka-Volterra max relative error: {rel_err}"

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_sir_vs_scipy(self, solver):
        """SIR model: compare smoothed mean against scipy."""
        x0 = np.array([0.99, 0.01, 0.0])
        tspan = (0.0, 30.0)
        N = 300

        ts, _, _, m_smooth, _ = solver(_vf_sir, x0, tspan=tspan, N=N, q=3)

        def vf_scipy(t, x, beta=0.5, gamma=0.1):
            return [
                -beta * x[0] * x[1],
                beta * x[0] * x[1] - gamma * x[1],
                gamma * x[1],
            ]

        y_ref = _scipy_reference(
            vf_scipy, [0.99, 0.01, 0.0], (0.0, 30.0), list(float(t) for t in ts)
        )

        max_abs_err = float(np.max(np.abs(m_smooth - y_ref)))
        # Measured 3.6e-11; the bound leaves ~140x headroom.
        assert max_abs_err < 5e-9, f"SIR max absolute error: {max_abs_err}"


# ---------------------------------------------------------------------------
# Category 3: Probabilistic Properties
# ---------------------------------------------------------------------------


class TestProbabilisticProperties:
    """Test properties specific to probabilistic ODE solvers."""

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_smoother_improves_on_filter(self, solver):
        """Smoothed solution should have lower or equal RMSE vs filter."""
        x0 = np.array([1.0])
        tspan = (0.0, 5.0)
        N = 100

        ts, m_filter, _, m_smooth, _ = solver(_vf_exp_decay, x0, tspan=tspan, N=N, q=3)
        y_exact = np.exp(-ts)[:, None]

        rmse_filter = float(np.sqrt(np.mean((m_filter - y_exact) ** 2)))
        rmse_smooth = float(np.sqrt(np.mean((m_smooth - y_exact) ** 2)))

        assert rmse_smooth <= rmse_filter + 1e-10, (
            f"Smoother RMSE ({rmse_smooth:.2e}) > filter RMSE ({rmse_filter:.2e})"
        )

    @pytest.mark.parametrize("solver", SOLVERS)
    def test_uncertainty_decreases_with_more_steps(self, solver):
        """More steps should produce smaller posterior uncertainty."""
        x0 = np.array([1.0])
        tspan = (0.0, 2.0)

        n_coarse, n_fine = 50, 200
        _, _, _, _, P_smooth_coarse = solver(
            _vf_exp_decay, x0, tspan=tspan, N=n_coarse, q=3
        )
        _, _, _, _, P_smooth_fine = solver(
            _vf_exp_decay, x0, tspan=tspan, N=n_fine, q=3
        )

        # Compare posterior variance at the same physical time t=1.0 (the grid
        # midpoint of each run), computed from N rather than hard-coded so the
        # test is robust to grid changes.
        var_coarse = float(np.sum(P_smooth_coarse[n_coarse // 2] ** 2))
        var_fine = float(np.sum(P_smooth_fine[n_fine // 2] ** 2))

        assert var_fine < var_coarse, (
            f"Fine variance ({var_fine:.2e}) >= coarse variance ({var_coarse:.2e})"
        )
