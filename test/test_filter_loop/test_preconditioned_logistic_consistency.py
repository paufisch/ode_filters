import jax.numpy as jnp
import numpy as np
import pytest

from ode_filters.filters.ODE_filter_loop import (
    ekf1_sqr_loop,
    ekf1_sqr_loop_preconditioned,
    rts_sqr_smoother_loop,
    rts_sqr_smoother_loop_preconditioned,
)
from ode_filters.measurement.measurement_models import ODEInformation
from ode_filters.priors.GMP_priors import (
    IWP,
    PrecondIWP,
    taylor_mode_initialization,
)


def _upper_cholesky(matrix: np.ndarray) -> np.ndarray:
    lower = np.linalg.cholesky(matrix)
    return lower.T


def _reconstruct_covariance(factors: np.ndarray) -> np.ndarray:
    return np.matmul(factors.transpose(0, 2, 1), factors)


def _logistic_vf(x):
    return x * (1.0 - x)


def _lotka_volterra_vf(x):
    return jnp.array(
        [
            2.0 * x[0] / 3.0 - 4.0 * x[0] * x[1] / 3.0,
            x[0] * x[1] - x[1],
        ]
    )


def _sir_vf(x, beta=0.5, gamma=0.1):
    return jnp.array(
        [
            -beta * x[0] * x[1],
            beta * x[0] * x[1] - gamma * x[1],
            gamma * x[1],
        ]
    )


_EXAMPLES = (
    {
        "name": "logistic",
        "vf": _logistic_vf,
        "x0": np.array([0.01]),
        "t_span": (0.0, 10.0),
        "xi_scale": 0.5,
        "num_steps": 21,
    },
    {
        "name": "lotka_volterra",
        "vf": _lotka_volterra_vf,
        "x0": np.array([1.0, 1.0]),
        "t_span": (0.0, 30.0),
        "xi_scale": 1.0,
        "num_steps": 60,
    },
    {
        "name": "sir",
        "vf": _sir_vf,
        "x0": np.array([0.99, 0.01, 0.0]),
        "t_span": (0.0, 100.0),
        "xi_scale": 1.0,
        "num_steps": 100,
    },
)


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda ex: ex["name"])
def test_preconditioned_matches_standard_outputs(example):
    q = 2
    vf = example["vf"]
    x0 = example["x0"]
    t0, t1 = example["t_span"]
    num_steps = example["num_steps"]

    d = x0.shape[0]
    D = d * (q + 1)
    xi = example["xi_scale"] * np.eye(d)

    mu_0, Sigma_0_sqr = taylor_mode_initialization(vf, x0, q)

    _, h = np.linspace(t0, t1, num_steps + 1, retstep=True)

    prior = IWP(q, d, Xi=xi)
    A_h = prior.A(h)
    Q_h_sqr = _upper_cholesky(prior.Q(h))
    b_h = np.zeros(D)

    prior_precond = PrecondIWP(q, d, Xi=xi)
    A_bar = prior_precond.A()
    Q_sqr_bar = _upper_cholesky(prior_precond.Q())
    T_h = prior_precond.T(h)
    b_bar = np.zeros(D)

    measure = ODEInformation(vf, d=d, q=q)
    g = measure.g
    jacobian_g = measure.jacobian_g

    z_sequence = np.zeros((num_steps, d))
    R_h_sqr = np.zeros((d, d))

    (
        m_seq_standard,
        P_seq_sqr_standard,
        _,
        _,
        G_back_standard,
        d_back_standard,
        P_back_sqr_standard,
        _,
        _,
    ) = ekf1_sqr_loop(
        mu_0,
        Sigma_0_sqr,
        A_h,
        b_h,
        Q_h_sqr,
        R_h_sqr,
        g,
        jacobian_g,
        z_sequence,
        num_steps,
    )

    (
        m_seq_precond,
        P_seq_sqr_precond,
        m_seq_bar,
        P_seq_sqr_bar,
        _,
        _,
        G_back_bar,
        d_back_bar,
        P_back_sqr_bar,
        _,
        _,
    ) = ekf1_sqr_loop_preconditioned(
        mu_0,
        Sigma_0_sqr,
        T_h,
        A_bar,
        b_bar,
        Q_sqr_bar,
        R_h_sqr,
        g,
        jacobian_g,
        z_sequence,
        num_steps,
    )

    m_smoothed_standard, P_smoothed_sqr_standard = rts_sqr_smoother_loop(
        m_seq_standard[-1],
        P_seq_sqr_standard[-1],
        G_back_standard,
        d_back_standard,
        P_back_sqr_standard,
        num_steps,
    )

    m_smoothed_precond, P_smoothed_sqr_precond = rts_sqr_smoother_loop_preconditioned(
        m_seq_precond[-1],
        P_seq_sqr_precond[-1],
        m_seq_bar[-1],
        P_seq_sqr_bar[-1],
        G_back_bar,
        d_back_bar,
        P_back_sqr_bar,
        num_steps,
        T_h,
    )

    P_seq_standard = _reconstruct_covariance(P_seq_sqr_standard)
    P_seq_precond = _reconstruct_covariance(P_seq_sqr_precond)
    P_smoothed_standard = _reconstruct_covariance(P_smoothed_sqr_standard)
    P_smoothed_precond = _reconstruct_covariance(P_smoothed_sqr_precond)

    assert m_seq_precond == pytest.approx(m_seq_standard, rel=1e-1)
    assert P_seq_precond == pytest.approx(P_seq_standard, rel=1e-1)
    assert m_smoothed_precond == pytest.approx(m_smoothed_standard, rel=1e-1)
    assert P_smoothed_precond == pytest.approx(P_smoothed_standard, rel=1e-1)
