from __future__ import annotations

from collections.abc import Callable

import jax.numpy as np
from jax import Array

from ..measurement.measurement_models import BaseODEInformation
from ..priors.gmp_priors import BasePrior
from .ode_filter_step import (
    ekf1_sqr_filter_step,
    ekf1_sqr_filter_step_preconditioned,
    rts_sqr_smoother_step,
    rts_sqr_smoother_step_preconditioned,
)

StateFunction = Callable[[Array], Array]
JacobianFunction = Callable[[Array], Array]

LoopResult = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]


def ekf1_sqr_loop(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
) -> LoopResult:
    """Run a square-root EKF over ``N`` observation steps.

    Args:
        mu_0: Initial state mean estimate.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (e.g., IWP or PrecondIWP).
        measure: Measurement model (e.g., ODEInformation or subclass).
        tspan: Time interval (t_start, t_end).
        N: Number of filter steps.

    Returns:
        Tuple of 9 arrays containing filtered estimates and intermediate results.
    """

    m_seq = [mu_0]
    P_seq_sqr = [Sigma_0_sqr]
    m_pred_seq = []
    P_pred_seq_sqr = []
    G_back_seq = []
    d_back_seq = []
    P_back_seq_sqr = []
    mz_seq = []
    Pz_seq_sqr = []

    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_h = prior.A(h)
    b_h = prior.b(h)
    Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T

    for i in range(N):
        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz, Pz_sqr),
            (m, P_sqr),
        ) = ekf1_sqr_filter_step(
            A_h,
            b_h,
            Q_h_sqr,
            m_seq[-1],
            P_seq_sqr[-1],
            measure,
            t=ts[i + 1],
        )

        m_pred_seq.append(m_pred)
        P_pred_seq_sqr.append(P_pred_sqr)
        G_back_seq.append(G_back)
        d_back_seq.append(d_back)
        P_back_seq_sqr.append(P_back_sqr)
        mz_seq.append(mz)
        Pz_seq_sqr.append(Pz_sqr)
        m_seq.append(m)
        P_seq_sqr.append(P_sqr)

    return (
        m_seq,
        P_seq_sqr,
        m_pred_seq,
        P_pred_seq_sqr,
        G_back_seq,
        d_back_seq,
        P_back_seq_sqr,
        mz_seq,
        Pz_seq_sqr,
    )


def rts_sqr_smoother_loop(
    m_N: Array,
    P_N_sqr: Array,
    G_back_seq: Array,
    d_back_seq: Array,
    P_back_seq_sqr: Array,
    N: int,
) -> tuple[Array, Array]:
    """Run a Rauch-Tung-Striebel smoother over ``N`` steps.

    Args:
        m_N: Final filtered state mean.
        P_N_sqr: Final filtered state covariance (square-root form).
        G_back_seq: Backward pass gain sequence from filter.
        d_back_seq: Backward pass offset sequence from filter.
        P_back_seq_sqr: Backward pass covariance sequence (square-root form) from filter.
        N: Number of smoothing steps.

    Returns:
        Tuple of smoothed state means and covariances (square-root form).
    """

    state_dim = m_N.shape[0]
    m_smooth = np.zeros((N + 1, state_dim))
    P_smooth_sqr = np.zeros((N + 1, state_dim, state_dim))
    m_smooth = m_smooth.at[-1].set(m_N)
    P_smooth_sqr = P_smooth_sqr.at[-1].set(P_N_sqr)

    for j in range(N - 1, -1, -1):
        m_j, P_j = rts_sqr_smoother_step(
            G_back_seq[j],
            d_back_seq[j],
            P_back_seq_sqr[j],
            m_smooth[j + 1],
            P_smooth_sqr[j + 1],
        )
        m_smooth = m_smooth.at[j].set(m_j)
        P_smooth_sqr = P_smooth_sqr.at[j].set(P_j)

    return m_smooth, P_smooth_sqr


def ekf1_sqr_loop_preconditioned(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
]:
    """Run a preconditioned square-root EKF over ``N`` observation steps.

    Args:
        mu_0: Initial state mean estimate.
        P_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (typically PrecondIWP).
        measure: Measurement model (e.g., ODEInformation or subclass).
        tspan: Time interval (t_start, t_end).
        N: Number of filter steps.

    Returns:
        Tuple of 11 arrays containing original and preconditioned estimates.
    """

    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    # For PrecondIWP, A() and b() don't use h, but accept it for API compatibility
    A_bar = prior.A()
    b_bar = prior.b()
    Q_sqr_bar = np.linalg.cholesky(prior.Q()).T
    T_h = prior.T(h)

    m_seq = [mu_0]
    P_seq_sqr = [P_0_sqr]
    m_seq_bar = [np.linalg.solve(T_h, mu_0)]
    P_seq_sqr_bar = [np.linalg.solve(T_h, P_0_sqr.T).T]

    m_pred_seq_bar = []
    P_pred_seq_sqr_bar = []
    G_back_seq_bar = []
    d_back_seq_bar = []
    P_back_seq_sqr_bar = []
    mz_seq = []
    Pz_seq_sqr = []

    for i in range(N):
        (
            (m_pred_seq_bar_i, P_pred_seq_sqr_bar_i),
            (G_back_seq_bar_i, d_back_seq_bar_i, P_back_seq_sqr_bar_i),
            (mz_seq_i, Pz_seq_sqr_i),
            (m_seq_bar_next, P_seq_sqr_bar_next),
            (m_seq_next, P_seq_sqr_next),
        ) = ekf1_sqr_filter_step_preconditioned(
            A_bar,
            b_bar,
            Q_sqr_bar,
            T_h,
            m_seq_bar[-1],
            P_seq_sqr_bar[-1],
            measure,
            t=ts[i + 1],
        )

        m_pred_seq_bar.append(m_pred_seq_bar_i)
        P_pred_seq_sqr_bar.append(P_pred_seq_sqr_bar_i)
        G_back_seq_bar.append(G_back_seq_bar_i)
        d_back_seq_bar.append(d_back_seq_bar_i)
        P_back_seq_sqr_bar.append(P_back_seq_sqr_bar_i)
        mz_seq.append(mz_seq_i)
        Pz_seq_sqr.append(Pz_seq_sqr_i)
        m_seq_bar.append(m_seq_bar_next)
        P_seq_sqr_bar.append(P_seq_sqr_bar_next)
        m_seq.append(m_seq_next)
        P_seq_sqr.append(P_seq_sqr_next)

    return (
        m_seq,
        P_seq_sqr,
        m_seq_bar,
        P_seq_sqr_bar,
        m_pred_seq_bar,
        P_pred_seq_sqr_bar,
        G_back_seq_bar,
        d_back_seq_bar,
        P_back_seq_sqr_bar,
        mz_seq,
        Pz_seq_sqr,
        T_h,
    )


def rts_sqr_smoother_loop_preconditioned(
    m_N: Array,
    P_N_sqr: Array,
    m_N_bar: Array,
    P_N_sqr_bar: Array,
    G_back_seq_bar: Array,
    d_back_seq_bar: Array,
    P_back_seq_sqr_bar: Array,
    N: int,
    T_h: Array,
) -> tuple[Array, Array]:
    """Run a preconditioned Rauch-Tung-Striebel smoother over ``N`` steps.

    Args:
        m_N: Final filtered state mean (original space).
        P_N_sqr: Final filtered state covariance (square-root form, original space).
        m_N_bar: Final filtered state mean (preconditioned space).
        P_N_sqr_bar: Final filtered state covariance (square-root form, preconditioned space).
        G_back_seq_bar: Backward pass gain sequence (preconditioned).
        d_back_seq_bar: Backward pass offset sequence (preconditioned).
        P_back_seq_sqr_bar: Backward pass covariance sequence (square-root form, preconditioned).
        N: Number of smoothing steps.
        T_h: Preconditioning transformation matrix.

    Returns:
        Smoothed state means and covariances (square-root form, original space).
    """

    state_dim = m_N.shape[0]

    m_smooth = np.zeros((N + 1, state_dim))
    P_smooth_sqr = np.zeros((N + 1, state_dim, state_dim))
    m_smooth = m_smooth.at[-1].set(m_N)
    P_smooth_sqr = P_smooth_sqr.at[-1].set(P_N_sqr)
    m_smooth_bar = np.zeros((N + 1, state_dim))
    P_smooth_sqr_bar = np.zeros((N + 1, state_dim, state_dim))
    m_smooth_bar = m_smooth_bar.at[-1].set(m_N_bar)
    P_smooth_sqr_bar = P_smooth_sqr_bar.at[-1].set(P_N_sqr_bar)

    for j in range(N - 1, -1, -1):
        (m_bar_j, P_bar_j), (m_j, P_j) = rts_sqr_smoother_step_preconditioned(
            G_back_seq_bar[j],
            d_back_seq_bar[j],
            P_back_seq_sqr_bar[j],
            m_smooth_bar[j + 1],
            P_smooth_sqr_bar[j + 1],
            T_h,
        )
        m_smooth_bar = m_smooth_bar.at[j].set(m_bar_j)
        P_smooth_sqr_bar = P_smooth_sqr_bar.at[j].set(P_bar_j)
        m_smooth = m_smooth.at[j].set(m_j)
        P_smooth_sqr = P_smooth_sqr.at[j].set(P_j)

    return m_smooth, P_smooth_sqr
