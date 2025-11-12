from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .ODE_filter_step import (
    ekf1_sqr_filter_step,
    ekf1_sqr_filter_step_preconditioned,
    rts_sqr_smoother_step,
    rts_sqr_smoother_step_preconditioned,
)

Array = np.ndarray
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


# constant step size model actually
def ekf1_sqr_loop(
    mu_0: Array,
    Sigma_0_sqr: Array,
    A_h: Array,
    b_h: Array,
    Q_h_sqr: Array,
    R_h_sqr: Array,
    measure: object,
    N: int,
    ts: Array,
) -> LoopResult:
    """Run a square-root EKF over ``N`` observation steps."""

    state_dim = mu_0.shape[0]
    obs_dim = R_h_sqr.shape[0]

    m_seq = np.empty((N + 1, state_dim))
    P_seq_sqr = np.empty((N + 1, state_dim, state_dim))
    m_pred_seq = np.empty((N, state_dim))
    P_pred_seq_sqr = np.empty((N, state_dim, state_dim))
    G_back_seq = np.empty((N, state_dim, state_dim))
    d_back_seq = np.empty((N, state_dim))
    P_back_seq_sqr = np.empty((N, state_dim, state_dim))
    mz_seq = np.empty((N, obs_dim))
    Pz_seq_sqr = np.empty((N, obs_dim, obs_dim))

    m_seq[0] = mu_0
    P_seq_sqr[0] = Sigma_0_sqr

    for i in range(N):
        (
            (m_pred_seq[i], P_pred_seq_sqr[i]),
            (G_back_seq[i], d_back_seq[i], P_back_seq_sqr[i]),
            (mz_seq[i], Pz_seq_sqr[i]),
            (m_seq[i + 1], P_seq_sqr[i + 1]),
        ) = ekf1_sqr_filter_step(
            A_h,
            b_h,
            Q_h_sqr,
            m_seq[i],
            P_seq_sqr[i],
            measure,
            R_h_sqr,
            t=ts[i + 1],
        )

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
    """Run a Rauch–Tung–Striebel smoother over ``N`` steps."""

    state_dim = m_N.shape[0]
    m_smooth = np.empty((N + 1, state_dim))
    P_smooth_sqr = np.empty((N + 1, state_dim, state_dim))
    m_smooth[-1] = m_N
    P_smooth_sqr[-1] = P_N_sqr

    for j in range(N - 1, -1, -1):
        (m_smooth[j], P_smooth_sqr[j]) = rts_sqr_smoother_step(
            G_back_seq[j],
            d_back_seq[j],
            P_back_seq_sqr[j],
            m_smooth[j + 1],
            P_smooth_sqr[j + 1],
        )

    return m_smooth, P_smooth_sqr


def ekf1_sqr_loop_preconditioned(
    mu_0: Array,
    Sigma_0_sqr: Array,
    T_h: Array,
    A_bar: Array,
    b_bar: Array,
    Q_sqr_bar: Array,
    R_h_sqr: Array,
    measure: object,
    N: int,
    ts: Array,
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
    """Run a preconditioned square-root EKF over ``N`` observation steps."""

    state_dim = mu_0.shape[0]
    obs_dim = R_h_sqr.shape[0]

    m_seq_bar = np.empty((N + 1, state_dim))
    P_seq_sqr_bar = np.empty((N + 1, state_dim, state_dim))
    m_seq = np.empty((N + 1, state_dim))
    P_seq_sqr = np.empty((N + 1, state_dim, state_dim))
    m_pred_seq_bar = np.empty((N, state_dim))
    P_pred_seq_sqr_bar = np.empty((N, state_dim, state_dim))
    G_back_seq_bar = np.empty((N, state_dim, state_dim))
    d_back_seq_bar = np.empty((N, state_dim))
    P_back_seq_sqr_bar = np.empty((N, state_dim, state_dim))
    mz_seq = np.empty((N, obs_dim))
    Pz_seq_sqr = np.empty((N, obs_dim, obs_dim))

    m_seq[0] = mu_0
    P_seq_sqr[0] = Sigma_0_sqr

    m_seq_bar[0] = np.linalg.solve(T_h, mu_0)
    P_seq_sqr_bar[0] = np.linalg.solve(T_h, Sigma_0_sqr.T).T

    for i in range(N):
        (
            (m_pred_seq_bar[i], P_pred_seq_sqr_bar[i]),
            (G_back_seq_bar[i], d_back_seq_bar[i], P_back_seq_sqr_bar[i]),
            (mz_seq[i], Pz_seq_sqr[i]),
            (m_seq_bar[i + 1], P_seq_sqr_bar[i + 1]),
            (m_seq[i + 1], P_seq_sqr[i + 1]),
        ) = ekf1_sqr_filter_step_preconditioned(
            A_bar,
            b_bar,
            Q_sqr_bar,
            T_h,
            m_seq_bar[i],
            P_seq_sqr_bar[i],
            measure,
            R_h_sqr,
            t=ts[i + 1],
        )

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
    """Run a preconditioned Rauch–Tung–Striebel smoother over ``N`` steps."""

    state_dim = m_N.shape[0]

    m_smooth = np.empty((N + 1, state_dim))
    P_smooth_sqr = np.empty((N + 1, state_dim, state_dim))
    m_smooth[-1] = m_N
    P_smooth_sqr[-1] = P_N_sqr
    m_smooth_bar = np.empty((N + 1, state_dim))
    P_smooth_sqr_bar = np.empty((N + 1, state_dim, state_dim))
    m_smooth_bar[-1] = m_N_bar
    P_smooth_sqr_bar[-1] = P_N_sqr_bar

    for j in range(N - 1, -1, -1):
        (m_smooth_bar[j], P_smooth_sqr_bar[j]), (m_smooth[j], P_smooth_sqr[j]) = (
            rts_sqr_smoother_step_preconditioned(
                G_back_seq_bar[j],
                d_back_seq_bar[j],
                P_back_seq_sqr_bar[j],
                m_smooth_bar[j + 1],
                P_smooth_sqr_bar[j + 1],
                T_h,
            )
        )

    return m_smooth, P_smooth_sqr
