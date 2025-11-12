from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.linalg import cholesky

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


def ekf1_sqr_loop(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: object,
    measure: object,
    tspan: tuple[float, float],
    N: int,
) -> LoopResult:
    """Run a square-root EKF over ``N`` observation steps."""

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
    Q_h_sqr = cholesky(prior.Q(h), upper=True)

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
    P_0_sqr: Array,
    prior: object,
    measure: object,
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
    """Run a preconditioned square-root EKF over ``N`` observation steps."""

    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_bar = prior.A()
    b_bar = prior.b()
    Q_sqr_bar = cholesky(prior.Q(), upper=True)
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
