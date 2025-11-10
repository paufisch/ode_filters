from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..inference.sqr_gaussian_inference import sqr_inversion, sqr_marginalization

Array = np.ndarray
StateFunction = Callable[[Array], Array]
JacobianFunction = Callable[[Array], Array]

FilterStepResult = tuple[
    tuple[Array, Array],
    tuple[Array, Array, Array],
    tuple[Array, Array],
    tuple[Array, Array],
]

PreconditionedFilterStepResult = tuple[
    tuple[Array, Array],
    tuple[Array, Array, Array],
    tuple[Array, Array],
    tuple[Array, Array],
    tuple[Array, Array],
]


# all covariance matrices are saved and propagated in sqr form.
# E.g. A = A_sqr.T @ A_sqr
def ekf1_sqr_filter_step(
    A_t: Array,
    b_t: Array,
    Q_t_sqr: Array,
    m_prev: Array,
    P_prev_sqr: Array,
    g: StateFunction,
    jacobian_g: JacobianFunction,
    z_observed_t: Array,
    R_t_sqr: Array,
) -> FilterStepResult:
    """Perform a single square-root EKF prediction and update step."""

    m_pred, P_pred_sqr = sqr_marginalization(A_t, b_t, Q_t_sqr, m_prev, P_prev_sqr)
    G_back, d_back, P_back_sqr = sqr_inversion(
        A_t, m_prev, P_prev_sqr, m_pred, P_pred_sqr, Q_t_sqr
    )

    H_t = jacobian_g(m_pred)
    c_t = g(m_pred) - H_t @ m_pred

    m_z, P_z_sqr = sqr_marginalization(H_t, c_t, R_t_sqr, m_pred, P_pred_sqr)
    K_t, d, P_t_sqr = sqr_inversion(H_t, m_pred, P_pred_sqr, m_z, P_z_sqr, R_t_sqr)
    m_t = K_t @ z_observed_t + d

    return (
        (m_pred, P_pred_sqr),
        (G_back, d_back, P_back_sqr),
        (m_z, P_z_sqr),
        (m_t, P_t_sqr),
    )


def rts_sqr_smoother_step(
    G_back: Array,
    d_back: Array,
    P_back_sqr: Array,
    m_s: Array,
    P_s_sqr: Array,
) -> tuple[Array, Array]:
    m_s_prev, P_s_prev_sqr = sqr_marginalization(
        G_back, d_back, P_back_sqr, m_s, P_s_sqr
    )
    return (m_s_prev, P_s_prev_sqr)


# pre conditioning version of ekf1_sqr_filter_step
# T is a preconditioner with x_bar = T^-1 x
# A, Q and b are stepsize independent in the transformed space
# the dependence is essentially absorbed into T
def ekf1_sqr_filter_step_preconditioned(
    A_bar: Array,
    b_bar: Array,
    Q_sqr_bar: Array,
    T_t: Array,
    m_prev_bar: Array,
    P_prev_sqr_bar: Array,
    g: StateFunction,
    jacobian_g: JacobianFunction,
    z_observed_t: Array,
    R_t_sqr: Array,
) -> PreconditionedFilterStepResult:
    """Perform a single preconditioned square-root EKF step."""

    m_pred_bar, P_pred_sqr_bar = sqr_marginalization(
        A_bar, b_bar, Q_sqr_bar, m_prev_bar, P_prev_sqr_bar
    )
    G_back_bar, d_back_bar, P_back_sqr_bar = sqr_inversion(
        A_bar, m_prev_bar, P_prev_sqr_bar, m_pred_bar, P_pred_sqr_bar, Q_sqr_bar
    )

    H_t_bar = jacobian_g(T_t @ m_pred_bar) @ T_t
    c_t = g(T_t @ m_pred_bar) - H_t_bar @ m_pred_bar

    m_z, P_z_sqr = sqr_marginalization(
        H_t_bar, c_t, R_t_sqr, m_pred_bar, P_pred_sqr_bar
    )
    K_t_bar, d_bar, P_t_sqr_bar = sqr_inversion(
        H_t_bar, m_pred_bar, P_pred_sqr_bar, m_z, P_z_sqr, R_t_sqr
    )
    m_t_bar = K_t_bar @ z_observed_t + d_bar

    m_t = T_t @ m_t_bar
    P_t_sqr = P_t_sqr_bar @ T_t.T

    return (
        (m_pred_bar, P_pred_sqr_bar),
        (G_back_bar, d_back_bar, P_back_sqr_bar),
        (m_z, P_z_sqr),
        (m_t_bar, P_t_sqr_bar),
        (m_t, P_t_sqr),
    )


def rts_sqr_smoother_step_preconditioned(
    G_back_bar: Array,
    d_back_bar: Array,
    P_back_sqr_bar: Array,
    m_s_bar: Array,
    P_s_sqr_bar: Array,
    T_t: Array,
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    m_s_prev_bar, P_s_prev_sqr_bar = sqr_marginalization(
        G_back_bar, d_back_bar, P_back_sqr_bar, m_s_bar, P_s_sqr_bar
    )
    m_s_prev = T_t @ m_s_prev_bar
    P_s_prev_sqr = P_s_prev_sqr_bar @ T_t.T
    return (m_s_prev_bar, P_s_prev_sqr_bar), (m_s_prev, P_s_prev_sqr)
