from __future__ import annotations

from collections.abc import Callable

import jax.numpy as np
from jax import Array

from ..inference.sqr_gaussian_inference import sqr_inversion, sqr_marginalization
from ..measurement.measurement_models import BaseODEInformation

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


# All covariance matrices are saved and propagated in square-root form.
# E.g. A = A_sqr.T @ A_sqr
def ekf1_sqr_filter_step(
    A_t: Array,
    b_t: Array,
    Q_t_sqr: Array,
    m_prev: Array,
    P_prev_sqr: Array,
    measure: BaseODEInformation,
    t: float = 0.0,
) -> FilterStepResult:
    """Perform a single square-root EKF prediction and update step.

    Args:
        A_t: State transition matrix for current step.
        b_t: Drift vector for current step.
        Q_t_sqr: Square-root of process noise covariance.
        m_prev: Previous state mean estimate.
        P_prev_sqr: Previous state covariance (square-root form).
        measure: Measurement model (e.g., ODEInformation or subclass).
        t: Current time (default 0.0).

    Returns:
        Tuple of 4 tuples containing prediction, backward pass, and update results.
    """

    m_pred, P_pred_sqr = sqr_marginalization(A_t, b_t, Q_t_sqr, m_prev, P_prev_sqr)
    # this is optional if only filtering is relevant
    G_back, d_back, P_back_sqr = sqr_inversion(
        A_t, m_prev, P_prev_sqr, m_pred, P_pred_sqr, Q_t_sqr
    )

    H_t, c_t = measure.linearize(m_pred, t=t)
    R_t_sqr = measure.get_noise(t=t)

    m_z, P_z_sqr = sqr_marginalization(H_t, c_t, R_t_sqr, m_pred, P_pred_sqr)
    _, d, P_t_sqr = sqr_inversion(H_t, m_pred, P_pred_sqr, m_z, P_z_sqr, R_t_sqr)
    m_t = d  # for no zero measurements: m_t = K_t @ z_observed_t + d

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
    """Perform a single Rauch-Tung-Striebel backward smoothing step.

    Args:
        G_back: Backward pass gain matrix.
        d_back: Backward pass offset vector.
        P_back_sqr: Backward pass covariance (square-root form).
        m_s: Smoothed state mean from next time step.
        P_s_sqr: Smoothed state covariance (square-root form) from next time step.

    Returns:
        Tuple of previous smoothed mean and covariance (square-root form).
    """
    m_s_prev, P_s_prev_sqr = sqr_marginalization(
        G_back, d_back, P_back_sqr, m_s, P_s_sqr
    )
    return (m_s_prev, P_s_prev_sqr)


# Preconditioned version of ekf1_sqr_filter_step
# T is a preconditioner with x_bar = T^-1 x
# A, Q and b are stepsize-independent in the transformed space
# The stepsize dependence is essentially absorbed into T
def ekf1_sqr_filter_step_preconditioned(
    A_bar: Array,
    b_bar: Array,
    Q_sqr_bar: Array,
    T_t: Array,
    m_prev_bar: Array,
    P_prev_sqr_bar: Array,
    measure: BaseODEInformation,
    t: float = 0.0,
) -> PreconditionedFilterStepResult:
    """Perform a single preconditioned square-root EKF step.

    Args:
        A_bar: Stepsize-independent state transition matrix.
        b_bar: Stepsize-independent drift vector.
        Q_sqr_bar: Square-root of stepsize-independent process noise covariance.
        T_t: Preconditioning transformation matrix for current step.
        m_prev_bar: Previous state mean estimate (preconditioned space).
        P_prev_sqr_bar: Previous state covariance (square-root form, preconditioned space).
        measure: Measurement model (e.g., ODEInformation or subclass).
        t: Current time (default 0.0).

    Returns:
        Tuple of 5 tuples with preconditioned and original-space results.
    """

    m_pred_bar, P_pred_sqr_bar = sqr_marginalization(
        A_bar, b_bar, Q_sqr_bar, m_prev_bar, P_prev_sqr_bar
    )
    G_back_bar, d_back_bar, P_back_sqr_bar = sqr_inversion(
        A_bar, m_prev_bar, P_prev_sqr_bar, m_pred_bar, P_pred_sqr_bar, Q_sqr_bar
    )

    H_t, c_t = measure.linearize(T_t @ m_pred_bar, t=t)
    H_t_bar = H_t @ T_t
    R_t_sqr = measure.get_noise(t=t)

    m_z, P_z_sqr = sqr_marginalization(
        H_t_bar, c_t, R_t_sqr, m_pred_bar, P_pred_sqr_bar
    )
    _, d_bar, P_t_sqr_bar = sqr_inversion(
        H_t_bar, m_pred_bar, P_pred_sqr_bar, m_z, P_z_sqr, R_t_sqr
    )
    m_t_bar = d_bar  # for non zero measurements: K_t_bar @ z_observed_t + d_bar

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
    """Perform a single preconditioned Rauch-Tung-Striebel backward smoothing step.

    Args:
        G_back_bar: Backward pass gain matrix (preconditioned space).
        d_back_bar: Backward pass offset vector (preconditioned space).
        P_back_sqr_bar: Backward pass covariance (square-root form, preconditioned space).
        m_s_bar: Smoothed state mean from next step (preconditioned space).
        P_s_sqr_bar: Smoothed state covariance (square-root form, preconditioned space).
        T_t: Preconditioning transformation matrix.

    Returns:
        Tuple containing:
        - (mean, covariance) in preconditioned space
        - (mean, covariance) in original space
    """
    m_s_prev_bar, P_s_prev_sqr_bar = sqr_marginalization(
        G_back_bar, d_back_bar, P_back_sqr_bar, m_s_bar, P_s_sqr_bar
    )
    m_s_prev = T_t @ m_s_prev_bar
    P_s_prev_sqr = P_s_prev_sqr_bar @ T_t.T
    return (m_s_prev_bar, P_s_prev_sqr_bar), (m_s_prev, P_s_prev_sqr)


# =============================================================================
# Sequential update filter step functions
# =============================================================================
# These functions apply the ODE update (ODE + Conservation) first, then
# optionally apply an observation update separately. The observation is
# provided as an external (H, c, R_sqr) tuple rather than being extracted
# from the measurement model, decoupling the ODE model from observation
# scheduling.

SeqFilterStepResult = tuple[
    tuple[Array, Array],  # (m_pred, P_pred_sqr)
    tuple[Array, Array, Array],  # (G_back, d_back, P_back_sqr)
    tuple[Array, Array],  # (mz_ode, Pz_ode_sqr) - ODE obs marginal
    tuple[Array, Array],  # (mz_obs, Pz_obs_sqr) - observation marginal
    tuple[Array, Array],  # (m_t, P_t_sqr) - final updated state
]

SeqPrecondFilterStepResult = tuple[
    tuple[Array, Array],  # (m_pred_bar, P_pred_sqr_bar)
    tuple[Array, Array, Array],  # (G_back_bar, d_back_bar, P_back_sqr_bar)
    tuple[Array, Array],  # (mz_ode, Pz_ode_sqr)
    tuple[Array, Array],  # (mz_obs, Pz_obs_sqr)
    tuple[Array, Array],  # (m_t_bar, P_t_sqr_bar)
    tuple[Array, Array],  # (m_t, P_t_sqr)
]


def ekf1_sqr_filter_step_sequential(
    A_t: Array,
    b_t: Array,
    Q_t_sqr: Array,
    m_prev: Array,
    P_prev_sqr: Array,
    measure: BaseODEInformation,
    t: float = 0.0,
    *,
    obs: tuple[Array, Array, Array] | None = None,
) -> SeqFilterStepResult:
    """Perform a sequential square-root EKF step: ODE update then observation.

    Unlike the joint update in ``ekf1_sqr_filter_step``, this function first
    incorporates ODE + Conservation information, then applies an optional
    observation update using the ODE-updated state as the prior.

    Args:
        A_t: State transition matrix for current step.
        b_t: Drift vector for current step.
        Q_t_sqr: Square-root of process noise covariance.
        m_prev: Previous state mean estimate.
        P_prev_sqr: Previous state covariance (square-root form).
        measure: Measurement model (e.g., ODEInformation or subclass).
        t: Current time (default 0.0).
        obs: Optional observation tuple ``(H_obs, c_obs, R_obs_sqr)``.
            When provided, the observation update uses the ODE-updated
            state ``(m_ode, P_ode_sqr)`` as its prior.

    Returns:
        Tuple of 5 tuples: prediction, backward pass, ODE observation
        marginal, observation marginal, and final updated state.
    """
    # Prediction
    m_pred, P_pred_sqr = sqr_marginalization(A_t, b_t, Q_t_sqr, m_prev, P_prev_sqr)
    G_back, d_back, P_back_sqr = sqr_inversion(
        A_t, m_prev, P_prev_sqr, m_pred, P_pred_sqr, Q_t_sqr
    )

    # ODE + Conservation update
    H_ode, c_ode = measure.linearize_fixed(m_pred, t=t)
    R_ode_sqr = measure.get_fixed_noise_sqr()
    mz_ode, Pz_ode_sqr = sqr_marginalization(
        H_ode, c_ode, R_ode_sqr, m_pred, P_pred_sqr
    )
    _, m_ode, P_ode_sqr = sqr_inversion(
        H_ode, m_pred, P_pred_sqr, mz_ode, Pz_ode_sqr, R_ode_sqr
    )

    # Observation update
    if obs is not None:
        H_obs, c_obs, R_obs_sqr = obs
        mz_obs, Pz_obs_sqr = sqr_marginalization(
            H_obs, c_obs, R_obs_sqr, m_ode, P_ode_sqr
        )
        _, m_t, P_t_sqr = sqr_inversion(
            H_obs, m_ode, P_ode_sqr, mz_obs, Pz_obs_sqr, R_obs_sqr
        )
    else:
        m_t = m_ode
        P_t_sqr = P_ode_sqr
        mz_obs = np.zeros(0)
        Pz_obs_sqr = np.zeros((0, 0))

    return (
        (m_pred, P_pred_sqr),
        (G_back, d_back, P_back_sqr),
        (mz_ode, Pz_ode_sqr),
        (mz_obs, Pz_obs_sqr),
        (m_t, P_t_sqr),
    )


def ekf1_sqr_filter_step_preconditioned_sequential(
    A_bar: Array,
    b_bar: Array,
    Q_sqr_bar: Array,
    T_t: Array,
    m_prev_bar: Array,
    P_prev_sqr_bar: Array,
    measure: BaseODEInformation,
    t: float = 0.0,
    *,
    obs: tuple[Array, Array, Array] | None = None,
) -> SeqPrecondFilterStepResult:
    """Perform a sequential preconditioned square-root EKF step.

    Args:
        A_bar: Stepsize-independent state transition matrix.
        b_bar: Stepsize-independent drift vector.
        Q_sqr_bar: Square-root of stepsize-independent process noise covariance.
        T_t: Preconditioning transformation matrix for current step.
        m_prev_bar: Previous state mean estimate (preconditioned space).
        P_prev_sqr_bar: Previous state covariance (square-root form, preconditioned).
        measure: Measurement model (e.g., ODEInformation or subclass).
        t: Current time (default 0.0).
        obs: Optional observation tuple ``(H_obs, c_obs, R_obs_sqr)``.

    Returns:
        Tuple of 6 tuples with preconditioned and original-space results.
    """
    # Prediction
    m_pred_bar, P_pred_sqr_bar = sqr_marginalization(
        A_bar, b_bar, Q_sqr_bar, m_prev_bar, P_prev_sqr_bar
    )
    G_back_bar, d_back_bar, P_back_sqr_bar = sqr_inversion(
        A_bar, m_prev_bar, P_prev_sqr_bar, m_pred_bar, P_pred_sqr_bar, Q_sqr_bar
    )

    # ODE + Conservation update (linearize in original space)
    H_ode, c_ode = measure.linearize_fixed(T_t @ m_pred_bar, t=t)
    H_ode_bar = H_ode @ T_t
    R_ode_sqr = measure.get_fixed_noise_sqr()
    mz_ode, Pz_ode_sqr = sqr_marginalization(
        H_ode_bar, c_ode, R_ode_sqr, m_pred_bar, P_pred_sqr_bar
    )
    _, m_ode_bar, P_ode_sqr_bar = sqr_inversion(
        H_ode_bar, m_pred_bar, P_pred_sqr_bar, mz_ode, Pz_ode_sqr, R_ode_sqr
    )

    # Observation update
    if obs is not None:
        H_obs, c_obs, R_obs_sqr = obs
        H_obs_bar = H_obs @ T_t
        mz_obs, Pz_obs_sqr = sqr_marginalization(
            H_obs_bar, c_obs, R_obs_sqr, m_ode_bar, P_ode_sqr_bar
        )
        _, m_t_bar, P_t_sqr_bar = sqr_inversion(
            H_obs_bar, m_ode_bar, P_ode_sqr_bar, mz_obs, Pz_obs_sqr, R_obs_sqr
        )
    else:
        m_t_bar = m_ode_bar
        P_t_sqr_bar = P_ode_sqr_bar
        mz_obs = np.zeros(0)
        Pz_obs_sqr = np.zeros((0, 0))

    m_t = T_t @ m_t_bar
    P_t_sqr = P_t_sqr_bar @ T_t.T

    return (
        (m_pred_bar, P_pred_sqr_bar),
        (G_back_bar, d_back_bar, P_back_sqr_bar),
        (mz_ode, Pz_ode_sqr),
        (mz_obs, Pz_obs_sqr),
        (m_t_bar, P_t_sqr_bar),
        (m_t, P_t_sqr),
    )
