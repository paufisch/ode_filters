"""Kalman filtering and smoothing utilities for ODE inference.

This module provides core implementations for linear and extended Kalman filters
with support for automatic differentiation via autograd.
"""

import autograd.numpy as np
from autograd import grad, jacobian


def filter(m_t_minus, P_t_minus, A, Q, H, R, y):
    """One step of the linear Kalman filter (prediction and update).

    Combines the prediction step (applying dynamics and process noise) with
    the update step (incorporating the observation).

    Args:
        m_t_minus: Predicted state mean (shape [n_state]).
        P_t_minus: Predicted state covariance (shape [n_state]).
        A: State transition matrix (shape [n_state, n_state]).
        Q: Process noise covariance (shape [n_state, n_state]).
        H: Observation matrix (shape [n_obs, n_state]).
        R: Observation noise covariance (shape [n_obs, n_obs]).
        y: Observation value (shape [n_obs]).

    Returns:
        (m_t, P_t): Updated state mean and covariance.
        (m_t_minus_pred, P_t_minus_pred): Predicted mean and covariance before update.
    """
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    z = y - H @ m_t_minus
    S = H @ P_t_minus @ H.T + R
    K = P_t_minus @ H.T @ np.linalg.inv(S)
    m_t = m_t_minus + K @ z 
    P_t = (np.eye(len(K)) - K @ H) @ P_t_minus
    return (m_t, P_t), (m_t_minus, P_t_minus)


def filter_affine(m_t_minus, P_t_minus, A, Q, H, c, R, y):
    """Kalman filter step with affine (linearized) observation model.

    Supports nonlinear observations via affine linearization: H @ x + c.
    Used in Extended Kalman Filters.

    Args:
        m_t_minus: Predicted state mean (shape [n_state]).
        P_t_minus: Predicted state covariance (shape [n_state, n_state]).
        A: State transition matrix (shape [n_state, n_state]).
        Q: Process noise covariance (shape [n_state, n_state]).
        H: Observation sensitivity matrix (shape [n_obs, n_state]).
        c: Affine observation offset (shape [n_obs]).
        R: Observation noise covariance (shape [n_obs, n_obs]).
        y: Observation value (shape [n_obs]).

    Returns:
        (m_t, P_t): Updated state mean and covariance.
        (m_t_minus_pred, P_t_minus_pred): Predicted mean and covariance.
    """
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    z = y - H @ m_t_minus - c
    S = H @ P_t_minus @ H.T + R
    K = P_t_minus @ H.T @ np.linalg.inv(S)
    m_t = m_t_minus + K @ z 
    P_t = (np.eye(len(K)) - K @ H) @ P_t_minus
    return (m_t, P_t), (m_t_minus, P_t_minus)


def filter_light(m_t_minus, P_t_minus, A, Q, H, R, y):
    """Lightweight Kalman filter step returning only updated states.

    Returns only the updated (filtered) estimates without predicted estimates.

    Args:
        m_t_minus: Predicted state mean (shape [n_state]).
        P_t_minus: Predicted state covariance (shape [n_state, n_state]).
        A: State transition matrix (shape [n_state, n_state]).
        Q: Process noise covariance (shape [n_state, n_state]).
        H: Observation matrix (shape [n_obs, n_state]).
        R: Observation noise covariance (shape [n_obs, n_obs]).
        y: Observation value (shape [n_obs]).

    Returns:
        m_t: Updated state mean (shape [n_state]).
        P_t: Updated state covariance (shape [n_state, n_state]).
    """
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    z = y - H @ m_t_minus
    S = H @ P_t_minus @ H.T + R
    K = P_t_minus @ H.T @ np.linalg.inv(S)
    m_t = m_t_minus + K @ z 
    P_t = (np.eye(len(K)) - K @ H) @ P_t_minus
    return m_t, P_t


def filter_light_affine(m_t_minus, P_t_minus, A, Q, H, c, R, y):
    """Lightweight affine Kalman filter step.

    Lightweight version of filter_affine, returning only updated estimates.

    Args:
        m_t_minus: Predicted state mean (shape [n_state]).
        P_t_minus: Predicted state covariance (shape [n_state, n_state]).
        A: State transition matrix (shape [n_state, n_state]).
        Q: Process noise covariance (shape [n_state, n_state]).
        H: Observation sensitivity matrix (shape [n_obs, n_state]).
        c: Affine observation offset (shape [n_obs]).
        R: Observation noise covariance (shape [n_obs, n_obs]).
        y: Observation value (shape [n_obs]).

    Returns:
        m_t: Updated state mean (shape [n_state]).
        P_t: Updated state covariance (shape [n_state, n_state]).
    """
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    z = y - H @ m_t_minus - c
    S = H @ P_t_minus @ H.T + R
    K = P_t_minus @ H.T @ np.linalg.inv(S)
    m_t = m_t_minus + K @ z 
    P_t = (np.eye(len(K)) - K @ H) @ P_t_minus
    return m_t, P_t


def future_prediction(m_t_minus, P_t_minus, A, Q):
    """Predict state one step forward without observation.

    Applies state transition and process noise only (no measurement update).

    Args:
        m_t_minus: Current state mean (shape [n_state]).
        P_t_minus: Current state covariance (shape [n_state, n_state]).
        A: State transition matrix (shape [n_state, n_state]).
        Q: Process noise covariance (shape [n_state, n_state]).

    Returns:
        m_t_minus: Next state mean (shape [n_state]).
        P_t_minus: Next state covariance (shape [n_state, n_state]).
    """
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    return m_t_minus, P_t_minus


def smoother(m_t, P_t, A, m_t_plus_1_bar, P_t_plus_1_bar, m_t_plus_1_s, P_t_plus_1_s):
    """One backward step of Rauch–Tung–Striebel (RTS) smoothing.

    Updates the smoothed estimate at time t using forward-filtered estimates
    and the already-smoothed estimates from time t+1.

    Args:
        m_t: Filtered mean at time t (shape [n_state]).
        P_t: Filtered covariance at time t (shape [n_state, n_state]).
        A: State transition matrix (shape [n_state, n_state]).
        m_t_plus_1_bar: Predicted mean at time t+1 (shape [n_state]).
        P_t_plus_1_bar: Predicted covariance at time t+1 (shape [n_state, n_state]).
        m_t_plus_1_s: Smoothed mean at time t+1 (shape [n_state]).
        P_t_plus_1_s: Smoothed covariance at time t+1 (shape [n_state, n_state]).

    Returns:
        m_t_s: Smoothed mean at time t (shape [n_state]).
        P_t_s: Smoothed covariance at time t (shape [n_state, n_state]).
    """
    G = P_t @ A.T @ np.linalg.inv(P_t_plus_1_bar)
    m_t_s = m_t + G @ (m_t_plus_1_s - m_t_plus_1_bar)
    P_t_s = P_t + G @ (P_t_plus_1_s - P_t_plus_1_bar) @ G.T
    return m_t_s, P_t_s