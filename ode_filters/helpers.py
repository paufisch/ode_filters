"""Kalman filtering and smoothing utilities for ODE inference.

This module provides core implementations of linear and extended Kalman filters,
smoothers, and prediction functions using autograd for automatic differentiation support.
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from autograd import grad, jacobian




def filter(m_t_minus, P_t_minus, A, Q, H, R, y):
    """One step of the linear Kalman filter (prediction and update).

    Combines the prediction step (applying dynamics and process noise) with
    the update step (incorporating the observation).

    Args:
        m_t_minus: Predicted state mean (shape [n_state]).
        P_t_minus: Predicted state covariance (shape [n_state, n_state]).
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


def predict(m_0, P_0, A_sequence, Q_sequence, H_sequence, y_sequence, R_sequence):
    """Run forward Kalman filter over a sequence of observations.

    Args:
        m_0: Initial state mean (shape [n_state]).
        P_0: Initial state covariance (shape [n_state, n_state]).
        A_sequence: State transition matrices per time step (list or array).
        Q_sequence: Process noise covariances per time step (list or array).
        H_sequence: Observation matrices per time step (list or array).
        y_sequence: Observations per time step (array shape [T, n_obs]).
        R_sequence: Observation noise covariances per time step (list or array).

    Returns:
        m_sequence: Filtered means, shape [T+1, n_state].
        P_sequence: Filtered covariances, shape [T+1, n_state, n_state].
    """

    # Initialize sequences with initial values
    m_sequence = [m_0]
    P_sequence = [P_0]
    
    # Get the number of time steps
    num_steps = len(A_sequence)
    
    # Iterate from t=1 onwards
    for t in range(num_steps):
        # Get current parameters
        A_t = A_sequence[t]
        Q_t = Q_sequence[t]
        H_t = H_sequence[t]
        y_t = y_sequence[t]
        R_t = R_sequence[t]
        
        # Get previous state estimates
        m_t_minus_1 = m_sequence[t]
        P_t_minus_1 = P_sequence[t]
        
        # Apply FILTER function
        (m_t, P_t), (m_t_minus, P_t_minus) = filter(m_t_minus_1, P_t_minus_1, A_t, Q_t, H_t, R_t, y_t)
        
        # Store results
        m_sequence.append(m_t)
        P_sequence.append(P_t)
    
    return m_sequence, P_sequence


# implementation of t out of t_max steps of the Kalman filter. The function computes the filtered estimate up to t and also computes the recursive naive rediction from t+1 to t_max. So far this only works for time invariant models, Since A, Q, H, R are all time invariant. Returns the already projeced estimate and future predictions
def kalman_filter(t, t_max, mu_0, Sigma_0, A, Q, H, R, z_sequence):

    m_sequence = [mu_0]
    P_sequence = [Sigma_0] 

    for i in range(t):
        m_prime, P_prime = filter_light(m_sequence[-1], P_sequence[-1], A, Q, H, R, z_sequence[i])
        m_sequence.append(m_prime)
        P_sequence.append(P_prime)

    m_future = [m_sequence[-1]]
    P_future = [P_sequence[-1]]
        
    for i in range(t+1, t_max):
        m_next, P_next = future_prediction(m_future[-1], P_future[-1], A, Q)
        m_future.append(m_next)
        P_future.append(P_next)

    m_projected = np.array(m_sequence)@H.T
    P_projected = H@np.array(P_sequence)@H.T
    m_projected = m_projected.squeeze()
    P_projected = P_projected.squeeze()

    m_future_projected = np.array(m_future)@H.T
    P_future_projected = H@np.array(P_future)@H.T
    m_future_projected = m_future_projected.squeeze()
    P_future_projected = P_future_projected.squeeze()

    return m_projected, P_projected, m_future_projected, P_future_projected


def extended_kalman_filter(t, t_max, mu_0, Sigma_0, A, Q, R, z_sequence, g, jacobian_g):

    m_sequence = [mu_0]
    P_sequence = [Sigma_0] 

    for i in range(t):
        linearization_point, _ = future_prediction(m_sequence[-1], P_sequence[-1], A, Q)
        #linearization_point = m_sequence[-1]
        H = jacobian_g(linearization_point).reshape(1,-1)
        c = g(linearization_point) - H @ linearization_point

        m_prime, P_prime = filter_light_affine(m_sequence[-1], P_sequence[-1], A, Q, H, c, R, z_sequence[i])
        m_sequence.append(m_prime)
        P_sequence.append(P_prime)

    m_future = [m_sequence[-1]]
    P_future = [P_sequence[-1]]
        
    for i in range(t+1, t_max):
        m_next, P_next = future_prediction(m_future[-1], P_future[-1], A, Q)
        m_future.append(m_next)
        P_future.append(P_next)

    m_projected = np.array(m_sequence)[:,0]
    P_projected = np.array(P_sequence)[:,0,0]
    m_projected = m_projected.squeeze()
    P_projected = P_projected.squeeze()

    m_future_projected = np.array(m_future)[:,0]
    P_future_projected = np.array(P_future)[:,0,0]
    m_future_projected = m_future_projected.squeeze()
    P_future_projected = P_future_projected.squeeze()

    return m_projected, P_projected, m_future_projected, P_future_projected


#instead of returning only the projected prediciton of the first computation this full filter iterator returns the full predicitons for the full state variable
def extended_kalman_filter_full(t, t_max, mu_0, Sigma_0, A, Q, R, z_sequence, g, jacobian_g):

    m_sequence = [mu_0]
    P_sequence = [Sigma_0] 

    for i in range(t):
        linearization_point = m_sequence[-1]
        H = jacobian_g(linearization_point).reshape(1,-1)
        c = g(linearization_point) - H @ linearization_point

        m_prime, P_prime = filter_light_affine(m_sequence[-1], P_sequence[-1], A, Q, H, c, R, z_sequence[i])
        m_sequence.append(m_prime)
        P_sequence.append(P_prime)

    m_future = [m_sequence[-1]]
    P_future = [P_sequence[-1]]
        
    for i in range(t+1, t_max):
        m_next, P_next = future_prediction(m_future[-1], P_future[-1], A, Q)
        m_future.append(m_next)
        P_future.append(P_next)

    m_projected = np.array(m_sequence)
    P_projected = np.array(P_sequence)
    m_projected = m_projected.squeeze()
    P_projected = P_projected.squeeze()

    m_future_projected = np.array(m_future)
    P_future_projected = np.array(P_future)
    m_future_projected = m_future_projected.squeeze()
    P_future_projected = P_future_projected.squeeze()

    return m_projected, P_projected, m_future_projected, P_future_projected


def plot_kalman_filter(t, ts, m_projected, P_projected, m_future_projected, P_future_projected, x_sequence=None, z_sequence=None, savefig=False, path="filter_imgs", x_lim=None, y_lim=None, loc_pos='upper right'):

    plt.figure(figsize=(8, 4))
    # plot the ground truth if given
    if x_sequence is not None:
        plt.plot(ts, x_sequence, label='ground truth', color='black', linestyle='--')
    # plot the observed data if given
    if z_sequence is not None:
        plt.scatter(ts[1:], z_sequence, color='black', s=20, alpha=0.3)
        plt.scatter(ts[1:t+1], z_sequence[:t], s=20, alpha=0.7, label='observed data')

    # plot the filtered estimate projeced to one dimesnion
    plt.plot(ts[:t+1], m_projected, color='red', alpha=0.7, label='filtered estimate')
    
    # Ensure variance is non-negative and calculate bounds safely
    P_safe = np.maximum(P_projected, 0)  # Clip negative variances to 0
    std_dev = np.sqrt(P_safe)
    lower_bound = m_projected - std_dev
    upper_bound = m_projected + std_dev
    plt.fill_between(ts[:t+1], lower_bound, upper_bound, color='red', alpha=0.2)

    plt.plot(ts[t:], m_future_projected, color='black', alpha=0.3, label='predicted future')
    
    # Ensure variance is non-negative and calculate bounds safely for future predictions
    P_future_safe = np.maximum(P_future_projected, 0)  # Clip negative variances to 0
    std_dev_future = np.sqrt(P_future_safe)
    lower_bound_future = m_future_projected - std_dev_future
    upper_bound_future = m_future_projected + std_dev_future
    plt.fill_between(ts[t:], lower_bound_future, upper_bound_future, color='black', alpha=0.2)

    #plt.axvline(x=t, color='black', alpha=0.3, label='current time')
    #plt.yticks(np.arange(-8, 2, 1))
    if x_lim != None:
        plt.xlim(x_lim)
    if y_lim != None:
        plt.ylim(y_lim)
    

    plt.xlabel('Time')
    plt.ylabel('State')
    plt.legend(loc=loc_pos)
    plt.grid(True)
    if savefig:
        plt.savefig(path + f'/kalman_filter_{t}.png')
    else:
        plt.show()
    plt.close()


def extended_kalman_smoother_full(mu_0, Sigma_0, A, Q, R, z_sequence, g, jacobian_g, projection_index: int = 0):
    """
    Run an Extended Kalman Filter (with affine linearization of the observation)
    followed by Rauch–Tung–Striebel smoothing.

    Args:
        mu_0: Initial mean vector (shape [state_dim])
        Sigma_0: Initial covariance matrix (shape [state_dim, state_dim])
        A: State transition matrix (shape [state_dim, state_dim])
        Q: Process noise covariance (shape [state_dim, state_dim])
        R: Observation noise covariance (shape [1, 1] or compatible)
        z_sequence: Observations array (shape [T] or [T, 1])
        g: Nonlinear observation function, g(x) -> scalar
        jacobian_g: Function that returns Jacobian of g at x (shape [state_dim])
        projection_index: Index of state dimension for 1D projection (default 0)

    Returns:
        m_sequence, P_sequence: Filtered means/covariances for t=0..T (arrays)
        m_smoothed, P_smoothed: Smoothed means/covariances for t=0..T (arrays)
        m_projected, P_projected: Filtered projections to the selected state index
        m_smoothed_projected, P_smoothed_projected: Smoothed projections
    """

    # Normalize observations
    z_sequence = np.asarray(z_sequence).reshape(-1)
    T = len(z_sequence)

    # Forward EKF (affine observation)
    m_sequence = [mu_0]
    P_sequence = [Sigma_0]
    m_sequence_predicted = [mu_0]
    P_sequence_predicted = [Sigma_0]

    for t in range(T):
        linearization_point = m_sequence[-1]
        H = jacobian_g(linearization_point).reshape(1, -1)
        c = g(linearization_point) - H @ linearization_point

        (m_t, P_t), (m_t_bar, P_t_bar) = filter_affine(
            m_sequence[-1], P_sequence[-1], A, Q, H, c, R, z_sequence[t]
        )

        m_sequence.append(m_t)
        P_sequence.append(P_t)
        m_sequence_predicted.append(m_t_bar)
        P_sequence_predicted.append(P_t_bar)

    m_sequence = np.array(m_sequence)
    P_sequence = np.array(P_sequence)
    m_sequence_predicted = np.array(m_sequence_predicted)
    P_sequence_predicted = np.array(P_sequence_predicted)

    # RTS smoothing (backward)
    m_smoothed = [m_sequence[-1]]
    P_smoothed = [P_sequence[-1]]
    for back in range(1, T + 1):
        idx = T - back
        m_prev_s, P_prev_s = smoother(
            m_sequence[idx],
            P_sequence[idx],
            A,
            m_sequence_predicted[idx + 1],
            P_sequence_predicted[idx + 1],
            m_smoothed[-1],
            P_smoothed[-1],
        )
        m_smoothed.append(m_prev_s)
        P_smoothed.append(P_prev_s)

    # Reverse to chronological order
    m_smoothed = m_smoothed[::-1]
    P_smoothed = P_smoothed[::-1]

    m_smoothed = np.array(m_smoothed)
    P_smoothed = np.array(P_smoothed)

    # Projections to 1D (selected state index)
    m_projected = m_sequence[:, projection_index].squeeze()
    P_projected = np.array(P_sequence)[:, projection_index, projection_index].squeeze()

    m_smoothed_projected = m_smoothed[:, projection_index].squeeze()
    P_smoothed_projected = np.array(P_smoothed)[:, projection_index, projection_index].squeeze()

    return (
        m_sequence,
        P_sequence,
        m_smoothed,
        P_smoothed,
        m_projected,
        P_projected,
        m_smoothed_projected,
        P_smoothed_projected,
    )


def kalman_smoother_full(mu_0, Sigma_0, A, Q, H, R, z_sequence):
    """
    Linear Kalman filter forward pass followed by RTS smoothing.

    Args:
        mu_0: Initial mean (shape [state_dim])
        Sigma_0: Initial covariance (shape [state_dim, state_dim])
        A: State transition (shape [state_dim, state_dim])
        Q: Process noise covariance (shape [state_dim, state_dim])
        H: Observation matrix (shape [1, state_dim])
        R: Observation noise covariance (shape [1, 1])
        z_sequence: Observations array (shape [T] or [T, 1])

    Returns:
        m_sequence, P_sequence: Filtered means/covariances (arrays length T+1)
        m_smoothed, P_smoothed: Smoothed means/covariances (arrays length T+1)
        m_projected, P_projected: Filtered 1D projection via H
        m_smoothed_projected, P_smoothed_projected: Smoothed 1D projection via H
    """

    z_sequence = np.asarray(z_sequence).reshape(-1)
    T = len(z_sequence)

    # Forward Kalman filter (linear)
    m_sequence = [mu_0]
    P_sequence = [Sigma_0]
    m_sequence_predicted = [mu_0]
    P_sequence_predicted = [Sigma_0]

    for t in range(T):
        (m_t, P_t), (m_t_bar, P_t_bar) = filter(
            m_sequence[-1], P_sequence[-1], A, Q, H, R, z_sequence[t]
        )
        m_sequence.append(m_t)
        P_sequence.append(P_t)
        m_sequence_predicted.append(m_t_bar)
        P_sequence_predicted.append(P_t_bar)

    m_sequence = np.array(m_sequence)
    P_sequence = np.array(P_sequence)
    m_sequence_predicted = np.array(m_sequence_predicted)
    P_sequence_predicted = np.array(P_sequence_predicted)

    # RTS smoothing
    m_smoothed = [m_sequence[-1]]
    P_smoothed = [P_sequence[-1]]
    for back in range(1, T + 1):
        idx = T - back
        m_prev_s, P_prev_s = smoother(
            m_sequence[idx],
            P_sequence[idx],
            A,
            m_sequence_predicted[idx + 1],
            P_sequence_predicted[idx + 1],
            m_smoothed[-1],
            P_smoothed[-1],
        )
        m_smoothed.append(m_prev_s)
        P_smoothed.append(P_prev_s)

    m_smoothed = m_smoothed[::-1]
    P_smoothed = P_smoothed[::-1]
    m_smoothed = np.array(m_smoothed)
    P_smoothed = np.array(P_smoothed)

    # 1D projection using H
    # Means: m @ H.T → shape (T+1, 1) → squeeze to (T+1,)
    m_projected = (m_sequence @ H.T).squeeze()
    m_smoothed_projected = (m_smoothed @ H.T).squeeze()

    # Variances: H P H^T → shape (T+1, 1, 1) → squeeze
    HP = np.einsum('ij,tjk->tik', H, P_sequence)
    P_projected = np.einsum('tik,kj->ti', HP, H.T).squeeze()

    HP_s = np.einsum('ij,tjk->tik', H, P_smoothed)
    P_smoothed_projected = np.einsum('tik,kj->ti', HP_s, H.T).squeeze()

    return (
        m_sequence,
        P_sequence,
        m_smoothed,
        P_smoothed,
        m_projected,
        P_projected,
        m_smoothed_projected,
        P_smoothed_projected,
    )