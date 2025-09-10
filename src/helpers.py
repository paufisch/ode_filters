import autograd.numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from autograd import grad, jacobian




def filter(m_t_minus, P_t_minus, A, Q, H, R, y):
    """
    One step of the Kalman filter.filter.


    Args:
        m_t_minus: The predicted mean.
        P_t_minus: The predicted covariance.
        A: The state transition matrix.
        Q: The process noise covariance.
        H: The observation matrix.
        R: The observation noise covariance.
        y: The observation.

    Returns:
        (m_t, P_t), (m_t_minus, P_t_minus): The filtered mean and covariance, and the predicted mean and covariance.
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
    """
    One step of the Kalman filter that allows for affine observations which is used for the extended Kalman filter.

    Args:
        m_t_minus: The predicted mean.
        P_t_minus: The predicted covariance.
        A: The state transition matrix.
        Q: The process noise covariance.
        H: The observation matrix.
        R: The observation noise covariance.
        y: The observation.

    Returns:
        (m_t, P_t), (m_t_minus, P_t_minus): The filtered mean and covariance, and the predicted mean and covariance.
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
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    z = y - H @ m_t_minus
    S = H @ P_t_minus @ H.T + R
    K = P_t_minus @ H.T @ np.linalg.inv(S)
    m_t = m_t_minus + K @ z 
    P_t = (np.eye(len(K)) - K @ H) @ P_t_minus
    return m_t, P_t


def filter_light_affine(m_t_minus, P_t_minus, A, Q, H, c, R, y):
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    z = y - H @ m_t_minus - c
    S = H @ P_t_minus @ H.T + R
    K = P_t_minus @ H.T @ np.linalg.inv(S)
    m_t = m_t_minus + K @ z 
    P_t = (np.eye(len(K)) - K @ H) @ P_t_minus
    return m_t, P_t


def future_prediction(m_t_minus, P_t_minus, A, Q):
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    return m_t_minus, P_t_minus


def smoother(m_t, P_t, A, m_t_plus_1_bar, P_t_plus_1_bar, m_t_plus_1_s, P_t_plus_1_s):
    G = P_t @ A.T @ np.linalg.inv(P_t_plus_1_bar)
    m_t_s = m_t + G @ (m_t_plus_1_s - m_t_plus_1_bar)
    P_t_s = P_t + G @ (P_t_plus_1_s - P_t_plus_1_bar) @ G.T
    return m_t_s, P_t_s


def predict(m_0, P_0, A_sequence, Q_sequence, H_sequence, y_sequence, R_sequence):

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


def plot_smoothed_estimate(
    t,
    ts,
    x_sequence,
    z_sequence,
    m_projected,
    P_projected,
    m_smoothed_projected,
    P_smoothed_projected,
    *,
    savefig: bool = False,
    path: str = "smoother_imgs",
    x_lim=None,
    y_lim=None,
    loc_pos: str = 'upper right',
):
    """
    Plot filtered and smoothed 1D projections with safe uncertainty bands.

    Args mirror the notebook version but live here for reuse.
    """
    plt.figure(figsize=(8, 4))

    # ground truth and observations
    if x_sequence is not None:
        plt.plot(x_sequence, label='ground truth', color='black', linestyle='--')
    if z_sequence is not None:
        plt.scatter(ts[1:], z_sequence, s=20, alpha=0.7, label='observed data')

    # filtered
    plt.plot(ts[:t+1], m_projected[:t+1], color='red', alpha=0.7, label='filtered estimate')
    P_f_safe = np.maximum(P_projected[:t+1], 0)
    std_f = np.sqrt(P_f_safe)
    lb_f = m_projected[:t+1] - std_f
    ub_f = m_projected[:t+1] + std_f
    fill_lower_f = np.minimum(lb_f, ub_f)
    fill_upper_f = np.maximum(lb_f, ub_f)
    plt.fill_between(ts[:t+1], fill_lower_f, fill_upper_f, color='red', alpha=0.2)

    # smoothed
    plt.plot(ts[t:], m_smoothed_projected[t:], color='blue', alpha=0.7, label='smoothed estimate')
    P_s_safe = np.maximum(P_smoothed_projected[t:], 0)
    std_s = np.sqrt(P_s_safe)
    lb_s = m_smoothed_projected[t:] - std_s
    ub_s = m_smoothed_projected[t:] + std_s
    fill_lower_s = np.minimum(lb_s, ub_s)
    fill_upper_s = np.maximum(lb_s, ub_s)
    plt.fill_between(ts[t:], fill_lower_s, fill_upper_s, color='blue', alpha=0.2)

    if y_lim is not None:
        plt.ylim(y_lim)
    if x_lim is not None:
        plt.xlim(x_lim)

    plt.xlabel('Time')
    plt.ylabel('State')
    if loc_pos is not None:
        plt.legend(loc=loc_pos)
    plt.grid(True)

    if savefig:
        plt.savefig(f'{path}/kalman_filter_{t}.png')
        plt.close()
    else:
        plt.show()
        plt.close()


def plot_generator_kalman_filter(
    t,
    ts,
    m_projected,
    P_projected,
    m_future_projected,
    P_future_projected,
    x_sequence=None,
    z_sequence=None,
    savefig=False,
    path="filter_imgs",
    x_lim=None,
    y_lim=None,
    loc_pos='upper right',
    ax=None,
):
    """
    Plot Kalman filter results on a provided axis or a new one.
    Mirrors the notebook utility but with safe fill_between handling.
    """

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        created_fig = True

    # ground truth
    if x_sequence is not None:
        ax.plot(ts, x_sequence, label='ground truth', color='black', linestyle='--')

    # observations
    if z_sequence is not None:
        ax.scatter(ts[1:], z_sequence, color='black', s=20, alpha=0.3)
        ax.scatter(ts[1:t+1], z_sequence[:t], s=20, alpha=1.0, label='observed data')

    # filtered
    ax.plot(ts[:t+1], m_projected, color='red', alpha=0.7, label='filtered estimate')
    P_safe = np.maximum(P_projected, 0)
    std_f = np.sqrt(P_safe)
    lb_f = m_projected - std_f
    ub_f = m_projected + std_f
    fill_lower_f = np.minimum(lb_f, ub_f)
    fill_upper_f = np.maximum(lb_f, ub_f)
    ax.fill_between(ts[:t+1], fill_lower_f, fill_upper_f, color='red', alpha=0.2)

    # future prediction
    ax.plot(ts[t:], m_future_projected, color='black', alpha=0.3, label='predicted future')
    P_future_safe = np.maximum(P_future_projected, 0)
    std_fp = np.sqrt(P_future_safe)
    lb_fp = m_future_projected - std_fp
    ub_fp = m_future_projected + std_fp
    fill_lower_fp = np.minimum(lb_fp, ub_fp)
    fill_upper_fp = np.maximum(lb_fp, ub_fp)
    ax.fill_between(ts[t:], fill_lower_fp, fill_upper_fp, color='black', alpha=0.2)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)

    ax.set_xlabel('Time')
    ax.set_ylabel('State')
    if loc_pos is not None:
        ax.legend(loc=loc_pos)
    ax.grid(True)

    if savefig and created_fig:
        fig.savefig(path + f'/kalman_filter_{t}.png')
        plt.close(fig)
    elif created_fig:
        plt.show()
        plt.close(fig)

    return ax


def plot_kalman_filter_subplots(
    step,
    ts,
    N,
    mu_0,
    Sigma_0,
    A,
    Q,
    R,
    z_sequence,
    g,
    jacobian_g,
    exact_solutions,  # list of callables [x_exact, x_prime_exact, x_prime_prime_exact]
    *,
    savefig: bool = False,
    img_path: str = "logistic_imgs",
):
    """
    Create a 4x1 subplot figure of EKF results for the first three Taylor modes
    and the residual measurement g(X) using the helper plot_generator_kalman_filter.

    Args:
        step: current filtering step (int)
        ts: time array (1D)
        N: total number of steps (int)
        mu_0, Sigma_0, A, Q, R: model parameters
        z_sequence: observation array length N (or N-1 depending on convention)
        g, jacobian_g: measurement function and its jacobian
        exact_solutions: list of callables evaluated on ts for ground truth
        savefig: whether to save the figure
        img_path: directory to save images to
    """
    # Run EKF to get full state sequences
    m_filtered, P_filtered, m_predicted, P_predicted = extended_kalman_filter_full(
        step, N, mu_0, Sigma_0, A, Q, R, z_sequence, g, jacobian_g
    )

    # Handle edge shapes for step=0 and step=N-1
    if step == 0:
        m_filtered = m_filtered.reshape(1, -1)
        P_filtered = P_filtered.reshape(1, *P_filtered.shape)
    if step == N - 1:
        m_predicted = m_predicted.reshape(1, -1)
        P_predicted = P_predicted.reshape(1, *P_predicted.shape)

    # Build figure
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))

    # First three components vs exact solutions
    for i in range(3):
        x_seq = exact_solutions[i](ts) if exact_solutions is not None else None
        plot_generator_kalman_filter(
            step,
            ts,
            m_filtered[:, i],
            P_filtered[:, i, i],
            m_predicted[:, i],
            P_predicted[:, i, i],
            x_sequence=x_seq,
            x_lim=[0, 10],
            y_lim=[0, 1],
            loc_pos=None,
            ax=axs[i],
        )
        axs[i].set_ylabel(f'$X^{{({i})}}(t)$')
        # ticks/labels
        axs[i].set_xticks([0, 10])
        axs[i].set_xticklabels([])
        axs[i].set_xlabel('')
        axs[i].grid(False)

    # z values and uncertainties
    z_filtered = m_filtered[:, 1] - m_filtered[:, 0] * (1 - m_filtered[:, 0])
    z_pred = m_predicted[:, 1] - m_predicted[:, 0] * (1 - m_predicted[:, 0])

    z_filtered_uncertainty = []
    for i in range(len(m_filtered)):
        J = jacobian_g(m_filtered[i, :])
        z_filtered_uncertainty.append(J @ P_filtered[i, :] @ J.T)
    z_filtered_uncertainty = np.array(z_filtered_uncertainty)

    z_pred_uncertainty = []
    for i in range(len(m_predicted)):
        Jp = jacobian_g(m_predicted[i, :])
        z_pred_uncertainty.append(Jp @ P_predicted[i, :] @ Jp.T)
    z_pred_uncertainty = np.array(z_pred_uncertainty)

    # Fourth subplot: residual g(X)
    plot_generator_kalman_filter(
        step,
        ts,
        z_filtered,
        z_filtered_uncertainty,
        z_pred,
        z_pred_uncertainty,
        x_sequence=np.zeros_like(ts),
        z_sequence=z_sequence[1:],
        x_lim=[0, 10],
        y_lim=[-0.1, 0.1],
        loc_pos='lower left',
        ax=axs[3],
    )
    axs[3].set_ylabel('$g(X(t))$')
    axs[3].set_xlabel('Time')
    axs[3].grid(False)
    axs[3].set_xticks([0, 10])
    axs[3].set_yticks([-0.1, 0, 0.1])

    # Custom y-lims/ticks for first three plots as in notebook
    axs[0].set_ylim([0, 1])
    axs[0].set_yticks([0, 1])
    axs[1].set_ylim([0, 0.4])
    axs[1].set_yticks([0, 0.4])
    axs[2].set_ylim([-0.4, 0.4])
    axs[2].set_yticks([-0.2, 0.2])

    plt.tight_layout()
    if savefig:
        plt.savefig(f"{img_path}/img_{step}.png")
        plt.close()
    else:
        plt.show()
    return fig, axs