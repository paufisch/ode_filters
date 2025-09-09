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