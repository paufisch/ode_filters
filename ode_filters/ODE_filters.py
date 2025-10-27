#single step of a kalman filter
from .gaussian_inference import *
from .sqr_gaussian_inference import *



def ekf1_filter_step(A_t, b_t, Q_t, R_t, mu_t, Sigma_t, g, jacobian_g, z_observed):
    """One step of the Extended Kalman Filter (EKF) with nonlinear observations.

    Performs prediction and update steps. Prediction applies linear dynamics;
    update linearizes the nonlinear observation model and updates using inversion.

    Args:
        A_t: State transition matrix (shape [n_state, n_state]).
        b_t: State transition offset (shape [n_state]).
        Q_t: Process noise covariance (shape [n_state, n_state]).
        R_t: Observation noise covariance (shape [1, 1] or scalar).
        mu_t: Current state mean (shape [n_state]).
        Sigma_t: Current state covariance (shape [n_state, n_state]).
        g: Nonlinear observation function.
        jacobian_g: Jacobian of g at linearization point.
        z_observed: Observed measurement value.

    Returns:
        (m_pred, P_pred): Predicted mean and covariance.
        (m_updated, P_updated): Updated (filtered) mean and covariance.
    """

    m_pred, P_pred = marginalization(A_t, b_t, Q_t, mu_t, Sigma_t)

    #linearize the observation model
    H_t = jacobian_g(m_pred)
    c_t = g(m_pred) - H_t @ m_pred

    # predict/estimate the next observationm: p(z_n|x_n)
    #z_expected, S_sqr = sqr_marginalization(H_n, c, R_h, m_next, P_next_sqr)

    #compute backward transition: p(z_n|x_n)
    #this actually implicitly first computes the observation estimate which is why the above line is commented

    #observe z:
    m_updated, P_updated = inversion(H_t, c_t, R_t, m_pred, P_pred, z_observed)
    
    return (m_pred, P_pred), (m_updated, P_updated)


#single step of a kalman filter
def ekf1_filter_step_stable(A_t, b_t, Q_t, R_t, mu_t, Sigma_t, g, jacobian_g, z_observed):
    """Numerically stable EKF step using square-root covariance representation.

    Uses Cholesky factors for improved numerical stability compared to
    ekf1_filter_step. Otherwise functionally equivalent.

    Args:
        A_t: State transition matrix (shape [n_state, n_state]).
        b_t: State transition offset (shape [n_state]).
        Q_t: Process noise Cholesky factor (shape [n_state, n_state]).
        R_t: Observation noise Cholesky factor (shape [1, 1] or scalar).
        mu_t: Current state mean (shape [n_state]).
        Sigma_t: Current state Cholesky factor (shape [n_state, n_state]).
        g: Nonlinear observation function.
        jacobian_g: Jacobian of g.
        z_observed: Observed measurement.

    Returns:
        (m_pred, P_pred): Predicted mean and Cholesky factor.
        (m_updated, P_updated): Updated mean and Cholesky factor.
    """

    m_pred, P_pred = sqr_marginalization(A_t, b_t, Q_t, mu_t, Sigma_t)

    #linearize the observation model
    H_t = jacobian_g(m_pred)
    c_t = g(m_pred) - H_t @ m_pred

    # predict/estimate the next observationm: p(z_n|x_n)
    #z_expected, S_sqr = sqr_marginalization(H_n, c, R_h, m_next, P_next_sqr)

    #compute backward transition: p(z_n|x_n)
    #this actually implicitly first computes the observation estimate which is why the above line is commented

    #observe z:
    m_updated, P_updated = sqr_inversion(H_t, c_t, R_t, m_pred, P_pred, z_observed)
    
    return (m_pred, P_pred), (m_updated, P_updated)


def compute_kalman_forward(mu_0, Sigma_0, A_h, b_h, Q_h, R_h, g, jacobian_g, z_sequence, N):
    """Forward EKF pass over full observation sequence.

    Iterates ekf1_filter_step over N observations, accumulating filtered and
    predicted state sequences.

    Args:
        mu_0: Initial state mean (shape [n_state]).
        Sigma_0: Initial state covariance (shape [n_state, n_state]).
        A_h, b_h, Q_h, R_h: System dynamics and noise parameters.
        g: Observation function.
        jacobian_g: Jacobian of g.
        z_sequence: Observations array (shape [N, n_obs]).
        N: Number of time steps.

    Returns:
        m_sequence: Filtered means, shape [N+1, n_state].
        P_sequence: Filtered covariances, shape [N+1, n_state, n_state].
        m_predictions: Predicted means, shape [N, n_state].
        P_predictions: Predicted covariances, shape [N, n_state, n_state].
    """
    #complete forward kalman filtering pass:
    m_sequence = [mu_0]
    P_sequence = [Sigma_0] 
    m_predictions = []
    P_predictions = []

    for i in range(N):
        #this index correspnds to the timestep 
        #print(ts[i+1])
        #h = ts[i+1]-ts[i]
        (m_pred_nxt, P_pred_nxt), (m_nxt, P_nxt) = ekf1_filter_step(A_h, b_h, Q_h, R_h, m_sequence[-1], P_sequence[-1], g, jacobian_g, z_sequence[i,:])
        m_sequence.append(m_nxt)
        P_sequence.append(P_nxt)
        m_predictions.append(m_pred_nxt)
        P_predictions.append(P_pred_nxt)


    m_sequence = np.array(m_sequence)
    P_sequence = np.array(P_sequence)
    m_predictions = np.array(m_predictions)
    P_predictions = np.array(P_predictions)

    return m_sequence, P_sequence, m_predictions, P_predictions


def compute_kalman_forward_stable(mu_0, Sigma_0, A_h, b_h, Q_h, R_h, g, jacobian_g, z_sequence, N):
    """Numerically stable forward EKF pass using square-root covariances.

    Like compute_kalman_forward but uses ekf1_filter_step_stable. Takes
    Q_h as a Cholesky factor and reconstructs full covariances at the end.

    Args:
        mu_0: Initial state mean (shape [n_state]).
        Sigma_0: Initial state Cholesky factor (shape [n_state, n_state]).
        A_h, b_h, Q_h, R_h: System parameters (Q_h should be Cholesky).
        g: Observation function.
        jacobian_g: Jacobian of g.
        z_sequence: Observations array (shape [N, n_obs]).
        N: Number of time steps.

    Returns:
        m_sequence: Filtered means, shape [N+1, n_state].
        P_sequence: Filtered full covariances, shape [N+1, n_state, n_state].
        m_predictions: Predicted means, shape [N, n_state].
        P_predictions: Predicted full covariances, shape [N, n_state, n_state].
    """
    #complete forward kalman filtering pass:
    Q_h = np.linalg.cholesky(Q_h).T
    #R_h = np.linalg.cholesky(R_h).T
    #Sigma_0 = np.linalg.cholesky(Sigma_0).T
    
    
    m_sequence = [mu_0]
    P_sequence = [Sigma_0] 
    m_predictions = []
    P_predictions = []

    for i in range(N):
        #this index correspnds to the timestep 
        #print(ts[i+1])
        #h = ts[i+1]-ts[i]
        (m_pred_nxt, P_pred_nxt), (m_nxt, P_nxt) = ekf1_filter_step_stable(A_h, b_h, Q_h, R_h, m_sequence[-1], P_sequence[-1], g, jacobian_g, z_sequence[i,:])
        m_sequence.append(m_nxt)
        P_sequence.append(P_nxt)
        m_predictions.append(m_pred_nxt)
        P_predictions.append(P_pred_nxt)


    m_sequence = np.array(m_sequence)
    P_sequence = np.array(P_sequence)
    m_predictions = np.array(m_predictions)
    P_predictions = np.array(P_predictions)

        # Important: square covariances 
    for i in range(P_sequence.shape[0]):
        P_sequence[i,...] = P_sequence[i,...].T @ P_sequence[i,...]

        # Important: square covariances 
    for i in range(P_predictions.shape[0]):
        P_predictions[i,...] = P_predictions[i,...].T @ P_predictions[i,...]

    return m_sequence, P_sequence, m_predictions, P_predictions


def kf_smoother_step(m_t, P_t, A, m_pred, P_pred, m_smooth, P_smooth):
    """One backward step of Rauch–Tung–Striebel (RTS) smoothing.

    Updates the smoothed estimate at time t using the filtered estimate and
    the smoothed estimate from time t+1 via the Kalman smoother gain.

    Args:
        m_t: Filtered mean at time t (shape [n_state]).
        P_t: Filtered covariance at time t (shape [n_state, n_state]).
        A: State transition matrix (shape [n_state, n_state]).
        m_pred: Predicted mean at time t+1 (shape [n_state]).
        P_pred: Predicted covariance at time t+1 (shape [n_state, n_state]).
        m_smooth: Smoothed mean at time t+1 (shape [n_state]).
        P_smooth: Smoothed covariance at time t+1 (shape [n_state, n_state]).

    Returns:
        m_t_s: Smoothed mean at time t (shape [n_state]).
        P_t_s: Smoothed covariance at time t (shape [n_state, n_state]).
    """
    
    #G = P_t @ A.T @ np.linalg.inv(P_pred)
    #instead of the naive inverse computation more efficiently:
    L, lower = cho_factor(P_pred, lower=True, check_finite=False)
    Y = cho_solve((L, lower), (P_t @ A.T).T, check_finite=False)
    G = Y.T
    m_t_s = m_t + G @ (m_smooth - m_pred)
    P_t_s = P_t + G @ (P_smooth - P_pred) @ G.T

    return m_t_s, P_t_s


def compute_kalman_backward(m_seq, P_seq, m_pred, P_pred, A_h, N):
    """Backward RTS smoothing pass.

    Applies kf_smoother_step iteratively backward in time to produce smoothed
    state estimates given filtered and predicted sequences.

    Args:
        m_seq: Filtered means (shape [N+1, n_state]).
        P_seq: Filtered covariances (shape [N+1, n_state, n_state]).
        m_pred: Predicted means (shape [N, n_state]).
        P_pred: Predicted covariances (shape [N, n_state, n_state]).
        A_h: State transition matrix (shape [n_state, n_state]).
        N: Number of time steps.

    Returns:
        m_smoothed: Smoothed means (shape [N+1, n_state]).
        P_smoothed: Smoothed covariances (shape [N+1, n_state, n_state]).
    """
    m_smoothed = [m_seq[-1]]
    P_smoothed = [P_seq[-1,...]]

    for i in range(1,N+1):
        m_t_s, P_t_s = kf_smoother_step(m_seq[N-i], P_seq[N-i,...], A_h, m_pred[-i], P_pred[-i,...], m_smoothed[-1], P_smoothed[-1])
        m_smoothed.append(m_t_s)
        P_smoothed.append(P_t_s)

    m_smoothed = np.array(m_smoothed)[::-1]
    P_smoothed = np.array(P_smoothed)[::-1]

    return m_smoothed, P_smoothed


def backward_transitions(m_seq, P_seq, m_pred, P_pred, A_h, N):
    """Compute backward transition parameters for sampling.

    Computes the linear transition parameters (G, d, Lambda) needed for
    backward sampling through the state sequence.

    Args:
        m_seq: Filtered means (shape [N+1, n_state]).
        P_seq: Filtered covariances (shape [N+1, n_state, n_state]).
        m_pred: Predicted means (shape [N, n_state]).
        P_pred: Predicted covariances (shape [N, n_state, n_state]).
        A_h: State transition matrix (shape [n_state, n_state]).
        N: Number of time steps.

    Returns:
        Gs: Backward gain matrices, shape [N, n_state, n_state].
        ds: Backward offsets, shape [N, n_state].
        Lambdas: Backward covariances, shape [N, n_state, n_state].
    """

    Gs, ds, Lambdas = [], [], []
    for i in range(1,N+1):
        G_nxt, d_nxt, Lambda_nxt = inversion2(A_h, m_seq[N-i], P_seq[N-i,...], m_pred[-i], P_pred[-i,...])
        Gs.append(G_nxt)
        ds.append(d_nxt)
        Lambdas.append(Lambda_nxt)

    Gs = np.array(Gs)
    ds = np.array(ds)
    Lambdas = np.array(Lambdas)
    
    return Gs, ds, Lambdas


def backward_sample_paths(num_samples, m_sequence, P_sequence, Gs, ds, Lambdas, N, seed=42):
    """Generate sample paths by backward sampling from the posterior.

    Samples from the joint distribution of state trajectories conditioned on
    all observations using the backward transition parameters.

    Args:
        num_samples: Number of trajectory samples to generate.
        m_sequence: Smoothed means from filter (shape [N+1, n_state]).
        P_sequence: Smoothed covariances (shape [N+1, n_state, n_state]).
        Gs: Backward gain matrices (shape [N, n_state, n_state]).
        ds: Backward offsets (shape [N, n_state]).
        Lambdas: Backward transition covariances (shape [N, n_state, n_state]).
        N: Number of time steps.
        seed: Random seed for reproducibility.

    Returns:
        X_s_samples: Sample paths, shape [num_samples, N+1, n_state].
    """

    np.random.seed(seed)
    X_s_samples = []
    for _ in range(num_samples):
        X_T = np.random.multivariate_normal(m_sequence[-1], P_sequence[-1])
        X_s = [X_T]
        for i in range(N):
            X_nxt = np.random.multivariate_normal(Gs[i] @ X_s[-1] + ds[i], Lambdas[i])
            X_s.append(X_nxt)
        X_s = np.array(X_s)[::-1]
        X_s_samples.append(X_s)
    X_s_samples = np.array(X_s_samples)  # shape: (num_samples, N+1, d)
    return X_s_samples


def compute_kalman_forward_with_backward_transitions(mu_0, Sigma_0, A_h, b_h, Q_h, R_h, g, jacobian_g, z_sequence, N):
    """Forward EKF pass with precomputed backward transition parameters.

    Combines forward filtering with computation of backward transition
    parameters in a single pass for efficiency.

    Args:
        mu_0: Initial state mean (shape [n_state]).
        Sigma_0: Initial state covariance (shape [n_state, n_state]).
        A_h, b_h, Q_h, R_h: System parameters.
        g: Observation function.
        jacobian_g: Jacobian of g.
        z_sequence: Observations array (shape [N, n_obs]).
        N: Number of time steps.

    Returns:
        m_sequence: Filtered means, shape [N+1, n_state].
        P_sequence: Filtered covariances, shape [N+1, n_state, n_state].
        m_predictions: Predicted means, shape [N, n_state].
        P_predictions: Predicted covariances, shape [N, n_state, n_state].
        Gs: Backward gains, shape [N, n_state, n_state].
        ds: Backward offsets, shape [N, n_state].
        Lambdas: Backward covariances, shape [N, n_state, n_state].
    """
    #complete forward kalman filtering pass:
    m_sequence = [mu_0]
    P_sequence = [Sigma_0] 
    m_predictions = []
    P_predictions = []
    Gs, ds, Lambdas = [], [], []

    for i in range(N):
        #this index correspnds to the timestep 
        #print(ts[i+1])
        #h = ts[i+1]-ts[i]
        (m_pred_nxt, P_pred_nxt), (m_nxt, P_nxt) = ekf1_filter_step(A_h, b_h, Q_h, R_h, m_sequence[-1], P_sequence[-1], g, jacobian_g, z_sequence[i,:])
        G_nxt, d_nxt, Lambda_nxt = inversion2(A_h, m_sequence[-1], P_sequence[-1], m_pred_nxt, P_pred_nxt)
        m_sequence.append(m_nxt)
        P_sequence.append(P_nxt)
        m_predictions.append(m_pred_nxt)
        P_predictions.append(P_pred_nxt)
        Gs.append(G_nxt)
        ds.append(d_nxt)
        Lambdas.append(Lambda_nxt)

    m_sequence = np.array(m_sequence)
    P_sequence = np.array(P_sequence)
    m_predictions = np.array(m_predictions)
    P_predictions = np.array(P_predictions)
    Gs = np.array(Gs)
    ds = np.array(ds)
    Lambdas = np.array(Lambdas)

    return m_sequence, P_sequence, m_predictions, P_predictions, Gs, ds, Lambdas


def compute_kalman_forward_with_backward_transitions_intermediate(mu_0, Sigma_0, A_h, b_h, Q_h, R_h, g, jacobian_g, z_sequence, N, M):
    """Forward pass with intermediate prediction steps and backward transitions.

    Like compute_kalman_forward_with_backward_transitions but includes M total
    steps (with N observations). Intermediate steps only predict without updating.

    Args:
        mu_0: Initial state mean (shape [n_state]).
        Sigma_0: Initial state covariance (shape [n_state, n_state]).
        A_h, b_h, Q_h, R_h: System parameters.
        g: Observation function.
        jacobian_g: Jacobian of g.
        z_sequence: Observations array (shape [N, n_obs]).
        N: Number of observations.
        M: Total number of time steps (must be a multiple of N).

    Returns:
        m_sequence: State means, shape [M+1, n_state].
        P_sequence: State covariances, shape [M+1, n_state, n_state].
        m_predictions: Predicted means, shape [M, n_state].
        P_predictions: Predicted covariances, shape [M, n_state, n_state].
        Gs: Backward gains, shape [M, n_state, n_state].
        ds: Backward offsets, shape [M, n_state].
        Lambdas: Backward covariances, shape [M, n_state, n_state].
    """
    #complete forward kalman filtering pass:
    m_sequence = [mu_0]
    P_sequence = [Sigma_0] 
    m_predictions = [mu_0]
    P_predictions = [Sigma_0]
    Gs, ds, Lambdas = [], [], []

    I = M//N
    for i in range(1,M+1):
        
        if i%I == 0:
            (m_pred_nxt, P_pred_nxt), (m_nxt, P_nxt) = ekf1_filter_step(A_h, b_h, Q_h, R_h, m_sequence[-1], P_sequence[-1], g, jacobian_g, z_sequence[i//I - 1,:])

        else:
            m_pred_nxt, P_pred_nxt = future_prediction(m_sequence[-1], P_sequence[-1], A_h, Q_h)
            m_nxt, P_nxt = m_pred_nxt, P_pred_nxt

        G_nxt, d_nxt, Lambda_nxt = inversion2(A_h, m_sequence[-1], P_sequence[-1], m_pred_nxt, P_pred_nxt)
        m_sequence.append(m_nxt)
        P_sequence.append(P_nxt)
        m_predictions.append(m_pred_nxt)
        P_predictions.append(P_pred_nxt)
        Gs.append(G_nxt)
        ds.append(d_nxt)
        Lambdas.append(Lambda_nxt)

    m_sequence = np.array(m_sequence)
    P_sequence = np.array(P_sequence)
    m_predictions = np.array(m_predictions)
    P_predictions = np.array(P_predictions)
    Gs = np.array(Gs)
    ds = np.array(ds)
    Lambdas = np.array(Lambdas)

    return m_sequence, P_sequence, m_predictions, P_predictions, Gs, ds, Lambdas


def future_prediction(m_t_minus, P_t_minus, A, Q):
    """Predict state distribution one step forward without observation.

    Applies the linear state transition and process noise to predict the
    state distribution at the next time step.

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


def predict_future(k, m_start, P_start, A_h, Q_h, N):
    """Predict future state distributions for N-k steps ahead.

    Iterates future_prediction to obtain the distribution at all future
    time points from step k to step N.

    Args:
        k: Current time step index.
        m_start: Current state mean (shape [n_state]).
        P_start: Current state covariance (shape [n_state, n_state]).
        A_h: State transition matrix (shape [n_state, n_state]).
        Q_h: Process noise covariance (shape [n_state, n_state]).
        N: Total number of time steps.

    Returns:
        m_future: Future means, shape [N-k+1, n_state].
        P_future: Future covariances, shape [N-k+1, n_state, n_state].
    """
    m_future = [m_start]
    P_future = [P_start]
    for _ in range(N-k):
        m_nxt, P_nxt = future_prediction(m_future[-1], P_future[-1], A_h, Q_h)
        m_future.append(m_nxt)
        P_future.append(P_nxt)

    m_future = np.array(m_future)
    P_future = np.array(P_future)
    return m_future, P_future

