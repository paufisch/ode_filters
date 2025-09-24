#single step of a kalman filter
from src.gaussian_inference import *
from src.sqr_gaussian_inference import *



def ekf1_filter_step(A_t, b_t, Q_t, R_t, mu_t, Sigma_t, g, jacobian_g, z_observed):

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
    
    #G = P_t @ A.T @ np.linalg.inv(P_pred)
    #instead of the naive inverse computation more efficiently:
    L, lower = cho_factor(P_pred, lower=True, check_finite=False)
    Y = cho_solve((L, lower), (P_t @ A.T).T, check_finite=False)
    G = Y.T
    m_t_s = m_t + G @ (m_smooth - m_pred)
    P_t_s = P_t + G @ (P_smooth - P_pred) @ G.T

    return m_t_s, P_t_s


def compute_kalman_backward(m_seq, P_seq, m_pred, P_pred, A_h, N):
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
    """
    Perform backward sampling for a given number of samples.

    Args:
        num_samples (int): Number of sample paths to generate.
        m_sequence (np.ndarray): Sequence of means from the smoother (T+1, d).
        P_sequence (np.ndarray): Sequence of covariances from the smoother (T+1, d, d).
        Gs (np.ndarray): Backward transition matrices (N, d, d).
        ds (np.ndarray): Backward transition offsets (N, d).
        Lambdas (np.ndarray): Backward transition covariances (N, d, d).
        N (int): Number of time steps (T).

    Returns:
        np.ndarray: Sampled paths of shape (num_samples, N+1, d).
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


def future_prediction(m_t_minus, P_t_minus, A, Q):
    m_t_minus = A @ m_t_minus
    P_t_minus = A @ P_t_minus @ A.T + Q
    return m_t_minus, P_t_minus


def predict_future(k, m_sequence, P_sequence, A_h, Q_h, N):
    m_future = [m_sequence[k,:]]
    P_future = [P_sequence[k,...]]
    for l in range(N-k):
        m_nxt, P_nxt = future_prediction(m_future[-1], P_future[-1], A_h, Q_h)
        m_future.append(m_nxt)
        P_future.append(P_nxt)

    m_future = np.array(m_future)
    P_future = np.array(P_future)
    return m_future, P_future

