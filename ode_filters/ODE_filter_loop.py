import numpy as np

from ode_filters.ODE_filter_step import ekf1_filter_step, rts_smoother_step


# extended kalman filter of order 1 with initialization:
# special case for fixed grid filtering
def ekf1_loop(mu_0, Sigma_0, A_h, b_h, Q_h, R_h, g, jacobian_g, z_sequence, N):
    # Determine dimensions from first iteration or function signature
    state_dim = mu_0.shape[0]
    obs_dim = z_sequence.shape[1]

    # Pre-allocate all arrays
    m_seq = np.empty((N + 1, state_dim))
    P_seq = np.empty((N + 1, state_dim, state_dim))
    m_pred_seq = np.empty((N, state_dim))
    P_pred_seq = np.empty((N, state_dim, state_dim))
    G_back_seq = np.empty((N, state_dim, state_dim))
    d_back_seq = np.empty((N, state_dim))
    P_back_seq = np.empty((N, state_dim, state_dim))
    mz_seq = np.empty((N, obs_dim))
    Pz_seq = np.empty((N, obs_dim, obs_dim))

    # Initialize first values
    m_seq[0] = mu_0
    P_seq[0] = Sigma_0

    # Fill in the loop
    for i in range(N):
        (
            (m_pred_seq[i], P_pred_seq[i]),
            (G_back_seq[i], d_back_seq[i], P_back_seq[i]),
            (mz_seq[i], Pz_seq[i]),
            (m_seq[i + 1], P_seq[i + 1]),
        ) = ekf1_filter_step(
            A_h, b_h, Q_h, m_seq[i], P_seq[i], g, jacobian_g, z_sequence[i], R_h
        )

    return (
        m_seq,
        P_seq,
        m_pred_seq,
        P_pred_seq,
        G_back_seq,
        d_back_seq,
        P_back_seq,
        mz_seq,
        Pz_seq,
    )


def rts_smoother_loop(m_N, P_N, G_back_seq, d_back_seq, P_back_seq, N):
    state_dim = m_N.shape[0]

    # Pre-allocate all arrays
    m_smooth = np.empty((N + 1, state_dim))
    P_smooth = np.empty((N + 1, state_dim, state_dim))
    m_smooth[-1] = m_N
    P_smooth[-1] = P_N

    for j in range(N, -1, -1):
        (m_smooth[j], P_smooth[j]) = rts_smoother_step(
            G_back_seq[j],
            d_back_seq[j],
            P_back_seq[j],
            m_smooth[j + 1],
            P_smooth[j + 1],
        )

    return m_smooth, P_smooth
