import numpy as np

from ode_filters.ODE_filter_step import (
    ekf1_sqr_filter_step,
    rts_sqr_smoother_step,
)


# extended kalman filter of order 1 with initialization:
# special case for fixed grid filtering
# also this is time invarant in the sense that A_h, b_h, Q_h and R_h are time invariant
def ekf1_sqr_loop(
    mu_0, Sigma_0_sqr, A_h, b_h, Q_h_sqr, R_h_sqr, g, jacobian_g, z_sequence, N
):
    # Determine dimensions from first iteration or function signature
    state_dim = mu_0.shape[0]
    obs_dim = z_sequence.shape[1]

    # Pre-allocate all arrays
    m_seq = np.empty((N + 1, state_dim))
    P_seq_sqr = np.empty((N + 1, state_dim, state_dim))
    m_pred_seq = np.empty((N, state_dim))
    P_pred_seq_sqr = np.empty((N, state_dim, state_dim))
    G_back_seq = np.empty((N, state_dim, state_dim))
    d_back_seq = np.empty((N, state_dim))
    P_back_seq_sqr = np.empty((N, state_dim, state_dim))
    mz_seq = np.empty((N, obs_dim))
    Pz_seq_sqr = np.empty((N, obs_dim, obs_dim))

    # Initialize first values
    m_seq[0] = mu_0
    P_seq_sqr[0] = Sigma_0_sqr

    # Fill in the loop
    for i in range(N):
        (
            (m_pred_seq[i], P_pred_seq_sqr[i]),
            (G_back_seq[i], d_back_seq[i], P_back_seq_sqr[i]),
            (mz_seq[i], Pz_seq_sqr[i]),
            (m_seq[i + 1], P_seq_sqr[i + 1]),
        ) = ekf1_sqr_filter_step(
            A_h,
            b_h,
            Q_h_sqr,
            m_seq[i],
            P_seq_sqr[i],
            g,
            jacobian_g,
            z_sequence[i],
            R_h_sqr,
        )

    # P_seq = np.matmul(np.transpose(P_seq_sqr, (0, 2, 1)), P_seq_sqr)
    # P_pred_seq = np.matmul(np.transpose(P_pred_seq_sqr, (0, 2, 1)), P_pred_seq_sqr)
    # Pz_seq = np.matmul(np.transpose(Pz_seq_sqr, (0, 2, 1)), Pz_seq_sqr)
    # P_back_seq = np.matmul(np.transpose(P_back_seq_sqr, (0, 2, 1)), P_back_seq_sqr)

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


def rts_sqr_smoother_loop(m_N, P_N_sqr, G_back_seq, d_back_seq, P_back_seq_sqr, N):
    state_dim = m_N.shape[0]

    # Pre-allocate all arrays
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
