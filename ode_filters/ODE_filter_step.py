from ode_filters.gaussian_inference import inversion, marginalization


def ekf1_filter_step(A_t, b_t, Q_t, m_prev, P_prev, g, jacobian_g, z_observed_t, R_t):
    """One Filter step"""

    m_pred, P_pred = marginalization(A_t, b_t, Q_t, m_prev, P_prev)
    G_back, d_back, P_back = inversion(A_t, m_prev, P_prev, m_pred, P_pred)

    H_t = jacobian_g(m_pred)
    c_t = g(m_pred) - H_t @ m_pred

    m_z, P_z = marginalization(H_t, c_t, R_t, m_pred, P_pred)
    K_t, d, P_t = inversion(H_t, m_pred, P_pred, m_z, P_z)
    m_t = K_t @ z_observed_t + d

    return (m_pred, P_pred), (G_back, d_back, P_back), (m_z, P_z), (m_t, P_t)


def rts_smoother_step(G_back, d_back, P_back, m_s, P_s):
    m_s_prev, P_s_prev = marginalization(G_back, d_back, P_back, m_s, P_s)
    return (m_s_prev, P_s_prev)
