from ode_filters.sqr_gaussian_inference import sqr_inversion, sqr_marginalization


# note that sqr inversion additionally takes Q/R as argument
# all covariance matrices are saved and propagated in sqr form.
# E.g. A = A_sqr.T @ A_sqr
def ekf1_sqr_filter_step(
    A_t, b_t, Q_t_sqr, m_prev, P_prev_sqr, g, jacobian_g, z_observed_t, R_t_sqr
):
    """One Filter step"""

    m_pred, P_pred_sqr = sqr_marginalization(A_t, b_t, Q_t_sqr, m_prev, P_prev_sqr)
    G_back, d_back, P_back_sqr = sqr_inversion(
        A_t, m_prev, P_prev_sqr, m_pred, P_pred_sqr, Q_t_sqr
    )

    H_t = jacobian_g(m_pred)
    c_t = g(m_pred) - H_t @ m_pred

    m_z, P_z_sqr = sqr_marginalization(H_t, c_t, R_t_sqr, m_pred, P_pred_sqr)
    K_t, d, P_t_sqr = sqr_inversion(H_t, m_pred, P_pred_sqr, m_z, P_z_sqr, R_t_sqr)
    m_t = K_t @ z_observed_t + d

    return (
        (m_pred, P_pred_sqr),
        (G_back, d_back, P_back_sqr),
        (m_z, P_z_sqr),
        (m_t, P_t_sqr),
    )


def rts_sqr_smoother_step(G_back, d_back, P_back_sqr, m_s, P_s_sqr):
    m_s_prev, P_s_prev_sqr = sqr_marginalization(
        G_back, d_back, P_back_sqr, m_s, P_s_sqr
    )
    return (m_s_prev, P_s_prev_sqr)
