import numpy as np
import pytest

from ode_filters.filters.ODE_filter_loop import (
    ekf1_sqr_loop,
    rts_sqr_smoother_loop,
)


def _linear_measurement(H, c):
    def g(x, *, t):
        return H @ x + c

    def jacobian(x, *, t):
        return

    return g, jacobian


def _reconstruct_covariance_sequence(factors):
    return np.matmul(factors.transpose(0, 2, 1), factors)


def _dense_filter_step(A, b, Q, m_prev, P_prev, g, jacobian_g, R):
    m_pred = A @ m_prev + b
    P_pred = A @ P_prev @ A.T + Q

    cross_cov = P_prev @ A.T
    G_back = np.linalg.solve(P_pred, cross_cov.T).T
    d_back = m_prev - G_back @ m_pred
    P_back = P_prev - G_back @ P_pred @ G_back.T

    H = jacobian_g(m_pred)
    c = g(m_pred) - H @ m_pred

    m_z = H @ m_pred + c
    P_z = H @ P_pred @ H.T + R

    innovation_cross = P_pred @ H.T
    K = np.linalg.solve(P_z, innovation_cross.T).T
    d = m_pred - K @ m_z
    P_post = P_pred - K @ P_z @ K.T
    m_post = d

    return (m_pred, P_pred), (G_back, d_back, P_back), (m_z, P_z), (m_post, P_post)


def ekf1_dense_loop(mu_0, Sigma_0, A, b, Q, R, g, jacobian_g, N):
    state_dim = mu_0.shape[0]
    obs_dim = R.shape[0]

    m_seq = np.empty((N + 1, state_dim))
    P_seq = np.empty((N + 1, state_dim, state_dim))
    m_pred_seq = np.empty((N, state_dim))
    P_pred_seq = np.empty((N, state_dim, state_dim))
    G_back_seq = np.empty((N, state_dim, state_dim))
    d_back_seq = np.empty((N, state_dim))
    P_back_seq = np.empty((N, state_dim, state_dim))
    mz_seq = np.empty((N, obs_dim))
    Pz_seq = np.empty((N, obs_dim, obs_dim))

    m_seq[0] = mu_0
    P_seq[0] = Sigma_0

    for i in range(N):
        (
            (m_pred_seq[i], P_pred_seq[i]),
            (G_back_seq[i], d_back_seq[i], P_back_seq[i]),
            (mz_seq[i], Pz_seq[i]),
            (m_seq[i + 1], P_seq[i + 1]),
        ) = _dense_filter_step(A, b, Q, m_seq[i], P_seq[i], g, jacobian_g, R)

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


def _dense_smoother_step(G_back, d_back, P_back, m_s, P_s):
    m_prev = G_back @ m_s + d_back
    P_prev = G_back @ P_s @ G_back.T + P_back
    return m_prev, P_prev


def rts_dense_smoother_loop(m_N, P_N, G_back_seq, d_back_seq, P_back_seq, N):
    state_dim = m_N.shape[0]

    m_smooth = np.empty((N + 1, state_dim))
    P_smooth = np.empty((N + 1, state_dim, state_dim))
    m_smooth[-1] = m_N
    P_smooth[-1] = P_N

    for j in range(N - 1, -1, -1):
        (m_smooth[j], P_smooth[j]) = _dense_smoother_step(
            G_back_seq[j],
            d_back_seq[j],
            P_back_seq[j],
            m_smooth[j + 1],
            P_smooth[j + 1],
        )

    return m_smooth, P_smooth


def test_ekf1_sqr_loop_matches_dense_linear_case():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    Q = np.array([[0.05, 0.0], [0.0, 0.02]])
    R = np.array([[0.1]])

    H = np.array([[1.0, 0.5]])
    c = np.array([0.2])

    mu_0 = np.array([0.0, 1.0])
    Sigma_0 = np.array([[0.4, 0.1], [0.1, 0.3]])

    num_steps = 3

    g, jacobian = _linear_measurement(
        H,
        c,
    )

    dense_results = ekf1_dense_loop(mu_0, Sigma_0, A, b, Q, R, g, jacobian, num_steps)

    Sigma_0_sqr = np.linalg.cholesky(Sigma_0, upper=True)
    Q_sqr = np.linalg.cholesky(Q, upper=True)
    R_sqr = np.linalg.cholesky(R, upper=True)

    sqr_results = ekf1_sqr_loop(
        mu_0, Sigma_0_sqr, A, b, Q_sqr, R_sqr, g, jacobian, num_steps
    )

    (
        dense_m_seq,
        dense_P_seq,
        dense_m_pred_seq,
        dense_P_pred_seq,
        dense_G_back_seq,
        dense_d_back_seq,
        dense_P_back_seq,
        dense_mz_seq,
        dense_Pz_seq,
    ) = dense_results

    (
        sqr_m_seq,
        sqr_P_seq_sqr,
        sqr_m_pred_seq,
        sqr_P_pred_seq_sqr,
        sqr_G_back_seq,
        sqr_d_back_seq,
        sqr_P_back_seq_sqr,
        sqr_mz_seq,
        sqr_Pz_seq_sqr,
    ) = sqr_results

    assert sqr_m_seq == pytest.approx(dense_m_seq, rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance_sequence(sqr_P_seq_sqr) == pytest.approx(
        dense_P_seq, rel=1e-12, abs=1e-12
    )

    assert sqr_m_pred_seq == pytest.approx(dense_m_pred_seq, rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance_sequence(sqr_P_pred_seq_sqr) == pytest.approx(
        dense_P_pred_seq, rel=1e-12, abs=1e-12
    )

    assert sqr_G_back_seq == pytest.approx(dense_G_back_seq, rel=1e-12, abs=1e-12)
    assert sqr_d_back_seq == pytest.approx(dense_d_back_seq, rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance_sequence(sqr_P_back_seq_sqr) == pytest.approx(
        dense_P_back_seq, rel=1e-12, abs=1e-12
    )

    assert sqr_mz_seq == pytest.approx(dense_mz_seq, rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance_sequence(sqr_Pz_seq_sqr) == pytest.approx(
        dense_Pz_seq, rel=1e-12, abs=1e-12
    )


def test_rts_sqr_smoother_loop_matches_dense_linear_case():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    Q = np.array([[0.05, 0.0], [0.0, 0.02]])
    R = np.array([[0.1]])

    H = np.array([[1.0, 0.5]])
    c = np.array([0.2])

    mu_0 = np.array([0.0, 1.0])
    Sigma_0 = np.array([[0.4, 0.1], [0.1, 0.3]])

    num_steps = 3

    g, jacobian = _linear_measurement(H, c)

    dense_results = ekf1_dense_loop(mu_0, Sigma_0, A, b, Q, R, g, jacobian, num_steps)

    Sigma_0_sqr = np.linalg.cholesky(Sigma_0, upper=True)
    Q_sqr = np.linalg.cholesky(Q, upper=True)
    R_sqr = np.linalg.cholesky(R, upper=True)

    sqr_results = ekf1_sqr_loop(
        mu_0, Sigma_0_sqr, A, b, Q_sqr, R_sqr, g, jacobian, num_steps
    )

    (
        dense_m_seq,
        dense_P_seq,
        _,
        _,
        dense_G_back_seq,
        dense_d_back_seq,
        dense_P_back_seq,
        _,
        _,
    ) = dense_results

    (
        sqr_m_seq,
        sqr_P_seq_sqr,
        _,
        _,
        sqr_G_back_seq,
        sqr_d_back_seq,
        sqr_P_back_seq_sqr,
        _,
        _,
    ) = sqr_results

    dense_m_smooth, dense_P_smooth = rts_dense_smoother_loop(
        dense_m_seq[-1],
        dense_P_seq[-1],
        dense_G_back_seq,
        dense_d_back_seq,
        dense_P_back_seq,
        num_steps - 1,
    )

    sqr_m_smooth, sqr_P_smooth_sqr = rts_sqr_smoother_loop(
        sqr_m_seq[-1],
        sqr_P_seq_sqr[-1],
        sqr_G_back_seq,
        sqr_d_back_seq,
        sqr_P_back_seq_sqr,
        num_steps - 1,
    )

    assert sqr_m_smooth == pytest.approx(dense_m_smooth, rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance_sequence(sqr_P_smooth_sqr) == pytest.approx(
        dense_P_smooth, rel=1e-12, abs=1e-12
    )
