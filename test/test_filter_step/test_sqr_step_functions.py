import numpy as np
import pytest

from ode_filters.ODE_filter_step import (
    ekf1_sqr_filter_step,
    rts_sqr_smoother_step,
)


def _linear_measurement(H, c):
    def g(x):
        return H @ x + c

    def jacobian(_):
        return H

    return g, jacobian


def _reconstruct_covariance(factor):
    return factor.T @ factor


def _dense_filter_step(A, b, Q, m_prev, P_prev, g, jacobian_g, z_observed, R):
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
    m_post = K @ z_observed + d

    return (m_pred, P_pred), (G_back, d_back, P_back), (m_z, P_z), (m_post, P_post)


def _dense_smoother_step(G_back, d_back, P_back, m_s, P_s):
    m_prev = G_back @ m_s + d_back
    P_prev = G_back @ P_s @ G_back.T + P_back
    return m_prev, P_prev


def test_ekf1_sqr_filter_step_matches_dense_linear_case():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    Q = np.array([[0.05, 0.0], [0.0, 0.02]])
    R = np.array([[0.1]])

    H = np.array([[1.0, 0.5]])
    c = np.array([0.2])

    m_prev = np.array([0.0, 1.0])
    P_prev = np.array([[0.4, 0.1], [0.1, 0.3]])
    z_observed = np.array([0.3])

    g, jacobian = _linear_measurement(H, c)

    dense_pred, dense_back, dense_obs, dense_post = _dense_filter_step(
        A, b, Q, m_prev, P_prev, g, jacobian, z_observed, R
    )

    P_prev_sqr = np.linalg.cholesky(P_prev, upper=True)
    Q_sqr = np.linalg.cholesky(Q, upper=True)
    R_sqr = np.linalg.cholesky(R, upper=True)

    sqr_pred, sqr_back, sqr_obs, sqr_post = ekf1_sqr_filter_step(
        A, b, Q_sqr, m_prev, P_prev_sqr, g, jacobian, z_observed, R_sqr
    )

    assert sqr_pred[0] == pytest.approx(dense_pred[0], rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance(sqr_pred[1]) == pytest.approx(
        dense_pred[1], rel=1e-12, abs=1e-12
    )

    assert sqr_back[0] == pytest.approx(dense_back[0], rel=1e-12, abs=1e-12)
    assert sqr_back[1] == pytest.approx(dense_back[1], rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance(sqr_back[2]) == pytest.approx(
        dense_back[2], rel=1e-12, abs=1e-12
    )

    assert sqr_obs[0] == pytest.approx(dense_obs[0], rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance(sqr_obs[1]) == pytest.approx(
        dense_obs[1], rel=1e-12, abs=1e-12
    )

    assert sqr_post[0] == pytest.approx(dense_post[0], rel=1e-12, abs=1e-12)
    assert _reconstruct_covariance(sqr_post[1]) == pytest.approx(
        dense_post[1], rel=1e-12, abs=1e-12
    )


def test_rts_sqr_smoother_step_matches_dense_linear_case():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    Q = np.array([[0.05, 0.0], [0.0, 0.02]])
    R = np.array([[0.1]])

    H = np.array([[1.0, 0.5]])
    c = np.array([0.2])

    m_prev = np.array([0.0, 1.0])
    P_prev = np.array([[0.4, 0.1], [0.1, 0.3]])
    z_observed = np.array([0.3])

    g, jacobian = _linear_measurement(H, c)

    dense_pred, dense_back, dense_obs, dense_post = _dense_filter_step(
        A, b, Q, m_prev, P_prev, g, jacobian, z_observed, R
    )

    P_prev_sqr = np.linalg.cholesky(P_prev, upper=True)
    Q_sqr = np.linalg.cholesky(Q, upper=True)
    R_sqr = np.linalg.cholesky(R, upper=True)

    sqr_pred, sqr_back, sqr_obs, sqr_post = ekf1_sqr_filter_step(
        A, b, Q_sqr, m_prev, P_prev_sqr, g, jacobian, z_observed, R_sqr
    )

    dense_smoothed_prev = _dense_smoother_step(
        dense_back[0], dense_back[1], dense_back[2], dense_post[0], dense_post[1]
    )

    sqr_smoothed_prev = rts_sqr_smoother_step(
        sqr_back[0], sqr_back[1], sqr_back[2], sqr_post[0], sqr_post[1]
    )

    assert sqr_smoothed_prev[0] == pytest.approx(
        dense_smoothed_prev[0], rel=1e-12, abs=1e-12
    )
    assert _reconstruct_covariance(sqr_smoothed_prev[1]) == pytest.approx(
        dense_smoothed_prev[1], rel=1e-12, abs=1e-12
    )
