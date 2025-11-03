import numpy as np
import pytest

from ode_filters.ODE_filter_loop import (
    ekf1_loop,
    ekf1_sqr_loop,
    rts_smoother_loop,
    rts_sqr_smoother_loop,
)


def _linear_measurement(H, c):
    def g(x):
        return H @ x + c

    def jacobian(_):
        return H

    return g, jacobian


def _reconstruct_covariance_sequence(factors):
    return np.matmul(factors.transpose(0, 2, 1), factors)


def test_ekf1_sqr_loop_matches_dense_linear_case():
    A = np.array([[1.0, 0.1], [0.0, 1.0]])
    b = np.array([0.0, 0.0])
    Q = np.array([[0.05, 0.0], [0.0, 0.02]])
    R = np.array([[0.1]])

    H = np.array([[1.0, 0.5]])
    c = np.array([0.2])

    mu_0 = np.array([0.0, 1.0])
    Sigma_0 = np.array([[0.4, 0.1], [0.1, 0.3]])

    z_sequence = np.array([[0.1], [0.2], [0.15]])
    num_steps = z_sequence.shape[0]

    g, jacobian = _linear_measurement(H, c)

    dense_results = ekf1_loop(
        mu_0, Sigma_0, A, b, Q, R, g, jacobian, z_sequence, num_steps
    )

    Sigma_0_sqr = np.linalg.cholesky(Sigma_0).T
    Q_sqr = np.linalg.cholesky(Q).T
    R_sqr = np.linalg.cholesky(R).T

    sqr_results = ekf1_sqr_loop(
        mu_0, Sigma_0_sqr, A, b, Q_sqr, R_sqr, g, jacobian, z_sequence, num_steps
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

    z_sequence = np.array([[0.1], [0.2], [0.15]])
    num_steps = z_sequence.shape[0]

    g, jacobian = _linear_measurement(H, c)

    dense_results = ekf1_loop(
        mu_0, Sigma_0, A, b, Q, R, g, jacobian, z_sequence, num_steps
    )

    Sigma_0_sqr = np.linalg.cholesky(Sigma_0).T
    Q_sqr = np.linalg.cholesky(Q).T
    R_sqr = np.linalg.cholesky(R).T

    sqr_results = ekf1_sqr_loop(
        mu_0, Sigma_0_sqr, A, b, Q_sqr, R_sqr, g, jacobian, z_sequence, num_steps
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

    dense_m_smooth, dense_P_smooth = rts_smoother_loop(
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
