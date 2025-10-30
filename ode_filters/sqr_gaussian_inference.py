from typing import Tuple

import numpy as np


def sqr_marginalization(
    A: np.ndarray,
    b: np.ndarray,
    Q_sqr: np.ndarray,
    mu: np.ndarray,
    Sigma_sqr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Marginalize out the linear transformation in a Gaussian model. Square root form to preserve positive definiteness.

    Computes the marginal distribution of z = Ax + b given p(x) ~ N(mu, Sigma)
    and p(z|x) ~ N(Ax + b, Q).
    The result is p(z) = N(mu_z, Sigma_z) with mu_z = A(mu)+b, Sigma_z = A(Sigma)A.T+Q

    Args:
        A: Linear transformation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q_sqr: Square root of Observation noise covariance (shape [n_obs, n_obs] or [n_obs]).
            If 1D array, will be converted to 2D.
        mu: Prior mean (shape [n_state]).
        Sigma_sqr: Square root of Prior covariance (shape [n_state, n_state]).

    Returns:
        mu_z: Marginal mean of z (shape [n_obs]).
        Sigma_z_sqr: Square root Marginal sqr covariance of z (shape [n_obs, n_obs]).

    Raises:
        ValueError: If input shapes are incompatible or invalid.

    """
    Q_sqr = np.atleast_2d(Q_sqr)

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[0]} rows but b has shape {b.shape[0]}. "
            "b must have the same number of elements as A has rows."
        )

    if Q_sqr.shape[0] != Q_sqr.shape[1]:
        raise ValueError(
            f"Shape mismatch: Q is expected to be of square shape but has shape {Q_sqr.shape}"
        )

    # Compute marginal statistics
    mu_z = A @ mu + b
    C = np.concatenate([A @ Sigma_sqr, Q_sqr], axis=0)
    _, Sigma_z_sqr = np.linalg.qr(C)

    return mu_z, Sigma_z_sqr


def sqr_inversion(
    A: np.ndarray,
    mu: np.ndarray,
    Sigma_sqr: np.ndarray,
    mu_z: np.ndarray,
    Sigma_z_sqr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inversion using square-root covariance representation for stability.

    Numerically stable Bayesian bayesian_update using Cholesky factors and QR decomposition. Returns the posterior mean and Cholesky factor.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        mu: Prior mean (shape [n_state]).
        Sigma_sqr: Square root of prior covariance (shape [n_state, n_state]).
        mu_z: Marginal observation mean (shape [n_obs]).
        Sigma_z_sqr: Square root of marginal observation covariance (shape [n_obs, n_obs]).

    Returns:
        G: Kalman gain matrix (shape [n_state, n_obs]).
        d: Posterior offset (shape [n_state]).
        Lambda_sqr: Posterior covariance (shape [n_state, n_state]).
    """

    Sigma_z_2d_sqr = np.atleast_2d(Sigma_z_sqr)
    Sigma_z_2d = Sigma_z_2d_sqr @ Sigma_z_2d_sqr.T
    Sigma = Sigma_sqr @ Sigma_sqr.T

    K = np.linalg.solve(Sigma_z_2d, (A @ Sigma)).T
    d = mu - K @ mu_z
    C = np.concatenate([-K @ Sigma_z_sqr, Sigma_sqr], axis=0)
    _, Lambda_sqr = np.linalg.qr(C)

    return K, d, Lambda_sqr
