from typing import Tuple

import numpy as np


def marginalization(
    A: np.ndarray, b: np.ndarray, Q: np.ndarray, mu: np.ndarray, Sigma: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Marginalize out the linear transformation in a Gaussian model.

    Computes the marginal distribution of z = Ax + b given p(x) ~ N(mu, Sigma)
    and p(z|x) ~ N(Ax + b, Q).
    The result is p(z) = N(mu_z, Sigma_z) with mu_z = A(mu)+b, Sigma_z = A(Sigma)A.T+Q

    Args:
        A: Linear transformation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q: Observation noise covariance (shape [n_obs, n_obs] or [n_obs]).
            If 1D array, will be converted to 2D.
        mu: Prior mean (shape [n_state]).
        Sigma: Prior covariance (shape [n_state, n_state]).

    Returns:
        mu_z: Marginal mean of z (shape [n_obs]).
        Sigma_z: Marginal covariance of z (shape [n_obs, n_obs]).

    Raises:
        ValueError: If input shapes are incompatible or invalid.

    Example:
        >>> A = np.array([[1.0, 0.0]])
        >>> b = np.array([0.0])
        >>> Q = np.array([[0.1]])
        >>> mu = np.array([1.0, 2.0])
        >>> Sigma = np.eye(2)
        >>> mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)
    """
    Q = np.atleast_2d(Q)

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[0]} rows but b has shape {b.shape[0]}. "
            "b must have the same number of elements as A has rows."
        )

    if Q.shape[0] != Q.shape[1]:
        raise ValueError(
            f"Shape mismatch: Q is expected to be of square shape but has shape {Q.shape}"
        )

    # Compute marginal statistics
    mu_z = A @ mu + b
    Sigma_z = A @ Sigma @ A.T + Q

    return mu_z, Sigma_z


# this is a full bayesian update including the marginalization first
def inversion(A, b, Q, mu, Sigma, z):
    """Compute the posterior of x given observation z using Bayesian inversion.

    Given p(x) ~ N(mu, Sigma), p(z|x) ~ N(Ax + b, Q), and an observation z,
    computes p(x|z) ~ N(G@z + d, Lambda) efficiently using Cholesky decomposition.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q: Observation noise covariance (shape [n_obs, n_obs]).
        mu: Prior mean (shape [n_state]).
        Sigma: Prior covariance (shape [n_state, n_state]).
        z: Observation value (shape [n_obs]).

    Returns:
        posterior_mean: Posterior mean N(G@z + d, Lambda) (shape [n_state]).
        Lambda: Posterior covariance (shape [n_state, n_state]).
    """

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)
    Sigma_z_2d = np.atleast_2d(Sigma_z)
    # Solve linear system instead of computing inverse explicitly
    # K = np.linalg.solve(Sigma_z_2d, (Sigma @ A.T).T).T
    K = np.linalg.solve(Sigma_z_2d, (A @ Sigma)).T

    d = mu - K @ mu_z
    Lambda = Sigma - K @ Sigma_z @ K.T  # == Sigma - K@A@Sigma
    return K @ z + d, Lambda


# in contrast to inversion1 above, this invertes the marginalizes distribution
def inversion2(A, mu, Sigma, mu_z, Sigma_z):
    """Compute posterior parameters without explicit observation value.

    Variant of inversion that returns the gain matrix G, offset d, and posterior
    covariance Lambda separately, given pre-computed marginal parameters.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        mu: Prior mean (shape [n_state]).
        Sigma: Prior covariance (shape [n_state, n_state]).
        mu_z: Marginal observation mean (shape [n_obs]).
        Sigma_z: Marginal observation covariance (shape [n_obs, n_obs]).

    Returns:
        G: Kalman gain matrix (shape [n_state, n_obs]).
        d: Posterior offset (shape [n_state]).
        Lambda: Posterior covariance (shape [n_state, n_state]).
    """

    Sigma_z_2d = np.atleast_2d(Sigma_z)
    # Solve linear system instead of computing inverse explicitly
    # K = np.linalg.solve(Sigma_z_2d, (Sigma @ A.T).T).T
    K = np.linalg.solve(Sigma_z_2d, (A @ Sigma)).T

    d = mu - K @ mu_z
    Lambda = Sigma - K @ Sigma_z @ K.T  # == Sigma - K@A@Sigma
    return K, d, Lambda
