from __future__ import annotations

import jax.numpy as np
from jax import Array


def sqr_marginalization(
    A: Array,
    b: Array,
    Q_sqr: Array,
    mu: Array,
    Sigma_sqr: Array,
) -> tuple[Array, Array]:
    """Marginalize out the linear transformation in a Gaussian model using square-root form.

    Computes the marginal distribution of z = Ax + b given p(x) ~ N(mu, Sigma)
    and p(z|x) ~ N(Ax + b, Q). The result is p(z) = N(mu_z, Sigma_z) where:
    - mu_z = A @ mu + b
    - Sigma_z = A @ Sigma @ A.T + Q

    The square-root form is preserved to maintain numerical stability.

    Args:
        A: Linear transformation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q_sqr: Square root of observation noise covariance. Shape [n_obs, n_obs] or [n_obs].
            If 1D array, will be converted to 2D.
        mu: Prior mean (shape [n_state]).
        Sigma_sqr: Square root of prior covariance (shape [n_state, n_state]).

    Returns:
        Tuple of (mu_z, Sigma_z_sqr) where:
        - mu_z is the marginal mean of z (shape [n_obs])
        - Sigma_z_sqr is the square root of marginal covariance of z (shape [n_obs, n_obs])

    Raises:
        ValueError: If input shapes are incompatible or invalid.
    """
    Q_sqr = np.atleast_2d(Q_sqr)
    Sigma_sqr = np.atleast_2d(Sigma_sqr)

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[0]} rows but b has shape {b.shape[0]}. "
            "b must have the same number of elements as A has rows."
        )

    if Q_sqr.shape[0] != Q_sqr.shape[1]:
        raise ValueError(
            f"Shape mismatch: Q is expected to be of square shape but has shape {Q_sqr.shape}"
        )

    if Sigma_sqr.shape[0] != Sigma_sqr.shape[1]:
        raise ValueError(
            f"Shape mismatch: Sigma_sqr is expected to be of square shape but has shape {Sigma_sqr.shape}"
        )

    if A.shape[1] != Sigma_sqr.shape[0]:
        raise ValueError(
            f"Shape mismatch: A and Sigma_sqr should have matching first shapes, but have shape A={A.shape}, Sigma_sqr={Sigma_sqr.shape}"
        )

    # Compute marginal statistics
    mu_z = A @ mu + b
    C = np.concatenate([Sigma_sqr @ A.T, Q_sqr], axis=0)
    _, Sigma_z_sqr = np.linalg.qr(C)

    return mu_z, Sigma_z_sqr


def sqr_inversion(
    A: Array,
    mu: Array,
    Sigma_sqr: Array,
    mu_z: Array,
    Sigma_z_sqr: Array,
    Q_sqr: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Numerically stable Bayesian inference using square-root representations.

    Performs Bayesian update of a Gaussian given a linear observation model, using
    Cholesky factors and QR decomposition to maintain numerical stability.

    Given p(x) ~ N(mu, Sigma) and p(z|x) ~ N(Ax + b, Q), computes the posterior
    p(x|z) in square-root form.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        mu: Prior mean (shape [n_state]).
        Sigma_sqr: Square root of prior covariance (shape [n_state, n_state]).
        mu_z: Marginal observation mean (shape [n_obs]).
        Sigma_z_sqr: Square root of marginal observation covariance (shape [n_obs, n_obs]).
        Q_sqr: Square root of measurement noise covariance (optional).

    Returns:
        Tuple of (G, d, Lambda_sqr) where:
        - G is the Kalman gain matrix (shape [n_state, n_obs])
        - d is the posterior offset/mean correction (shape [n_state])
        - Lambda_sqr is the posterior covariance square root (shape [n_state, n_state])
    """
    if Q_sqr is None:
        Q_sqr = np.zeros_like(Sigma_z_sqr)

    n_state = A.shape[1]

    Sigma_z_sqr = np.atleast_2d(Sigma_z_sqr)
    Sigma_z = Sigma_z_sqr.T @ Sigma_z_sqr
    Sigma = Sigma_sqr.T @ Sigma_sqr

    K = np.linalg.solve(Sigma_z, A @ Sigma).T
    d = mu - K @ mu_z
    B = np.eye(n_state) - K @ A
    C = np.concatenate([Sigma_sqr @ B.T, (Q_sqr @ K.T).reshape(-1, n_state)], axis=0)
    _, Lambda_sqr = np.linalg.qr(C)

    return K, d, Lambda_sqr
