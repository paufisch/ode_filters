"""Square-root covariance rescaling utilities for post-hoc calibration.

After a fixed-step run with ``sigma = 1``, calibrating the trajectory amounts
to multiplying every covariance by sigma^2. In square-root form this is a
multiplication of every ``P_sqr`` by ``sqrt(sigma^2)``.
"""

from __future__ import annotations

import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike


def rescale_sqr(P_sqr: Array, sigma_sqr: ArrayLike) -> Array:
    """Rescale a single square-root covariance by sqrt(sigma^2).

    If ``P = P_sqr.T @ P_sqr``, then ``(sqrt(s) * P_sqr).T @ (sqrt(s) * P_sqr)
    == s * P``. Used to apply a per-step or global sigma to stored covariances.

    Args:
        P_sqr: Square-root covariance.
        sigma_sqr: Non-negative scalar; the variance multiplier.

    Returns:
        Rescaled square-root covariance, same shape as ``P_sqr``.
    """
    return np.sqrt(np.asarray(sigma_sqr)) * P_sqr


def rescale_sqr_seq(P_seq_sqr: ArrayLike, sigma_sqr: ArrayLike) -> Array:
    """Rescale a sequence of square-root covariances.

    Args:
        P_seq_sqr: Stacked square-root covariances, shape ``[N, ...]``.
        sigma_sqr: Either a scalar (applied globally) or a 1-D array of
            length ``N`` (applied per step).

    Returns:
        Rescaled stack, same shape as ``P_seq_sqr``.
    """
    P_arr = np.asarray(P_seq_sqr)
    s = np.asarray(sigma_sqr)
    factor = np.sqrt(s)
    if factor.ndim == 0:
        return factor * P_arr
    if factor.ndim == 1 and factor.shape[0] == P_arr.shape[0]:
        return factor.reshape((-1,) + (1,) * (P_arr.ndim - 1)) * P_arr
    raise ValueError(
        f"sigma_sqr must be scalar or shape ({P_arr.shape[0]},); got {s.shape}"
    )
