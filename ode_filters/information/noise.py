"""Noise covariances given as a variance, per-row variances or a matrix."""

from __future__ import annotations

import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _check_noise_covariance


def noise_matrix(value: ArrayLike, dim: int, name: str) -> Array:
    """A ``[dim, dim]`` covariance from a scalar, a length-``dim`` vector or a matrix.

    Raises:
        ValueError: For another shape, or for a concrete covariance the filters
            cannot factorize (see ``_check_noise_covariance``).
    """
    value = np.asarray(value, dtype=float)
    if value.ndim == 0:
        R = value * np.eye(dim)
    elif value.ndim == 1 and value.shape == (dim,):
        R = np.diag(value)
    elif value.shape == (dim, dim):
        R = value
    else:
        raise ValueError(
            f"{name} must be a scalar, a length-{dim} vector or a ({dim}, {dim}) "
            f"matrix, got shape {value.shape}."
        )
    _check_noise_covariance(R, name)
    return R
