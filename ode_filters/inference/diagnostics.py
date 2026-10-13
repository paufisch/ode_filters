"""Consistency diagnostics and predictive distributions.

A filter whose model is consistent with the data produces whitened innovations
``v_n = Pz_sqr_n^-T mz_n`` that are independent standard normal: their squared
norms (the normalized innovation squared, NIS) are chi-squared with as many
degrees of freedom as the step has rows, and they are uncorrelated in time.
Systematic departures point at a misspecified prior, diffusion or noise.
"""

from __future__ import annotations

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian import Gaussian, marginalize


def whitened_residuals(mz: ArrayLike, Pz_sqr: ArrayLike) -> Array:
    """Whitened innovations ``Pz_sqr^-T mz`` per step.

    Args:
        mz: Innovation means, shape ``[N, k]`` (e.g. ``result.mz_obs``).
        Pz_sqr: Their square-root covariances, shape ``[N, k, k]``.

    Returns:
        Shape ``[N, k]``; independent standard normal under a consistent model.
        Rows without data (zero mean, unit variance in ``mz_obs``) are zero.
    """
    mz, Pz_sqr = np.asarray(mz), np.asarray(Pz_sqr)
    return jax.vmap(lambda m, S: jax.scipy.linalg.solve_triangular(S.T, m, lower=True))(
        mz, Pz_sqr
    )


def nis(mz: ArrayLike, Pz_sqr: ArrayLike) -> Array:
    """Normalized innovation squared per step, ``mz.T Pz^-1 mz``.

    Chi-squared with ``k`` degrees of freedom under a consistent model (``k`` the
    number of rows with data at that step; rows without data contribute zero).

    Returns:
        Shape ``[N]``.
    """
    v = whitened_residuals(mz, Pz_sqr)
    return np.sum(v * v, axis=-1)


def residual_autocorrelation(whitened: ArrayLike, max_lag: int) -> Array:
    """Sample autocorrelation of whitened residuals, per component.

    Args:
        whitened: Whitened residuals, shape ``[N, k]`` (``whitened_residuals``).
        max_lag: Largest lag.

    Returns:
        Shape ``[max_lag, k]``: lags ``1..max_lag``. Under a consistent model the
        entries are close to zero, within about ``2 / sqrt(N)``.
    """
    v = np.asarray(whitened)
    v = v - v.mean(axis=0)
    variance = np.sum(v * v, axis=0)
    lags = [
        np.sum(v[lag:] * v[:-lag], axis=0) / variance for lag in range(1, max_lag + 1)
    ]
    return np.stack(lags)


def observation_predictive(
    marginals: Gaussian, H: ArrayLike, R_sqr: ArrayLike | None = None
) -> Gaussian:
    """Predictive distribution of linear observations ``y = H x + v``.

    Args:
        marginals: State marginals, stacked (e.g. ``rts_smoother(result)`` for
            the posterior predictive, or ``interpolate(...)`` at any times).
        H: Observation matrix, shape ``[k, D]``.
        R_sqr: Square root of the observation noise, ``[k, k]``; ``None`` for
            the noise-free prediction of ``H x``.

    Returns:
        The predictive ``Gaussian`` per marginal: means ``[M, k]``, square-root
        covariances ``[M, k, k]``.
    """
    H = np.asarray(H)
    k = H.shape[0]
    noise = np.zeros((k, k)) if R_sqr is None else np.asarray(R_sqr)
    return jax.vmap(lambda m, P: marginalize(H, np.zeros(k), noise, Gaussian(m, P)))(
        marginals.m, marginals.P_sqr
    )
