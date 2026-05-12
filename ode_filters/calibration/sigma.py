"""Diffusion-scale estimators for probabilistic ODE filters.

All estimators consume the predicted-observation marginal ``(m_z, P_z_sqr)``
in the same square-root convention as the rest of the library:
``S = P_z_sqr.T @ P_z_sqr``. The whitened residual is

    v = P_z_sqr^{-T} @ m_z,    v.T @ v == m_z.T @ S^{-1} @ m_z.

References:
    Bosch, Tronarp, Hennig. "Calibrated Adaptive Probabilistic ODE Solvers."
    AISTATS 2021. Equations (32) and surrounding for the per-step quasi-MLE.
"""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike


def quasi_mle_sigma_sqr(m_z: Array, P_z_sqr: Array) -> Array:
    """Per-step quasi-MLE of the scalar diffusion sigma^2 (Bosch et al. 2021).

    Computes

        sigma_hat^2 = (1 / d) * m_z.T @ S^{-1} @ m_z

    where ``d = m_z.shape[0]`` and ``S = P_z_sqr.T @ P_z_sqr`` is the
    *prior-propagated* predicted-observation covariance.

    **Important: ``P_z_sqr`` must exclude any measurement noise R.** The
    quasi-MLE assumes ``S`` scales linearly with ``sigma^2`` (i.e. ``S =
    sigma^2 * H P_pred H.T``). If ``S`` is the full EKF-step output
    ``H P_pred H.T + R`` with ``R != 0``, this estimator conflates the
    diffusion scale with the measurement noise and is biased. Cf. probnum's
    ``meas_rv_error_free`` and probdiffeq's noise-free observed RV, which
    make the same exclusion.

    For pure ODE-information filtering (``measure.get_noise`` returns zero --
    the typical case) the distinction is moot. For data-assimilation
    workflows with ``R != 0``, pre-compute the noise-free covariance by
    calling :func:`~ode_filters.inference.sqr_gaussian_inference.sqr_marginalization`
    with a zero ``Q_sqr``.

    Args:
        m_z: Predicted observation mean (a.k.a. ODE residual mean), shape ``[d]``.
        P_z_sqr: Square-root of the *noise-free* predicted observation
            covariance, shape ``[d, d]`` (upper-triangular).

    Returns:
        Scalar JAX array containing the per-step quasi-MLE of sigma^2.
    """
    v = jax.scipy.linalg.solve_triangular(P_z_sqr.T, m_z, lower=True)
    d = m_z.shape[0]
    return (v @ v) / d


def posthoc_mle_sigma_sqr(mz_seq: ArrayLike, Pz_seq_sqr: ArrayLike) -> Array:
    """Closed-form MLE of sigma^2 over a fixed-step trajectory.

    Computes

        sigma_hat^2 = (1 / (N * d)) * sum_n m_z^{(n)} . T @ S_n^{-1} @ m_z^{(n)}

    which is the joint MLE under a constant sigma assumption. Equivalent to the
    mean of :func:`quasi_mle_sigma_sqr` applied step-by-step.

    The same R=0 caveat as :func:`quasi_mle_sigma_sqr` applies: ``Pz_seq_sqr``
    must be the *prior-propagated* predicted-observation covariance, with no
    measurement noise mixed in. For the typical fixed-step ODE-information
    run, ``Pz_seq_sqr`` from the loop output already satisfies this because
    ``ODEInformation`` returns ``R = 0`` by default.

    Args:
        mz_seq: Sequence of predicted observation means, shape ``[N, d]``.
        Pz_seq_sqr: Sequence of square-root observation covariances, shape
            ``[N, d, d]``.

    Returns:
        Scalar JAX array containing the MLE of sigma^2.
    """
    mz_arr = np.asarray(mz_seq)
    Pz_arr = np.asarray(Pz_seq_sqr)
    sigma_sqr_seq = jax.vmap(quasi_mle_sigma_sqr)(mz_arr, Pz_arr)
    return np.mean(sigma_sqr_seq)


def aggregate_sigma_sqr(
    sigma_sqr_seq: ArrayLike,
    *,
    kind: Literal["mean", "last", "running"] = "mean",
) -> Array:
    """Aggregate a sequence of per-step sigma^2 estimates.

    Args:
        sigma_sqr_seq: Per-step quasi-MLE estimates, shape ``[N]``.
        kind: Aggregation method.

            - ``"mean"``: arithmetic mean (closed-form joint MLE under constant
              sigma; equivalent to :func:`posthoc_mle_sigma_sqr`).
            - ``"last"``: use only the final per-step estimate.
            - ``"running"``: per-step running mean
              ``sigma_hat^2_n = (1/n) sum_{k<=n} sigma^2_k``. Returns an array
              of shape ``[N]`` rather than a scalar.

    Returns:
        Scalar JAX array (``"mean"``, ``"last"``) or 1-D array (``"running"``).
    """
    arr = np.asarray(sigma_sqr_seq)
    if kind == "mean":
        return np.mean(arr)
    if kind == "last":
        return arr[-1]
    if kind == "running":
        return np.cumsum(arr) / np.arange(1, arr.shape[0] + 1)
    raise ValueError(f"Unknown aggregator kind: {kind!r}")
