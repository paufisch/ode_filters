"""Diffusion-scale estimators for probabilistic ODE filters.

All estimators consume the predicted-observation marginal ``(m_z, P_z_sqr)``
in the same square-root convention as the rest of the library:
``S = P_z_sqr.T @ P_z_sqr``. The whitened residual is

    v = P_z_sqr^{-T} @ m_z,    v.T @ v == m_z.T @ S^{-1} @ m_z.

Two conventions for the residual covariance ``S`` correspond to two
different generative-model assumptions:

- ``S = H Q(h) H.T`` (process-noise only). The implicit model is
  ``m_z ~ N(0, sigma^2 H Q(h) H.T)``, i.e. the residual is treated as if
  it arose purely from the current step's process noise scaled by
  ``sigma^2``. This is the Bosch-Tronarp-Hennig 2021 per-step quasi-MLE
  (Eq. 32). The estimate is meant to be baked into the current step's
  ``Q_h`` *before* propagation; past steps keep their own calibration.
  :func:`quasi_mle_sigma_sqr_from_Q` is the convenience wrapper.

- ``S = H P_pred H.T`` (full noise-free residual cov, including
  ``A P_prev A.T``). The implicit model is ``sigma^2`` multiplying every
  covariance globally. Useful for *post-hoc* calibration of a sigma=1
  trajectory; cf. :func:`posthoc_mle_sigma_sqr`.

References:
    Bosch, Tronarp, Hennig. "Calibrated Adaptive Probabilistic ODE Solvers."
    AISTATS 2021. Equation (32) and surrounding for the per-step quasi-MLE.
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

    where ``d = m_z.shape[0]`` and ``S = P_z_sqr.T @ P_z_sqr`` is a
    *noise-free* predicted-observation covariance under the implicit
    assumption that ``S`` scales linearly with ``sigma^2``. Which ``S`` is
    "correct" depends on the calibration mode -- see the module docstring.

    **``P_z_sqr`` must exclude any measurement noise R.** The estimator
    treats ``S`` as scaling linearly with ``sigma^2``; adding ``R`` into the
    denominator inflates it by a constant and biases ``sigma_hat^2`` down.
    For pure ODE-information filtering (``measure.get_noise`` returns zero
    -- the typical case) the distinction is moot. For data-assimilation
    workflows with ``R != 0``, pre-compute the noise-free covariance via
    :func:`~ode_filters.inference.sqr_gaussian_inference.sqr_marginalization`
    with a zero ``R_sqr`` (or use :func:`quasi_mle_sigma_sqr_from_Q`, which
    builds ``H Q H.T`` directly and never sees ``R``).

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


def _sqr_HQH(H: Array, Q_sqr: Array) -> Array:
    """Upper-triangular square root of ``H @ Q @ H.T``.

    ``Q = Q_sqr.T @ Q_sqr``, so ``H Q H.T = (Q_sqr @ H.T).T @ (Q_sqr @ H.T)``;
    a thin QR of ``Q_sqr @ H.T`` returns the upper-triangular factor.
    """
    M = Q_sqr @ H.T
    _, R = np.linalg.qr(M)
    return R


def quasi_mle_sigma_sqr_from_Q(m_z: Array, H: Array, Q_sqr: Array) -> Array:
    """Per-step quasi-MLE of sigma^2 in the *dynamic-diffusion* convention.

    Uses ``S = H Q(h) H.T`` -- i.e. only the current step's process-noise
    propagation -- as the residual covariance. This is the Bosch-Tronarp-
    Hennig 2021 per-step quasi-MLE (Eq. 32). The resulting ``sigma_hat^2``
    is intended to be baked into ``Q_h`` for the current step's
    propagation; past-step contributions in ``A P_prev A.T`` keep their
    own previously-applied scaling.

    Args:
        m_z: Predicted observation mean (residual), shape ``[d]``.
        H: Linearised observation Jacobian, shape ``[d, n]``.
        Q_sqr: Square-root of the per-step process noise, shape ``[n, n]``.

    Returns:
        Scalar JAX array containing ``sigma_hat^2``.
    """
    Pz_Q_sqr = _sqr_HQH(H, Q_sqr)
    return quasi_mle_sigma_sqr(m_z, Pz_Q_sqr)


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
