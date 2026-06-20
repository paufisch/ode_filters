"""Square-root covariance rescaling utilities for post-hoc calibration.

After a fixed-step run with ``sigma = 1``, calibrating the trajectory amounts
to multiplying every covariance by sigma^2. In square-root form this is a
multiplication of every ``P_sqr`` by ``sqrt(sigma^2)``.

For joint / latent-force priors (``JointPrior`` / ``PrecondJointPrior``) the
diffusion ``sigma^2`` belongs to the ODE-*state* block only; the hidden/input
block must be left untouched (Schmidt et al. 2021). Pass ``prior=`` to route the
rescaling through ``prior.apply_state_sigma_to_cov_sqr``, which scales only the
state block for joint priors (and the whole covariance for non-joint priors, so
the result is identical to the plain path there).
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike


class _StateRescalable(Protocol):
    """A prior exposing the block-aware post-hoc rescaling hook."""

    def apply_state_sigma_to_cov_sqr(
        self, P_sqr: Array, sigma_sqr: ArrayLike
    ) -> Array: ...


def rescale_sqr(
    P_sqr: Array, sigma_sqr: ArrayLike, *, prior: _StateRescalable | None = None
) -> Array:
    """Rescale a single square-root covariance by sqrt(sigma^2).

    If ``P = P_sqr.T @ P_sqr``, then ``(sqrt(s) * P_sqr).T @ (sqrt(s) * P_sqr)
    == s * P``. Used to apply a per-step or global sigma to stored covariances.

    Args:
        P_sqr: Square-root covariance.
        sigma_sqr: Non-negative scalar; the variance multiplier.
        prior: Optional prior. When given, the rescaling is routed through
            ``prior.apply_state_sigma_to_cov_sqr`` so that joint /
            latent-force priors scale only the ODE-state block and leave the
            hidden/input block unchanged. For non-joint priors this is identical
            to the default whole-covariance scaling. **Pass this for any
            ``JointPrior`` / ``PrecondJointPrior`` result** -- omitting it scales
            the input block too, which is incorrect.

    Returns:
        Rescaled square-root covariance, same shape as ``P_sqr``.
    """
    if prior is not None:
        return prior.apply_state_sigma_to_cov_sqr(P_sqr, sigma_sqr)
    return np.sqrt(np.asarray(sigma_sqr)) * P_sqr


def rescale_sqr_seq(
    P_seq_sqr: ArrayLike,
    sigma_sqr: ArrayLike,
    *,
    prior: _StateRescalable | None = None,
) -> Array:
    """Rescale a sequence of square-root covariances.

    Args:
        P_seq_sqr: Stacked square-root covariances, shape ``[N, ...]``.
        sigma_sqr: Either a scalar (applied globally) or a 1-D array of
            length ``N`` (applied per step).
        prior: Optional prior; see :func:`rescale_sqr`. When given, joint /
            latent-force priors rescale only the state block.

    Returns:
        Rescaled stack, same shape as ``P_seq_sqr``.
    """
    P_arr = np.asarray(P_seq_sqr)
    s = np.asarray(sigma_sqr)

    if prior is not None:
        if s.ndim == 0:
            return jax.vmap(lambda P: prior.apply_state_sigma_to_cov_sqr(P, s))(P_arr)
        if s.ndim == 1 and s.shape[0] == P_arr.shape[0]:
            return jax.vmap(prior.apply_state_sigma_to_cov_sqr)(P_arr, s)
        raise ValueError(
            f"sigma_sqr must be scalar or shape ({P_arr.shape[0]},); got {s.shape}"
        )

    factor = np.sqrt(s)
    if factor.ndim == 0:
        return factor * P_arr
    if factor.ndim == 1 and factor.shape[0] == P_arr.shape[0]:
        return factor.reshape((-1,) + (1,) * (P_arr.ndim - 1)) * P_arr
    raise ValueError(
        f"sigma_sqr must be scalar or shape ({P_arr.shape[0]},); got {s.shape}"
    )
