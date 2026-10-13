"""Data observations: ``Measurement``, ``ObsModel`` and ``prepare_observations``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _check_noise_covariance, _safe_cholesky_sqr

# Default measurement noise for observations
DEFAULT_MEASUREMENT_NOISE = 1e-6

# Tolerances for matching a query time t against Measurement.z_t. Set to
# absorb the ~1e-8 drift between numpy.linspace and jax.numpy.linspace on
# typical grids. Re-exported so the adaptive loop's "did we land on a
# measurement time?" check uses the same tolerance as find_index.
MEASUREMENT_TIME_RTOL = 1e-5
MEASUREMENT_TIME_ATOL = 1e-7


class ObsModel(NamedTuple):
    """Pre-computed observation data for sequential filtering.

    Describes linear, time-invariant observations where the observation
    matrix ``H`` and noise covariance are constant across time and only
    the measurement values vary per step.  Built by
    ``prepare_observations`` from a list of ``Measurement``
    objects and a time grid.

    Attributes:
        H: Observation Jacobian (constant), shape ``[obs_dim, state_dim]``.
        R_sqr: Square-root noise covariance for active observations
            (constant), shape ``[obs_dim, obs_dim]``.
        c_seq: Observation offsets per step, shape ``[N, obs_dim]``.
            Equal to ``-z[idx]`` when the observation is active, zero
            otherwise.
        mask: Boolean mask for active dimensions, shape ``[N, obs_dim]``.
    """

    H: Array
    R_sqr: Array
    c_seq: Array
    mask: Array


def prepare_observations(
    observations: list[Measurement],
    E0: ArrayLike,
    ts: ArrayLike,
) -> ObsModel | None:
    """Build an ``ObsModel`` from ``Measurement`` objects.

    Assumes linear, time-invariant observation matrices (``A`` and noise
    are constant across time; only the measurement values ``z`` vary).

    Args:
        observations: List of Measurement constraints.
        E0: State extraction matrix, shape ``[d, state_dim]``.
        ts: Time grid of shape ``[N+1]`` (includes initial time ``t0``).
            The filter uses ``ts[1:]`` for the ``N`` filter steps.

    Returns:
        An ``ObsModel``, or ``None`` when *observations* is empty.
    """
    if not observations:
        return None

    E0 = np.asarray(E0)
    ts = np.asarray(ts)
    N = len(ts) - 1
    state_dim = E0.shape[1]
    max_obs_dim = sum(m.dim for m in observations)

    # --- Constant Jacobian and noise (computed once) ---
    H = np.zeros((max_obs_dim, state_dim))
    R_block = np.zeros((max_obs_dim, max_obs_dim))
    offset = 0
    for m in observations:
        H_m = m.A if m.full_state else m.A @ E0
        H = H.at[offset : offset + m.dim, :].set(H_m)
        R_block = R_block.at[offset : offset + m.dim, offset : offset + m.dim].set(
            m.get_noise_matrix()
        )
        offset += m.dim

    R_sqr = _safe_cholesky_sqr(R_block, max_obs_dim)

    # --- Per-step offsets and masks ---
    c_list: list[Array] = []
    mask_list: list[Array] = []

    for i in range(N):
        t = float(ts[i + 1])
        c_i = np.zeros(max_obs_dim)
        mask_i = np.zeros(max_obs_dim, dtype=bool)

        offset = 0
        for m in observations:
            idx = m.find_index(t)
            if idx is not None:
                c_i = c_i.at[offset : offset + m.dim].set(-m.z[idx])
                mask_i = mask_i.at[offset : offset + m.dim].set(True)
            offset += m.dim

        c_list.append(c_i)
        mask_list.append(mask_i)

    mask = np.stack(mask_list)

    # Guard against asynchronous / partial-dimension observations. The masked
    # update consumed by the scan / adaptive paths is all-or-nothing per step
    # (it gates on ``mask.any()``), so a step where only *some* of the stacked
    # measurements fire would silently condition the inactive rows on a
    # fabricated "observe 0". Such a step has a mask row that is neither all-on
    # nor all-off. Reject it loudly until per-dimension masking is implemented
    # (see ROADMAP: observation-API consolidation).
    row_any = mask.any(axis=1)
    row_all = mask.all(axis=1)
    if bool(np.any(row_any & ~row_all)):
        raise NotImplementedError(
            "Asynchronous / partial observations are not supported by the "
            "fixed-shape masked update: at some time step only a subset of the "
            "stacked measurements fire, which would corrupt the inactive rows. "
            "Use measurements that share the same observation times, or the "
            "embedded-Measurement Python path (sqr_adaptive_loop). "
            "Per-dimension masking is planned."
        )

    return ObsModel(
        H=H,
        R_sqr=R_sqr,
        c_seq=np.stack(c_list),
        mask=mask,
    )


@dataclass(frozen=True)
class Measurement:
    """Time-varying linear measurement: A @ x = z[t] (active only at specified times).

    This constraint is only active when the filter's current time matches one of
    the measurement times in z_t. Time matching uses tolerance-based comparison
    (not exact equality) to handle floating-point discrepancies.

    Important:
        When creating time grids, use the same linspace implementation (preferably
        jax.numpy.linspace) for both measurement times (z_t) and filter time steps.
        NumPy and JAX linspace can produce slightly different values (~1e-8 differences)
        which may cause measurements to be missed with exact comparison.

    Args:
        A: Measurement matrix (shape [k, d] or [k, state_dim] if full_state=True).
        z: Measurement values (shape [n, k]).
        z_t: Measurement times (shape [n]). Should use jax.numpy.linspace for
            consistency with the filter's internal time grid.
        noise: Measurement noise variance (scalar or [k] or [k, k]).
        full_state: If True, A operates on the full state X instead of x = E0 @ X.
    """

    A: Array
    z: Array
    z_t: Array
    noise: float | Array = DEFAULT_MEASUREMENT_NOISE
    full_state: bool = False

    def __post_init__(self) -> None:
        for name in ("A", "z", "z_t"):
            object.__setattr__(self, name, np.asarray(getattr(self, name)))
        if self.A.ndim != 2:
            raise ValueError(f"'A' must be 2D, got {self.A.ndim}D.")
        if self.z.ndim != 2 or self.z.shape[1] != self.A.shape[0]:
            raise ValueError(
                f"'z' must be 2D with shape (n, {self.A.shape[0]}), got {self.z.shape}."
            )
        # Normalize z_t to 1D
        z_t = self.z_t
        if z_t.ndim == 2 and z_t.shape[1] == 1:
            object.__setattr__(self, "z_t", z_t.reshape(-1))
        elif z_t.ndim != 1:
            raise ValueError(
                f"'z_t' must be 1D shape (n,) or 2D shape (n, 1), got {z_t.shape}."
            )
        if self.z_t.shape[0] != self.z.shape[0]:
            raise ValueError(
                f"'z_t' length must match number of measurements {self.z.shape[0]}, "
                f"got {self.z_t.shape[0]}."
            )
        noise = self.get_noise_matrix()
        if noise.shape != (self.dim, self.dim):
            raise ValueError(
                f"'noise' must be a scalar, a length-{self.dim} vector or a "
                f"({self.dim}, {self.dim}) matrix, got shape {np.shape(self.noise)}."
            )
        _check_noise_covariance(noise, "Measurement noise")

    def find_index(
        self,
        t: float,
        rtol: float = MEASUREMENT_TIME_RTOL,
        atol: float = MEASUREMENT_TIME_ATOL,
    ) -> int | None:
        """Find measurement index for time t, or None if not found.

        Uses binary search with tolerance for robust floating-point comparison.
        Times are assumed to be sorted.

        Note: Default tolerances are set to handle discrepancies between
        NumPy and JAX linspace implementations, which can differ by ~1e-8
        for typical time grids.
        """
        z_t = onp.asarray(self.z_t)
        t_val = float(t)
        idx = int(onp.searchsorted(z_t, t_val))
        # Check the found position and neighbors for a close match
        for i in [idx, idx - 1]:
            if 0 <= i < len(z_t) and onp.isclose(z_t[i], t_val, rtol=rtol, atol=atol):
                return i
        return None

    def residual(self, x: Array, t: float) -> Array | None:
        """Compute residual: A @ x - z[t], or None if no measurement at t."""
        idx = self.find_index(t)
        if idx is None:
            return None
        return self.A @ x - self.z[idx]

    def jacobian(self, t: float) -> Array | None:
        """Return Jacobian (constant): A, or None if no measurement at t."""
        if self.find_index(t) is None:
            return None
        return self.A

    @property
    def dim(self) -> int:
        """Dimension of measurement output."""
        return int(self.A.shape[0])

    def get_noise_matrix(self) -> Array:
        """Get noise covariance matrix."""
        noise = np.asarray(self.noise)
        if noise.ndim == 0:
            return noise * np.eye(self.dim)
        elif noise.ndim == 1:
            return np.diag(noise)
        return noise
