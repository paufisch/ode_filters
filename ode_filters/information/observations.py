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

# Tolerances for matching measurement times to grid nodes, absorbing the ~1e-8
# drift between numpy.linspace and jax.numpy.linspace on typical grids.
MEASUREMENT_TIME_RTOL = 1e-5
MEASUREMENT_TIME_ATOL = 1e-7


class ObsModel(NamedTuple):
    """Data observations on a time grid, for the filters' ``obs_model``.

    Linear observations ``y = H x + v`` with a constant ``H`` and noise; only the
    data vary per step. Built by ``prepare_observations``.

    Attributes:
        H: Observation matrix, shape ``[obs_dim, state_dim]``.
        R_sqr: Square-root noise covariance, shape ``[obs_dim, obs_dim]``.
        c_seq: Offsets per filter step, ``-y`` where a row has data and zero
            elsewhere, shape ``[N, obs_dim]``.
        mask: Which rows have data at each step, shape ``[N, obs_dim]``. Rows
            without data are ignored, so measurements can have different times.
    """

    H: Array
    R_sqr: Array
    c_seq: Array
    mask: Array


def _grid_steps(times: onp.ndarray, steps: onp.ndarray) -> onp.ndarray:
    """Index of the grid step ending at each time; ``-1`` where none does."""
    idx = onp.clip(onp.searchsorted(steps, times), 0, len(steps) - 1)
    out = onp.full(times.shape, -1)
    for candidate in (idx, onp.clip(idx - 1, 0, len(steps) - 1)):
        hit = onp.isclose(
            steps[candidate],
            times,
            rtol=MEASUREMENT_TIME_RTOL,
            atol=MEASUREMENT_TIME_ATOL,
        )
        out = onp.where((out < 0) & hit, candidate, out)
    return out


def prepare_observations(
    observations: list[Measurement],
    E0: ArrayLike,
    ts: ArrayLike,
) -> ObsModel | None:
    """Build an ``ObsModel`` on the grid ``ts`` from ``Measurement`` objects.

    Filter step ``n`` updates at ``ts[n + 1]``, so every measurement time must be
    a node of ``ts[1:]`` (matched with a small tolerance). Measurements may have
    different times: at each step only the rows with data are conditioned on.
    Off-grid data need the grid to include their times, e.g.
    ``ts = np.union1d(ts, z_t)``; the filters accept non-uniform grids. The time
    matching runs on the host, so the times must be concrete, while the data
    values may be traced (e.g. differentiated).

    Args:
        observations: The measurements.
        E0: State extraction matrix, shape ``[d, state_dim]`` (for measurements
            on ``E0 @ state``).
        ts: Time grid of shape ``[N + 1]``, the solver's grid.

    Returns:
        An ``ObsModel``, or ``None`` when ``observations`` is empty.

    Raises:
        ValueError: If a measurement time is not a node of ``ts[1:]``.
    """
    if not observations:
        return None

    E0 = np.asarray(E0)
    steps = onp.asarray(ts, dtype=float)[1:]
    n_steps = steps.shape[0]
    state_dim = E0.shape[1]
    obs_dim = sum(m.dim for m in observations)

    H = np.zeros((obs_dim, state_dim))
    R = np.zeros((obs_dim, obs_dim))
    c_seq = np.zeros((n_steps, obs_dim))
    mask = onp.zeros((n_steps, obs_dim), dtype=bool)
    offset = 0
    for m in observations:
        rows = slice(offset, offset + m.dim)
        H = H.at[rows, :].set(m.A if m.full_state else m.A @ E0)
        R = R.at[rows, rows].set(m.get_noise_matrix())
        times = onp.asarray(m.z_t, dtype=float)
        idx = _grid_steps(times, steps)
        if onp.any(idx < 0):
            missing = times[idx < 0]
            raise ValueError(
                f"Measurement times {missing.tolist()} are not nodes of the grid "
                "ts[1:]. Include them in the grid (e.g. ts = np.union1d(ts, z_t); "
                "the filters accept non-uniform grids); data at the initial time "
                "ts[0] belong in the initial state."
            )
        c_seq = c_seq.at[idx, rows].set(-np.asarray(m.z))
        mask[idx, rows] = True
        offset += m.dim

    return ObsModel(
        H=H,
        R_sqr=_safe_cholesky_sqr(R, obs_dim),
        c_seq=c_seq,
        mask=np.asarray(mask),
    )


@dataclass(frozen=True)
class Measurement:
    """Noisy linear data ``z[n] = A @ x(z_t[n]) + v`` at the times ``z_t``.

    Passed to the solvers through ``prepare_observations``, which matches the
    times to the solver's grid with a small tolerance (absorbing the ~1e-8
    differences between NumPy and JAX ``linspace``).

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
