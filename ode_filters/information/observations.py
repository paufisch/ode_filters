"""Data observations: ``Measurement``, ``ObsModel`` and ``prepare_observations``."""

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _safe_cholesky_sqr
from .noise import noise_matrix

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
        R = R.at[rows, rows].set(m.R)
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


class Measurement(eqx.Module):
    """Noisy linear data ``z[n] = A @ x(z_t[n]) + v`` at the times ``z_t``.

    Passed to the solvers through ``prepare_observations``, which matches the
    times to the solver's grid with a small tolerance (absorbing the ~1e-8
    differences between NumPy and JAX ``linspace``).

    Attributes:
        A: Measurement matrix, shape ``[k, d]`` (on ``x = E0 @ state``) or
            ``[k, state_dim]`` with ``full_state=True``.
        z: Data, shape ``[n, k]``.
        z_t: Times of the data, shape ``[n]`` (``[n, 1]`` is flattened).
        R: Noise covariance, ``[k, k]``; given as ``noise=`` a variance, per-row
            variances or a covariance.
        full_state: Whether ``A`` acts on the full state instead of ``x``.
    """

    A: Array
    z: Array
    z_t: Array
    R: Array
    full_state: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        A: ArrayLike,
        z: ArrayLike,
        z_t: ArrayLike,
        noise: ArrayLike = DEFAULT_MEASUREMENT_NOISE,
        full_state: bool = False,
    ):
        A, z, z_t = np.asarray(A), np.asarray(z), np.asarray(z_t)
        if A.ndim != 2:
            raise ValueError(f"'A' must be 2D, got {A.ndim}D.")
        if z.ndim != 2 or z.shape[1] != A.shape[0]:
            raise ValueError(
                f"'z' must be 2D with shape (n, {A.shape[0]}), got {z.shape}."
            )
        if z_t.ndim == 2 and z_t.shape[1] == 1:
            z_t = z_t.reshape(-1)
        elif z_t.ndim != 1:
            raise ValueError(
                f"'z_t' must be 1D shape (n,) or 2D shape (n, 1), got {z_t.shape}."
            )
        if z_t.shape[0] != z.shape[0]:
            raise ValueError(
                f"'z_t' length must match number of measurements {z.shape[0]}, "
                f"got {z_t.shape[0]}."
            )
        self.A, self.z, self.z_t = A, z, z_t
        self.R = noise_matrix(noise, A.shape[0], "Measurement noise")
        self.full_state = full_state

    @property
    def dim(self) -> int:
        """Number of measured quantities ``k``."""
        return int(self.A.shape[0])
