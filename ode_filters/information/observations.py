"""Data observations: ``Measurement``, ``ObsModel`` and ``prepare_observations``.

``prepare_observations`` places each datum on the solver's grid ``ts``. A time
that matches a node of the grid (within ``1e-7 + 1e-5 |t|``) is that node. Any
other time in ``(ts[0], ts[-1]]`` becomes a *data-only node*: the filter
predicts to it and conditions on the data there, without an update on the ODE
information, so the ODE information stays that of the grid. The ``ObsModel``
then records its *timeline*, the grid with the data-only nodes inserted, and
the solvers return their results on it.

The matching runs on the host, with NumPy. A ``Measurement`` keeps concrete
times as a NumPy array, so it can be built and ``prepare_observations`` called
inside ``jax.jit`` (or ``grad``, ``vmap``) while its data are traced.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import equinox as eqx
import jax
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
    """Data observations for the solvers' ``obs_model``.

    Linear observations ``y = H x + v`` with a constant ``H`` and noise; only the
    data vary per step. Built by ``prepare_observations``. Step ``n`` ends at
    node ``n + 1`` of the solve. Without data-only nodes these are the nodes of
    the solver's grid ``ts``, and ``t`` and ``grid_node`` are ``None``. With
    data-only nodes they are the nodes of the timeline ``t``, the grid with a
    node at each data time between its nodes; a solver checks that the grid
    nodes of the timeline are its ``ts``.

    Several data of one measurement at one node occupy copies of the
    measurement's rows, as many as its largest number of data at one node. At a
    step the copies without data are masked.

    Attributes:
        H: Observation matrix, shape ``[obs_dim, state_dim]``.
        R_sqr: Square-root noise covariance, shape ``[obs_dim, obs_dim]``.
        c_seq: Offsets per step, ``-y`` where a row has data and zero
            elsewhere, shape ``[E, obs_dim]``.
        mask: Which rows have data at each step, shape ``[E, obs_dim]``. Rows
            without data are ignored, so measurements can have different times.
        t: The timeline, shape ``[E + 1]``: the grid with the data-only nodes,
            a NumPy array unless traced. ``None`` when every datum lies on a
            node of the grid; the steps are then those of the grid.
        grid_node: Whether step ``n`` ends at a node of the grid (``True``) or
            at a data-only node (``False``), shape ``[E]``; ``None`` with ``t``.
    """

    H: Array
    R_sqr: Array
    c_seq: Array
    mask: Array
    t: Array | onp.ndarray | None = None
    grid_node: Array | onp.ndarray | None = None

    def without_data(self) -> ObsModel:
        """This ``ObsModel`` without its data: no rows, on the same nodes.

        A solver given it runs the pass without data on the nodes of this
        ``ObsModel``, its data-only nodes included, and reports no data
        evidence (``log_likelihood_obs`` is ``None``). This is the pass without
        data that ``fenrir_loglik(data_free, obs_model)`` and
        ``conditional_loglik(data, data_free)`` take. With data-only nodes, a
        pass with ``obs_model=None`` runs on the grid alone and does not match
        this ``ObsModel``.

        Returns:
            An ``ObsModel`` with no rows (``obs_dim = 0``) and the same ``t``
            and ``grid_node``.
        """
        n_steps = self.c_seq.shape[0]
        return ObsModel(
            H=np.zeros((0, self.H.shape[1]), dtype=self.H.dtype),
            R_sqr=np.zeros((0, 0), dtype=self.R_sqr.dtype),
            c_seq=np.zeros((n_steps, 0), dtype=self.c_seq.dtype),
            mask=np.zeros((n_steps, 0), dtype=bool),
            t=self.t,
            grid_node=self.grid_node,
        )


def _host(values: ArrayLike, name: str) -> onp.ndarray:
    """``values`` as a new NumPy float array; a ``TypeError`` if they are traced."""
    try:
        return onp.array(values, dtype=float)
    except jax.errors.TracerArrayConversionError:
        raise TypeError(
            f"{name} are traced, but the time matching runs on the host and needs "
            "them concrete. Inside jax.jit, eqx.filter_jit, grad or vmap, give "
            "them as NumPy arrays, Python numbers or arrays that the traced "
            "function closes over. Arguments of the traced function, values "
            "computed in it and the times of a Measurement passed in as an "
            "argument are traced. The data z may be traced."
        ) from None


def _match_nodes(times: onp.ndarray, nodes: onp.ndarray) -> onp.ndarray:
    """Index of the node that each time matches; ``-1`` where none does."""
    idx = onp.clip(onp.searchsorted(nodes, times), 0, len(nodes) - 1)
    out = onp.full(times.shape, -1)
    for candidate in (idx, onp.clip(idx - 1, 0, len(nodes) - 1)):
        hit = onp.isclose(
            nodes[candidate],
            times,
            rtol=MEASUREMENT_TIME_RTOL,
            atol=MEASUREMENT_TIME_ATOL,
        )
        out = onp.where((out < 0) & hit, candidate, out)
    return out


def _timeline(
    grid: onp.ndarray, own_times: list[onp.ndarray]
) -> tuple[onp.ndarray, onp.ndarray | None, list[onp.ndarray]]:
    """The nodes of the solve, and the step that ends at each measurement time.

    Returns ``(nodes, grid_node, steps)``. ``nodes`` is the grid with a node at
    each time that matches no node of it; ``grid_node`` marks the steps that end
    at a node of the grid, and is ``None`` when no time is off the grid.
    ``steps[i]`` is the step index (node index minus one) of each time of
    measurement ``i``.
    """
    times = onp.unique(onp.concatenate([onp.zeros(0), *own_times]))
    node = _match_nodes(times, grid)
    early = times[(node == 0) | ((node < 0) & (times < grid[0]))]
    if early.size:
        raise ValueError(
            f"Measurement times {early.tolist()} lie at or before the initial time "
            f"ts[0] = {grid[0]}. Data at the initial time belong in the initial "
            "state: condition m_0 and P_0_sqr on them."
        )
    late = times[(node < 0) & (times > grid[-1])]
    if late.size:
        raise ValueError(
            f"Measurement times {late.tolist()} lie after the final time ts[-1] = "
            f"{grid[-1]}; extend ts to cover them."
        )
    off = times[node < 0]
    # An off-grid time within the matching tolerance of the first time of the
    # previous group joins that group's node.
    shared = off.copy()
    for i in range(1, off.shape[0]):
        if onp.isclose(
            shared[i - 1],
            off[i],
            rtol=MEASUREMENT_TIME_RTOL,
            atol=MEASUREMENT_TIME_ATOL,
        ):
            shared[i] = shared[i - 1]
    nodes = onp.union1d(grid, shared)
    position = onp.empty(times.shape, dtype=int)
    on_grid = node >= 0
    position[on_grid] = onp.searchsorted(nodes, grid[node[on_grid]])
    position[~on_grid] = onp.searchsorted(nodes, shared)
    steps = [position[onp.searchsorted(times, own)] - 1 for own in own_times]
    grid_node = onp.isin(nodes[1:], grid[1:]) if off.size else None
    return nodes, grid_node, steps


def _copies(steps: onp.ndarray) -> tuple[int, onp.ndarray]:
    """The number of row copies a measurement needs, and the copy of each datum.

    A datum takes the first copy that no earlier datum at its step has taken.
    """
    taken: dict[int, int] = {}
    copy = onp.empty(steps.shape, dtype=int)
    for i, step in enumerate(steps.tolist()):
        n = taken.get(step, 0)
        copy[i] = n
        taken[step] = n + 1
    return max(taken.values(), default=1), copy


def prepare_observations(
    observations: Sequence[Measurement],
    E0: ArrayLike,
    ts: ArrayLike,
) -> ObsModel | None:
    """Build an ``ObsModel`` for the solver's grid ``ts`` from ``Measurement`` objects.

    The usual way to pass data is the list of ``Measurement`` objects itself.
    The solvers take it as ``measurements=``, and ``marginal_loglik``, ``fit``
    and ``information_criteria`` as their data; each converts it with this
    function. Call it where code needs the ``ObsModel`` itself: for
    ``fenrir_loglik`` and ``conditional_loglik`` called directly, with
    ``obs_model.without_data()`` for the pass without data, or to reuse one
    conversion in several solves on the same grid.

    Every measurement time ``t`` must lie in ``(ts[0], ts[-1]]``. A time that
    matches a node of ``ts`` within ``1e-7 + 1e-5 |t|`` is conditioned on at
    that node, after the ODE information. Any other time becomes a data-only
    node: the filter predicts to it, conditions on the data and continues
    without an update on the ODE information, so the ODE information is that of
    the grid ``ts`` whatever the data times. Times within the tolerance of each
    other share one data-only node. With data-only nodes, the ``ObsModel``
    records its timeline (``t``, ``grid_node``) and the solvers return their
    results on it. When every time is a node of ``ts``, ``t`` is ``None``.

    Measurements may have different times: at each step only the rows with
    data are conditioned on. Several data of one measurement at one node are
    all conditioned on, through copies of the measurement's rows.

    The time matching runs on the host, so the times and ``ts`` must be
    concrete, while the data values (and ``A``, the noise and ``E0``) may be
    traced: the measurements and the ``ObsModel`` can be built inside a jitted,
    differentiated or vmapped function (see ``Measurement``).

    Args:
        observations: The measurements.
        E0: Selector of the solution ``x`` in the state, shape
            ``[d, state_dim]``. The ``A`` of a measurement acts on
            ``E0 @ state`` unless it has ``full_state=True``. Pass
            ``prior.E0_state``, which for a ``JointPrior`` selects the solution
            without the input.
        ts: The solver's time grid, shape ``[N + 1]``, strictly increasing;
            ``ts[0]`` is the time of the initial state.

    Returns:
        An ``ObsModel``, or ``None`` when ``observations`` is empty.

    Raises:
        ValueError: For an invalid grid, or a measurement time at or before
            ``ts[0]`` or after ``ts[-1]``.
        TypeError: If ``ts`` or the measurement times are traced.
    """
    if not observations:
        return None

    E0 = np.asarray(E0)
    grid = _host(ts, "The nodes of the grid ts")
    if grid.ndim != 1 or grid.shape[0] < 2:
        raise ValueError(
            f"ts must be a 1-D array of at least two times, got shape {grid.shape}."
        )
    if not onp.all(onp.diff(grid) > 0):
        raise ValueError("ts must be strictly increasing.")
    own_times = [_host(m.z_t, "Measurement times") for m in observations]
    nodes, grid_node, steps = _timeline(grid, own_times)
    copies = [_copies(s) for s in steps]
    n_steps = nodes.shape[0] - 1
    state_dim = E0.shape[1]
    obs_dim = sum(n * m.dim for (n, _), m in zip(copies, observations, strict=True))

    H = np.zeros((obs_dim, state_dim))
    R = np.zeros((obs_dim, obs_dim))
    c_seq = np.zeros((n_steps, obs_dim))
    mask = onp.zeros((n_steps, obs_dim), dtype=bool)
    offset = 0
    for m, step, (n_copies, copy) in zip(observations, steps, copies, strict=True):
        k = m.dim
        H_m = m.A if m.full_state else m.A @ E0
        for j in range(n_copies):
            block = slice(offset + j * k, offset + (j + 1) * k)
            H = H.at[block, :].set(H_m)
            R = R.at[block, block].set(m.R)
        rows = offset + k * copy[:, None] + onp.arange(k)[None, :]
        c_seq = c_seq.at[step[:, None], rows].set(-np.asarray(m.z))
        mask[step[:, None], rows] = True
        offset += n_copies * k

    return ObsModel(
        H=H,
        R_sqr=_safe_cholesky_sqr(R, obs_dim),
        c_seq=c_seq,
        mask=np.asarray(mask),
        t=None if grid_node is None else nodes,
        grid_node=grid_node,
    )


class Measurement(eqx.Module):
    """Noisy linear data ``z[n] = A @ x(z_t[n]) + v`` at the times ``z_t``.

    Passed to the solvers as ``measurements=``, or through
    ``prepare_observations``, which matches the times to the solver's grid with
    a small tolerance (absorbing the ~1e-8 differences between NumPy and JAX
    ``linspace``). A time between two grid nodes becomes a data-only node.

    The matching runs on the host, so concrete times are kept as a NumPy array.
    They stay concrete when the ``Measurement`` is built inside ``jax.jit`` (or
    ``eqx.filter_jit``, ``grad``, ``vmap``) from NumPy arrays, Python numbers or
    arrays that the traced function closes over, while ``z``, ``A`` and the
    noise may be traced. Times that are traced (an argument of the traced
    function, a value computed in it, or the times of a ``Measurement`` passed
    in as an argument) are kept as they are, and ``prepare_observations``
    raises a ``TypeError`` for them.

    Attributes:
        A: Measurement matrix, shape ``[k, d]``, on the solution
            ``x = E0 @ state`` with ``E0 = prior.E0_state`` (for a
            ``JointPrior``, the solution without the input); or
            ``[k, state_dim]`` with ``full_state=True``.
        z: Data, shape ``[n, k]``.
        z_t: Times of the data, shape ``[n]`` (``[n, 1]`` is flattened); a NumPy
            array unless traced.
        R: Noise covariance, ``[k, k]``; given as ``noise=`` a variance, per-row
            variances or a covariance.
        full_state: Whether ``A`` acts on the full state instead of ``x``.
    """

    A: Array
    z: Array
    z_t: Array | onp.ndarray
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
        A, z = np.asarray(A), np.asarray(z)
        # A new NumPy copy, as check_grid makes of a grid (see there).
        try:
            z_t = onp.array(z_t, dtype=float)
        except jax.errors.TracerArrayConversionError:
            z_t = np.asarray(z_t, dtype=float)
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
