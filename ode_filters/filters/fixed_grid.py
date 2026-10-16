"""Filters over a given time grid: ``gaussian_filter`` and ``affine_filter``.

Both scan ``filter_step`` (``ode_filters.filters.kernel``) over the steps of
the grid ``ts`` and differ only in the ODE information of a step:
``gaussian_filter`` linearizes the measurement model at each prediction with a
``Correction``; ``affine_filter`` takes a frozen ``AffineSurrogate``, one
affine model per step, and so is an exact square-root Kalman filter of that
linear-Gaussian model.

The grid may be non-uniform. On a concrete uniform grid the prior's transition
is computed once; otherwise once per step. A grid is concrete when it is a NumPy
array or a JAX array that is not traced, which includes an array that a jitted
function closes over. A grid is traced when it is, for example, an argument of
a jitted function. Data (an ``ObsModel`` from ``prepare_observations`` on the
same grid, or ``measurements=``) are conditioned on after the ODE information
at each step.

**Data-only nodes.** Data at times between the grid nodes give the ``ObsModel``
data-only nodes (see ``prepare_observations``), and the filter then runs over
its timeline, the grid with these nodes inserted. A step that ends at a
data-only node predicts to it and conditions on the data there. It has no
update on the ODE information: its ODE evidence is zero, and its ODE
innovation is reported with zero mean and an identity factor. A step that ends
at a grid node updates on the ODE information, linearized at its prediction,
and then on the data, as on the grid. The diffusion of a grid interval is
estimated once, at the interval's start, from the prediction over the whole
interval, as without data; every step inside the interval propagates with it
and reports it in ``sigma_sqr`` and ``diffusion``. An interval without
data-only nodes is one step, computed as on the grid alone. The result lives on
the timeline (``result.t``), so ``rts_smoother``, ``sample_posterior`` and
``interpolate`` apply unchanged.

Reverse-mode differentiation of the scan stores the intermediates of every
step. With ``checkpoint=True`` the step is wrapped in ``jax.checkpoint``, so the
backward pass stores only the state the scan carries from step to step and
recomputes the intermediates, at the cost of one more forward pass.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..calibration.policies import (
    Calibration,
    CalibrationMode,
    NoCalibration,
    as_calibration,
)
from ..gaussian import AffineModel, Gaussian, log_likelihood
from ..information import Measurement, MeasurementModel, ObsModel
from ..information.observations import (
    MEASUREMENT_TIME_ATOL,
    MEASUREMENT_TIME_RTOL,
    prepare_observations,
)
from ..linearization.correction import Correction, TaylorCorrection
from ..linearization.surrogate import AffineSurrogate
from ..priors.base import BasePrior, Transition
from .kernel import (
    FrozenInformation,
    Information,
    OnlineInformation,
    StepOutput,
    estimate_diffusion,
    filter_step,
    observe,
    predict,
)
from .result import FilterResult

# ``jax.checkpoint`` is the public name in every supported JAX version (0.11
# removed ``jax.ad_checkpoint.checkpoint``), but the type stubs of JAX 0.8 do
# not list it as exported.
_checkpoint = jax.checkpoint  # pyright: ignore[reportPrivateImportUsage]


def check_grid(ts: ArrayLike) -> Array | onp.ndarray:
    """Validate a time grid: 1-D, at least two nodes, increasing where concrete.

    A concrete grid comes back as a NumPy array, a traced one as a JAX array. The
    NumPy array stays concrete inside ``jax.jit``, where converting it to a JAX
    array would make it a tracer, so ``grid_transitions`` still sees a uniform
    grid as uniform. It is a new copy on every call: JAX keys its device copy of
    a captured NumPy constant by the array object and keeps it across a change
    of ``jax_enable_x64`` (see ``ode_filters.priors._constants``), and
    ``onp.asarray`` returns the same object for the same grid.
    """
    try:
        ts_host = onp.array(ts, dtype=float)
    except jax.errors.TracerArrayConversionError:
        ts_host = None
    grid = np.asarray(ts, dtype=float) if ts_host is None else ts_host
    if grid.ndim != 1 or grid.shape[0] < 2:
        raise ValueError(
            f"ts must be a 1-D array of at least two times, got shape {grid.shape}."
        )
    if ts_host is not None and not onp.all(onp.diff(ts_host) > 0):
        raise ValueError("ts must be strictly increasing.")
    return grid


def grid_transitions(
    prior: BasePrior, ts: Array | onp.ndarray
) -> tuple[Transition | None, Transition | None]:
    """The prior's transitions over the steps of ``ts``.

    Returns ``(constant, per_step)``, exactly one of them not ``None``. A
    concrete, uniform grid needs one transition (computed once, with the step
    ``(t_K - t_0) / (K - 1)``); otherwise there is one per step, stacked.
    """
    n_steps = ts.shape[0] - 1
    try:
        ts_host = onp.asarray(ts)
    except jax.errors.TracerArrayConversionError:
        ts_host = None
    if ts_host is not None:
        h = (ts_host[-1] - ts_host[0]) / n_steps
        if onp.allclose(onp.diff(ts_host), h, rtol=1e-10, atol=0.0):
            return prior.transition(np.asarray(h)), None
    return None, jax.vmap(prior.transition)(np.diff(ts))


def check_obs_model(obs_model: ObsModel, n_steps: int, nodes: str = "the grid") -> None:
    """Raise if ``obs_model`` was built for another number of steps."""
    if obs_model.c_seq.shape[0] != n_steps or obs_model.mask.shape[0] != n_steps:
        raise ValueError(
            f"obs_model has {obs_model.c_seq.shape[0]} steps but {nodes} has "
            f"{n_steps}; build it with prepare_observations(measurements, E0, ts) "
            "on the same grid."
        )


class Timeline(NamedTuple):
    """The steps of a solve over a timeline with data-only nodes.

    Attributes:
        t: The nodes, shape ``[E + 1]``. Its grid nodes are the values of the
            solver's ``ts``.
        grid_node: Whether step ``n`` ends at a node of the grid, shape ``[E]``.
        interval: The grid interval of step ``n``, shape ``[E]``: ``k`` for a
            step between ``ts[k]`` and ``ts[k + 1]``.
        first: Whether step ``n`` starts at a node of the grid, shape ``[E]``.
    """

    t: Array | onp.ndarray
    grid_node: Array | onp.ndarray
    interval: Array | onp.ndarray
    first: Array | onp.ndarray


def _concrete(values: ArrayLike, dtype: type) -> onp.ndarray | None:
    """``values`` as a new NumPy array, or ``None`` if they are traced."""
    try:
        return onp.array(values, dtype=dtype)
    except jax.errors.TracerArrayConversionError:
        return None


def _timeline_on(
    ts: Array | onp.ndarray, t: ArrayLike, grid_node: ArrayLike
) -> Timeline:
    """The timeline ``(t, grid_node)`` of an ``ObsModel``, checked against ``ts``.

    The grid nodes of the timeline are replaced by the values of ``ts``, which
    may differ from those it was built on within the matching tolerance. Where
    everything is concrete, the check is on the values; otherwise on the shapes
    only.
    """
    n_steps = ts.shape[0] - 1
    t_host = _concrete(t, float)
    node_host = _concrete(grid_node, bool)
    ts_host = _concrete(ts, float)
    shape_t, shape_node = onp.shape(t), onp.shape(grid_node)
    if len(shape_t) != 1 or shape_node != (shape_t[0] - 1,):
        raise ValueError(
            f"obs_model.t must be 1-D and obs_model.grid_node have one entry per "
            f"step; got shapes {shape_t} and {shape_node}."
        )
    if t_host is None or node_host is None or ts_host is None:
        nodes_in = np.asarray(t, dtype=float)
        node = np.asarray(grid_node, dtype=bool)
        grid = np.asarray(ts, dtype=float)
        end = np.cumsum(node.astype(int))
        at_node = np.where(node, grid[np.minimum(end, n_steps)], nodes_in[1:])
        return Timeline(
            t=np.concatenate([grid[:1], at_node]),
            grid_node=node,
            interval=end - node.astype(int),
            first=np.concatenate([np.ones(1, dtype=bool), node[:-1]]),
        )
    grid_t = onp.concatenate([t_host[:1], t_host[1:][node_host]])
    nodes = t_host.copy()
    if grid_t.shape == ts_host.shape:
        nodes[0] = ts_host[0]
        nodes[1:][node_host] = ts_host[1:]
    if (
        grid_t.shape != ts_host.shape
        or not onp.allclose(
            grid_t, ts_host, rtol=MEASUREMENT_TIME_RTOL, atol=MEASUREMENT_TIME_ATOL
        )
        or not onp.all(onp.diff(nodes) > 0)
    ):
        raise ValueError(
            f"obs_model belongs to another grid: the {grid_t.shape[0]} grid nodes "
            f"of its timeline are not the {n_steps + 1} nodes of ts. Build it "
            "with prepare_observations(measurements, E0, ts) on the solver's grid."
        )
    steps = node_host.astype(int)
    return Timeline(
        t=nodes,
        grid_node=node_host,
        interval=onp.cumsum(steps) - steps,
        first=onp.concatenate([[True], node_host[:-1]]),
    )


def resolve_observations(
    obs_model: ObsModel | None, ts: Array | onp.ndarray
) -> tuple[ObsModel | None, Timeline | None]:
    """Check ``obs_model`` against the grid ``ts``; return its data and timeline.

    Returns ``(data, timeline)``. ``data`` is ``None`` without data, which
    includes an ``ObsModel`` without rows (``ObsModel.without_data``).
    ``timeline`` is ``None`` unless the ``ObsModel`` has data-only nodes.

    Raises:
        ValueError: If ``obs_model`` was built for another grid.
    """
    if obs_model is None:
        return None, None
    data = obs_model if obs_model.H.shape[0] else None
    if obs_model.t is None and obs_model.grid_node is None:
        check_obs_model(obs_model, ts.shape[0] - 1)
        return data, None
    if obs_model.t is None or obs_model.grid_node is None:
        raise ValueError(
            "obs_model.t and obs_model.grid_node must both be given or both be None."
        )
    timeline = _timeline_on(ts, obs_model.t, obs_model.grid_node)
    check_obs_model(obs_model, timeline.grid_node.shape[0], "its timeline")
    return data, timeline


def observations_from(
    obs_model: ObsModel | None,
    measurements: Sequence[Measurement] | None,
    prior: BasePrior,
    ts: Array | onp.ndarray,
) -> ObsModel | None:
    """The data of a solve, given as an ``obs_model`` or as ``measurements``.

    Raises:
        ValueError: If both are given.
    """
    if measurements is None:
        return obs_model
    if obs_model is not None:
        raise ValueError("Pass the data as obs_model= or as measurements=, not both.")
    return prepare_observations(measurements, prior.E0_state, ts)


def check_surrogate(surrogate: AffineSurrogate, n_steps: int, state_dim: int) -> None:
    """Raise unless ``surrogate`` holds one affine model per step on the state."""
    H, c, R_sqr = surrogate
    if H.ndim != 3 or H.shape[0] != n_steps:
        raise ValueError(
            f"surrogate.H must have shape [N, m, D] with N = {n_steps} steps, got "
            f"{H.shape}."
        )
    if H.shape[2] != state_dim:
        raise ValueError(
            f"surrogate.H acts on states of dimension {H.shape[2]}, but m_0 has "
            f"dimension {state_dim}."
        )
    if c.shape != H.shape[:2] or R_sqr.shape != (*H.shape[:2], H.shape[1]):
        raise ValueError(
            f"surrogate.c and surrogate.R_sqr must have shapes {H.shape[:2]} and "
            f"{(*H.shape[:2], H.shape[1])}, got {c.shape} and {R_sqr.shape}."
        )


def timeline_transitions(
    prior: BasePrior, ts: Array | onp.ndarray, timeline: Timeline
) -> tuple[Transition, Transition]:
    """The transitions of a timeline solve, one per step, stacked.

    Returns ``(interval, step)``: the transition over the grid interval of each
    step, which calibrates the diffusion, and the transition over the step
    itself, which predicts. A step that spans its whole interval predicts with
    the interval's transition, as on the grid alone.
    """
    n = timeline.grid_node.shape[0]
    constant, per_interval = grid_transitions(prior, ts)
    if per_interval is None:
        assert constant is not None
        interval = jax.tree.map(lambda a: np.broadcast_to(a, (n, *a.shape)), constant)
    else:
        interval = jax.tree.map(lambda a: a[timeline.interval], per_interval)
    sub_steps = jax.vmap(prior.transition)(np.diff(np.asarray(timeline.t)))
    whole = np.asarray(timeline.first) & np.asarray(timeline.grid_node)

    def pick(a: Array, b: Array) -> Array:
        return np.where(whole.reshape(n, *(1,) * (a.ndim - 1)), a, b)

    return interval, jax.tree.map(pick, interval, sub_steps)


def _result(nodes: Array | onp.ndarray, x_0: Gaussian, out: StepOutput) -> FilterResult:
    """The ``FilterResult`` of a scan of steps from ``x_0`` over ``nodes``."""

    def with_initial(first: Array, rest: Array) -> Array:
        return np.concatenate([first[None], rest], axis=0)

    innovation_obs, ll_obs = out.innovation_obs, out.log_likelihood_obs
    return FilterResult(
        t=np.asarray(nodes),
        m=with_initial(x_0.m, out.filtered.m),
        P_sqr=with_initial(x_0.P_sqr, out.filtered.P_sqr),
        log_likelihood=np.sum(out.log_likelihood),
        log_likelihood_obs=None if ll_obs is None else np.sum(ll_obs),
        sigma_sqr=out.sigma_sqr,
        diffusion=out.diffusion,
        m_pred=out.predicted.m,
        P_pred_sqr=out.predicted.P_sqr,
        G_back=out.backward.G,
        d_back=out.backward.d,
        P_back_sqr=out.backward.P_sqr,
        mz=out.innovation.m,
        Pz_sqr=out.innovation.P_sqr,
        mz_obs=None if innovation_obs is None else innovation_obs.m,
        Pz_obs_sqr=None if innovation_obs is None else innovation_obs.P_sqr,
    )


def run_filter(
    m_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    ts: Array | onp.ndarray,
    *,
    calibration: Calibration,
    information: Information | None = None,
    surrogate: AffineSurrogate | None = None,
    obs_model: ObsModel | None = None,
    checkpoint: bool = False,
) -> FilterResult:
    """Scan ``filter_step`` over the grid ``ts``, or over the timeline of the data.

    The ODE information is either ``information`` (linearized online) or
    ``surrogate`` (one frozen affine model per grid step, stacked).
    ``checkpoint`` wraps the scan body in ``jax.checkpoint``. An ``obs_model``
    with data-only nodes runs the scan over its timeline (``run_timeline``).
    """
    data, timeline = resolve_observations(obs_model, ts)
    if timeline is not None:
        return run_timeline(
            m_0,
            P_0_sqr,
            prior,
            ts,
            timeline,
            calibration=calibration,
            information=information,
            surrogate=surrogate,
            obs_model=data,
            checkpoint=checkpoint,
        )
    constant, per_step = grid_transitions(prior, ts)
    observations = None if data is None else (data.c_seq, data.mask)

    def body(x: Gaussian, inputs):
        t, transition, model, step_data = inputs
        step_information = (
            information if model is None else FrozenInformation(AffineModel(*model))
        )
        assert step_information is not None
        observation = mask = None
        if step_data is not None:
            assert data is not None
            c, mask = step_data
            observation = AffineModel(data.H, c, data.R_sqr)
        step_transition = constant if transition is None else transition
        assert step_transition is not None
        out = filter_step(
            x,
            step_transition,
            step_information,
            t=t,
            calibration=calibration,
            scale_noise=prior.apply_state_sigma_sqr,
            observation=observation,
            mask=mask,
        )
        return out.filtered, out

    # The scan already keeps the recomputation from being merged with the
    # forward pass, so jax.checkpoint need not prevent CSE (see its docs).
    step = _checkpoint(body, prevent_cse=False) if checkpoint else body
    x_0 = Gaussian(np.asarray(m_0), np.asarray(P_0_sqr))
    _, out = jax.lax.scan(step, x_0, (ts[1:], per_step, surrogate, observations))
    return _result(ts, x_0, out)


def run_timeline(
    m_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    ts: Array | onp.ndarray,
    timeline: Timeline,
    *,
    calibration: Calibration,
    information: Information | None = None,
    surrogate: AffineSurrogate | None = None,
    obs_model: ObsModel | None = None,
    checkpoint: bool = False,
) -> FilterResult:
    """Scan the steps of a timeline with data-only nodes (see the module docstring).

    The step at the start of a grid interval estimates the interval's diffusion
    with ``estimate_diffusion`` and the interval's transition; the scan carries
    it to the later steps of the interval. Only the steps that end at a grid
    node update on the ODE information (with the surrogate model of their
    interval, if given).
    """
    interval_transitions, step_transitions = timeline_transitions(prior, ts, timeline)
    nodes = np.asarray(timeline.t)
    interval_ends = np.asarray(ts)[1:][timeline.interval]
    models = None
    if surrogate is not None:
        models = jax.tree.map(lambda a: a[timeline.interval], tuple(surrogate))
    observations = None if obs_model is None else (obs_model.c_seq, obs_model.mask)
    x_0 = Gaussian(np.asarray(m_0), np.asarray(P_0_sqr))

    def information_of(model) -> Information:
        if model is None:
            assert information is not None
            return information
        return FrozenInformation(AffineModel(*model))

    def ode_update(x: Gaussian, t: Array, model) -> tuple[Gaussian, Gaussian, Array]:
        step_information = information_of(model)
        linearized = step_information.linearize(x.m, t=t)
        filtered, innovation = step_information.condition(x, linearized, t=t)
        return filtered, innovation, log_likelihood(innovation)

    def diffusion_of(x: Gaussian, transition: Transition, t: Array, model):
        sigma_sqr, diffusion, _ = estimate_diffusion(
            x, transition, information_of(model), t=t, calibration=calibration
        )
        return sigma_sqr, diffusion

    def first_of(tree):
        return jax.tree.map(lambda a: a[0], tree)

    model_0 = None if models is None else first_of(models)
    update_shape = jax.eval_shape(ode_update, x_0, nodes[1], model_0)
    diffusion_shape = jax.eval_shape(
        diffusion_of, x_0, first_of(interval_transitions), nodes[1], model_0
    )
    innovation_shape = update_shape[1].m
    no_innovation = Gaussian(
        np.zeros(innovation_shape.shape, innovation_shape.dtype),
        np.eye(innovation_shape.shape[0], dtype=innovation_shape.dtype),
    )
    no_evidence = np.zeros((), update_shape[2].dtype)

    def body(carry, inputs):
        x, sigma_prev, diffusion_prev = carry
        t, t_interval, first, at_grid, interval, transition, model, data = inputs
        sigma_sqr, diffusion = jax.lax.cond(
            first,
            lambda: diffusion_of(x, interval, t_interval, model),
            lambda: (sigma_prev, diffusion_prev),
        )
        Q_sqr = prior.apply_state_sigma_sqr(transition.Q_sqr, diffusion)
        predicted, backward = predict(x, transition._replace(Q_sqr=Q_sqr))
        filtered, innovation, ll = jax.lax.cond(
            at_grid,
            lambda: ode_update(predicted, t, model),
            lambda: (predicted, no_innovation, no_evidence),
        )
        out = StepOutput(
            predicted=predicted,
            backward=backward,
            innovation=innovation,
            filtered=filtered,
            sigma_sqr=sigma_sqr,
            diffusion=diffusion,
            log_likelihood=ll,
        )
        if data is not None:
            assert obs_model is not None
            c, mask = data
            filtered, innovation_obs, ll_obs = observe(
                filtered, AffineModel(obs_model.H, c, obs_model.R_sqr), mask
            )
            out = out._replace(
                filtered=filtered,
                innovation_obs=innovation_obs,
                log_likelihood_obs=ll_obs,
            )
        return (filtered, sigma_sqr, diffusion), out

    step = _checkpoint(body, prevent_cse=False) if checkpoint else body
    init = (
        x_0,
        *(np.zeros(s.shape, s.dtype) for s in diffusion_shape),
    )
    _, out = jax.lax.scan(
        step,
        init,
        (
            nodes[1:],
            interval_ends,
            timeline.first,
            timeline.grid_node,
            interval_transitions,
            step_transitions,
            models,
            observations,
        ),
    )
    return _result(nodes, x_0, out)


def gaussian_filter(
    m_0: ArrayLike,
    P_0_sqr: ArrayLike,
    prior: BasePrior,
    measure: MeasurementModel,
    ts: ArrayLike,
    *,
    correction: Correction | None = None,
    calibration: Calibration | CalibrationMode = "dynamic",
    obs_model: ObsModel | None = None,
    measurements: Sequence[Measurement] | None = None,
    checkpoint: bool = False,
) -> FilterResult:
    """Gaussian ODE filter over the time grid ``ts``.

    Each step predicts with the prior, calibrates the diffusion, updates on the
    ODE information linearized by ``correction`` at the prediction, and then on
    the data, if any (see ``ode_filters.filters.kernel``). The grid may be
    non-uniform; on a uniform one the prior's transition is computed once.

    Data between the grid nodes add data-only nodes, where the filter
    conditions on the data without an update on the ODE information. The
    diffusion of a grid interval is estimated at its start from the prediction
    over the whole interval, and its steps propagate with it. The result then
    lives on the timeline, the grid with the data-only nodes (see
    ``ode_filters.filters.fixed_grid``).

    Args:
        m_0: Initial mean, shape ``[D]``.
        P_0_sqr: Initial square-root covariance, shape ``[D, D]``.
        prior: Gauss-Markov prior (plain or preconditioned).
        measure: ODE-information model (the ODE plus conservation laws).
        ts: Time grid, shape ``[K]``, strictly increasing; ``ts[0]`` is the
            time of the initial state.
        correction: Linearization (``TaylorCorrection`` /
            ``IteratedTaylorCorrection`` / ``QuadratureCorrection``); ``None``
            is EK1.
        calibration: Diffusion calibration policy, or one of ``"dynamic"``
            (default), ``"none"``, ``"diagonal"``, ``"diagonal_ekf0"``.
        obs_model: Data for the grid ``ts``, from
            ``prepare_observations(measurements, prior.E0_state, ts)``.
        measurements: The data as ``Measurement`` objects, in place of
            ``obs_model``. They are converted with
            ``prepare_observations(measurements, prior.E0_state, ts)``, so the
            ``A`` of a measurement acts on the solution
            ``x = prior.E0_state @ state`` (for a ``JointPrior``, the solution
            without the input), or on the full state with ``full_state=True``.
        checkpoint: Wrap each step in ``jax.checkpoint``. Reverse-mode
            differentiation then stores only the state carried between steps
            and recomputes each step's intermediates in the backward pass,
            which costs one more forward pass. The values are identical and
            the gradients agree to round-off. Default ``False``.

    Returns:
        A ``FilterResult`` over ``ts``, or over the timeline of the data if
        they have data-only nodes.

    Raises:
        ValueError: For an invalid grid, an ``obs_model`` built for another
            grid, both ``obs_model`` and ``measurements``, invalid measurement
            times, or a calibration the prior does not support.
    """
    ts = check_grid(ts)
    calibration = as_calibration(calibration)
    calibration.validate(prior)
    obs_model = observations_from(obs_model, measurements, prior, ts)
    information = OnlineInformation(
        measure, TaylorCorrection() if correction is None else correction
    )
    return run_filter(
        np.asarray(m_0),
        np.asarray(P_0_sqr),
        prior,
        ts,
        calibration=calibration,
        information=information,
        obs_model=obs_model,
        checkpoint=checkpoint,
    )


def affine_filter(
    m_0: ArrayLike,
    P_0_sqr: ArrayLike,
    prior: BasePrior,
    surrogate: AffineSurrogate,
    ts: ArrayLike,
    *,
    obs_model: ObsModel | None = None,
    measurements: Sequence[Measurement] | None = None,
    checkpoint: bool = False,
) -> FilterResult:
    """Square-root Kalman filter of a frozen linearization.

    Runs the same steps as ``gaussian_filter``, but the ODE information of step
    ``i`` is the given affine model ``surrogate.model(i)`` instead of a
    linearization at the prediction. The diffusion is fixed (the prior carries
    its scale); ``sigma_sqr`` reports the quasi-MLE over all rows of the
    surrogate. The result supports everything a ``gaussian_filter`` result does,
    e.g. ``rts_smoother``. With data-only nodes, the surrogate keeps one model
    per step of the grid ``ts``, which updates at the grid node that ends the
    step, and the result lives on the timeline.

    Args:
        m_0: Initial mean, shape ``[D]``.
        P_0_sqr: Initial square-root covariance, shape ``[D, D]``.
        prior: Gauss-Markov prior (plain or preconditioned).
        surrogate: One affine model per step of ``ts`` (see
            ``AffineSurrogate``), e.g. from ``taylor_surrogate`` or
            ``slr_surrogate``.
        ts: Time grid, shape ``[N + 1]``.
        obs_model: Data for the grid ``ts``, from
            ``prepare_observations(measurements, prior.E0_state, ts)``.
        measurements: The data as ``Measurement`` objects, in place of
            ``obs_model`` (see ``gaussian_filter``).
        checkpoint: Wrap each step in ``jax.checkpoint`` (see
            ``gaussian_filter``). Default ``False``.

    Returns:
        A ``FilterResult`` over ``ts``, or over the timeline of the data if
        they have data-only nodes.

    Raises:
        ValueError: If the surrogate does not have one model per step, or its
            shapes do not match each other or the state; for an ``obs_model``
            built for another grid, or both ``obs_model`` and ``measurements``.
    """
    ts = check_grid(ts)
    m_0 = np.asarray(m_0)
    check_surrogate(surrogate, ts.shape[0] - 1, m_0.shape[0])
    obs_model = observations_from(obs_model, measurements, prior, ts)
    return run_filter(
        m_0,
        np.asarray(P_0_sqr),
        prior,
        ts,
        calibration=NoCalibration(),
        surrogate=surrogate,
        obs_model=obs_model,
        checkpoint=checkpoint,
    )
