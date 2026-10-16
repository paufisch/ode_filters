"""Filters over a given time grid: ``gaussian_filter`` and ``affine_filter``.

Both scan ``filter_step`` (``ode_filters.filters.kernel``) over the steps of
the grid ``ts`` and differ only in the ODE information of a step:
``gaussian_filter`` linearizes the measurement model at each prediction with a
``Correction``; ``affine_filter`` takes a frozen ``AffineSurrogate``, one
affine model per step, and so is an exact square-root Kalman filter of that
linear-Gaussian model.

The grid may be non-uniform. On a concrete uniform grid the prior's transition
is computed once; otherwise once per step. A grid is concrete when it is a NumPy
array or a JAX array that is not traced -- also when a jitted function closes
over it -- and traced when it is, e.g., an argument of a jitted function. Data
(an ``ObsModel`` from ``prepare_observations`` on the same grid) are conditioned
on after the ODE information at each step.

Reverse-mode differentiation of the scan stores the intermediates of every
step. With ``checkpoint=True`` the step is wrapped in ``jax.checkpoint``, so the
backward pass stores only the state the scan carries from step to step and
recomputes the intermediates, at the cost of one more forward pass.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp
from jax import Array, ad_checkpoint
from jax.typing import ArrayLike

from ..calibration.policies import (
    Calibration,
    CalibrationMode,
    NoCalibration,
    as_calibration,
)
from ..gaussian import AffineModel, Gaussian
from ..information import MeasurementModel, ObsModel
from ..linearization.correction import Correction, TaylorCorrection
from ..linearization.surrogate import AffineSurrogate
from ..priors.base import BasePrior, Transition
from .kernel import FrozenInformation, Information, OnlineInformation, filter_step
from .result import FilterResult


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


def check_obs_model(obs_model: ObsModel, n_steps: int) -> None:
    """Raise if ``obs_model`` was built for a grid with another number of steps."""
    if obs_model.c_seq.shape[0] != n_steps or obs_model.mask.shape[0] != n_steps:
        raise ValueError(
            f"obs_model has {obs_model.c_seq.shape[0]} steps but the grid has "
            f"{n_steps}; build it with prepare_observations(measurements, E0, ts) "
            "on the same grid."
        )


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
    """Scan ``filter_step`` over the grid ``ts``.

    The ODE information is either ``information`` (linearized online) or
    ``surrogate`` (one frozen affine model per step, stacked). ``checkpoint``
    wraps the scan body in ``jax.checkpoint``.
    """
    n_steps = ts.shape[0] - 1
    constant, per_step = grid_transitions(prior, ts)
    if obs_model is not None:
        check_obs_model(obs_model, n_steps)
        observations = (obs_model.c_seq, obs_model.mask)
    else:
        observations = None

    def body(x: Gaussian, inputs):
        t, transition, model, data = inputs
        step_information = (
            information if model is None else FrozenInformation(AffineModel(*model))
        )
        assert step_information is not None
        observation = mask = None
        if data is not None:
            assert obs_model is not None
            c, mask = data
            observation = AffineModel(obs_model.H, c, obs_model.R_sqr)
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
    step = ad_checkpoint.checkpoint(body, prevent_cse=False) if checkpoint else body
    x_0 = Gaussian(np.asarray(m_0), np.asarray(P_0_sqr))
    _, out = jax.lax.scan(step, x_0, (ts[1:], per_step, surrogate, observations))

    def with_initial(first: Array, rest: Array) -> Array:
        return np.concatenate([first[None], rest], axis=0)

    innovation_obs, ll_obs = out.innovation_obs, out.log_likelihood_obs
    return FilterResult(
        t=np.asarray(ts),
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
    checkpoint: bool = False,
) -> FilterResult:
    """Gaussian ODE filter over the time grid ``ts``.

    Each step predicts with the prior, calibrates the diffusion, updates on the
    ODE information linearized by ``correction`` at the prediction, and then on
    the data of ``obs_model``, if any (see ``ode_filters.filters.kernel``). The
    grid may be non-uniform; on a uniform one the prior's transition is computed
    once.

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
        obs_model: Data on the same grid, from
            ``prepare_observations(measurements, E0, ts)``.
        checkpoint: Wrap each step in ``jax.checkpoint``. Reverse-mode
            differentiation then stores only the state carried between steps
            and recomputes each step's intermediates in the backward pass,
            which costs one more forward pass. The values are identical and
            the gradients agree to round-off. Default ``False``.

    Returns:
        A ``FilterResult`` over ``ts``.

    Raises:
        ValueError: For an invalid grid, an ``obs_model`` of another length, or
            a calibration the prior does not support.
    """
    ts = check_grid(ts)
    calibration = as_calibration(calibration)
    calibration.validate(prior)
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
    checkpoint: bool = False,
) -> FilterResult:
    """Square-root Kalman filter of a frozen linearization.

    Runs the same steps as ``gaussian_filter``, but the ODE information of step
    ``i`` is the given affine model ``surrogate.model(i)`` instead of a
    linearization at the prediction. The diffusion is fixed (the prior carries
    its scale); ``sigma_sqr`` reports the quasi-MLE over all rows of the
    surrogate. The result supports everything a ``gaussian_filter`` result does,
    e.g. ``rts_smoother``.

    Args:
        m_0: Initial mean, shape ``[D]``.
        P_0_sqr: Initial square-root covariance, shape ``[D, D]``.
        prior: Gauss-Markov prior (plain or preconditioned).
        surrogate: One affine model per step (see ``AffineSurrogate``), e.g.
            from ``taylor_surrogate`` or ``slr_surrogate``.
        ts: Time grid, shape ``[N + 1]``.
        obs_model: Data on the same grid, from
            ``prepare_observations(measurements, E0, ts)``.
        checkpoint: Wrap each step in ``jax.checkpoint`` (see
            ``gaussian_filter``). Default ``False``.

    Returns:
        A ``FilterResult`` over ``ts``.

    Raises:
        ValueError: If the surrogate does not have one model per step, or its
            shapes do not match each other or the state.
    """
    ts = check_grid(ts)
    m_0 = np.asarray(m_0)
    check_surrogate(surrogate, ts.shape[0] - 1, m_0.shape[0])
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
