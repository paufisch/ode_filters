"""Differentiable marginal likelihood for ODE-parameter inference (Layer 1).

This is the pure, equinox-free inference core: a function of a *parameter pytree*
``theta`` that returns the data marginal log-likelihood, suitable for
gradient-based optimization (Optax) or sampling (NumPyro / BlackJAX). The
ergonomic, object-oriented ``fit(model, ...)`` wrapper (Layer 2) builds on top of
this.

The design passes the parameters as an argument. ``theta`` carries the
differentiable parameters and flows into the vector field and the initial state
through the problem's ``build`` callable, and the static configuration (prior,
grid, calibration) lives in an ``InferenceProblem``. Only ``theta`` is
differentiated. ``model`` and ``data`` are closed over or held static
(``model`` holds a Python callable and is therefore not a traceable argument).

The likelihood is evaluated with ``gaussian_filter`` on the problem's time
grid. By default the returned scalar is the *observation* channel,
``FilterResult.log_likelihood_obs``: the data evidence of the sequential filter,
each observation conditioned on the ODE information up to its own time. Pass
``channel="ode"`` or ``channel="both"`` to get the ODE-information evidence
instead of it or alongside it. Methods that weight the two channels separately
need the pair. The conditional evidence ``log p(Y | Z)``, which conditions the
data on *all* of the ODE information, is the ``"fenrir"`` or ``"dalton"``
channel (see ``ode_filters.inference.evidence``). Calibration defaults to
``"none"`` (fixed diffusion) because dynamic calibration absorbs misfit and
confounds the likelihood over ``theta``.

``InferenceProblem.prior_fn`` extends the same core to *prior* hyperparameters
(diffusion scale, length scale): when set, the prior is built from ``theta``
inside the traced region rather than held static.

``InferenceProblem.adaptive`` switches the solve to adaptive steps: with
``AdaptiveSettings``, ``gaussian_filter_adaptive`` runs with the problem's grid
as its save grid, which also carries the data. The channels ``"obs"``,
``"ode"``, ``"both"`` and ``"fenrir"`` are available on this path.

The data are an ``ObsModel`` for the problem's grid or a sequence of
``Measurement`` objects. Data between the grid nodes become data-only nodes
(see ``prepare_observations``), on both paths and for every channel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, NamedTuple, overload

import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..calibration.policies import Calibration, CalibrationMode
from ..filters.adaptive import SigmaInError, gaussian_filter_adaptive
from ..filters.adaptive_controller import StepSizeController
from ..filters.fixed_grid import gaussian_filter
from ..filters.result import FilterResult
from ..information import (
    BaseODEInformation,
    Measurement,
    ObsModel,
    prepare_observations,
)
from ..linearization import Correction
from ..priors import BasePrior
from .evidence import conditional_loglik, fenrir_loglik
from .parameters import unwrap


class AdaptiveSettings(NamedTuple):
    """Options of ``gaussian_filter_adaptive`` for ``marginal_loglik``.

    In ``InferenceProblem.adaptive``, they make ``marginal_loglik`` run
    ``gaussian_filter_adaptive`` with ``save_at=model.ts``. The solver chooses
    its steps between the nodes of ``model.ts``, lands on each node and on each
    data time between them, and conditions on the data there. Each field is passed on as the solver's
    argument of the same name, and the defaults are the solver's. The
    tolerances set the accuracy of the likelihood.

    Attributes:
        atol: Absolute tolerance of the local-error estimate.
        rtol: Relative tolerance of the local-error estimate.
        h_init: First trial step; ``None`` is ``(ts[-1] - ts[0]) / 100``.
        max_step: Upper bound on every step; ``None`` leaves the next node of
            ``model.ts`` as the only bound.
        controller: Step-size controller; ``None`` is
            ``PIController(order=prior.q)``.
        sigma_in_error: The diffusion in the error estimate, ``"per_step"`` or
            ``"running_mean"``.
        max_steps: Cap on the attempts per interval of ``model.ts``. A solve
            that reaches it fails, and ``marginal_loglik`` returns NaN.
    """

    atol: ArrayLike = 1e-4
    rtol: ArrayLike = 1e-2
    h_init: ArrayLike | None = None
    max_step: ArrayLike | None = None
    controller: StepSizeController | None = None
    sigma_in_error: SigmaInError = "per_step"
    max_steps: int = 4096


class InferenceProblem(NamedTuple):
    """Static specification mapping a parameter pytree to solver inputs.

    Attributes:
        build: Callable ``theta -> (m_0, P_0_sqr, measure)`` constructing the
            (differentiable) initial Gaussian and measurement model from the
            parameters. This is the single problem-specific bridge; everything
            else is static.
        prior: Gauss-Markov prior (e.g. ``IWP``).
        ts: Time grid, shape ``[K]``, for which the data are prepared. With
            ``adaptive``, the save grid of the adaptive solver, which steps
            between its nodes.
        calibration: Diffusion calibration policy or shorthand. Defaults to
            ``"none"`` (fixed diffusion), the recommended setting for parameter
            inference.
        correction: Linearization strategy (a ``Correction``);
            ``None`` defaults to EK1. Use e.g. ``TaylorCorrection(order=0)`` or
            ``IteratedTaylorCorrection()`` to fit with EK0 / IEKF.
        prior_fn: Optional callable ``theta -> prior``, for fitting the
            hyperparameters of the prior (diffusion scale, length scale) instead
            of or in addition to parameters of the vector field. When given it
            takes precedence over the static ``prior``, and the prior is built
            inside the traced region so its hyperparameters are differentiated.
            When ``None`` (the default) the static ``prior`` is used unchanged.
        adaptive: ``None`` (the default) solves on the fixed grid ``ts`` with
            ``gaussian_filter``. ``AdaptiveSettings`` solve with
            ``gaussian_filter_adaptive`` and these settings, with ``ts`` as the
            save grid; the solver also lands on the data times between its
            nodes. The channel ``"dalton"`` is not available on this path.
    """

    build: Callable[[Any], tuple[Array, Array, BaseODEInformation]]
    prior: BasePrior
    ts: Array
    calibration: Calibration | CalibrationMode = "none"
    correction: Correction | None = None
    prior_fn: Callable[[Any], BasePrior] | None = None
    adaptive: AdaptiveSettings | None = None


_CHANNELS = ("obs", "ode", "both", "fenrir", "dalton")


def _checked(result: FilterResult, value: Array) -> Array:
    """``value``, or NaN where an adaptive solve failed (missed a save time)."""
    if result.success is None:  # the fixed grid
        return value
    return np.where(result.success, value, np.nan)


@overload
def marginal_loglik(
    theta: Any,
    data: ObsModel | Sequence[Measurement],
    *,
    model: InferenceProblem,
    channel: Literal["obs", "fenrir", "dalton"] = ...,
) -> Array: ...


# The ODE channel is the one mode that needs no observations, so it is the only
# overload accepting ``data=None``; the other two keep observations mandatory.
@overload
def marginal_loglik(
    theta: Any,
    data: ObsModel | Sequence[Measurement] | None,
    *,
    model: InferenceProblem,
    channel: Literal["ode"],
) -> Array: ...


@overload
def marginal_loglik(
    theta: Any,
    data: ObsModel | Sequence[Measurement],
    *,
    model: InferenceProblem,
    channel: Literal["both"],
) -> tuple[Array, Array]: ...


def _require_data(data: object, channel: str) -> None:
    """Raise unless there are data or the channel needs none."""
    if data is None and channel != "ode":
        raise ValueError(
            f"marginal_loglik(channel={channel!r}) requires observations; pass an "
            "ObsModel or a sequence of Measurement objects, or use channel='ode'."
        )


def marginal_loglik(
    theta: Any,
    data: ObsModel | Sequence[Measurement] | None,
    *,
    model: InferenceProblem,
    channel: str = "obs",
) -> Array | tuple[Array, Array]:
    """Marginal log-likelihood for gradient-based ODE-parameter inference.

    Args:
        theta: Differentiable parameter pytree, consumed by ``model.build`` (and
            by ``model.prior_fn``, if set). May contain
            ``PositiveReal`` / ``Real``
            wrappers, which are unwrapped automatically.
        data: Observations as an ``ObsModel`` for ``model.ts`` (e.g. from
            ``prepare_observations``), whose measured values live in
            ``data.c_seq``, or as a sequence of ``Measurement`` objects. These
            are converted with
            ``prepare_observations(data, prior.E0_state, model.ts)``, where
            ``prior`` is ``model.prior`` or ``model.prior_fn(theta)``. Data
            between the nodes of ``model.ts`` become data-only nodes. Required
            unless ``channel="ode"``, which needs no observations and so
            accepts ``None``.
        model: Static ``InferenceProblem`` (closed over, not traced).
        channel: Which evidence channel to return.

            - ``"obs"`` (default): ``FilterResult.log_likelihood_obs``, the data
              evidence of the sequential filter. A scalar.
            - ``"ode"``: the ODE-defect (residual) marginal log-likelihood. A
              scalar; does not require observations. With data-only nodes it
              is the evidence of the ODE information at the grid nodes, given
              the data before each of them.
            - ``"both"``: the tuple ``(ll_ode, ll_obs)``. The two channels are
              sums over different numbers of terms, ``N`` filter steps and
              ``K`` observations respectively. Choose their weights explicitly
              when combining them, because in the unweighted sum the influence
              of the residual channel grows with the grid density.
            - ``"fenrir"``: the conditional evidence ``log p(Y | Z)`` by Fenrir:
              one pass without data, then ``fenrir_loglik``.
            - ``"dalton"``: ``log p(Y | Z)`` by Dalton: a pass with and one
              without data, then ``conditional_loglik``. With one-shot
              linearizations it differs from Fenrir's value (see
              ``ode_filters.inference.evidence``).

            The passes without data of ``"fenrir"`` and ``"dalton"`` run on the
            nodes of the data, ``data.without_data()``, which include any
            data-only nodes.

            With adaptive steps (``model.adaptive``), the ODE information
            ``Z`` lives on the steps the solver chooses, and ``"ode"`` sums
            over those steps. ``"fenrir"`` runs the pass without data with
            ``smoother=True`` and filters the data backwards through its
            backward conditionals, one per interval between landing points.
            ``"dalton"`` raises: the passes with and without data choose
            different steps, so their ODE evidences condition on ODE
            information at different times, and their difference is not
            ``log p(Y | Z)``.

    Returns:
        A scalar, or ``(ll_ode, ll_obs)`` for ``channel="both"``. Maximize over ``theta`` (or minimize the negative)
        to fit parameters. ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` over
        ``theta`` are all supported. An adaptive solve that misses a save time
        (``max_steps`` reached) or has a non-finite evidence returns NaN.

    Raises:
        ValueError: If ``channel`` is unknown, if ``data`` is ``None`` or empty
            for a channel that needs observations, for invalid measurement
            times, or for ``channel="dalton"`` with adaptive steps.
    """
    if channel not in _CHANNELS:
        raise ValueError(f"channel must be one of {_CHANNELS}; got {channel!r}.")
    _require_data(data, channel)
    adaptive = model.adaptive
    if adaptive is not None and channel == "dalton":
        raise ValueError(
            "channel='dalton' is not available with adaptive steps: the passes "
            "with and without data choose different steps, so their difference "
            "is not log p(Y | Z). Use channel='fenrir', or adaptive=None."
        )

    # Resolve any unconstrained-parameter wrappers (identity on plain arrays).
    theta = unwrap(theta)
    m_0, P_0_sqr, measure = model.build(theta)
    # prior_fn puts the prior's hyperparameters inside the traced region.
    prior = model.prior if model.prior_fn is None else model.prior_fn(theta)
    obs = data
    if data is not None and not isinstance(data, ObsModel):
        obs = prepare_observations(data, prior.E0_state, model.ts)
        _require_data(obs, channel)
    assert obs is None or isinstance(obs, ObsModel)

    def run(obs_model: ObsModel | None, *, smoother: bool = False) -> FilterResult:
        if adaptive is None:
            return gaussian_filter(
                m_0,
                P_0_sqr,
                prior,
                measure,
                model.ts,
                calibration=model.calibration,
                obs_model=obs_model,
                correction=model.correction,
            )
        return gaussian_filter_adaptive(
            m_0,
            P_0_sqr,
            prior,
            measure,
            model.ts,
            correction=model.correction,
            calibration=model.calibration,
            obs_model=obs_model,
            atol=adaptive.atol,
            rtol=adaptive.rtol,
            h_init=adaptive.h_init,
            max_step=adaptive.max_step,
            controller=adaptive.controller,
            sigma_in_error=adaptive.sigma_in_error,
            max_steps=adaptive.max_steps,
            smoother=smoother,
        )

    if channel in ("fenrir", "dalton"):
        assert obs is not None
        # Fenrir needs the backward conditionals of the pass without data.
        data_free = run(obs.without_data(), smoother=channel == "fenrir")
        if channel == "fenrir":
            return _checked(data_free, fenrir_loglik(data_free, obs))
        return conditional_loglik(run(obs), data_free)
    result = run(obs)
    ll_ode = _checked(result, result.log_likelihood)
    if channel == "ode":
        return ll_ode
    # data was provided, so the filter always fills in the obs-likelihood.
    assert result.log_likelihood_obs is not None
    ll_obs = _checked(result, result.log_likelihood_obs)
    if channel == "obs":
        return ll_obs
    return ll_ode, ll_obs
