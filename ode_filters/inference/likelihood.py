"""Differentiable marginal likelihood for ODE-parameter inference (Layer 1).

This is the pure, equinox-free inference core: a function of a *parameter pytree*
``theta`` that returns the data marginal log-likelihood, suitable for
gradient-based optimization (Optax) or sampling (NumPyro / BlackJAX). The
ergonomic, object-oriented ``fit(model, ...)`` wrapper (Layer 2) builds on top of
this.

Design -- *params-as-argument*: ``theta`` carries the differentiable parameters
and flows into the vector field / initial state via the problem's ``build``
callable; the static configuration (prior, grid, calibration) lives in an
``InferenceProblem``. Only ``theta`` is differentiated -- ``model`` and
``data`` are closed over / held static (``model`` holds a Python callable and so
is not itself a traceable argument).

The likelihood is evaluated with ``gaussian_filter`` on the problem's time
grid. By default the returned scalar is the *observation* channel,
``FilterResult.log_likelihood_obs``: the data evidence of the sequential filter,
each observation conditioned on the ODE information up to its own time. Pass
``channel="ode"`` or ``channel="both"`` to get the ODE-information evidence
instead of, or alongside, it -- methods that weight the two channels separately
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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple, overload

import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..calibration.policies import Calibration, CalibrationMode
from ..filters.adaptive import SigmaInError, gaussian_filter_adaptive
from ..filters.adaptive_controller import StepSizeController
from ..filters.fixed_grid import gaussian_filter
from ..filters.result import FilterResult
from ..information import BaseODEInformation, ObsModel
from ..linearization import Correction
from ..priors import BasePrior
from .evidence import conditional_loglik, fenrir_loglik
from .parameters import unwrap


class AdaptiveSettings(NamedTuple):
    """Options of ``gaussian_filter_adaptive`` for ``marginal_loglik``.

    In ``InferenceProblem.adaptive``, they make ``marginal_loglik`` run
    ``gaussian_filter_adaptive`` with ``save_at=model.ts``. The solver chooses
    its steps between the nodes of ``model.ts``, lands on each node and
    conditions on the data there. Each field is passed on as the solver's
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
        ts: Time grid, shape ``[K]``; the observations' grid. With
            ``adaptive``, the save grid of the adaptive solver, which steps
            between its nodes.
        calibration: Diffusion calibration policy or shorthand. Defaults to
            ``"none"`` (fixed diffusion) -- recommended for parameter inference.
        correction: Linearization strategy (a ``Correction``);
            ``None`` defaults to EK1. Use e.g. ``TaylorCorrection(order=0)`` or
            ``IteratedTaylorCorrection()`` to fit with EK0 / IEKF.
        prior_fn: Optional callable ``theta -> prior``, for fitting the *prior's*
            hyperparameters (diffusion scale, length scale) rather than -- or in
            addition to -- parameters of the vector field. When given it takes
            precedence over the static ``prior``, and the prior is built inside
            the traced region so its hyperparameters are differentiated. When
            ``None`` (the default) the static ``prior`` is used unchanged.
        adaptive: ``None`` (the default) solves on the fixed grid ``ts`` with
            ``gaussian_filter``. ``AdaptiveSettings`` solve with
            ``gaussian_filter_adaptive`` and these settings, with ``ts`` as the
            save grid, where the data must lie. The channel ``"dalton"`` is
            not available on this path.
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
    data: ObsModel,
    *,
    model: InferenceProblem,
    channel: Literal["obs", "fenrir", "dalton"] = ...,
) -> Array: ...


# The ODE channel is the one mode that needs no observations, so it is the only
# overload accepting ``data=None``; the other two keep observations mandatory.
@overload
def marginal_loglik(
    theta: Any,
    data: ObsModel | None,
    *,
    model: InferenceProblem,
    channel: Literal["ode"],
) -> Array: ...


@overload
def marginal_loglik(
    theta: Any,
    data: ObsModel,
    *,
    model: InferenceProblem,
    channel: Literal["both"],
) -> tuple[Array, Array]: ...


def marginal_loglik(
    theta: Any,
    data: ObsModel | None,
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
        data: Observations as an ``ObsModel`` (e.g. from
            ``prepare_observations``); the measured values live in
            ``data.c_seq``. Required unless ``channel="ode"``, which needs no
            observations and so accepts ``None``.
        model: Static ``InferenceProblem`` (closed over, not traced).
        channel: Which evidence channel to return.

            - ``"obs"`` (default): ``FilterResult.log_likelihood_obs``, the data
              evidence of the sequential filter. A scalar.
            - ``"ode"``: the ODE-defect (residual) marginal log-likelihood. A
              scalar; does not require observations.
            - ``"both"``: the tuple ``(ll_ode, ll_obs)``. The two channels are
              summed over *different* numbers of terms -- ``N`` filter steps and
              ``K`` observations respectively -- so weight them deliberately
              when combining; the raw sum makes the residual channel's influence
              scale with the grid density.
            - ``"fenrir"``: the conditional evidence ``log p(Y | Z)`` by Fenrir:
              one pass without data, then ``fenrir_loglik``.
            - ``"dalton"``: ``log p(Y | Z)`` by Dalton: a pass with and one
              without data, then ``conditional_loglik``. With one-shot
              linearizations it differs from Fenrir's value (see
              ``ode_filters.inference.evidence``).

            With adaptive steps (``model.adaptive``), the ODE information
            ``Z`` lives on the steps the solver chooses, and ``"ode"`` sums
            over those steps. ``"fenrir"`` runs the pass without data with
            ``smoother=True`` and filters the data backwards through its
            backward conditionals, one per interval of ``model.ts``.
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
        ValueError: If ``channel`` is unknown, if ``data`` is ``None`` for a
            channel that needs observations, or for ``channel="dalton"`` with
            adaptive steps.
    """
    if channel not in _CHANNELS:
        raise ValueError(f"channel must be one of {_CHANNELS}; got {channel!r}.")
    if data is None and channel != "ode":
        raise ValueError(
            f"marginal_loglik(channel={channel!r}) requires observations; pass an "
            "ObsModel built via prepare_observations(...), or use channel='ode'."
        )
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

    def run(obs_model: ObsModel | None) -> FilterResult:
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
            # Fenrir needs the backward conditionals of the pass without data.
            smoother=obs_model is None and channel == "fenrir",
        )

    if channel in ("fenrir", "dalton"):
        assert data is not None
        data_free = run(None)
        if channel == "fenrir":
            return _checked(data_free, fenrir_loglik(data_free, data))
        return conditional_loglik(run(data), data_free)
    result = run(data)
    ll_ode = _checked(result, result.log_likelihood)
    if channel == "ode":
        return ll_ode
    # data was provided, so the filter always fills in the obs-likelihood.
    assert result.log_likelihood_obs is not None
    ll_obs = _checked(result, result.log_likelihood_obs)
    if channel == "obs":
        return ll_obs
    return ll_ode, ll_obs
