"""Differentiable marginal likelihood for ODE-parameter inference (Layer 1).

This is the pure, equinox-free inference core: a function of a *parameter pytree*
``theta`` that returns the data marginal log-likelihood, suitable for
gradient-based optimization (Optax) or sampling (NumPyro / BlackJAX). The
ergonomic, object-oriented ``fit(model, ...)`` wrapper (Layer 2) builds on top of
this.

Design -- *params-as-argument*: ``theta`` carries the differentiable parameters
and flows into the vector field / initial state via the problem's ``build``
callable; the static configuration (prior, grid, calibration) lives in an
:class:`InferenceProblem`. Only ``theta`` is differentiated -- ``model`` and
``data`` are closed over / held static (``model`` holds a Python callable and so
is not itself a traceable argument).

The likelihood is evaluated on the fixed-grid ``jax.lax.scan`` loop
(:func:`sqr_loop_dynamic_scan` with an ``obs_model``), which is the only
jit/grad/vmap-safe path -- the Python-loop and adaptive drivers are not (ROADMAP
rule R6). By default the returned scalar is the *observation* marginal
log-likelihood: the data evidence under the ODE-constrained Gauss-Markov model
(the Fenrir objective). Pass ``channel="ode"`` or ``channel="both"`` to get the
ODE-defect (residual) evidence instead of, or alongside, the data evidence --
methods that weight the two channels separately (split / hybrid / tiered
hyperparameter selection) need the pair. Calibration defaults to ``"none"``
(fixed diffusion) because dynamic calibration absorbs misfit and confounds the
likelihood over ``theta``.

``InferenceProblem.prior_fn`` extends the same core to *prior* hyperparameters
(diffusion scale, length scale): when set, the prior is built from ``theta``
inside the traced region rather than held static.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, cast, overload

from jax import Array

from ..measurement.measurement_models import BaseODEInformation, ObsModel
from .parameters import unwrap

if TYPE_CHECKING:
    from ..filters.ode_filter_adaptive import CalibrationMode


class InferenceProblem(NamedTuple):
    """Static specification mapping a parameter pytree to solver inputs.

    Attributes:
        build: Callable ``theta -> (mu_0, Sigma_0_sqr, measure)`` constructing the
            (differentiable) initial Gaussian and measurement model from the
            parameters. This is the single problem-specific bridge; everything
            else is static.
        prior: Gauss-Markov prior (e.g. :class:`IWP`).
        tspan: Time interval ``(t0, t1)`` (a tuple, for jit-static hashing).
        N: Number of fixed-grid steps.
        calibration: Diffusion calibration mode. Defaults to ``"none"`` (fixed
            diffusion) -- recommended for parameter inference.
        min_sigma_sqr: Lower bound passed through to the loop.
        correction: Linearization strategy (a :class:`~ode_filters.Correction`);
            ``None`` defaults to EK1. Use e.g. ``TaylorCorrection(order=0)`` or
            ``IteratedTaylorCorrection()`` to fit with EK0 / IEKF.
        prior_fn: Optional callable ``theta -> prior``, for fitting the *prior's*
            hyperparameters (diffusion scale, length scale) rather than -- or in
            addition to -- parameters of the vector field. When given it takes
            precedence over the static ``prior``, and the prior is built inside
            the traced region so its hyperparameters are differentiated. When
            ``None`` (the default) the static ``prior`` is used unchanged.
    """

    build: Callable[[Any], tuple[Array, Array, BaseODEInformation]]
    prior: Any
    tspan: tuple[float, float]
    N: int
    calibration: str = "none"
    min_sigma_sqr: float = 0.0
    correction: Any = None
    prior_fn: Callable[[Any], Any] | None = None


@overload
def marginal_loglik(
    theta: Any,
    data: ObsModel,
    *,
    model: InferenceProblem,
    channel: Literal["obs", "ode"] = ...,
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
    data: ObsModel,
    *,
    model: InferenceProblem,
    channel: str = "obs",
) -> Array | tuple[Array, Array]:
    """Marginal log-likelihood for gradient-based ODE-parameter inference.

    Args:
        theta: Differentiable parameter pytree, consumed by ``model.build`` (and
            by ``model.prior_fn``, if set). May contain
            :class:`~ode_filters.PositiveReal` / :class:`~ode_filters.Real`
            wrappers, which are unwrapped automatically.
        data: Observations as an :class:`ObsModel` (e.g. from
            :func:`prepare_observations`); the measured values live in
            ``data.c_seq``. Required unless ``channel="ode"``.
        model: Static :class:`InferenceProblem` (closed over, not traced).
        channel: Which evidence channel to return.

            - ``"obs"`` (default): the observation marginal log-likelihood --
              the data evidence under the ODE-constrained model (the Fenrir
              objective). A scalar.
            - ``"ode"``: the ODE-defect (residual) marginal log-likelihood. A
              scalar; does not require observations.
            - ``"both"``: the tuple ``(ll_ode, ll_obs)``. The two channels are
              summed over *different* numbers of terms -- ``N`` filter steps and
              ``K`` observations respectively -- so weight them deliberately
              when combining; the raw sum makes the residual channel's influence
              scale with the grid density.

    Returns:
        A scalar for ``channel="obs"`` / ``"ode"``, or ``(ll_ode, ll_obs)`` for
        ``channel="both"``. Maximize over ``theta`` (or minimize the negative)
        to fit parameters. ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` over
        ``theta`` are all supported.

    Raises:
        ValueError: If ``channel`` is not one of the three modes, or if ``data``
            is ``None`` for a channel that needs observations.
    """
    # Imported lazily to avoid an import cycle: the filter depends on the
    # low-level inference primitives in this package.
    from ..filters.gaussian_filter import gaussian_filter

    if channel not in ("obs", "ode", "both"):
        raise ValueError(
            f"channel must be one of 'obs', 'ode', 'both'; got {channel!r}."
        )
    if data is None and channel != "ode":
        raise ValueError(
            f"marginal_loglik(channel={channel!r}) requires observations; pass an "
            "ObsModel built via prepare_observations(...), or use channel='ode'."
        )

    # Resolve any unconstrained-parameter wrappers (identity on plain arrays).
    theta = unwrap(theta)
    mu_0, Sigma_0_sqr, measure = model.build(theta)
    # prior_fn puts the prior's hyperparameters inside the traced region.
    prior = model.prior if model.prior_fn is None else model.prior_fn(theta)
    result = gaussian_filter(
        mu_0,
        Sigma_0_sqr,
        prior,
        measure,
        model.tspan,
        model.N,
        # InferenceProblem.calibration is a plain str; the value is validated by
        # the solver, so narrow it to the literal mode the filter expects.
        calibration=cast("CalibrationMode", model.calibration),
        min_sigma_sqr=model.min_sigma_sqr,
        obs_model=data,
        correction=model.correction,
    )
    if channel == "ode":
        return result.log_likelihood
    # data was provided, so the filter always fills in the obs-likelihood.
    assert result.log_likelihood_obs is not None
    if channel == "obs":
        return result.log_likelihood_obs
    return result.log_likelihood, result.log_likelihood_obs
