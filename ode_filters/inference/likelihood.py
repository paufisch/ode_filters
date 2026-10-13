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
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple, overload

from jax import Array

from ..calibration.policies import Calibration, CalibrationMode
from ..filters.fixed_grid import gaussian_filter
from ..information import BaseODEInformation, ObsModel
from .evidence import conditional_loglik, fenrir_loglik
from .parameters import unwrap


class InferenceProblem(NamedTuple):
    """Static specification mapping a parameter pytree to solver inputs.

    Attributes:
        build: Callable ``theta -> (m_0, P_0_sqr, measure)`` constructing the
            (differentiable) initial Gaussian and measurement model from the
            parameters. This is the single problem-specific bridge; everything
            else is static.
        prior: Gauss-Markov prior (e.g. ``IWP``).
        ts: Time grid, shape ``[K]``; the observations' grid.
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
    """

    build: Callable[[Any], tuple[Array, Array, BaseODEInformation]]
    prior: Any
    ts: Array
    calibration: Calibration | CalibrationMode = "none"
    correction: Any = None
    prior_fn: Callable[[Any], Any] | None = None


_CHANNELS = ("obs", "ode", "both", "fenrir", "dalton")


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

    Returns:
        A scalar, or ``(ll_ode, ll_obs)`` for ``channel="both"``. Maximize over ``theta`` (or minimize the negative)
        to fit parameters. ``jax.jit`` / ``jax.grad`` / ``jax.vmap`` over
        ``theta`` are all supported.

    Raises:
        ValueError: If ``channel`` is unknown, or if ``data`` is ``None`` for a
            channel that needs observations.
    """
    if channel not in _CHANNELS:
        raise ValueError(f"channel must be one of {_CHANNELS}; got {channel!r}.")
    if data is None and channel != "ode":
        raise ValueError(
            f"marginal_loglik(channel={channel!r}) requires observations; pass an "
            "ObsModel built via prepare_observations(...), or use channel='ode'."
        )

    # Resolve any unconstrained-parameter wrappers (identity on plain arrays).
    theta = unwrap(theta)
    m_0, P_0_sqr, measure = model.build(theta)
    # prior_fn puts the prior's hyperparameters inside the traced region.
    prior = model.prior if model.prior_fn is None else model.prior_fn(theta)

    def run(obs_model: ObsModel | None):
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

    if channel in ("fenrir", "dalton"):
        assert data is not None
        data_free = run(None)
        if channel == "fenrir":
            return fenrir_loglik(data_free, data)
        return conditional_loglik(run(data), data_free)
    result = run(data)
    if channel == "ode":
        return result.log_likelihood
    # data was provided, so the filter always fills in the obs-likelihood.
    assert result.log_likelihood_obs is not None
    if channel == "obs":
        return result.log_likelihood_obs
    return result.log_likelihood, result.log_likelihood_obs
