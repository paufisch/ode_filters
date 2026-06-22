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
rule R6). The returned scalar is the *observation* marginal log-likelihood: the
data evidence under the ODE-constrained Gauss-Markov model (the Fenrir objective).
Calibration defaults to ``"none"`` (fixed diffusion) because dynamic calibration
absorbs misfit and confounds the likelihood over ``theta``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

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
    """

    build: Callable[[Any], tuple[Array, Array, BaseODEInformation]]
    prior: Any
    tspan: tuple[float, float]
    N: int
    calibration: str = "none"
    min_sigma_sqr: float = 0.0
    correction: Any = None


def marginal_loglik(theta: Any, data: ObsModel, *, model: InferenceProblem) -> Array:
    """Data marginal log-likelihood for gradient-based ODE-parameter inference.

    Args:
        theta: Differentiable parameter pytree, consumed by ``model.build``. May
            contain :class:`~ode_filters.PositiveReal` / :class:`~ode_filters.Real`
            wrappers, which are unwrapped automatically.
        data: Observations as an :class:`ObsModel` (e.g. from
            :func:`prepare_observations`); the measured values live in
            ``data.c_seq``.
        model: Static :class:`InferenceProblem` (closed over, not traced).

    Returns:
        Scalar observation marginal log-likelihood. Maximize over ``theta`` (or
        minimize its negative) to fit parameters. ``jax.jit`` / ``jax.grad`` /
        ``jax.vmap`` over ``theta`` are all supported.

    Raises:
        ValueError: If ``data`` is ``None`` (observations are required).
    """
    # Imported lazily to avoid an import cycle: the filter depends on the
    # low-level inference primitives in this package.
    from ..filters.gaussian_filter import gaussian_filter

    if data is None:
        raise ValueError(
            "marginal_loglik requires observations; pass an ObsModel built via "
            "prepare_observations(...)."
        )

    # Resolve any unconstrained-parameter wrappers (identity on plain arrays).
    theta = unwrap(theta)
    mu_0, Sigma_0_sqr, measure = model.build(theta)
    result = gaussian_filter(
        mu_0,
        Sigma_0_sqr,
        model.prior,
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
    # obs_model was provided, so the filter always fills in the obs-likelihood.
    assert result.log_likelihood_obs is not None
    return result.log_likelihood_obs
