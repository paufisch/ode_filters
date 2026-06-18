"""Ergonomic object API for ODE-parameter inference (Layer 2).

:class:`ODEFilter` bundles the problem structure (vector field, prior, grid) with
the differentiable parameters as a single ``equinox.Module``, and :func:`fit`
optimizes it against data -- the GPJax ``fit(model, ...)`` pattern. This is sugar
over the pure Layer-1 core (:func:`marginal_loglik`): ``ODEFilter.loglik`` builds
an :class:`InferenceProblem` from the module and calls the core, so there is a
single source of truth for the likelihood.

Only ``ode_params`` is a trainable (array) leaf; the vector field, prior, grid and
calibration are static configuration, and the initial Gaussian is produced by a
static ``init_fn(params)`` callable -- so ``eqx.filter_value_and_grad``
differentiates exactly the parameters.

For anything beyond a plain first-order ODE (hidden-state / joint priors, custom
measurement models, fitting the initial condition), use :func:`marginal_loglik`
with a custom :class:`InferenceProblem` directly. Selectable linearization schemes
(EK0/EK1) in the inference path await the observation loop becoming
correction-aware (ROADMAP P0.5).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as np
from jax import Array

from ..measurement.measurement_models import ObsModel, ODEInformation
from .parameter_inference import InferenceProblem, marginal_loglik


class ODEFilter(eqx.Module):
    """A fittable first-order probabilistic ODE solver.

    Attributes:
        vf: Vector field ``vf(x, params, *, t) -> dx/dt`` (``params`` is the
            differentiable parameter pytree, ``ode_params``).
        init_fn: Callable ``params -> (mu_0, Sigma_0_sqr)`` producing the initial
            Gaussian (square-root covariance). May ignore ``params`` for a fixed
            initial condition.
        prior: Gauss-Markov prior (e.g. :class:`IWP`).
        tspan: Time interval ``(t0, t1)``.
        N: Number of fixed-grid steps.
        ode_params: The differentiable parameters (the only trainable leaf).
        calibration: Diffusion calibration mode (default ``"none"``).
    """

    vf: Callable = eqx.field(static=True)
    init_fn: Callable = eqx.field(static=True)
    prior: Any = eqx.field(static=True)
    tspan: tuple[float, float] = eqx.field(static=True)
    N: int = eqx.field(static=True)
    ode_params: Array
    calibration: str = eqx.field(static=True, default="none")

    def loglik(self, data: ObsModel) -> Array:
        """Data marginal log-likelihood at the current parameters."""

        def build(params):
            def vf(x, *, t):
                return self.vf(x, params, t=t)

            measure = ODEInformation(vf, self.prior.E0, self.prior.E1)
            mu_0, Sigma_0_sqr = self.init_fn(params)
            return mu_0, Sigma_0_sqr, measure

        problem = InferenceProblem(
            build=build,
            prior=self.prior,
            tspan=self.tspan,
            N=self.N,
            calibration=self.calibration,
        )
        return marginal_loglik(self.ode_params, data, model=problem)


def fit(
    model: ODEFilter,
    data: ObsModel,
    optim: Any,
    *,
    steps: int = 200,
) -> tuple[ODEFilter, Array]:
    """Fit an :class:`ODEFilter` by maximizing the data marginal log-likelihood.

    Args:
        model: Initial :class:`ODEFilter`.
        data: Observations as an :class:`ObsModel`.
        optim: An Optax ``GradientTransformation`` (anything exposing
            ``init`` / ``update``); the library does not import Optax itself.
        steps: Number of optimization steps.

    Returns:
        Tuple ``(fitted_model, losses)`` where ``losses`` is the per-step negative
        log-likelihood (shape ``[steps]``).
    """
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(model, opt_state):
        loss, grads = eqx.filter_value_and_grad(lambda m: -m.loglik(data))(model)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    losses = []
    for _ in range(steps):
        model, opt_state, loss = make_step(model, opt_state)
        losses.append(loss)
    return model, np.asarray(losses)
