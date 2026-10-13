"""Ergonomic object API for ODE-parameter inference (Layer 2).

``ODEFilter`` bundles the problem structure (vector field, prior, grid) with
the differentiable parameters as a single ``equinox.Module``, and ``fit``
optimizes it against data -- the GPJax ``fit(model, ...)`` pattern. This is sugar
over the pure Layer-1 core (``marginal_loglik``): ``ODEFilter.loglik`` builds
an ``InferenceProblem`` from the module and calls the core, so there is a
single source of truth for the likelihood.

``fit`` trains ``ode_params`` only; the vector field, prior, grid and
calibration stay fixed, and the initial Gaussian is produced by a static
``init_fn(params)`` callable.

For anything beyond a plain first-order ODE (hidden-state / joint priors, custom
measurement models, fitting the initial condition), use ``marginal_loglik``
with a custom ``InferenceProblem`` directly. Pass ``correction=`` to fit with
a different linearization scheme (e.g. ``TaylorCorrection(order=0)`` for EK0 or
``IteratedTaylorCorrection()`` for IEKF); it defaults to EK1.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax import Array

from ..calibration.policies import Calibration, CalibrationMode
from ..information import ObsModel, ODEInformation
from .likelihood import InferenceProblem, marginal_loglik


def _as_static_grid(ts: Any) -> tuple[float, ...]:
    return tuple(float(t) for t in onp.asarray(ts))


class ODEFilter(eqx.Module):
    """A fittable first-order probabilistic ODE solver.

    Attributes:
        vf: Vector field ``vf(x, params, *, t) -> dx/dt`` (``params`` is the
            differentiable parameter pytree, ``ode_params``).
        init_fn: Callable ``params -> (m_0, P_0_sqr)`` producing the initial
            Gaussian (square-root covariance). May ignore ``params`` for a fixed
            initial condition.
        prior: Gauss-Markov prior (e.g. ``IWP``); fixed during ``fit``.
        ts: Time grid, shape ``[K]``; stored as a tuple, so it is static
            configuration rather than a trainable leaf.
        ode_params: The parameters ``fit`` trains.
        calibration: Diffusion calibration policy or shorthand (default ``"none"``).
    """

    vf: Callable = eqx.field(static=True)
    init_fn: Callable = eqx.field(static=True)
    prior: Any
    ts: tuple[float, ...] = eqx.field(static=True, converter=_as_static_grid)
    ode_params: Array
    calibration: Calibration | CalibrationMode = eqx.field(static=True, default="none")
    correction: Any = eqx.field(static=True, default=None)

    def loglik(self, data: ObsModel) -> Array:
        """Data marginal log-likelihood at the current parameters."""

        def build(params):
            def vf(x, *, t):
                return self.vf(x, params, t=t)

            measure = ODEInformation(vf, self.prior.E0, self.prior.E1)
            m_0, P_0_sqr = self.init_fn(params)
            return m_0, P_0_sqr, measure

        problem = InferenceProblem(
            build=build,
            prior=self.prior,
            ts=np.asarray(self.ts),
            calibration=self.calibration,
            correction=self.correction,
        )
        return marginal_loglik(self.ode_params, data, model=problem)


def fit(
    model: ODEFilter,
    data: ObsModel,
    optim: Any,
    *,
    steps: int = 200,
) -> tuple[ODEFilter, Array]:
    """Fit an ``ODEFilter`` by maximizing the data marginal log-likelihood.

    Args:
        model: Initial ``ODEFilter``.
        data: Observations as an ``ObsModel``.
        optim: An Optax ``GradientTransformation`` (anything exposing
            ``init`` / ``update``); the library does not import Optax itself.
        steps: Number of optimization steps.

    Returns:
        Tuple ``(fitted_model, losses)`` where ``losses`` is the per-step negative
        log-likelihood (shape ``[steps]``).
    """
    # Only ode_params is trained; the prior and everything else stay fixed.
    trainable = jax.tree.map(lambda _: False, model)
    trainable = eqx.tree_at(
        lambda m: m.ode_params,
        trainable,
        replace=jax.tree.map(eqx.is_inexact_array, model.ode_params),
    )
    params, fixed = eqx.partition(model, trainable)
    opt_state = optim.init(params)

    @eqx.filter_jit
    def make_step(params, opt_state):
        def loss_fn(p):
            return -eqx.combine(p, fixed).loglik(data)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(params)
        updates, opt_state = optim.update(grads, opt_state)
        return eqx.apply_updates(params, updates), opt_state, loss

    losses = []
    for _ in range(steps):
        params, opt_state, loss = make_step(params, opt_state)
        losses.append(loss)
    return eqx.combine(params, fixed), np.asarray(losses)


def fit_scipy(
    objective: Callable[[Any], Array],
    theta0: Any,
    *,
    method: str = "L-BFGS-B",
    **options: Any,
) -> tuple[Any, Any]:
    """Minimize ``objective(theta)`` with SciPy, using JAX gradients.

    A two-tier alternative to the Optax ``fit``: quasi-Newton methods such as
    L-BFGS-B converge in far fewer evaluations on the smooth, low-dimensional
    objectives of hyperparameter selection. ``theta`` may be any pytree of
    arrays (it is flattened for SciPy and unflattened for the objective); the
    objective and its gradient are jitted once.

    Args:
        objective: Scalar function to minimize, e.g. ``lambda th:
            -marginal_loglik(th, data, model=problem)``.
        theta0: Initial parameters (a pytree of arrays).
        method: A gradient-based ``scipy.optimize.minimize`` method.
        **options: Passed to ``scipy.optimize.minimize`` (e.g. ``bounds``,
            ``tol``, ``options``).

    Returns:
        ``(theta_hat, result)``: the minimizer as a pytree like ``theta0``, and
        SciPy's ``OptimizeResult``.

    Raises:
        ImportError: If SciPy is not installed (``pip install ode-filters[inference]``).
    """
    try:
        from scipy.optimize import minimize
    except ImportError as err:  # pragma: no cover - scipy is a dev dependency
        raise ImportError(
            "fit_scipy needs SciPy: pip install 'ode-filters[inference]'."
        ) from err
    from jax.flatten_util import ravel_pytree

    flat0, unravel = ravel_pytree(theta0)
    value_and_grad = jax.jit(jax.value_and_grad(lambda flat: objective(unravel(flat))))

    def fun(flat):
        value, grad = value_and_grad(np.asarray(flat))
        return float(value), onp.asarray(grad, dtype=float)

    result = minimize(
        fun, onp.asarray(flat0, dtype=float), jac=True, method=method, **options
    )
    return unravel(np.asarray(result.x)), result
