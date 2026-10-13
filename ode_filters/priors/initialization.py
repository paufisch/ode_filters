"""Initial states for the filters: Taylor-mode initialization."""

from __future__ import annotations

from operator import index

import jax
import jax.experimental.jet
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from .base import VectorField


def taylor_mode_initialization(
    vf: VectorField,
    inits: ArrayLike | tuple[ArrayLike, ...],
    q: int,
    t0: float = 0.0,
    order: int = 1,
) -> tuple[Array, Array]:
    """Return flattened Taylor-mode coefficients produced via JAX Jet.

    Args:
        vf: Vector field whose Taylor coefficients are required.
            For order=1: vf(x, *, t) -> dx/dt
            For order=2: vf(x, dx, *, t) -> d²x/dt²
        inits: Initial value(s) around which the expansion takes place.
            For order=1: x0 (initial state)
            For order=2: (x0, dx0) tuple of initial state and velocity
        q: Number of higher-order coefficients to compute.
        t0: Initial time for Taylor expansion (default 0.0).
        order: ODE order (1 or 2, default 1).

    Returns:
        Tuple of (coefficients, covariance) where:
        - coefficients are the flattened Taylor coefficients as numpy array
        - covariance is a zero covariance matrix as numpy array
    """
    if not callable(vf):
        raise TypeError("vf must be callable.")
    q = index(q)
    if q < 0:
        raise ValueError("q must be a non-negative integer.")
    order = index(order)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")

    # Normalize inits to a tuple of initial derivatives: (x0,) or (x0, dx0)
    if order == 1:
        init_derivs = (np.asarray(inits),)
    else:
        if not isinstance(inits, (tuple, list)) or len(inits) != order:
            raise ValueError(
                f"For order={order}, inits must be a tuple of {order} arrays."
            )
        init_derivs = tuple(np.asarray(x) for x in inits)
        if not all(x.shape == init_derivs[0].shape for x in init_derivs):
            raise ValueError("All initial derivatives must have the same shape.")

    d = init_derivs[0].shape[0]

    # Augmented state: [x0, dx0, ...] and augmented vector field
    # For order k: y = [x, v1, ..., v_{k-1}], dy/dt = [v1, ..., v_{k-1}, vf(...)]
    base_state = np.concatenate(init_derivs)

    def aug_vf(y):
        derivs = [y[i * d : (i + 1) * d] for i in range(order)]
        highest_deriv = vf(*derivs, t=t0)
        return np.concatenate([*derivs[1:], highest_deriv])

    coefficients: list[Array] = [init_derivs[0]]
    series_terms: list[Array] = []

    for _ in range(q):
        primals_out, series_out = jax.experimental.jet.jet(
            aug_vf,
            primals=(base_state,),
            series=(tuple(series_terms),),
        )

        updated_series = [np.asarray(primals_out)]
        updated_series.extend(np.asarray(term) for term in series_out)
        series_terms = updated_series

        # Extract x-part (first d components) from the augmented derivative
        coefficients.append(series_terms[-1][:d])

    leaves = jax.tree_util.tree_leaves(coefficients)
    init = np.concatenate([np.ravel(arr) for arr in leaves])
    D = init.shape[0]
    return init, np.zeros((D, D))
