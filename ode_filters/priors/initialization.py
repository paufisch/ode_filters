"""Initial states for the filters: Taylor-mode initialization.

``taylor_mode_initialization`` gives the exact derivatives of the initial value
(a Dirac on the state). ``joint_taylor_initialization`` does the same for an ODE
driven by a hidden input with a prior of its own, and carries the input's
uncertainty into the state block consistently.
"""

from __future__ import annotations

from operator import index

import jax
import jax.experimental.jet
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from .base import VectorField
from .joint import JointPrior


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


def joint_taylor_initialization(
    vf: VectorField,
    inits: ArrayLike | tuple[ArrayLike, ArrayLike],
    *,
    prior: JointPrior,
    t0: float = 0.0,
    order: int = 1,
    u0: ArrayLike | None = None,
    input_cov_sqr: ArrayLike | None = None,
    state_cov_sqr: ArrayLike | None = None,
) -> tuple[Array, Array]:
    """A consistent initial law for an ODE driven by a hidden input.

    The input block has mean ``[u0, 0, ..., 0]`` and covariance
    ``S_u = input_cov_sqr.T @ input_cov_sqr``, by default the input prior's
    stationary law. The state block's mean is the Taylor-mode initialization of
    the ODE driven by the input's truncated Taylor polynomial, whose
    coefficients are the input block: the augmented system
    ``[x, (x'), u, u', ..., u^(q_u)]`` with ``u^(j)' = u^(j+1)`` and
    ``u^(q_u)' = 0`` is expanded, and the state's derivatives
    ``[x, x', ..., x^(q_x)]`` are read off. Their Jacobian ``J`` with respect to
    the input block (forward-mode autodiff at the input mean) carries the
    input's spread into the state block::

        P_0 = [[S_x + J S_u J.T, J S_u], [S_u J.T, S_u]].

    ``J`` is exact for an input that enters the vector field additively and a
    linearization otherwise. As in ``taylor_mode_initialization``, the vector
    field is evaluated at the fixed time ``t0``.

    Args:
        vf: Vector field with the hidden-input signature, ``vf(x, u, *, t)``
            (``order=1``) or ``vf(x, dx, u, *, t)`` (``order=2``), as for
            ``ODEInformationWithHidden`` / ``SecondOrderODEInformationWithHidden``.
        inits: ``x0``, or ``(x0, dx0)`` for ``order=2``.
        prior: The ``JointPrior`` of state and input.
        t0: Initial time.
        order: Order of the ODE (1 or 2).
        u0: Mean of the input at ``t0`` (default zero, the stationary mean).
        input_cov_sqr: Square root of the input block's covariance, shape
            ``[D_u, D_u]``; defaults to the input prior's
            ``stationary_cov_sqr()``.
        state_cov_sqr: Square root of the state block's own covariance, shape
            ``[D_x, D_x]`` (default zero: the Taylor Dirac).

    Returns:
        ``(m_0, P_0_sqr)`` in the ``JointPrior`` state order (the state block,
        then the input block). ``P_0_sqr`` is the square factor
        ``[[R_x, 0], [R_u J.T, R_u]]`` of ``P_0``, not a triangular one: every
        consumer uses only ``P_0_sqr.T @ P_0_sqr``, and a QR through the zero
        rows of the Dirac state block would have no derivative.

    Raises:
        TypeError: If ``prior`` is not a ``JointPrior``.
        ValueError: If ``order`` is not 1 or 2, ``inits`` does not match it, or
            no ``input_cov_sqr`` is given for an input prior without a
            stationary law.
    """
    if not isinstance(prior, JointPrior):
        raise TypeError(f"prior must be a JointPrior, got {type(prior).__name__}.")
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")
    q_x, d_x = prior.prior_x.q, prior.prior_x.d
    q_u, d_u = prior.prior_u.q, prior.prior_u.d
    D_x, D_u = prior.prior_x.state_dim, prior.prior_u.state_dim

    if order == 1:
        initial = [np.asarray(inits, dtype=float).reshape(-1)]
    else:
        if not isinstance(inits, (tuple, list)) or len(inits) != 2:
            raise ValueError("For order=2, inits must be a tuple (x0, dx0).")
        initial = [np.asarray(v, dtype=float).reshape(-1) for v in inits]
    n_state = order * d_x

    if input_cov_sqr is None:
        stationary = getattr(prior.prior_u, "stationary_cov_sqr", None)
        if stationary is None:
            raise ValueError(
                f"The input prior {type(prior.prior_u).__name__} has no stationary "
                "law: pass input_cov_sqr."
            )
        input_cov_sqr = stationary()
    R_u = np.asarray(input_cov_sqr)
    R_x = np.zeros((D_x, D_x)) if state_cov_sqr is None else np.asarray(state_cov_sqr)
    u_mean = np.zeros(D_u)
    if u0 is not None:
        u_mean = u_mean.at[:d_u].set(np.asarray(u0, dtype=float).reshape(-1))

    def state_derivatives(u_block: Array) -> Array:
        def augmented(y, *, t):
            u_derivs = [
                y[n_state + j * d_u : n_state + (j + 1) * d_u] for j in range(q_u + 1)
            ]
            if order == 2:
                x, v = y[:d_x], y[d_x:n_state]
                top = np.concatenate([v, vf(x, v, u_derivs[0], t=t)])
            else:
                top = vf(y[:d_x], u_derivs[0], t=t)
            return np.concatenate([top, *u_derivs[1:], np.zeros(d_u)])

        y0 = np.concatenate([*initial, u_block])
        m, _ = taylor_mode_initialization(augmented, y0, q=q_x, t0=t0)
        return m.reshape(q_x + 1, -1)[:, :d_x].reshape(-1)

    m_x = state_derivatives(u_mean)
    J = jax.jacfwd(state_derivatives)(u_mean)
    P_0_sqr = np.block([[R_x, np.zeros((D_x, D_u))], [R_u @ J.T, R_u]])
    return np.concatenate([m_x, u_mean]), P_0_sqr
