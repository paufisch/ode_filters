from typing import Callable

import jax
import jax.numpy as jnp
from jax import Array


class ODEInformation:
    """Evaluation and differential information for ODE measurement models."""

    def __init__(self, vf: Callable[[Array], Array], d: int = 1, q: int = 1):
        """Initialize the measurement model.

        Args:
            vf: Vector field of the underlying ODE. It must map a state of
                shape ``(d,)`` to an array of the same shape.
            d: Dimension of the state space. Must be positive.
            q: Order of the differential equation. Must be at least one.
        """

        if d <= 0:
            raise ValueError("'d' must be positive.")
        if q < 1:
            raise ValueError("'q' must be at least one.")

        # Define projection matrices
        eye_d = jnp.eye(d, dtype=jnp.float32)
        basis = jnp.eye(q + 1, dtype=jnp.float32)
        self._E0 = jnp.kron(basis[0:1], eye_d)
        self._E1 = jnp.kron(basis[1:2], eye_d)

        self._vf = vf
        self._d = d
        self._q = q
        self._state_dim = (q + 1) * d
        self._jacobian_fn = jax.jacfwd(self.g)

    def g(self, state: Array) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: One-dimensional array of length ``(q + 1) * d`` containing the
                stacked state derivatives.

        Returns:
            A length-``d`` array with the measurement model evaluation.
        """

        state_arr = self._validate_state(state)
        projected = self._E0 @ state_arr
        return self._E1 @ state_arr - self._vf(projected)

    def jacobian_g(self, state: Array) -> Array:
        """Return the Jacobian of the observation model at ``state``.

        Args:
            state: One-dimensional array of length ``(q + 1) * d`` containing the
                stacked state derivatives.

        Returns:
            A ``(d, (q + 1) * d)`` array containing the Jacobian.
        """

        state_arr = self._validate_state(state)
        return self._jacobian_fn(state_arr)

    def _validate_state(self, state: Array) -> Array:
        """Return a validated one-dimensional state array.

        Args:
            state: Candidate state array.

        Returns:
            A one-dimensional JAX array with dtype ``float32`` and length
            ``(q + 1) * d``.
        """

        state_arr = jnp.asarray(state, dtype=jnp.float32)
        if state_arr.ndim != 1:
            raise ValueError("'state' must be a one-dimensional array.")
        if state_arr.shape[0] != self._state_dim:
            raise ValueError(
                f"'state' must have length {self._state_dim}, got {state_arr.shape[0]}."
            )
        return state_arr
