from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array


class ODEInformation:
    """Evaluation and differential information for ODE measurement models."""

    def __init__(self, vf: Callable[[Array, float], Array], d: int = 1, q: int = 1):
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
        self._R = jnp.zeros((d, d))
        self._state_dim = (q + 1) * d
        self._jacobian_vf = jax.jacfwd(self._vf)

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: One-dimensional array of length ``(q + 1) * d`` containing the
                stacked state derivatives.

        Returns:
            A length-``d`` array with the measurement model evaluation.
        """

        state_arr = self._validate_state(state)
        projected = self._E0 @ state_arr
        return self._E1 @ state_arr - self._vf(projected, t=t)

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``.

        Args:
            state: One-dimensional array of length ``(q + 1) * d`` containing the
                stacked state derivatives.

        Returns:
            A ``(d, (q + 1) * d)`` array containing the Jacobian.
        """

        state_arr = self._validate_state(state)
        return self._E1 - self._jacobian_vf(self._E0 @ state_arr, t=t) @ self._E0

    def linearize(self, state: Array, *, t: float) -> [Array, Array]:
        """First order taylor linearization"""
        H_t = self.jacobian_g(state, t=t)
        c_t = self.g(state, t=t) - H_t @ state
        return H_t, c_t

    def get_noise(self, *, t: float) -> Array:
        return self._R

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


class ODEconservation(ODEInformation):
    """Evaluation and differential information for ODE measurement models."""

    def __init__(
        self,
        vf: Callable[[Array, float], Array],
        A,
        p,
        d: int = 1,
        q: int = 1,
    ):
        """Initialize with ODE information plus linear measurements.

        Extends ODEInformation with additional linear measurements of the form
        A @ x(t) = p.

        Args:
            vf, d, q: See ODEInformation for documentation of these parameters.
            A: Measurement matrix for linear observations.
            z: Measurement values.
            z_t: Time points corresponding to measurements.
        """
        super().__init__(vf, d, q)
        if A.shape[0] != p.shape[0]:
            raise ValueError(
                f"A.shape[0] ({A.shape[0]}) must match p.shape[0] ({p.shape[0]})."
            )
        self._A = A
        self._p = p
        self._k = p.shape[0]
        self._R = jnp.zeros((d + self._k, d + self._k))

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: One-dimensional array of length ``(q + 1) * d`` containing the
                stacked state derivatives.

        """
        # Get ODE information from parent class
        ode_info = super().g(state, t=t)
        return jnp.concatenate([ode_info, self._A @ self._E0 @ state - self._p])

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``."""

        ode_jacobi = super().jacobian_g(state, t=t)
        return jnp.concatenate([ode_jacobi, self._A @ self._E0])


class ODEmeasurement(ODEInformation):
    """Evaluation and differential information for ODE measurement models."""

    def __init__(
        self,
        vf: Callable[[Array, float], Array],
        A,
        z,
        z_t,
        d: int = 1,
        q: int = 1,
    ):
        """Initialize with ODE information plus linear measurements.

        Extends ODEInformation with additional linear measurements of the form
        A @ x(t) = z(t).

        Args:
            vf, d, q: See ODEInformation for documentation of these parameters.
            A: Measurement matrix for linear observations.
            z: Measurement values.
            z_t: Time points corresponding to measurements.
        """
        super().__init__(vf, d, q)
        self._A = A
        self._z = z
        self._z_t = z_t
        self._k = A.shape[0]
        self._R_measure = jnp.zeros(
            (d + self._k, d + self._k)
        )  # TODO: this should be modifiyable especially for noisy measurements

        # Check that A is 2D and has d columns
        if self._A.ndim != 2 or self._A.shape[1] != d:
            raise ValueError(
                f"'A' must be 2D with shape (k, {d}), got {self._A.shape}."
            )

        # Check that z is 2D and matches A's row dimension
        if self._z.ndim != 2 or self._z.shape[1] != self._A.shape[0]:
            raise ValueError(
                f"'z' must be 2D with shape (n, {self._A.shape[0]}), got {self._z.shape}."
            )

        # Check z_t shape and flatten if needed
        if self._z_t.ndim == 1:
            if self._z_t.shape[0] != self._z.shape[0]:
                raise ValueError(
                    f"'z_t' length must match number of measurements {self._z.shape[0]}, "
                    f"got {self._z_t.shape[0]}."
                )
        elif self._z_t.ndim == 2 and self._z_t.shape[1] == 1:
            if self._z_t.shape[0] != self._z.shape[0]:
                raise ValueError(
                    f"'z_t' length must match number of measurements {self._z.shape[0]}, "
                    f"got {self._z_t.shape[0]}."
                )
            self._z_t = self._z_t.flatten()
        else:
            raise ValueError(
                f"'z_t' must be 1D shape (n,) or 2D shape (n, 1), got {self._z_t.shape}."
            )

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: One-dimensional array of length ``(q + 1) * d`` containing the
                stacked state derivatives.

        """
        # Get ODE information from parent class
        ode_info = super().g(state, t=t)

        # Add measurement information if at a measurement time
        if t in self._z_t:
            idx = jnp.where(self._z_t == t)[0][0]
            measure = self._z[idx]
            return jnp.concatenate([ode_info, self._A @ self._E0 @ state - measure])

        return ode_info

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``."""

        # state_arr = self._validate_state(state)
        # return self._E1 - self._jacobian_vf(self._E0 @ state_arr, t=t) @ self._E0
        ode_jacobi = super().jacobian_g(state, t=t)

        if t in self._z_t:
            return jnp.concatenate([ode_jacobi, self._A @ self._E0])

        return ode_jacobi

    def get_noise(self, *, t: float) -> Array:
        if t in self._z_t:
            return self._R_measure
        return self._R


class ODEconservationmeasurement(ODEconservation):
    """Evaluation and differential information for ODE measurement models."""

    def __init__(
        self,
        vf: Callable[[Array, float], Array],
        A: Array,
        z: Array,
        z_t: Array,
        C: Array,
        p: Array,
        d: int = 1,
        q: int = 1,
    ):
        """Initialize with ODE information plus linear measurements.

        Extends ODEInformation with additional linear measurements of the form
        A @ x(t) = z(t).

        Args:
            vf, d, q: See ODEInformation for documentation of these parameters.
            A: Measurement matrix for linear observations.
            z: Measurement values.
            z_t: Time points corresponding to measurements.
        """
        super().__init__(vf, C, p, d, q)
        self._A = A
        self._z = z
        self._z_t = z_t
        self._k = A.shape[0]
        self._R_measure = jnp.zeros(
            (d + self._k, d + self._k)
        )  # TODO: this should be modifiyable especially for noisy measurements

        # Check that A is 2D and has d columns
        if self._A.ndim != 2 or self._A.shape[1] != d:
            raise ValueError(
                f"'A' must be 2D with shape (k, {d}), got {self._A.shape}."
            )

        # Check that z is 2D and matches A's row dimension
        if self._z.ndim != 2 or self._z.shape[1] != self._A.shape[0]:
            raise ValueError(
                f"'z' must be 2D with shape (n, {self._A.shape[0]}), got {self._z.shape}."
            )

        # Check z_t shape and flatten if needed
        if self._z_t.ndim == 1:
            if self._z_t.shape[0] != self._z.shape[0]:
                raise ValueError(
                    f"'z_t' length must match number of measurements {self._z.shape[0]}, "
                    f"got {self._z_t.shape[0]}."
                )
        elif self._z_t.ndim == 2 and self._z_t.shape[1] == 1:
            if self._z_t.shape[0] != self._z.shape[0]:
                raise ValueError(
                    f"'z_t' length must match number of measurements {self._z.shape[0]}, "
                    f"got {self._z_t.shape[0]}."
                )
            self._z_t = self._z_t.flatten()
        else:
            raise ValueError(
                f"'z_t' must be 1D shape (n,) or 2D shape (n, 1), got {self._z_t.shape}."
            )

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: One-dimensional array of length ``(q + 1) * d`` containing the
                stacked state derivatives.

        """
        # Get ODE information from parent class
        ode_info = super().g(state, t=t)

        # Add measurement information if at a measurement time
        if t in self._z_t:
            idx = jnp.where(self._z_t == t)[0][0]
            measure = self._z[idx]
            return jnp.concatenate([ode_info, self._A @ self._E0 @ state - measure])

        return ode_info

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``."""

        # state_arr = self._validate_state(state)
        # return self._E1 - self._jacobian_vf(self._E0 @ state_arr, t=t) @ self._E0
        ode_jacobi = super().jacobian_g(state, t=t)

        if t in self._z_t:
            return jnp.concatenate([ode_jacobi, self._A @ self._E0])

        return ode_jacobi

    def get_noise(self, *, t: float) -> Array:
        if t in self._z_t:
            return self._R_measure
        return self._R
