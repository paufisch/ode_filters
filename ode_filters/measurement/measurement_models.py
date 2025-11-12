from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class BaseODEInformation(ABC):
    """Abstract base class for ODE measurement models."""

    def __init__(self, vf: Callable[[Array, float], Array], d: int = 1, q: int = 1):
        if d <= 0:
            raise ValueError("'d' must be positive.")
        if q < 1:
            raise ValueError("'q' must be at least one.")

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

    @abstractmethod
    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector."""

    @abstractmethod
    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``."""

    @abstractmethod
    def get_noise(self, *, t: float) -> Array:
        """Return the measurement noise matrix at time ``t``."""

    def linearize(self, state: Array, *, t: float) -> tuple[Array, Array]:
        state_arr = self._validate_state(state)
        H_t = self.jacobian_g(state_arr, t=t)
        c_t = self.g(state_arr, t=t) - H_t @ state_arr
        return H_t, c_t

    def _validate_state(self, state: Array) -> Array:
        state_arr = jnp.asarray(state, dtype=jnp.float32)
        if state_arr.ndim != 1:
            raise ValueError("'state' must be a one-dimensional array.")
        if state_arr.shape[0] != self._state_dim:
            raise ValueError(
                f"'state' must have length {self._state_dim}, got {state_arr.shape[0]}."
            )
        return state_arr


class ODEInformation(BaseODEInformation):
    """Baseline ODE measurement model without additional constraints."""

    def g(self, state: Array, *, t: float) -> Array:
        state_arr = self._validate_state(state)
        projected = self._E0 @ state_arr
        return self._E1 @ state_arr - self._vf(projected, t=t)

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        state_arr = self._validate_state(state)
        return self._E1 - self._jacobian_vf(self._E0 @ state_arr, t=t) @ self._E0

    def get_noise(self, *, t: float) -> Array:
        return self._R


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
        """Initialize with ODE information plus linear conservation law.

        Extends ODEInformation with additional linear conservation law of the form
        A @ x(t) = p.

        Args:
            vf, d, q: See ODEInformation for documentation of these parameters.
            A: Measurement matrix for linear observations.
            z: Measurement values.
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


class LinearMeasurementBase:
    """Shared utilities for models with additional linear measurements."""

    def _setup_linear_measurements(self, A: Array, z: Array, z_t: Array) -> None:
        A_arr = jnp.asarray(A)
        if A_arr.ndim != 2 or A_arr.shape[1] != self._d:
            raise ValueError(
                f"'A' must be 2D with shape (k, {self._d}), got {A_arr.shape}."
            )

        z_arr = jnp.asarray(z)
        if z_arr.ndim != 2 or z_arr.shape[1] != A_arr.shape[0]:
            raise ValueError(
                f"'z' must be 2D with shape (n, {A_arr.shape[0]}), got {z_arr.shape}."
            )

        z_t_arr = np.asarray(z_t)
        if z_t_arr.ndim == 1:
            pass
        elif z_t_arr.ndim == 2 and z_t_arr.shape[1] == 1:
            z_t_arr = z_t_arr.reshape(-1)
        else:
            raise ValueError(
                f"'z_t' must be 1D shape (n,) or 2D shape (n, 1), got {z_t_arr.shape}."
            )

        if z_t_arr.shape[0] != z_arr.shape[0]:
            raise ValueError(
                f"'z_t' length must match number of measurements {z_arr.shape[0]}, "
                f"got {z_t_arr.shape[0]}."
            )

        self._A_meas = A_arr
        self._z_meas = z_arr
        self._z_t_meas = z_t_arr
        self._measurement_dim = int(A_arr.shape[0])
        base_dim = int(self._R.shape[0])
        self._R_measure = jnp.zeros(
            (base_dim + self._measurement_dim, base_dim + self._measurement_dim),
            dtype=self._R.dtype,
        )

    def _measurement_index(self, t: float) -> int | None:
        matches = np.where(self._z_t_meas == t)[0]
        if matches.size == 0:
            return None
        return int(matches[0])

    def _measurement_residual(self, state: Array, idx: int) -> Array:
        return self._A_meas @ self._E0 @ state - self._z_meas[idx]

    def _measurement_jacobian(self) -> Array:
        return self._A_meas @ self._E0

    def _measurement_noise(self) -> Array:
        return self._R_measure


class ODEmeasurement(LinearMeasurementBase, ODEInformation):
    """Evaluation and differential information for ODE measurement models."""

    def __init__(
        self,
        vf: Callable[[Array, float], Array],
        A: Array,
        z: Array,
        z_t: Array,
        d: int = 1,
        q: int = 1,
    ):
        super().__init__(vf, d, q)
        self._setup_linear_measurements(A, z, z_t)

    def g(self, state: Array, *, t: float) -> Array:
        ode_info = super().g(state, t=t)
        idx = self._measurement_index(t)
        if idx is None:
            return ode_info
        residual = self._measurement_residual(state, idx)
        return jnp.concatenate([ode_info, residual])

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        ode_jacobi = super().jacobian_g(state, t=t)
        idx = self._measurement_index(t)
        if idx is None:
            return ode_jacobi
        return jnp.concatenate([ode_jacobi, self._measurement_jacobian()])

    def get_noise(self, *, t: float) -> Array:
        idx = self._measurement_index(t)
        if idx is None:
            return super().get_noise(t=t)
        return self._measurement_noise()


class ODEconservationmeasurement(LinearMeasurementBase, ODEconservation):
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
        super().__init__(vf, C, p, d, q)
        self._setup_linear_measurements(A, z, z_t)
        self._A_lin = self._A_meas
        self._z = self._z_meas
        self._z_t = self._z_t_meas
        self._k_lin = self._measurement_dim

    def g(self, state: Array, *, t: float) -> Array:
        ode_info = super().g(state, t=t)
        idx = self._measurement_index(t)
        if idx is None:
            return ode_info
        residual = self._measurement_residual(state, idx)
        return jnp.concatenate([ode_info, residual])

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        ode_jacobi = super().jacobian_g(state, t=t)
        idx = self._measurement_index(t)
        if idx is None:
            return ode_jacobi
        return jnp.concatenate([ode_jacobi, self._measurement_jacobian()])

    def get_noise(self, *, t: float) -> Array:
        idx = self._measurement_index(t)
        if idx is None:
            return super().get_noise(t=t)
        return self._measurement_noise()
