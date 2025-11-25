from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike


class BaseODEInformation(ABC):
    """Abstract base class for ODE measurement models."""

    def __init__(self, vf: Callable[[Array, float], Array], d: int = 1, q: int = 1):
        if d <= 0:
            raise ValueError("'d' must be positive.")
        if q < 1:
            raise ValueError("'q' must be at least one.")

        eye_d = np.eye(d, dtype=np.float32)
        basis = np.eye(q + 1, dtype=np.float32)
        self._E0 = np.kron(basis[0:1], eye_d)
        self._E1 = np.kron(basis[1:2], eye_d)

        self._vf = vf
        self._d = d
        self._q = q
        self._R = np.zeros((d, d))
        self._state_dim = (q + 1) * d
        self._jacobian_vf = jax.jacfwd(self._vf)

    @abstractmethod
    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: State vector of length ``(q + 1) * d``.
            t: Current time.

        Returns:
            Observation model evaluation.
        """

    @abstractmethod
    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``.

        Args:
            state: State vector of length ``(q + 1) * d``.
            t: Current time.

        Returns:
            Jacobian matrix of the observation model.
        """

    @abstractmethod
    def get_noise(self, *, t: float) -> Array:
        """Return the measurement noise matrix at time ``t``.

        Args:
            t: Current time.

        Returns:
            Measurement noise covariance matrix.
        """

    def linearize(self, state: Array, *, t: float) -> tuple[Array, Array]:
        """Linearize the observation model around the given state.

        Args:
            state: State vector to linearize around.
            t: Current time.

        Returns:
            Tuple of (H_t, c_t) where:
            - H_t is the Jacobian matrix
            - c_t is the constant term (observation offset)
        """
        state_arr = self._validate_state(state)
        H_t = self.jacobian_g(state_arr, t=t)
        c_t = self.g(state_arr, t=t) - H_t @ state_arr
        return H_t, c_t

    def _validate_state(self, state: Array) -> Array:
        """Validate and convert state to required format.

        Args:
            state: State vector to validate.

        Returns:
            Validated state as float32 JAX array.

        Raises:
            ValueError: If state is not 1D or has incorrect length.
        """
        state_arr = np.asarray(state, dtype=np.float32)
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
        """Evaluate the observation model for a flattened state vector.

        Returns the difference between the first derivative and the vector field.
        """
        state_arr = self._validate_state(state)
        projected = self._E0 @ state_arr
        return self._E1 @ state_arr - self._vf(projected, t=t)

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``."""
        state_arr = self._validate_state(state)
        return self._E1 - self._jacobian_vf(self._E0 @ state_arr, t=t) @ self._E0

    def get_noise(self, *, t: float) -> Array:
        """Return the measurement noise matrix at time ``t``."""
        return self._R


class ODEconservation(ODEInformation):
    """Evaluation and differential information for ODE measurement models."""

    def __init__(
        self,
        vf: Callable[[Array, float], Array],
        A: Array,
        p: Array,
        d: int = 1,
        q: int = 1,
    ):
        """Initialize with ODE information plus linear conservation law.

        Extends ODEInformation with additional linear conservation law of the form
        A @ x(t) = p.

        Args:
            vf: Vector field function with signature f(x, t) -> dx/dt.
            A: Constraint matrix for linear conservation law (shape [k, d]).
            p: Conservation law constant values (shape [k]).
            d: State dimension (default 1).
            q: Order of derivatives (default 1).
        """
        super().__init__(vf, d, q)
        if A.shape[0] != p.shape[0]:
            raise ValueError(
                f"A.shape[0] ({A.shape[0]}) must match p.shape[0] ({p.shape[0]})."
            )
        self._A = A
        self._p = p
        self._k = p.shape[0]
        self._R = np.zeros((d + self._k, d + self._k))

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: One-dimensional array of length ``(q + 1) * d`` containing the
                stacked state derivatives.
            t: Current time.

        Returns:
            Observation model including ODE information and conservation constraint.
        """
        # Get ODE information from parent class
        ode_info = super().g(state, t=t)
        return np.concatenate([ode_info, self._A @ self._E0 @ state - self._p])

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``.

        Args:
            state: State vector of length ``(q + 1) * d``.
            t: Current time.

        Returns:
            Jacobian matrix including ODE and conservation constraint terms.
        """
        ode_jacobi = super().jacobian_g(state, t=t)
        return np.concatenate([ode_jacobi, self._A @ self._E0])


class LinearMeasurementBase:
    """Shared utilities for models with additional linear measurements.

    This is a mixin class that requires the subclass to define:
    - self._d: int - State dimension
    - self._R: Array - Base noise covariance matrix
    - self._E0: Array - State extraction matrix
    """

    def _setup_linear_measurements(
        self, A: ArrayLike, z: ArrayLike, z_t: ArrayLike
    ) -> None:
        if not hasattr(self, "_d"):
            raise ValueError("Subclass must define self._d (state dimension).")
        if not hasattr(self, "_R"):
            raise ValueError("Subclass must define self._R (noise covariance matrix).")
        if not hasattr(self, "_E0"):
            raise ValueError("Subclass must define self._E0 (state extraction matrix).")

        A_arr = np.asarray(A)
        if A_arr.ndim != 2 or A_arr.shape[1] != self._d:
            raise ValueError(
                f"'A' must be 2D with shape (k, {self._d}), got {A_arr.shape}."
            )

        z_arr = np.asarray(z)
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
        self._R_measure = np.zeros(
            (base_dim + self._measurement_dim, base_dim + self._measurement_dim),
            dtype=self._R.dtype,
        )

    def _measurement_index(self, t: float) -> int | None:
        """Find the measurement index for the given time ``t``.

        Args:
            t: Time value to search for.

        Returns:
            Index of the measurement at time t, or None if not found.
        """
        matches = np.where(self._z_t_meas == t)[0]
        if matches.size == 0:
            return None
        return int(matches[0])

    def _measurement_residual(self, state: Array, idx: int) -> Array:
        """Compute measurement residual for the given state and measurement index.

        Args:
            state: Current state vector.
            idx: Index of the measurement in the measurement arrays.

        Returns:
            Measurement residual: A @ state - z[idx].
        """
        return self._A_meas @ self._E0 @ state - self._z_meas[idx]

    def _measurement_jacobian(self) -> Array:
        """Get the Jacobian matrix for linear measurements.

        Returns:
            Measurement Jacobian matrix: A @ E0.
        """
        return self._A_meas @ self._E0

    def _measurement_noise(self) -> Array:
        """Get the measurement noise covariance including measurement term.

        Returns:
            Combined measurement noise covariance matrix.
        """
        return self._R_measure


class ODEmeasurement(LinearMeasurementBase, ODEInformation):
    """ODE information combined with optional linear measurements."""

    def __init__(
        self,
        vf: Callable[[Array, float], Array],
        A: Array,
        z: Array,
        z_t: Array,
        d: int = 1,
        q: int = 1,
    ):
        """Initialize ODE measurement model with linear measurements.

        Args:
            vf: Vector field function f(x, t) -> dx/dt.
            A: Measurement matrix (shape [k, d]).
            z: Measurement values (shape [n, k]).
            z_t: Measurement times (shape [n]).
            d: State dimension (default 1).
            q: Order of derivatives (default 1).
        """
        super().__init__(vf, d, q)
        self._setup_linear_measurements(A, z, z_t)

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate observation model with optional measurement term."""
        ode_info = super().g(state, t=t)
        idx = self._measurement_index(t)
        if idx is None:
            return ode_info
        residual = self._measurement_residual(state, idx)
        return np.concatenate([ode_info, residual])

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return Jacobian including optional measurement term."""
        ode_jacobi = super().jacobian_g(state, t=t)
        idx = self._measurement_index(t)
        if idx is None:
            return ode_jacobi
        return np.concatenate([ode_jacobi, self._measurement_jacobian()])

    def get_noise(self, *, t: float) -> Array:
        """Get measurement noise including optional measurement term."""
        idx = self._measurement_index(t)
        if idx is None:
            return super().get_noise(t=t)
        return self._measurement_noise()


class ODEconservationmeasurement(LinearMeasurementBase, ODEconservation):
    """ODE with conservation law combined with optional linear measurements."""

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
        """Initialize ODE measurement model with conservation law and linear measurements.

        Args:
            vf: Vector field function f(x, t) -> dx/dt.
            A: Measurement matrix for linear measurements (shape [k, d]).
            z: Measurement values (shape [n, k]).
            z_t: Measurement times (shape [n]).
            C: Constraint matrix for conservation law (shape [m, d]).
            p: Conservation law values (shape [m]).
            d: State dimension (default 1).
            q: Order of derivatives (default 1).
        """
        super().__init__(vf, C, p, d, q)
        self._setup_linear_measurements(A, z, z_t)
        self._A_lin = self._A_meas
        self._z = self._z_meas
        self._z_t = self._z_t_meas
        self._k_lin = self._measurement_dim

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate observation model with conservation law and optional measurements."""
        ode_info = super().g(state, t=t)
        idx = self._measurement_index(t)
        if idx is None:
            return ode_info
        residual = self._measurement_residual(state, idx)
        return np.concatenate([ode_info, residual])

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return Jacobian with conservation law and optional measurements."""
        ode_jacobi = super().jacobian_g(state, t=t)
        idx = self._measurement_index(t)
        if idx is None:
            return ode_jacobi
        return np.concatenate([ode_jacobi, self._measurement_jacobian()])

    def get_noise(self, *, t: float) -> Array:
        """Get measurement noise with conservation law and optional measurements."""
        idx = self._measurement_index(t)
        if idx is None:
            return super().get_noise(t=t)
        return self._measurement_noise()
