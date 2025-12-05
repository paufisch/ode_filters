from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

# Default measurement noise for observations
DEFAULT_MEASUREMENT_NOISE = 1e-6


class BaseODEInformation(ABC):
    """Abstract base class for ODE measurement models."""

    def __init__(
        self,
        vf: Callable[[Array, float], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E2: ArrayLike | None = None,
        order: int = 1,
    ):
        """Initialize the ODE measurement model.

        Args:
            vf: Vector field function.
                For order=1: vf(x, *, t) -> dx/dt
                For order=2: vf(x, dx, *, t) -> d²x/dt²
            E0: State extraction matrix (shape [d, (q+1)*d]).
            E1: First derivative extraction matrix (shape [d, (q+1)*d]).
            E2: Second derivative extraction matrix (shape [d, (q+1)*d]).
                Required for order=2, ignored for order=1.
            order: ODE order (1 or 2, default 1).
        """
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2.")
        if order == 2 and E2 is None:
            raise ValueError("E2 is required for order=2.")

        self._vf = vf
        self._order = order
        self._E0 = np.asarray(E0)
        self._E1 = np.asarray(E1)
        self._E2 = np.asarray(E2) if E2 is not None else None
        self._d = self._E0.shape[0]
        self._state_dim = self._E0.shape[1]

        # The constraint matrix is E1 for order 1, E2 for order 2
        self._E_constraint = self._E1 if order == 1 else self._E2
        self._R = np.zeros((self._E_constraint.shape[0], self._E_constraint.shape[0]))

        # Precompile Jacobian functions
        if order == 1:
            self._jacobian_vf = jax.jacfwd(self._vf)
        else:
            self._jacobian_vf_x = jax.jacfwd(self._vf, argnums=0)
            self._jacobian_vf_v = jax.jacfwd(self._vf, argnums=1)

    @property
    def R(self) -> Array:
        """Measurement noise covariance matrix."""
        return self._R

    @R.setter
    def R(self, value: ArrayLike) -> None:
        """Set the measurement noise covariance matrix.

        Args:
            value: Noise specification. Can be:
                - Scalar: Applied to all diagonal entries
                - 1D array: Used as diagonal values
                - 2D array: Used as full covariance matrix

        Raises:
            ValueError: If value has incorrect shape.
        """
        R_arr = np.asarray(value)
        expected_dim = self._R.shape[0]

        if R_arr.ndim == 0:
            # Scalar - apply to all diagonal entries
            self._R = float(R_arr) * np.eye(expected_dim)
        elif R_arr.ndim == 1:
            # Vector - set as diagonal
            if R_arr.shape[0] != expected_dim:
                raise ValueError(
                    f"Diagonal values must have length {expected_dim}, got {R_arr.shape[0]}."
                )
            self._R = np.diag(R_arr)
        elif R_arr.ndim == 2:
            # Full matrix
            expected_shape = (expected_dim, expected_dim)
            if R_arr.shape != expected_shape:
                raise ValueError(
                    f"R must have shape {expected_shape}, got {R_arr.shape}."
                )
            self._R = R_arr
        else:
            raise ValueError("R must be scalar, 1D, or 2D array.")

    @abstractmethod
    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: State vector of length matching ``E0.shape[1]``.
            t: Current time.

        Returns:
            Observation model evaluation.
        """

    @abstractmethod
    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``.

        Args:
            state: State vector of length matching ``E0.shape[1]``.
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

    # this is the EFK1 linearization
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

        For order=1: Returns E1 @ state - vf(E0 @ state)
        For order=2: Returns E2 @ state - vf(E0 @ state, E1 @ state)
        """
        state_arr = self._validate_state(state)
        x = self._E0 @ state_arr

        if self._order == 1:
            vf_eval = self._vf(x, t=t)
        else:
            v = self._E1 @ state_arr
            vf_eval = self._vf(x, v, t=t)

        return self._E_constraint @ state_arr - vf_eval

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``."""
        state_arr = self._validate_state(state)
        x = self._E0 @ state_arr

        if self._order == 1:
            jac_vf = self._jacobian_vf(x, t=t)
            return self._E_constraint - jac_vf @ self._E0
        else:
            v = self._E1 @ state_arr
            jac_x = self._jacobian_vf_x(x, v, t=t)
            jac_v = self._jacobian_vf_v(x, v, t=t)
            return self._E_constraint - jac_x @ self._E0 - jac_v @ self._E1

    def get_noise(self, *, t: float) -> Array:
        """Return the measurement noise matrix at time ``t``."""
        return self._R


class ODEconservation(ODEInformation):
    """Evaluation and differential information for ODE measurement models."""

    def __init__(
        self,
        vf: Callable[[Array, float], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        A: Array,
        p: Array,
        E2: ArrayLike | None = None,
        order: int = 1,
    ):
        """Initialize with ODE information plus linear conservation law.

        Extends ODEInformation with additional linear conservation law of the form
        A @ x(t) = p. Where x(t) = E0 @ X(t) is the projected state.

        Args:
            vf: Vector field function.
                For order=1: f(x, t) -> dx/dt
                For order=2: f(x, dx, t) -> d²x/dt²
            E0: State extraction matrix (shape [d, (q+1)*d]).
            E1: Derivative extraction matrix (shape [d, (q+1)*d]).
            A: Constraint matrix for linear conservation law (shape [k, d]).
            p: Conservation law constant values (shape [k]).
            E2: Second derivative extraction matrix (required for order=2).
            order: ODE order (1 or 2, default 1).
        """
        super().__init__(vf, E0, E1, E2=E2, order=order)
        if A.shape[0] != p.shape[0]:
            raise ValueError(
                f"A.shape[0] ({A.shape[0]}) must match p.shape[0] ({p.shape[0]})."
            )
        if A.shape[1] != self._E0.shape[0]:
            raise ValueError(
                f"A.shape[1] ({A.shape[1]}) must match E0.shape[0] ({self._E0.shape[0]})."
            )

        self._A = A
        self._p = p
        self._k = p.shape[0]  # shape of conservation information measurement
        self._m = self._d + self._k  # shape of total measurement
        self._R = np.zeros((self._m, self._m))

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: One-dimensional array of length matching ``E0.shape[1]``
                containing the stacked state derivatives.
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
            state: State vector of length matching ``E0.shape[1]``.
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
        self,
        A: ArrayLike,
        z: ArrayLike,
        z_t: ArrayLike,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
    ) -> None:
        """Set up linear measurements with optional measurement noise.

        Args:
            A: Measurement matrix (shape [k, d]).
            z: Measurement values (shape [n, k]).
            z_t: Measurement times (shape [n]).
            measurement_noise: Default noise variance for measurements (default: 1e-6).
        """
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
        total_dim = base_dim + self._measurement_dim

        # Create combined noise matrix with default measurement noise
        self._R_measure = np.zeros((total_dim, total_dim), dtype=self._R.dtype)
        # Copy base noise (ODE/conservation part)
        self._R_measure = self._R_measure.at[:base_dim, :base_dim].set(self._R)
        # Set default measurement noise on diagonal for measurement part
        for i in range(self._measurement_dim):
            self._R_measure = self._R_measure.at[base_dim + i, base_dim + i].set(
                measurement_noise
            )

    @property
    def R_measure(self) -> Array:
        """Combined noise covariance matrix (ODE + measurement)."""
        return self._R_measure

    @R_measure.setter
    def R_measure(self, value: ArrayLike) -> None:
        """Set the combined measurement noise covariance matrix.

        Args:
            value: New noise covariance matrix. Must match the expected shape.

        Raises:
            ValueError: If value has incorrect shape.
        """
        R_arr = np.asarray(value)
        expected_shape = self._R_measure.shape
        if R_arr.shape != expected_shape:
            raise ValueError(f"R must have shape {expected_shape}, got {R_arr.shape}.")
        self._R_measure = R_arr

    def set_measurement_noise(self, noise: float | ArrayLike) -> None:
        """Set the measurement noise for the observation part only.

        Args:
            noise: Either a scalar (applied to all measurement dimensions)
                   or a vector of length measurement_dim for diagonal noise,
                   or a full (measurement_dim, measurement_dim) matrix.
        """
        base_dim = int(self._R.shape[0])
        noise_arr = np.asarray(noise)

        if noise_arr.ndim == 0:
            # Scalar - apply to all diagonal entries
            for i in range(self._measurement_dim):
                self._R_measure = self._R_measure.at[base_dim + i, base_dim + i].set(
                    float(noise_arr)
                )
        elif noise_arr.ndim == 1:
            # Vector - set diagonal
            if noise_arr.shape[0] != self._measurement_dim:
                raise ValueError(
                    f"noise vector must have length {self._measurement_dim}, "
                    f"got {noise_arr.shape[0]}."
                )
            for i in range(self._measurement_dim):
                self._R_measure = self._R_measure.at[base_dim + i, base_dim + i].set(
                    noise_arr[i]
                )
        elif noise_arr.ndim == 2:
            # Full matrix
            if noise_arr.shape != (self._measurement_dim, self._measurement_dim):
                raise ValueError(
                    f"noise matrix must have shape "
                    f"({self._measurement_dim}, {self._measurement_dim}), "
                    f"got {noise_arr.shape}."
                )
            self._R_measure = self._R_measure.at[base_dim:, base_dim:].set(noise_arr)
        else:
            raise ValueError("noise must be scalar, 1D, or 2D array.")

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
        E0: ArrayLike,
        E1: ArrayLike,
        A: Array,
        z: Array,
        z_t: Array,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
        E2: ArrayLike | None = None,
        order: int = 1,
    ):
        """Initialize ODE measurement model with linear measurements.

        Args:
            vf: Vector field function.
                For order=1: f(x, t) -> dx/dt
                For order=2: f(x, dx, t) -> d²x/dt²
            E0: State extraction matrix (shape [d, (q+1)*d]).
            E1: Derivative extraction matrix (shape [d, (q+1)*d]).
            A: Measurement matrix (shape [k, d]).
            z: Measurement values (shape [n, k]).
            z_t: Measurement times (shape [n]).
            measurement_noise: Default noise variance for measurements (default: 1e-6).
            E2: Second derivative extraction matrix (required for order=2).
            order: ODE order (1 or 2, default 1).
        """
        super().__init__(vf, E0, E1, E2=E2, order=order)
        self._setup_linear_measurements(A, z, z_t, measurement_noise)

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
        E0: ArrayLike,
        E1: ArrayLike,
        C: Array,
        p: Array,
        A: Array,
        z: Array,
        z_t: Array,
        measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
        E2: ArrayLike | None = None,
        order: int = 1,
    ):
        """Initialize ODE measurement model with conservation law and linear measurements.

        Args:
            vf: Vector field function.
                For order=1: f(x, t) -> dx/dt
                For order=2: f(x, dx, t) -> d²x/dt²
            E0: State extraction matrix (shape [d, (q+1)*d]).
            E1: Derivative extraction matrix (shape [d, (q+1)*d]).
            C: Constraint matrix for conservation law (shape [m, d]).
            p: Conservation law values (shape [m]).
            A: Measurement matrix for linear measurements (shape [k, d]).
            z: Measurement values (shape [n, k]).
            z_t: Measurement times (shape [n]).
            measurement_noise: Default noise variance for measurements (default: 1e-6).
            E2: Second derivative extraction matrix (required for order=2).
            order: ODE order (1 or 2, default 1).
        """
        super().__init__(vf, E0, E1, C, p, E2=E2, order=order)
        self._setup_linear_measurements(A, z, z_t, measurement_noise)
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
