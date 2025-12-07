from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

# Default measurement noise for observations
DEFAULT_MEASUREMENT_NOISE = 1e-6


@dataclass(frozen=True)
class Conservation:
    """Conservation constraint: A @ x = p (always active).

    Args:
        A: Constraint matrix (shape [k, d]).
        p: Target values (shape [k]).
    """

    A: Array
    p: Array

    def __post_init__(self) -> None:
        if self.A.shape[0] != self.p.shape[0]:
            raise ValueError(
                f"A.shape[0] ({self.A.shape[0]}) must match p.shape[0] ({self.p.shape[0]})."
            )

    def residual(self, x: Array) -> Array:
        """Compute residual: A @ x - p."""
        return self.A @ x - self.p

    def jacobian(self) -> Array:
        """Return Jacobian (constant): A."""
        return self.A

    @property
    def dim(self) -> int:
        """Dimension of constraint output."""
        return int(self.p.shape[0])


@dataclass(frozen=True)
class Measurement:
    """Time-varying linear measurement: A @ x = z[t] (active only at specified times).

    Args:
        A: Measurement matrix (shape [k, d]).
        z: Measurement values (shape [n, k]).
        z_t: Measurement times (shape [n]).
        noise: Measurement noise variance (scalar or [k] or [k, k]).
    """

    A: Array
    z: Array
    z_t: Array
    noise: float | Array = DEFAULT_MEASUREMENT_NOISE

    def __post_init__(self) -> None:
        if self.A.ndim != 2:
            raise ValueError(f"'A' must be 2D, got {self.A.ndim}D.")
        if self.z.ndim != 2 or self.z.shape[1] != self.A.shape[0]:
            raise ValueError(
                f"'z' must be 2D with shape (n, {self.A.shape[0]}), got {self.z.shape}."
            )
        z_t = self.z_t
        if z_t.ndim == 2 and z_t.shape[1] == 1:
            # Allow (n, 1) shape, will be reshaped on access
            pass
        elif z_t.ndim != 1:
            raise ValueError(
                f"'z_t' must be 1D shape (n,) or 2D shape (n, 1), got {z_t.shape}."
            )
        if self._get_times().shape[0] != self.z.shape[0]:
            raise ValueError(
                f"'z_t' length must match number of measurements {self.z.shape[0]}, "
                f"got {self._get_times().shape[0]}."
            )

    def _get_times(self) -> Array:
        """Get times array, handling (n, 1) shape."""
        if self.z_t.ndim == 2:
            return self.z_t.reshape(-1)
        return self.z_t

    def find_index(self, t: float) -> int | None:
        """Find measurement index for time t, or None if not found."""
        matches = np.where(self._get_times() == t)[0]
        if matches.size == 0:
            return None
        return int(matches[0])

    def residual(self, x: Array, t: float) -> Array | None:
        """Compute residual: A @ x - z[t], or None if no measurement at t."""
        idx = self.find_index(t)
        if idx is None:
            return None
        return self.A @ x - self.z[idx]

    def jacobian(self, t: float) -> Array | None:
        """Return Jacobian (constant): A, or None if no measurement at t."""
        if self.find_index(t) is None:
            return None
        return self.A

    @property
    def dim(self) -> int:
        """Dimension of measurement output."""
        return int(self.A.shape[0])

    def get_noise_matrix(self) -> Array:
        """Get noise covariance matrix."""
        noise = np.asarray(self.noise)
        if noise.ndim == 0:
            return float(noise) * np.eye(self.dim)
        elif noise.ndim == 1:
            return np.diag(noise)
        return noise


class BaseODEInformation(ABC):
    """Abstract base class for ODE measurement models."""

    _vf: Callable
    _E0: Array
    _E1: Array
    _E_constraint: Array
    _d: int
    _state_dim: int
    _base_R: Array
    _constraints: list[Conservation | Measurement]

    def _init_constraints(
        self,
        constraints: list[Conservation | Measurement] | None,
    ) -> None:
        """Initialize constraint list and validate."""
        self._constraints = constraints or []
        for c in self._constraints:
            if isinstance(c, Conservation) and c.A.shape[1] != self._d:
                raise ValueError(
                    f"Conservation A.shape[1] ({c.A.shape[1]}) must match "
                    f"state dimension ({self._d})."
                )
            if isinstance(c, Measurement) and c.A.shape[1] != self._d:
                raise ValueError(
                    f"Measurement A.shape[1] ({c.A.shape[1]}) must match "
                    f"state dimension ({self._d})."
                )

    def _get_active_constraints(
        self, t: float
    ) -> tuple[list[Conservation], list[Measurement]]:
        """Get constraints active at time t."""
        conservations = [c for c in self._constraints if isinstance(c, Conservation)]
        measurements = [
            c
            for c in self._constraints
            if isinstance(c, Measurement) and c.find_index(t) is not None
        ]
        return conservations, measurements

    def _build_noise_matrix(self, t: float) -> Array:
        """Build combined noise matrix for all active constraints at time t."""
        conservations, measurements = self._get_active_constraints(t)
        cons_dim = sum(c.dim for c in conservations)
        meas_dim = sum(m.dim for m in measurements)
        total_dim = self._d + cons_dim + meas_dim

        R = np.zeros((total_dim, total_dim))
        # Base ODE noise (usually zero)
        R = R.at[: self._d, : self._d].set(self._base_R)
        # Conservation noise (zero)
        # Measurement noise
        offset = self._d + cons_dim
        for m in measurements:
            R = R.at[offset : offset + m.dim, offset : offset + m.dim].set(
                m.get_noise_matrix()
            )
            offset += m.dim
        return R

    @property
    def R(self) -> Array:
        """Base measurement noise covariance matrix (ODE part only)."""
        return self._base_R

    @R.setter
    def R(self, value: ArrayLike) -> None:
        """Set the base measurement noise covariance matrix.

        Args:
            value: Noise specification. Can be:
                - Scalar: Applied to all diagonal entries
                - 1D array: Used as diagonal values
                - 2D array: Used as full covariance matrix
        """
        R_arr = np.asarray(value)
        expected_dim = self._base_R.shape[0]

        if R_arr.ndim == 0:
            self._base_R = float(R_arr) * np.eye(expected_dim)
        elif R_arr.ndim == 1:
            if R_arr.shape[0] != expected_dim:
                raise ValueError(
                    f"Diagonal values must have length {expected_dim}, "
                    f"got {R_arr.shape[0]}."
                )
            self._base_R = np.diag(R_arr)
        elif R_arr.ndim == 2:
            expected_shape = (expected_dim, expected_dim)
            if R_arr.shape != expected_shape:
                raise ValueError(
                    f"R must have shape {expected_shape}, got {R_arr.shape}."
                )
            self._base_R = R_arr
        else:
            raise ValueError("R must be scalar, 1D, or 2D array.")

    @abstractmethod
    def _ode_residual(self, state: Array, *, t: float) -> Array:
        """Compute the ODE residual."""

    @abstractmethod
    def _ode_jacobian(self, state: Array, *, t: float) -> Array:
        """Compute the ODE Jacobian."""

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the observation model for a flattened state vector.

        Args:
            state: State vector of length matching ``E0.shape[1]``.
            t: Current time.

        Returns:
            Observation model evaluation (ODE + active constraints).
        """
        state_arr = self._validate_state(state)
        x = self._E0 @ state_arr

        # ODE residual
        residuals = [self._ode_residual(state_arr, t=t)]

        # Constraint residuals
        conservations, measurements = self._get_active_constraints(t)
        for c in conservations:
            residuals.append(c.residual(x))
        for m in measurements:
            res = m.residual(x, t)
            if res is not None:
                residuals.append(res)

        return np.concatenate(residuals) if len(residuals) > 1 else residuals[0]

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Return the Jacobian of the observation model at ``state``.

        Args:
            state: State vector of length matching ``E0.shape[1]``.
            t: Current time.

        Returns:
            Jacobian matrix of the observation model.
        """
        state_arr = self._validate_state(state)

        # ODE Jacobian
        jacobians = [self._ode_jacobian(state_arr, t=t)]

        # Constraint Jacobians (need to compose with E0)
        conservations, measurements = self._get_active_constraints(t)
        for c in conservations:
            jacobians.append(c.jacobian() @ self._E0)
        for m in measurements:
            jac = m.jacobian(t)
            if jac is not None:
                jacobians.append(jac @ self._E0)

        return np.concatenate(jacobians) if len(jacobians) > 1 else jacobians[0]

    def get_noise(self, *, t: float) -> Array:
        """Return the measurement noise matrix at time ``t``.

        Args:
            t: Current time.

        Returns:
            Measurement noise covariance matrix.
        """
        return self._build_noise_matrix(t)

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
        """Validate and convert state to required format."""
        state_arr = np.asarray(state, dtype=np.float32)
        if state_arr.ndim != 1:
            raise ValueError("'state' must be a one-dimensional array.")
        if state_arr.shape[0] != self._state_dim:
            raise ValueError(
                f"'state' must have length {self._state_dim}, got {state_arr.shape[0]}."
            )
        return state_arr


# =============================================================================
# First-order ODE classes
# =============================================================================


class ODEInformation(BaseODEInformation):
    """First-order ODE measurement model: dx/dt = f(x, t).

    Args:
        vf: Vector field function vf(x, *, t) -> dx/dt.
        E0: State extraction matrix (shape [d, D]).
        E1: First derivative extraction matrix (shape [d, D]).
        constraints: Optional list of Conservation and/or Measurement constraints.
    """

    def __init__(
        self,
        vf: Callable[[Array], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        constraints: list[Conservation | Measurement] | None = None,
    ):
        self._vf = vf
        self._E0 = np.asarray(E0)
        self._E1 = np.asarray(E1)
        self._d = self._E0.shape[0]
        self._state_dim = self._E0.shape[1]
        self._E_constraint = self._E1
        self._base_R = np.zeros((self._d, self._d))
        self._jacobian_vf = jax.jacfwd(self._vf)
        self._init_constraints(constraints)

    def _ode_residual(self, state: Array, *, t: float) -> Array:
        x = self._E0 @ state
        vf_eval = self._vf(x, t=t)
        return self._E_constraint @ state - vf_eval

    def _ode_jacobian(self, state: Array, *, t: float) -> Array:
        x = self._E0 @ state
        jac_vf = self._jacobian_vf(x, t=t)
        return self._E_constraint - jac_vf @ self._E0


class ODEInformationWithHidden(BaseODEInformation):
    """First-order ODE with hidden states: dx/dt = f(x, u, t).

    For joint state-parameter estimation where u is a hidden parameter
    that appears in the dynamics but evolves according to its own prior.

    Args:
        vf: Vector field function vf(x, u, *, t) -> dx/dt.
        E0: State extraction matrix for x (shape [d_x, D]).
        E1: First derivative extraction matrix (shape [d_x, D]).
        E0_hidden: Hidden state extraction matrix for u (shape [d_u, D]).
        constraints: Optional list of Conservation and/or Measurement constraints.
    """

    def __init__(
        self,
        vf: Callable[[Array, Array], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E0_hidden: ArrayLike,
        constraints: list[Conservation | Measurement] | None = None,
    ):
        self._vf = vf
        self._E0 = np.asarray(E0)
        self._E1 = np.asarray(E1)
        self._E0_hidden = np.asarray(E0_hidden)
        self._d = self._E0.shape[0]
        self._state_dim = self._E0.shape[1]
        self._E_constraint = self._E1
        self._base_R = np.zeros((self._d, self._d))
        self._jacobian_vf_x = jax.jacfwd(self._vf, argnums=0)
        self._jacobian_vf_u = jax.jacfwd(self._vf, argnums=1)
        self._init_constraints(constraints)

    def _ode_residual(self, state: Array, *, t: float) -> Array:
        x = self._E0 @ state
        u = self._E0_hidden @ state
        vf_eval = self._vf(x, u, t=t)
        return self._E_constraint @ state - vf_eval

    def _ode_jacobian(self, state: Array, *, t: float) -> Array:
        x = self._E0 @ state
        u = self._E0_hidden @ state
        jac_vf_x = self._jacobian_vf_x(x, u, t=t)
        jac_vf_u = self._jacobian_vf_u(x, u, t=t)
        return self._E_constraint - jac_vf_x @ self._E0 - jac_vf_u @ self._E0_hidden


# =============================================================================
# Second-order ODE classes
# =============================================================================


class SecondOrderODEInformation(BaseODEInformation):
    """Second-order ODE measurement model: d^2x/dt^2 = f(x, v, t).

    Args:
        vf: Vector field function vf(x, v, *, t) -> d^2x/dt^2.
        E0: State extraction matrix (shape [d, D]).
        E1: First derivative extraction matrix (shape [d, D]).
        E2: Second derivative extraction matrix (shape [d, D]).
        constraints: Optional list of Conservation and/or Measurement constraints.
    """

    def __init__(
        self,
        vf: Callable[[Array, Array], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E2: ArrayLike,
        constraints: list[Conservation | Measurement] | None = None,
    ):
        self._vf = vf
        self._E0 = np.asarray(E0)
        self._E1 = np.asarray(E1)
        self._E2 = np.asarray(E2)
        self._d = self._E0.shape[0]
        self._state_dim = self._E0.shape[1]
        self._E_constraint = self._E2
        self._base_R = np.zeros((self._d, self._d))
        self._jacobian_vf_x = jax.jacfwd(self._vf, argnums=0)
        self._jacobian_vf_v = jax.jacfwd(self._vf, argnums=1)
        self._init_constraints(constraints)

    def _ode_residual(self, state: Array, *, t: float) -> Array:
        x = self._E0 @ state
        v = self._E1 @ state
        vf_eval = self._vf(x, v, t=t)
        return self._E_constraint @ state - vf_eval

    def _ode_jacobian(self, state: Array, *, t: float) -> Array:
        x = self._E0 @ state
        v = self._E1 @ state
        jac_x = self._jacobian_vf_x(x, v, t=t)
        jac_v = self._jacobian_vf_v(x, v, t=t)
        return self._E_constraint - jac_x @ self._E0 - jac_v @ self._E1


class SecondOrderODEInformationWithHidden(BaseODEInformation):
    """Second-order ODE with hidden states: d^2x/dt^2 = f(x, v, u, t).

    For joint state-parameter estimation where u is a hidden parameter
    that appears in the dynamics but evolves according to its own prior.

    Args:
        vf: Vector field function vf(x, v, u, *, t) -> d^2x/dt^2.
        E0: State extraction matrix for x (shape [d_x, D]).
        E1: First derivative extraction matrix (shape [d_x, D]).
        E2: Second derivative extraction matrix (shape [d_x, D]).
        E0_hidden: Hidden state extraction matrix for u (shape [d_u, D]).
        constraints: Optional list of Conservation and/or Measurement constraints.
    """

    def __init__(
        self,
        vf: Callable[[Array, Array, Array], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E2: ArrayLike,
        E0_hidden: ArrayLike,
        constraints: list[Conservation | Measurement] | None = None,
    ):
        self._vf = vf
        self._E0 = np.asarray(E0)
        self._E1 = np.asarray(E1)
        self._E2 = np.asarray(E2)
        self._E0_hidden = np.asarray(E0_hidden)
        self._d = self._E0.shape[0]
        self._state_dim = self._E0.shape[1]
        self._E_constraint = self._E2
        self._base_R = np.zeros((self._d, self._d))
        self._jacobian_vf_x = jax.jacfwd(self._vf, argnums=0)
        self._jacobian_vf_v = jax.jacfwd(self._vf, argnums=1)
        self._jacobian_vf_u = jax.jacfwd(self._vf, argnums=2)
        self._init_constraints(constraints)

    def _ode_residual(self, state: Array, *, t: float) -> Array:
        x = self._E0 @ state
        v = self._E1 @ state
        u = self._E0_hidden @ state
        vf_eval = self._vf(x, v, u, t=t)
        return self._E_constraint @ state - vf_eval

    def _ode_jacobian(self, state: Array, *, t: float) -> Array:
        x = self._E0 @ state
        v = self._E1 @ state
        u = self._E0_hidden @ state
        jac_x = self._jacobian_vf_x(x, v, u, t=t)
        jac_v = self._jacobian_vf_v(x, v, u, t=t)
        jac_u = self._jacobian_vf_u(x, v, u, t=t)
        return (
            self._E_constraint
            - jac_x @ self._E0
            - jac_v @ self._E1
            - jac_u @ self._E0_hidden
        )


# =============================================================================
# Convenience factory functions for backward compatibility
# =============================================================================


def ODEconservation(
    vf: Callable,
    E0: ArrayLike,
    E1: ArrayLike,
    A: Array,
    p: Array,
) -> ODEInformation:
    """Create first-order ODE model with conservation constraint.

    Args:
        vf: Vector field function vf(x, *, t) -> dx/dt.
        E0: State extraction matrix.
        E1: Derivative extraction matrix.
        A: Conservation constraint matrix (shape [k, d]).
        p: Conservation target values (shape [k]).

    Returns:
        ODEInformation with conservation constraint.
    """
    return ODEInformation(vf, E0, E1, constraints=[Conservation(A, p)])


def SecondOrderODEconservation(
    vf: Callable,
    E0: ArrayLike,
    E1: ArrayLike,
    E2: ArrayLike,
    A: Array,
    p: Array,
) -> SecondOrderODEInformation:
    """Create second-order ODE model with conservation constraint.

    Args:
        vf: Vector field function vf(x, v, *, t) -> d^2x/dt^2.
        E0: State extraction matrix.
        E1: First derivative extraction matrix.
        E2: Second derivative extraction matrix.
        A: Conservation constraint matrix (shape [k, d]).
        p: Conservation target values (shape [k]).

    Returns:
        SecondOrderODEInformation with conservation constraint.
    """
    return SecondOrderODEInformation(vf, E0, E1, E2, constraints=[Conservation(A, p)])


def ODEmeasurement(
    vf: Callable,
    E0: ArrayLike,
    E1: ArrayLike,
    A: Array,
    z: Array,
    z_t: Array,
    measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
) -> ODEInformation:
    """Create first-order ODE model with linear measurements.

    Args:
        vf: Vector field function vf(x, *, t) -> dx/dt.
        E0: State extraction matrix.
        E1: Derivative extraction matrix.
        A: Measurement matrix (shape [k, d]).
        z: Measurement values (shape [n, k]).
        z_t: Measurement times (shape [n]).
        measurement_noise: Measurement noise variance.

    Returns:
        ODEInformation with measurement constraint.
    """
    return ODEInformation(
        vf, E0, E1, constraints=[Measurement(A, z, z_t, measurement_noise)]
    )


def SecondOrderODEmeasurement(
    vf: Callable,
    E0: ArrayLike,
    E1: ArrayLike,
    E2: ArrayLike,
    A: Array,
    z: Array,
    z_t: Array,
    measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
) -> SecondOrderODEInformation:
    """Create second-order ODE model with linear measurements.

    Args:
        vf: Vector field function vf(x, v, *, t) -> d^2x/dt^2.
        E0: State extraction matrix.
        E1: First derivative extraction matrix.
        E2: Second derivative extraction matrix.
        A: Measurement matrix (shape [k, d]).
        z: Measurement values (shape [n, k]).
        z_t: Measurement times (shape [n]).
        measurement_noise: Measurement noise variance.

    Returns:
        SecondOrderODEInformation with measurement constraint.
    """
    return SecondOrderODEInformation(
        vf, E0, E1, E2, constraints=[Measurement(A, z, z_t, measurement_noise)]
    )


def ODEconservationmeasurement(
    vf: Callable,
    E0: ArrayLike,
    E1: ArrayLike,
    C: Array,
    p: Array,
    A: Array,
    z: Array,
    z_t: Array,
    measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
) -> ODEInformation:
    """Create first-order ODE model with conservation and measurements.

    Args:
        vf: Vector field function vf(x, *, t) -> dx/dt.
        E0: State extraction matrix.
        E1: Derivative extraction matrix.
        C: Conservation constraint matrix (shape [m, d]).
        p: Conservation target values (shape [m]).
        A: Measurement matrix (shape [k, d]).
        z: Measurement values (shape [n, k]).
        z_t: Measurement times (shape [n]).
        measurement_noise: Measurement noise variance.

    Returns:
        ODEInformation with conservation and measurement constraints.
    """
    return ODEInformation(
        vf,
        E0,
        E1,
        constraints=[Conservation(C, p), Measurement(A, z, z_t, measurement_noise)],
    )


def SecondOrderODEconservationmeasurement(
    vf: Callable,
    E0: ArrayLike,
    E1: ArrayLike,
    E2: ArrayLike,
    C: Array,
    p: Array,
    A: Array,
    z: Array,
    z_t: Array,
    measurement_noise: float = DEFAULT_MEASUREMENT_NOISE,
) -> SecondOrderODEInformation:
    """Create second-order ODE model with conservation and measurements.

    Args:
        vf: Vector field function vf(x, v, *, t) -> d^2x/dt^2.
        E0: State extraction matrix.
        E1: First derivative extraction matrix.
        E2: Second derivative extraction matrix.
        C: Conservation constraint matrix (shape [m, d]).
        p: Conservation target values (shape [m]).
        A: Measurement matrix (shape [k, d]).
        z: Measurement values (shape [n, k]).
        z_t: Measurement times (shape [n]).
        measurement_noise: Measurement noise variance.

    Returns:
        SecondOrderODEInformation with conservation and measurement constraints.
    """
    return SecondOrderODEInformation(
        vf,
        E0,
        E1,
        E2,
        constraints=[Conservation(C, p), Measurement(A, z, z_t, measurement_noise)],
    )
