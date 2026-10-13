"""ODE information operators: the ODE residual plus conservation laws."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _check_noise_covariance, _safe_cholesky_sqr
from .observations import Measurement


def _check_selectors(**selectors: ArrayLike | None) -> None:
    """Validate the selection matrices of an ODE-information model.

    All must be 2-D with one column per state entry, and ``E1`` / ``E2`` need one
    row per solution component, like ``E0``; ``E0_hidden`` may have any number
    of rows. Catches e.g. a sliced ``E1`` that would otherwise broadcast
    silently into a wrong residual.
    """
    shapes = {}
    for name, E in selectors.items():
        if E is None:
            hint = " (a prior provides E2 only for q >= 2)" if name == "E2" else ""
            raise ValueError(f"{name} is None{hint}.")
        shape = np.shape(E)
        if len(shape) != 2:
            raise ValueError(
                f"{name} must be a 2-D selection matrix, got shape {shape}."
            )
        shapes[name] = shape
    if len({shape[1] for shape in shapes.values()}) > 1:
        listed = ", ".join(f"{name}: {shape}" for name, shape in shapes.items())
        raise ValueError(
            f"Selection matrices must all have one column per state entry; got {listed}."
        )
    d = shapes["E0"][0]
    for name in ("E1", "E2"):
        if name in shapes and shapes[name][0] != d:
            raise ValueError(
                f"{name} must have as many rows as E0 ({d}), got shape {shapes[name]}."
            )


@dataclass(frozen=True)
class Conservation:
    """Conservation constraint: A @ x = p (always active).

    Args:
        A: Constraint matrix (shape [k, d] or [k, state_dim] if full_state=True).
        p: Target values (shape [k]).
        full_state: If True, A operates on the full state X instead of x = E0 @ X.
    """

    A: Array
    p: Array
    full_state: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "A", np.asarray(self.A))
        object.__setattr__(self, "p", np.asarray(self.p))
        if self.A.ndim != 2 or self.p.ndim != 1:
            raise ValueError(
                f"Conservation needs a 2-D 'A' and a 1-D 'p', got shapes "
                f"{self.A.shape} and {self.p.shape}."
            )
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


class BaseODEInformation(ABC):
    """Abstract base class for ODE measurement models."""

    _vf: Callable
    _E0: Array
    _E1: Array
    _E_constraint: Array
    _d: int
    _state_dim: int
    _base_R: Array
    _constraints: list[Conservation]

    def _init_constraints(
        self,
        constraints: list[Conservation] | None,
    ) -> None:
        """Initialize constraint list and validate.

        Only ``Conservation`` constraints (noiseless, always-on model
        structure) are accepted. Data observations are *not* part of the
        measurement model -- pass them as an ``obs_model`` built with
        ``prepare_observations``.
        """
        # A copy, so appending to the caller's list later cannot change the model.
        self._constraints = list(constraints or [])
        for c in self._constraints:
            if isinstance(c, Conservation):
                expected = self._state_dim if c.full_state else self._d
                label = "full state" if c.full_state else "state"
                if c.A.shape[1] != expected:
                    raise ValueError(
                        f"Conservation A.shape[1] ({c.A.shape[1]}) must match "
                        f"{label} dimension ({expected})."
                    )
            elif isinstance(c, Measurement):
                raise TypeError(
                    "Measurement is no longer accepted as a measure constraint. "
                    "Data observations go through obs_model: build it with "
                    "prepare_observations([Measurement(...)], E0, ts) and pass "
                    "obs_model= to the solver. `measure` carries only the ODE "
                    "information and Conservation constraints."
                )
            else:
                raise TypeError(
                    f"constraints must be Conservation objects, got {type(c).__name__}."
                )

    def _conservations(self) -> list[Conservation]:
        """Conservation constraints registered on this measure (always active)."""
        return [c for c in self._constraints if isinstance(c, Conservation)]

    def _build_noise_matrix(self, t: ArrayLike) -> Array:
        """Build combined noise matrix for all active constraints at time t."""
        conservations = self._conservations()
        cons_dim = sum(c.dim for c in conservations)
        total_dim = self._d + cons_dim

        R = np.zeros((total_dim, total_dim))
        # Base ODE noise (usually zero)
        R = R.at[: self._d, : self._d].set(self._base_R)
        # Conservation noise (zero)
        return R

    @property
    def ode_dim(self) -> int:
        """Number of ODE-defect rows at the top of ``linearize()`` output.

        Calibration and step-size adaptation should use only
        ``H_t[:ode_dim]`` and ``mz_pred[:ode_dim]`` so that conservation
        and observation residuals (which live on different scales than the
        ODE defect) cannot bias the diffusion estimate. See Bosch,
        Tronarp, Hennig (2022), *Pick-and-Mix Information Operators*,
        sec. 3 and ProbNumDiffEq.jl's ``perform_step!`` for the matching
        design choice.
        """
        return self._d

    @property
    def E_constraint(self) -> Array:
        """Selection matrix for the highest-derivative ODE-defect rows.

        This is the Jacobian of the *linear* part of the ODE residual (``E1``
        for first-order systems, ``E2`` for second-order). A zeroth-order (EK0)
        correction uses it directly as the ODE-row Jacobian (vector field
        treated as locally constant); the first-order (EK1) Jacobian
        additionally subtracts the vector-field Jacobian (see
        ``_ode_jacobian``). Shape ``[ode_dim, state_dim]``.
        """
        return self._E_constraint

    @property
    def E_args(self) -> Array:
        """Selection matrix for the coordinates the vector field reads.

        The residual is affine in the state *except* through ``E_args @ state``:
        for a first-order ODE the vector field sees only ``E0 @ state``, for a
        second-order one it sees ``(E0 @ state, E1 @ state)``, and the hidden
        variants add ``E0_hidden``. Conservation constraints are affine and so
        never enter here.

        Statistical linearisation integrates over this projected subspace only
        (see ``slr_linearize``),
        which is what keeps the quadrature dimension equal to the ODE dimension
        rather than the full state dimension.

        The base implementation returns ``E0``, correct for first-order models.
        A custom subclass whose ``_ode_residual`` reads more than ``E0 @ state``
        **must override this**, or SLR-based corrections will silently miss the
        spread in the omitted directions; validate with
        ``check_arg_projection``.
        Shape ``[p, state_dim]``.
        """
        return self._E0

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
            self._base_R = R_arr * np.eye(expected_dim)
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
        _check_noise_covariance(self._base_R, "R")

    @abstractmethod
    def _ode_residual(self, state: Array, *, t: ArrayLike) -> Array:
        """Compute the ODE residual."""

    @abstractmethod
    def _ode_jacobian(self, state: Array, *, t: ArrayLike) -> Array:
        """Compute the ODE Jacobian."""

    def g(self, state: Array, *, t: ArrayLike) -> Array:
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
        for c in self._conservations():
            residuals.append(c.residual(state_arr if c.full_state else x))

        return np.concatenate(residuals) if len(residuals) > 1 else residuals[0]

    def jacobian_g(self, state: Array, *, t: ArrayLike) -> Array:
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

        # Constraint Jacobians (compose with E0 unless full_state)
        for c in self._conservations():
            jacobians.append(c.jacobian() if c.full_state else c.jacobian() @ self._E0)

        return np.concatenate(jacobians) if len(jacobians) > 1 else jacobians[0]

    def get_noise(self, *, t: ArrayLike) -> Array:
        """Return the square root of the measurement noise covariance at time ``t``.

        Returns an upper-triangular matrix L such that L.T @ L ≈ R,
        where R is the full noise covariance.

        Args:
            t: Current time.

        Returns:
            Upper-triangular square root of the noise covariance matrix.
        """
        R = self._build_noise_matrix(t)
        return _safe_cholesky_sqr(R, R.shape[0])

    def linearize(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
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
        x = self._E0 @ state_arr

        # ODE part (single evaluation of vf)
        H_ode = self._ode_jacobian(state_arr, t=t)
        g_ode = self._ode_residual(state_arr, t=t)
        jacobians = [H_ode]
        residuals = [g_ode]

        # Constraints (single iteration)
        for c in self._conservations():
            jacobians.append(c.jacobian() if c.full_state else c.jacobian() @ self._E0)
            residuals.append(c.residual(state_arr if c.full_state else x))

        H_t = np.concatenate(jacobians) if len(jacobians) > 1 else jacobians[0]
        g_val = np.concatenate(residuals) if len(residuals) > 1 else residuals[0]
        c_t = g_val - H_t @ state_arr
        return H_t, c_t

    def _validate_state(self, state: Array) -> Array:
        """Validate and convert state to required format."""
        state_arr = np.asarray(state)
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
        constraints: Optional list of Conservation constraints. (Data observations
            are *not* measure constraints -- pass them as an ``obs_model`` built
            with ``prepare_observations``; a ``Measurement`` here raises TypeError.)
    """

    def __init__(
        self,
        vf: Callable[..., Array],
        E0: ArrayLike,
        E1: ArrayLike,
        constraints: list[Conservation] | None = None,
    ):
        _check_selectors(E0=E0, E1=E1)
        self._vf = vf
        self._E0 = np.asarray(E0)
        self._E1 = np.asarray(E1)
        self._d = self._E0.shape[0]
        self._state_dim = self._E0.shape[1]
        self._E_constraint = self._E1
        self._base_R = np.zeros((self._d, self._d))
        self._jacobian_vf = jax.jacfwd(self._vf)
        self._init_constraints(constraints)

    def _ode_residual(self, state: Array, *, t: ArrayLike) -> Array:
        x = self._E0 @ state
        vf_eval = self._vf(x, t=t)
        return self._E_constraint @ state - vf_eval

    def _ode_jacobian(self, state: Array, *, t: ArrayLike) -> Array:
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
        constraints: Optional list of Conservation constraints. (Data observations
            are *not* measure constraints -- pass them as an ``obs_model`` built
            with ``prepare_observations``; a ``Measurement`` here raises TypeError.)
    """

    def __init__(
        self,
        vf: Callable[[Array, Array], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E0_hidden: ArrayLike,
        constraints: list[Conservation] | None = None,
    ):
        _check_selectors(E0=E0, E1=E1, E0_hidden=E0_hidden)
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

    @property
    def E_args(self) -> Array:
        """Vector-field argument selector: ``[E0; E0_hidden]`` (reads ``x``, ``u``)."""
        return np.concatenate([self._E0, self._E0_hidden], axis=0)

    def _ode_residual(self, state: Array, *, t: ArrayLike) -> Array:
        x = self._E0 @ state
        u = self._E0_hidden @ state
        vf_eval = self._vf(x, u, t=t)
        return self._E_constraint @ state - vf_eval

    def _ode_jacobian(self, state: Array, *, t: ArrayLike) -> Array:
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
        constraints: Optional list of Conservation constraints. (Data observations
            are *not* measure constraints -- pass them as an ``obs_model`` built
            with ``prepare_observations``; a ``Measurement`` here raises TypeError.)
    """

    def __init__(
        self,
        vf: Callable[[Array, Array], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E2: ArrayLike,
        constraints: list[Conservation] | None = None,
    ):
        _check_selectors(E0=E0, E1=E1, E2=E2)
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

    @property
    def E_args(self) -> Array:
        """Vector-field argument selector: ``[E0; E1]`` (reads ``x``, ``v``)."""
        return np.concatenate([self._E0, self._E1], axis=0)

    def _ode_residual(self, state: Array, *, t: ArrayLike) -> Array:
        x = self._E0 @ state
        v = self._E1 @ state
        vf_eval = self._vf(x, v, t=t)
        return self._E_constraint @ state - vf_eval

    def _ode_jacobian(self, state: Array, *, t: ArrayLike) -> Array:
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
        constraints: Optional list of Conservation constraints. (Data observations
            are *not* measure constraints -- pass them as an ``obs_model`` built
            with ``prepare_observations``; a ``Measurement`` here raises TypeError.)
    """

    def __init__(
        self,
        vf: Callable[[Array, Array, Array], Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E2: ArrayLike,
        E0_hidden: ArrayLike,
        constraints: list[Conservation] | None = None,
    ):
        _check_selectors(E0=E0, E1=E1, E2=E2, E0_hidden=E0_hidden)
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

    @property
    def E_args(self) -> Array:
        """Vector-field argument selector: ``[E0; E1; E0_hidden]``."""
        return np.concatenate([self._E0, self._E1, self._E0_hidden], axis=0)

    def _ode_residual(self, state: Array, *, t: ArrayLike) -> Array:
        x = self._E0 @ state
        v = self._E1 @ state
        u = self._E0_hidden @ state
        vf_eval = self._vf(x, v, u, t=t)
        return self._E_constraint @ state - vf_eval

    def _ode_jacobian(self, state: Array, *, t: ArrayLike) -> Array:
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
