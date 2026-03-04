from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

# Default measurement noise for observations
DEFAULT_MEASUREMENT_NOISE = 1e-6

# Large variance for inactive measurement rows (effectively disables them)
_INACTIVE_VARIANCE = 1e12

# Small jitter for Cholesky of PSD matrices with zero eigenvalues
_JITTER = 1e-32


def _safe_cholesky_sqr(R: Array, dim: int) -> Array:
    """Upper-triangular Cholesky of a PSD matrix that may have zero eigenvalues.

    Returns the upper-triangular square root L such that L.T @ L = R + jitter*I.
    """
    return np.linalg.cholesky(R + _JITTER * np.eye(dim)).T


class ScanData(NamedTuple):
    """Pre-computed data for scan-based filtering.

    This structure holds all measurement-related data pre-computed for each
    time step, enabling efficient use with jax.lax.scan. Constant matrices
    (measurement Jacobians, conservation Jacobians, noise) are stored once,
    while only per-step data (measurement offsets, noise selection, masks) is
    stored per time step.

    Attributes:
        c_meas: Measurement residual offsets, shape [N, max_meas_dim].
            Zero for inactive measurements. Per-step because z values differ.
        R_meas_sqr: Square root of measurement noise covariance,
            shape [N, max_meas_dim, max_meas_dim]. Per-step because active
            rows use actual noise sqrt while inactive rows use large variance.
        H_meas: Measurement Jacobians (constant across steps),
            shape [max_meas_dim, state_dim]. Stacked m.A @ E0 for all
            Measurement constraints.
        H_cons: Conservation Jacobians (constant across steps),
            shape [cons_dim, state_dim]. Stacked c.A @ E0 for all
            Conservation constraints. Empty (0 rows) if no conservations.
        R_fixed_sqr: Square root of fixed noise (ODE + Conservation),
            shape [fixed_dim, fixed_dim]. Constant across steps.
        obs_mask: Boolean mask for active observations, shape [N, max_obs_dim].
            True for active rows (ODE + Conservation always active).
        max_obs_dim: Total observation dimension (d + cons_dim + max_meas_dim).
        fixed_dim: Fixed observation dimension (d + cons_dim, always active).
        ts: Time values for each step, shape [N].
    """

    c_meas: Array
    R_meas_sqr: Array
    H_meas: Array
    H_cons: Array
    R_fixed_sqr: Array
    obs_mask: Array
    max_obs_dim: int
    fixed_dim: int
    ts: Array


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

    This constraint is only active when the filter's current time matches one of
    the measurement times in z_t. Time matching uses tolerance-based comparison
    (not exact equality) to handle floating-point discrepancies.

    Important:
        When creating time grids, use the same linspace implementation (preferably
        jax.numpy.linspace) for both measurement times (z_t) and filter time steps.
        NumPy and JAX linspace can produce slightly different values (~1e-8 differences)
        which may cause measurements to be missed with exact comparison.

    Args:
        A: Measurement matrix (shape [k, d]).
        z: Measurement values (shape [n, k]).
        z_t: Measurement times (shape [n]). Should use jax.numpy.linspace for
            consistency with the filter's internal time grid.
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
        # Normalize z_t to 1D
        z_t = self.z_t
        if z_t.ndim == 2 and z_t.shape[1] == 1:
            object.__setattr__(self, "z_t", z_t.reshape(-1))
        elif z_t.ndim != 1:
            raise ValueError(
                f"'z_t' must be 1D shape (n,) or 2D shape (n, 1), got {z_t.shape}."
            )
        if self.z_t.shape[0] != self.z.shape[0]:
            raise ValueError(
                f"'z_t' length must match number of measurements {self.z.shape[0]}, "
                f"got {self.z_t.shape[0]}."
            )

    def find_index(
        self, t: float, rtol: float = 1e-5, atol: float = 1e-7
    ) -> int | None:
        """Find measurement index for time t, or None if not found.

        Uses binary search with tolerance for robust floating-point comparison.
        Times are assumed to be sorted.

        Note: Default tolerances (rtol=1e-5, atol=1e-7) are set to handle
        discrepancies between NumPy and JAX linspace implementations, which
        can differ by ~1e-8 for typical time grids.
        """
        idx = np.searchsorted(self.z_t, t)
        # Check the found position and neighbors for a close match
        for i in [idx, idx - 1]:
            if 0 <= i < len(self.z_t) and np.isclose(
                self.z_t[i], t, rtol=rtol, atol=atol
            ):
                return int(i)
        return None

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
            return noise * np.eye(self.dim)
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
        x = self._E0 @ state_arr

        # ODE part (single evaluation of vf)
        H_ode = self._ode_jacobian(state_arr, t=t)
        g_ode = self._ode_residual(state_arr, t=t)
        jacobians = [H_ode]
        residuals = [g_ode]

        # Constraints (single iteration)
        conservations, measurements = self._get_active_constraints(t)
        for c in conservations:
            jacobians.append(c.jacobian() @ self._E0)
            residuals.append(c.residual(x))
        for m in measurements:
            jac = m.jacobian(t)
            if jac is not None:
                jacobians.append(jac @ self._E0)
                res = m.residual(x, t)
                if res is not None:
                    residuals.append(res)

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

    # =========================================================================
    # Scan-compatible methods for jax.lax.scan
    # =========================================================================

    def _get_fixed_dim(self) -> int:
        """Get the fixed observation dimension (ODE + Conservation)."""
        conservations = [c for c in self._constraints if isinstance(c, Conservation)]
        cons_dim = sum(c.dim for c in conservations)
        return self._d + cons_dim

    def _get_max_meas_dim(self) -> int:
        """Get the maximum measurement dimension across all Measurement constraints."""
        measurements = [c for c in self._constraints if isinstance(c, Measurement)]
        return sum(m.dim for m in measurements)

    def _linearize_fixed(
        self, state: Array, *, t: float, H_cons: Array | None = None
    ) -> tuple[Array, Array]:
        """Linearize ODE + Conservation parts only (fixed shape, no find_index).

        This method computes the Jacobian and residual for the ODE constraint
        and any Conservation constraints, which have fixed dimensions regardless
        of time. It does not include Measurement constraints.

        Args:
            state: State vector to linearize around.
            t: Current time.
            H_cons: Pre-computed conservation Jacobian (stacked c.A @ E0).
                If provided, avoids recomputing the constant conservation
                Jacobians. Shape [cons_dim, state_dim].

        Returns:
            Tuple of (H_fixed, c_fixed) where:
            - H_fixed has shape [fixed_dim, state_dim]
            - c_fixed has shape [fixed_dim]
        """
        state_arr = self._validate_state(state)
        x = self._E0 @ state_arr

        # ODE Jacobian and residual
        H_ode = self._ode_jacobian(state_arr, t=t)
        g_ode = self._ode_residual(state_arr, t=t)

        # Conservation constraints (always active)
        conservations = [c for c in self._constraints if isinstance(c, Conservation)]
        if conservations:
            # Use pre-computed Jacobian if available, otherwise compute
            if H_cons is None:
                H_cons = np.concatenate(
                    [c.jacobian() @ self._E0 for c in conservations], axis=0
                )
            g_cons = np.concatenate([c.residual(x) for c in conservations])
            H_fixed = np.concatenate([H_ode, H_cons], axis=0)
            g_fixed = np.concatenate([g_ode, g_cons])
        else:
            H_fixed = H_ode
            g_fixed = g_ode

        c_fixed = g_fixed - H_fixed @ state_arr
        return H_fixed, c_fixed

    def _get_fixed_noise_sqr(self) -> Array:
        """Get the square root of noise for fixed observations (ODE + Conservation).

        Returns upper-triangular Cholesky factor L such that L.T @ L ≈ R_fixed.
        """
        fixed_dim = self._get_fixed_dim()
        R_fixed = np.zeros((fixed_dim, fixed_dim))
        R_fixed = R_fixed.at[: self._d, : self._d].set(self._base_R)
        return _safe_cholesky_sqr(R_fixed, fixed_dim)

    def prepare_scan_data(self, ts: Array) -> ScanData:
        """Pre-compute measurement data for all time steps.

        This method iterates over the time array with concrete Python values,
        calling find_index normally. The result is a ScanData structure with
        padded arrays suitable for use with jax.lax.scan.

        Constant data (measurement Jacobians, conservation Jacobians, noise
        matrices) is stored once. Only per-step data (measurement offsets c_meas,
        noise selection R_meas_sqr, observation masks) varies across time steps.

        Args:
            ts: Time array of shape [N+1] (includes initial time t0).
                The filter uses ts[1:] for the N filter steps.

        Returns:
            ScanData with pre-computed data for scan-based filtering.
        """
        N = len(ts) - 1
        fixed_dim = self._get_fixed_dim()
        max_meas_dim = self._get_max_meas_dim()
        max_obs_dim = fixed_dim + max_meas_dim

        measurements = [c for c in self._constraints if isinstance(c, Measurement)]
        conservations = [c for c in self._constraints if isinstance(c, Conservation)]

        # --- Constant data (computed once) ---

        # Stacked measurement Jacobian: all m.A @ E0
        H_meas_const = np.zeros((max_meas_dim, self._state_dim))
        R_meas_block = np.zeros((max_meas_dim, max_meas_dim))
        offset = 0
        for m in measurements:
            H_meas_const = H_meas_const.at[offset : offset + m.dim, :].set(
                m.A @ self._E0
            )
            R_meas_block = R_meas_block.at[
                offset : offset + m.dim, offset : offset + m.dim
            ].set(m.get_noise_matrix())
            offset += m.dim

        # Square root of active measurement noise
        R_meas_sqr_active = (
            _safe_cholesky_sqr(R_meas_block, max_meas_dim)
            if max_meas_dim > 0
            else np.zeros((0, 0))
        )
        # Square root of inactive measurement noise (large variance)
        R_meas_sqr_inactive = np.sqrt(_INACTIVE_VARIANCE) * np.eye(max_meas_dim)

        # Stacked conservation Jacobian: all c.A @ E0
        cons_dim = sum(c.dim for c in conservations)
        H_cons = np.zeros((cons_dim, self._state_dim))
        offset = 0
        for c in conservations:
            H_cons = H_cons.at[offset : offset + c.dim, :].set(c.jacobian() @ self._E0)
            offset += c.dim

        # Fixed noise square root (ODE + Conservation)
        R_fixed_sqr = self._get_fixed_noise_sqr()

        # --- Per-step data ---

        c_meas_list = []
        R_meas_sqr_list = []
        obs_mask_list = []

        for i in range(N):
            t = float(ts[i + 1])

            c_meas_i = np.zeros(max_meas_dim)
            meas_mask_i = np.zeros(max_meas_dim, dtype=bool)
            R_meas_sqr_i = R_meas_sqr_inactive.copy()

            offset = 0
            for m in measurements:
                idx = m.find_index(t)
                if idx is not None:
                    c_meas_i = c_meas_i.at[offset : offset + m.dim].set(-m.z[idx])
                    meas_mask_i = meas_mask_i.at[offset : offset + m.dim].set(True)
                    R_meas_sqr_i = R_meas_sqr_i.at[
                        offset : offset + m.dim, offset : offset + m.dim
                    ].set(
                        R_meas_sqr_active[
                            offset : offset + m.dim, offset : offset + m.dim
                        ]
                    )
                offset += m.dim

            obs_mask_i = np.concatenate([np.ones(fixed_dim, dtype=bool), meas_mask_i])
            c_meas_list.append(c_meas_i)
            R_meas_sqr_list.append(R_meas_sqr_i)
            obs_mask_list.append(obs_mask_i)

        return ScanData(
            c_meas=np.stack(c_meas_list)
            if c_meas_list
            else np.zeros((0, max_meas_dim)),
            R_meas_sqr=np.stack(R_meas_sqr_list)
            if R_meas_sqr_list
            else np.zeros((0, max_meas_dim, max_meas_dim)),
            H_meas=H_meas_const,
            H_cons=H_cons,
            R_fixed_sqr=R_fixed_sqr,
            obs_mask=np.stack(obs_mask_list)
            if obs_mask_list
            else np.zeros((0, max_obs_dim), dtype=bool),
            max_obs_dim=max_obs_dim,
            fixed_dim=fixed_dim,
            ts=ts[1:],
        )

    def linearize_scan(
        self, state: Array, step_idx: int, scan_data: ScanData
    ) -> tuple[Array, Array, Array]:
        """Linearize the observation model for use in jax.lax.scan.

        This method computes the full linearization by combining:
        1. State-dependent ODE + Conservation rows (computed fresh, but with
           pre-computed conservation Jacobians from scan_data)
        2. Pre-computed measurement rows (indexed from scan_data)

        Args:
            state: State vector to linearize around.
            step_idx: Current step index (0 to N-1).
            scan_data: Pre-computed measurement data from prepare_scan_data.

        Returns:
            Tuple of (H_full, c_full, R_full_sqr) where:
            - H_full has shape [max_obs_dim, state_dim]
            - c_full has shape [max_obs_dim]
            - R_full_sqr has shape [max_obs_dim, max_obs_dim] (square root)
        """
        t = scan_data.ts[step_idx]

        # Compute state-dependent fixed part, reusing pre-computed H_cons
        H_fixed, c_fixed = self._linearize_fixed(state, t=t, H_cons=scan_data.H_cons)

        # Measurement part: constant H, per-step c and R_sqr
        H_meas = scan_data.H_meas
        c_meas = scan_data.c_meas[step_idx]
        R_meas_sqr = scan_data.R_meas_sqr[step_idx]

        # Concatenate to form full observation model
        H_full = np.concatenate([H_fixed, H_meas], axis=0)
        c_full = np.concatenate([c_fixed, c_meas])

        # Build block-diagonal noise square root
        R_fixed_sqr = scan_data.R_fixed_sqr
        fixed_dim = H_fixed.shape[0]
        max_meas_dim = H_meas.shape[0]
        max_obs_dim = fixed_dim + max_meas_dim
        R_full_sqr = np.zeros((max_obs_dim, max_obs_dim))
        R_full_sqr = R_full_sqr.at[:fixed_dim, :fixed_dim].set(R_fixed_sqr)
        R_full_sqr = R_full_sqr.at[fixed_dim:, fixed_dim:].set(R_meas_sqr)

        return H_full, c_full, R_full_sqr


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
# Black-box and transformed measurement models
# =============================================================================


class BlackBoxMeasurement:
    """Black-box measurement model with autodiff Jacobian computation.

    Allows users to define an arbitrary measurement function g(state, *, t)
    and automatically computes the Jacobian via JAX autodiff.

    Args:
        g_func: Measurement function g(state, *, t) -> observation.
            Must be a differentiable function compatible with JAX.
        state_dim: Dimension of the state vector.
        obs_dim: Dimension of the observation vector.
        noise: Measurement noise (scalar, 1D diagonal, or 2D covariance matrix).
            Default is 0.0 (no noise).

    Example:
        >>> def custom_g(state, *, t):
        ...     # Nonlinear observation: squared position + velocity
        ...     return jnp.array([state[0]**2, state[1]])
        >>> measure = BlackBoxMeasurement(custom_g, state_dim=4, obs_dim=2)
    """

    def __init__(
        self,
        g_func: Callable[[Array], Array],
        state_dim: int,
        obs_dim: int,
        noise: float | ArrayLike = 0.0,
    ):
        if not callable(g_func):
            raise TypeError("'g_func' must be callable.")
        if not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError("'state_dim' must be a positive integer.")
        if not isinstance(obs_dim, int) or obs_dim <= 0:
            raise ValueError("'obs_dim' must be a positive integer.")

        self._g_func = g_func
        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._jacobian_g_func = jax.jacrev(lambda s, t: g_func(s, t=t))

        # Initialize noise matrix
        noise_arr = np.asarray(noise)
        if noise_arr.ndim == 0:
            self._R = noise_arr * np.eye(obs_dim)
        elif noise_arr.ndim == 1:
            if noise_arr.shape[0] != obs_dim:
                raise ValueError(
                    f"Diagonal noise must have length {obs_dim}, got {noise_arr.shape[0]}."
                )
            self._R = np.diag(noise_arr)
        elif noise_arr.ndim == 2:
            if noise_arr.shape != (obs_dim, obs_dim):
                raise ValueError(
                    f"Noise matrix must have shape ({obs_dim}, {obs_dim}), "
                    f"got {noise_arr.shape}."
                )
            self._R = noise_arr
        else:
            raise ValueError("'noise' must be scalar, 1D, or 2D array.")

    @property
    def R(self) -> Array:
        """Measurement noise covariance matrix."""
        return self._R

    @R.setter
    def R(self, value: ArrayLike) -> None:
        """Set the measurement noise covariance matrix."""
        R_arr = np.asarray(value)
        if R_arr.ndim == 0:
            self._R = R_arr * np.eye(self._obs_dim)
        elif R_arr.ndim == 1:
            if R_arr.shape[0] != self._obs_dim:
                raise ValueError(
                    f"Diagonal noise must have length {self._obs_dim}, "
                    f"got {R_arr.shape[0]}."
                )
            self._R = np.diag(R_arr)
        elif R_arr.ndim == 2:
            if R_arr.shape != (self._obs_dim, self._obs_dim):
                raise ValueError(
                    f"Noise matrix must have shape ({self._obs_dim}, {self._obs_dim}), "
                    f"got {R_arr.shape}."
                )
            self._R = R_arr
        else:
            raise ValueError("'noise' must be scalar, 1D, or 2D array.")

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the measurement function.

        Args:
            state: State vector of length state_dim.
            t: Current time.

        Returns:
            Observation vector of length obs_dim.
        """
        state_arr = self._validate_state(state)
        return self._g_func(state_arr, t=t)

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Compute Jacobian of the measurement function via autodiff.

        Args:
            state: State vector of length state_dim.
            t: Current time.

        Returns:
            Jacobian matrix of shape (obs_dim, state_dim).
        """
        state_arr = self._validate_state(state)
        return self._jacobian_g_func(state_arr, t)

    def get_noise(self, *, t: float) -> Array:
        """Return the square root of the measurement noise covariance.

        Returns an upper-triangular matrix L such that L.T @ L ≈ R.

        Args:
            t: Current time (unused, included for API consistency).

        Returns:
            Upper-triangular square root of the noise covariance matrix.
        """
        return _safe_cholesky_sqr(self._R, self._obs_dim)

    def linearize(self, state: Array, *, t: float) -> tuple[Array, Array]:
        """Linearize the measurement model around the given state.

        Args:
            state: State vector to linearize around.
            t: Current time.

        Returns:
            Tuple of (H_t, c_t) where:
            - H_t is the Jacobian matrix (shape [obs_dim, state_dim])
            - c_t is the constant term (observation offset)
        """
        state_arr = self._validate_state(state)
        # Use VJP to get value and Jacobian in a single forward pass
        g_val, vjp_fn = jax.vjp(lambda s: self._g_func(s, t=t), state_arr)
        H_t = jax.vmap(lambda v: vjp_fn(v)[0])(np.eye(self._obs_dim))
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


class TransformedMeasurement:
    """Wrapper that applies a nonlinear state transformation before measurement.

    Given a base measurement model and a transformation sigma(state),
    this class computes g(sigma(state)) with proper chain-rule Jacobian:
        J_total = J_g(sigma(state)) @ J_sigma(state)

    This is useful for:
    - Nonlinear coordinate transformations
    - Feature extraction before measurement
    - Applying learned transformations to the state

    Args:
        base_model: Base measurement model with g, jacobian_g, get_noise, linearize.
        sigma: State transformation function sigma(state) -> transformed_state.
            Must be differentiable and compatible with JAX.
        use_autodiff_jacobian: If True (default), compute J_sigma via autodiff.
            If False, expect sigma_jacobian to be provided.
        sigma_jacobian: Optional explicit Jacobian function for sigma.
            If provided and use_autodiff_jacobian=False, this will be used
            instead of autodiff.

    Example:
        >>> # Apply softmax transformation to state before ODE measurement
        >>> def softmax_transform(state):
        ...     # Transform first 3 components via softmax
        ...     x = state[:3]
        ...     x_soft = jax.nn.softmax(x)
        ...     return state.at[:3].set(x_soft)
        >>> base = ODEInformation(vf, E0, E1)
        >>> transformed = TransformedMeasurement(base, softmax_transform)
    """

    def __init__(
        self,
        base_model: BaseODEInformation | BlackBoxMeasurement,
        sigma: Callable[[Array], Array],
        use_autodiff_jacobian: bool = True,
        sigma_jacobian: Callable[[Array], Array] | None = None,
    ):
        if not callable(sigma):
            raise TypeError("'sigma' must be callable.")

        self._base = base_model
        self._sigma = sigma
        self._use_autodiff = use_autodiff_jacobian

        if use_autodiff_jacobian:
            self._jacobian_sigma = jax.jacfwd(sigma)
        elif sigma_jacobian is not None:
            self._jacobian_sigma = sigma_jacobian
        else:
            raise ValueError(
                "Must provide 'sigma_jacobian' when 'use_autodiff_jacobian=False'."
            )

    @property
    def R(self) -> Array:
        """Measurement noise covariance matrix (delegated to base model)."""
        return self._base.R

    @R.setter
    def R(self, value: ArrayLike) -> None:
        """Set the measurement noise covariance matrix on the base model."""
        self._base.R = value

    def g(self, state: Array, *, t: float) -> Array:
        """Evaluate the measurement function on transformed state.

        Computes g(sigma(state), t).

        Args:
            state: Original state vector.
            t: Current time.

        Returns:
            Observation vector.
        """
        transformed = self._sigma(state)
        return self._base.g(transformed, t=t)

    def jacobian_g(self, state: Array, *, t: float) -> Array:
        """Compute Jacobian with chain rule: J_g(sigma(state)) @ J_sigma(state).

        Args:
            state: Original state vector.
            t: Current time.

        Returns:
            Jacobian matrix of the composed measurement model.
        """
        transformed = self._sigma(state)
        J_base = self._base.jacobian_g(transformed, t=t)
        J_sigma = self._jacobian_sigma(state)
        return J_base @ J_sigma

    def get_noise(self, *, t: float) -> Array:
        """Return the square root of the noise covariance (delegated to base model).

        Args:
            t: Current time.

        Returns:
            Upper-triangular square root of the noise covariance matrix.
        """
        return self._base.get_noise(t=t)

    def linearize(self, state: Array, *, t: float) -> tuple[Array, Array]:
        """Linearize the transformed measurement model.

        Args:
            state: Original state vector to linearize around.
            t: Current time.

        Returns:
            Tuple of (H_t, c_t) where:
            - H_t is the Jacobian of the composed model
            - c_t is the constant term (observation offset)
        """
        transformed = self._sigma(state)
        J_sigma = self._jacobian_sigma(state)
        J_base = self._base.jacobian_g(transformed, t=t)
        g_val = self._base.g(transformed, t=t)
        H_t = J_base @ J_sigma
        c_t = g_val - H_t @ state
        return H_t, c_t


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
