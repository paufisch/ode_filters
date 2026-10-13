"""Black-box and transformed measurement models."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _check_noise_covariance, _safe_cholesky_sqr
from .ode_information import BaseODEInformation

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
        ode_dim: Number of leading rows of ``g`` that are ODE-defect residuals.
            The solvers condition on every row (``g(state) = 0``) but estimate
            the diffusion ``sigma^2`` only from these rows. Defaults to
            ``obs_dim``, i.e. ``g`` is the ODE residual itself; pass a smaller
            value when ``g`` stacks further (e.g. conservation) rows after it.

    Example:
        >>> def custom_g(state, *, t):
        ...     # Nonlinear observation: squared position + velocity
        ...     return jnp.array([state[0]**2, state[1]])
        >>> measure = BlackBoxMeasurement(custom_g, state_dim=4, obs_dim=2)
    """

    def __init__(
        self,
        g_func: Callable[..., Array],
        state_dim: int,
        obs_dim: int,
        noise: float | ArrayLike = 0.0,
        *,
        ode_dim: int | None = None,
    ):
        if not callable(g_func):
            raise TypeError("'g_func' must be callable.")
        if not isinstance(state_dim, int) or state_dim <= 0:
            raise ValueError("'state_dim' must be a positive integer.")
        if not isinstance(obs_dim, int) or obs_dim <= 0:
            raise ValueError("'obs_dim' must be a positive integer.")
        if ode_dim is None:
            ode_dim = obs_dim
        if not isinstance(ode_dim, int) or not 0 < ode_dim <= obs_dim:
            raise ValueError(
                f"'ode_dim' must be an integer in [1, obs_dim={obs_dim}]; "
                f"got {ode_dim!r}."
            )

        self._g_func = g_func
        self._state_dim = state_dim
        self._obs_dim = obs_dim
        self._ode_dim = ode_dim
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
        _check_noise_covariance(self._R, "noise")

    @property
    def ode_dim(self) -> int:
        """Number of leading rows of ``g`` that drive diffusion calibration.

        The filter loops slice ``H[:ode_dim]`` / ``mz[:ode_dim]`` for the
        ``sigma^2`` estimate (see the ``ode_dim`` constructor argument).
        """
        return self._ode_dim

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
        _check_noise_covariance(self._R, "R")

    def g(self, state: Array, *, t: ArrayLike) -> Array:
        """Evaluate the measurement function.

        Args:
            state: State vector of length state_dim.
            t: Current time.

        Returns:
            Observation vector of length obs_dim.
        """
        state_arr = self._validate_state(state)
        return self._g_func(state_arr, t=t)

    def jacobian_g(self, state: Array, *, t: ArrayLike) -> Array:
        """Compute Jacobian of the measurement function via autodiff.

        Args:
            state: State vector of length state_dim.
            t: Current time.

        Returns:
            Jacobian matrix of shape (obs_dim, state_dim).
        """
        state_arr = self._validate_state(state)
        return self._jacobian_g_func(state_arr, t)

    def get_noise(self, *, t: ArrayLike) -> Array:
        """Return the square root of the measurement noise covariance.

        Returns an upper-triangular matrix L such that L.T @ L ≈ R.

        Args:
            t: Current time (unused, included for API consistency).

        Returns:
            Upper-triangular square root of the noise covariance matrix.
        """
        return _safe_cholesky_sqr(self._R, self._obs_dim)

    def linearize(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
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
        use_autodiff_jacobian: Whether to compute ``J_sigma`` by autodiff. Defaults
            to autodiff exactly when no ``sigma_jacobian`` is given; passing
            ``True`` together with a ``sigma_jacobian``, or ``False`` without
            one, raises.
        sigma_jacobian: Optional explicit Jacobian function for ``sigma``; used
            instead of autodiff when given.

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
        use_autodiff_jacobian: bool | None = None,
        sigma_jacobian: Callable[[Array], Array] | None = None,
    ):
        if not callable(sigma):
            raise TypeError("'sigma' must be callable.")
        if use_autodiff_jacobian is None:
            use_autodiff_jacobian = sigma_jacobian is None
        if use_autodiff_jacobian and sigma_jacobian is not None:
            raise ValueError(
                "Got both 'use_autodiff_jacobian=True' and a 'sigma_jacobian'; "
                "pass only one of them."
            )
        if not use_autodiff_jacobian and sigma_jacobian is None:
            raise ValueError(
                "Must provide 'sigma_jacobian' when 'use_autodiff_jacobian=False'."
            )

        self._base = base_model
        self._sigma = sigma
        self._use_autodiff = use_autodiff_jacobian
        self._jacobian_sigma = (
            jax.jacfwd(sigma) if sigma_jacobian is None else sigma_jacobian
        )

    @property
    def R(self) -> Array:
        """Measurement noise covariance matrix (delegated to base model)."""
        return self._base.R

    @R.setter
    def R(self, value: ArrayLike) -> None:
        """Set the measurement noise covariance matrix on the base model."""
        self._base.R = value

    @property
    def ode_dim(self) -> int:
        """Number of ODE-defect rows (delegated to the base model).

        A state transformation reparameterises *where* the residual is
        evaluated; it does not change how many of the rows are ODE defect and
        how many are constraints or observations. The filter loops read this to
        slice ``H_t[:ode_dim]`` for calibration, so without the delegation a
        transformed measure cannot be run through them at all.

        Both base types declare it (a ``BlackBoxMeasurement`` through its
        ``ode_dim`` argument).
        """
        return self._base.ode_dim

    @property
    def E_args(self) -> Array:
        """Argument selector, delegated to the base model.

        Valid whenever ``sigma`` acts within the coordinates the base already
        reads -- the composed residual ``g(sigma(state))`` is then still affine
        in the state except through ``E_args @ state``. That covers the usual
        case of a link function on one component (e.g. a positivity transform on
        a latent input).

        It matters for cost, not just tidiness: ``slr_linearize`` falls back to
        the identity when ``E_args`` is absent, which is correct but quadratures
        over the *whole* state -- ``n_nodes ** state_dim`` nodes instead of
        ``n_nodes ** len(args)``. For a joint latent-force state that is the
        difference between 256 nodes and 17 million.

        If ``sigma`` mixes in coordinates the base does not read, override this
        with the union of the two sets.

        Raises:
            AttributeError: If the base model declares no argument selector
                (e.g. a ``BlackBoxMeasurement``). Raising -- rather than
                guessing one -- is what keeps ``slr_linearize``'s identity
                fallback correct, since it probes with ``getattr``.
        """
        if not isinstance(self._base, BaseODEInformation):
            raise AttributeError(
                f"{type(self._base).__name__} base has no 'E_args': it declares no "
                "argument selector to project the quadrature onto."
            )
        return self._base.E_args

    def g(self, state: Array, *, t: ArrayLike) -> Array:
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

    def jacobian_g(self, state: Array, *, t: ArrayLike) -> Array:
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

    def get_noise(self, *, t: ArrayLike) -> Array:
        """Return the square root of the noise covariance (delegated to base model).

        Args:
            t: Current time.

        Returns:
            Upper-triangular square root of the noise covariance matrix.
        """
        return self._base.get_noise(t=t)

    def linearize(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
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
