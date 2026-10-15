"""Black-box and transformed measurement models."""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _safe_cholesky_sqr
from .noise import noise_matrix
from .ode_information import BaseODEInformation, check_state
from .protocol import MeasurementModel


class BlackBoxMeasurement(eqx.Module):
    """A measurement model from an arbitrary residual ``g(state, *, t)``.

    The Jacobian comes from automatic differentiation. As a solver's ``measure``,
    ``g`` plays the role of the ODE residual: the filter conditions on
    ``g(state) = 0``.

    Args:
        g_func: Residual ``g(state, *, t) -> [obs_dim]``, differentiable in JAX.
        state_dim: Dimension of the state.
        obs_dim: Dimension of the residual.
        noise: Noise of the residual: a variance, per-row variances or a
            covariance (default zero).
        ode_dim: Number of leading rows of ``g`` that are ODE residual; they
            alone drive the diffusion calibration. Defaults to ``obs_dim``;
            pass a smaller value when ``g`` stacks further rows (e.g.
            conservation laws) after the ODE residual.

    Example:
        >>> def custom_g(state, *, t):
        ...     return np.array([state[1] - state[0] ** 2])
        >>> measure = BlackBoxMeasurement(custom_g, state_dim=3, obs_dim=1)
    """

    g_func: Callable[..., Array] = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    ode_dim: int = eqx.field(static=True)
    R: Array

    def __init__(
        self,
        g_func: Callable[..., Array],
        state_dim: int,
        obs_dim: int,
        noise: ArrayLike = 0.0,
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
        self.g_func = g_func
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.ode_dim = ode_dim
        self.R = noise_matrix(noise, obs_dim, "noise")

    def g(self, state: Array, *, t: ArrayLike) -> Array:
        """The residual at ``state``, shape ``[obs_dim]``."""
        return self.g_func(self._validate_state(state), t=t)

    def jacobian_g(self, state: Array, *, t: ArrayLike) -> Array:
        """The Jacobian of the residual, shape ``[obs_dim, state_dim]``."""
        state = self._validate_state(state)
        return jax.jacrev(lambda s: self.g_func(s, t=t))(state)

    def noise_sqr(self, *, t: ArrayLike) -> Array:
        """Upper-triangular square root of the noise covariance."""
        return _safe_cholesky_sqr(self.R, self.obs_dim)

    def with_noise(self, noise: ArrayLike) -> BlackBoxMeasurement:
        """A copy with the noise ``noise`` (variance, per-row variances or matrix)."""
        return eqx.tree_at(
            lambda m: m.R, self, noise_matrix(noise, self.obs_dim, "noise")
        )

    def linearize(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
        """First-order linearization ``(H, c)`` from one VJP pass.

        Args:
            state: Linearization point.
            t: Time.

        Returns:
            ``(H, c)`` with ``H x + c`` the first-order expansion of ``g``.
        """
        state = self._validate_state(state)
        value, vjp = jax.vjp(lambda s: self.g_func(s, t=t), state)
        H = jax.vmap(lambda v: vjp(v)[0])(np.eye(self.obs_dim))
        return H, value - H @ state

    def _validate_state(self, state: Array) -> Array:
        return check_state(state, self.state_dim)


class TransformedMeasurement(eqx.Module):
    """A measurement model evaluated at a transformed state: ``g(sigma(state))``.

    The Jacobian follows the chain rule, ``J_g(sigma(state)) @ J_sigma(state)``.
    The noise is the base model's, unchanged: it must already be expressed in
    the base model's (post-transform) coordinates.

    Uses: a link function on a component (e.g. a positivity transform of a
    latent input), a nonlinear change of coordinates, feature extraction.

    Args:
        base: The measurement model evaluated at the transformed state.
        sigma: The transformation ``sigma(state) -> state``.
        sigma_jacobian: Its Jacobian; computed by automatic differentiation when
            not given.

    Example:
        >>> def softmax_transform(state):
        ...     return state.at[:3].set(jax.nn.softmax(state[:3]))
        >>> transformed = TransformedMeasurement(
        ...     ODEInformation(vf, E0, E1), softmax_transform
        ... )
    """

    base: MeasurementModel
    sigma: Callable[[Array], Array] = eqx.field(static=True)
    sigma_jacobian: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(
        self,
        base: MeasurementModel,
        sigma: Callable[[Array], Array],
        sigma_jacobian: Callable[[Array], Array] | None = None,
    ):
        if not callable(sigma):
            raise TypeError("'sigma' must be callable.")
        self.base = base
        self.sigma = sigma
        self.sigma_jacobian = (
            jax.jacfwd(sigma) if sigma_jacobian is None else sigma_jacobian
        )

    @property
    def ode_dim(self) -> int:
        """Number of ODE-residual rows, the base model's.

        A transformation changes where the residual is evaluated, not how many
        of its rows are ODE residual.
        """
        return self.base.ode_dim

    @property
    def R(self) -> Array:
        """The base model's noise covariance."""
        return self.base.R  # type: ignore[attr-defined]

    @property
    def E_args(self) -> Array:
        """Argument selector, the base model's.

        Valid whenever ``sigma`` acts within the coordinates the base already
        reads, as a link function on one component does: the composed residual
        is then still affine in the state except through ``E_args @ state``.
        It matters for cost: without it, ``slr_linearize`` integrates over the
        whole state (``n_nodes ** state_dim`` nodes instead of
        ``n_nodes ** len(args)``). If ``sigma`` mixes in coordinates the base
        does not read, subclass and override this with the union of the two.

        Raises:
            AttributeError: If the base declares no argument selector (e.g. a
                ``BlackBoxMeasurement``); ``slr_linearize`` then integrates over
                the full state, which is correct.
        """
        if not isinstance(self.base, BaseODEInformation):
            raise AttributeError(
                f"{type(self.base).__name__} base has no 'E_args': it declares no "
                "argument selector to project the quadrature onto."
            )
        return self.base.E_args

    def g(self, state: Array, *, t: ArrayLike) -> Array:
        """``g(sigma(state))``."""
        return self.base.g(self.sigma(state), t=t)

    def jacobian_g(self, state: Array, *, t: ArrayLike) -> Array:
        """``J_g(sigma(state)) @ J_sigma(state)``."""
        return self.base.jacobian_g(self.sigma(state), t=t) @ self.sigma_jacobian(state)

    def noise_sqr(self, *, t: ArrayLike) -> Array:
        """The base model's noise factor."""
        return self.base.noise_sqr(t=t)

    def with_noise(self, noise: ArrayLike) -> TransformedMeasurement:
        """A copy whose base model carries the noise ``noise``."""
        return eqx.tree_at(lambda m: m.base, self, self.base.with_noise(noise))  # type: ignore[attr-defined]

    def linearize(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
        """First-order linearization of the composed residual.

        Args:
            state: Linearization point (untransformed).
            t: Time.

        Returns:
            ``(H, c)``.
        """
        H = self.jacobian_g(state, t=t)
        return H, self.g(state, t=t) - H @ state
