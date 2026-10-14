"""Base class of the Gauss-Markov priors, and the transition a filter step uses."""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _safe_sqrt

# vf is called positionally with 1 (order 1) or 2 (order 2) derivatives plus a
# keyword-only ``t``; ``Callable[..., Array]`` captures both arities.
VectorField = Callable[..., Array]


class Transition(NamedTuple):
    """One step of a prior, ``x' = A x + b + w`` with ``w ~ N(0, Q_sqr.T @ Q_sqr)``.

    These are the matrices a filter step computes with. For a plain prior they
    act on the state itself (``T_diag`` is ``None``). A preconditioned prior
    returns them in the coordinates ``x_bar = x / T_diag`` (elementwise), in
    which they are better conditioned; the filter changes into those coordinates
    for the prediction and back, so its results are always in the original
    coordinates.

    Attributes:
        A: Transition matrix, shape ``[D, D]``.
        b: Drift, shape ``[D]``.
        Q_sqr: Upper-triangular square-root process noise, shape ``[D, D]``.
        T_diag: Diagonal of the preconditioner ``T(h)`` (``x = T x_bar``), shape
            ``[D]``, or ``None`` for a plain prior.
    """

    A: Array
    b: Array
    Q_sqr: Array
    T_diag: Array | None = None


def check_order_and_dimension(q: int, d: int) -> None:
    """Validate a prior's smoothness order ``q`` and dimension ``d``."""
    if not isinstance(q, int) or isinstance(q, bool):
        raise TypeError("q must be an integer.")
    if q < 0:
        raise ValueError("q must be non-negative.")
    if not isinstance(d, int) or isinstance(d, bool):
        raise TypeError("d must be an integer.")
    if d <= 0:
        raise ValueError("d must be positive.")


def component_matrix(Xi: ArrayLike | None, d: int) -> Array:
    """The component-correlation matrix ``Xi``: the identity by default."""
    xi = np.eye(d) if Xi is None else np.asarray(Xi, dtype=float)
    if xi.shape != (d, d):
        raise ValueError(f"Xi must have shape ({d}, {d}), got {xi.shape}.")
    return xi


def diagonal_flag(xi: Array) -> bool | None:
    """Whether ``xi`` is diagonal; ``None`` when it is traced and cannot be read."""
    try:
        x = onp.asarray(xi)
    except jax.errors.TracerArrayConversionError:
        return None
    return bool(onp.allclose(x - onp.diag(onp.diag(x)), 0.0))


def _selector(q: int, d: int, k: int) -> Array:
    """Selector of the ``k``-th derivative of a derivative-major state."""
    return np.kron(np.eye(q + 1)[k : k + 1], np.eye(d))


class BasePrior(eqx.Module):
    """Base class of the Gauss-Markov priors.

    A prior models a ``d``-dimensional solution ``x(t)`` together with its first
    ``q`` derivatives, stacked derivative-major into the state
    ``Y = [x, x', ..., x^(q)]`` of length ``(q+1)*d``. Over a step of size ``h`` it
    is the linear-Gaussian transition ``Y(t+h) = A(h) Y(t) + b(h) + noise`` with
    process noise covariance ``Q(h) = Q_sqr(h).T @ Q_sqr(h)``. Subclasses define
    ``A`` and ``Q_sqr``; the filters consume ``transition(h)``.

    Priors are ``equinox`` modules: their hyperparameters (``Xi``, a length
    scale, a rate) are pytree leaves, so a prior can be differentiated, vmapped
    or passed through ``equinox.filter_jit`` like any array; the orders and
    dimensions are static.

    Attributes:
        q: Number of modelled derivatives (the smoothness order).
        d: Dimension of the solution ``x``.
        xi: Component-correlation matrix of the driving noise, shape ``[d, d]``.
            It sets the scale of the prior uncertainty (together with the
            calibrated ``sigma^2``).
        xi_diagonal: Whether ``xi`` was diagonal when the prior was built
            (``None`` if it was built from a traced ``Xi``). Static, so the
            per-component calibration can check it under ``jit``, where ``xi``
            itself is traced. Replacing ``xi`` with ``equinox.tree_at`` does not
            update it; build a new prior instead.
    """

    q: eqx.AbstractVar[int]
    d: eqx.AbstractVar[int]
    xi: eqx.AbstractVar[Array]
    xi_diagonal: bool | None = eqx.field(static=True, default=None, kw_only=True)

    @property
    def state_dim(self) -> int:
        """Dimension ``D`` of the state."""
        return (self.q + 1) * self.d

    @property
    def E0(self) -> Array:
        """Selector of the solution ``x``, shape ``[d, D]``."""
        return _selector(self.q, self.d, 0)

    @property
    def E1(self) -> Array:
        """Selector of ``x'``, shape ``[d, D]``."""
        return _selector(self.q, self.d, 1)

    @property
    def E2(self) -> Array | None:
        """Selector of ``x''``, shape ``[d, D]``, or ``None`` if ``q < 2``."""
        return _selector(self.q, self.d, 2) if self.q >= 2 else None

    @abc.abstractmethod
    def A(self, h: ArrayLike) -> Array:
        """Transition matrix over a step ``h``, shape ``[D, D]``."""

    @abc.abstractmethod
    def Q_sqr(self, h: ArrayLike) -> Array:
        """Upper-triangular square root of the process noise: ``Q_sqr.T @ Q_sqr == Q(h)``."""

    def b(self, h: ArrayLike) -> Array:
        """Drift over a step ``h``, shape ``[D]`` (zero for these priors)."""
        return np.zeros(self.state_dim)

    def Q(self, h: ArrayLike) -> Array:
        """Process noise covariance over a step ``h``, shape ``[D, D]``.

        Prefer ``Q_sqr``: at high order the dense ``Q(h)`` is severely
        ill-conditioned, while its square-root factor stays accurate.
        """
        Q_sqr = self.Q_sqr(h)
        return Q_sqr.T @ Q_sqr

    def transition(self, h: ArrayLike) -> Transition:
        """The matrices a filter step of size ``h`` computes with.

        Plain priors return ``A(h)``, ``b(h)`` and ``Q_sqr(h)``; preconditioned
        priors override this to return them in preconditioned coordinates (see
        ``Transition``).
        """
        return Transition(self.A(h), self.b(h), self.Q_sqr(h))

    # ------------------------------------------------------------------
    # The state block: the part of the state the diffusion sigma^2 scales.
    # For a plain prior that is the whole state; a JointPrior overrides.
    # ------------------------------------------------------------------

    @property
    def E0_state(self) -> Array:
        """Selector of the ODE solution in the state.

        The adaptive error estimate measures this part only (Bosch, Tronarp &
        Hennig 2022, Sec. 3). It is ``E0`` for a plain prior.
        """
        return self.E0

    @property
    def xi_state(self) -> Array:
        """``Xi`` of the state block (per-component calibration needs it diagonal)."""
        return self.xi

    @property
    def xi_state_diagonal(self) -> bool | None:
        """Whether ``xi_state`` was diagonal at construction (``None``: unknown)."""
        return self.xi_diagonal

    @property
    def _state_block(self) -> tuple[int, int, int]:
        """``(D_x, d_x, q_x)`` of the block the diffusion scales: its leading columns."""
        return self.state_dim, self.d, self.q

    def _state_scale(self, sigma_sqr: ArrayLike) -> Array:
        """Column scaling ``sqrt(sigma^2)`` of the state block, length ``D_x``."""
        D_x, d_x, q_x = self._state_block
        sigma_sqr = np.asarray(sigma_sqr)
        if sigma_sqr.ndim == 0:
            return np.full(D_x, _safe_sqrt(sigma_sqr))
        if sigma_sqr.shape != (d_x,):
            raise ValueError(
                f"sigma_sqr must be scalar or length-{d_x} vector; "
                f"got shape {sigma_sqr.shape}."
            )
        return np.tile(_safe_sqrt(sigma_sqr), q_x + 1)

    def apply_state_sigma_sqr(self, Q_sqr: Array, sigma_sqr: ArrayLike) -> Array:
        """Scale the state block of a square-root process noise by ``sigma^2``.

        A scalar scales the whole state block; a length-``d`` vector (the
        per-component calibration) scales each component's columns across all
        ``q + 1`` derivatives. A joint prior's input block is left alone
        (Schmidt et al. 2021).
        """
        D_x = self._state_block[0]
        return Q_sqr.at[:, :D_x].multiply(self._state_scale(sigma_sqr)[None, :])

    def apply_state_sigma_to_cov_sqr(self, P_sqr: Array, sigma_sqr: ArrayLike) -> Array:
        """Scale the state block of a square-root covariance by ``sigma^2`` (post hoc)."""
        return self.apply_state_sigma_sqr(P_sqr, sigma_sqr)

    def sample(
        self,
        key: Array,
        ts: ArrayLike,
        *,
        m_0: ArrayLike | None = None,
        P_0_sqr: ArrayLike | None = None,
    ) -> Array:
        """A sample path of the prior at the times ``ts``.

        Exact: each interval is the prior's transition over ``h_i = ts[i+1] -
        ts[i]`` (``ts`` may be non-uniform). The key is split into one key for
        the initial state and one per interval.

        Args:
            key: A ``jax.random`` key.
            ts: Increasing times, shape ``[K]``.
            m_0: Mean of the initial state; required unless the prior has a
                stationary law (``stationary_cov_sqr``), which is then used with
                zero mean.
            P_0_sqr: Square-root covariance of the initial state (with ``m_0``).

        Returns:
            The path, shape ``[K, D]``.

        Raises:
            ValueError: If no initial law is given and the prior has no
                stationary one, or only one of ``m_0`` / ``P_0_sqr`` is given.
        """
        ts = np.asarray(ts, dtype=float)
        if (m_0 is None) != (P_0_sqr is None):
            raise ValueError("Pass both m_0 and P_0_sqr, or neither.")
        if m_0 is None:
            stationary = getattr(self, "stationary_cov_sqr", None)
            if stationary is None:
                raise ValueError(
                    f"{type(self).__name__} has no stationary law: pass m_0 and P_0_sqr."
                )
            m_0, P_0_sqr = np.zeros(self.state_dim), stationary()
        m_0, P_0_sqr = np.asarray(m_0), np.asarray(P_0_sqr)
        keys = jax.random.split(key, ts.shape[0])
        x_0 = m_0 + jax.random.normal(keys[0], m_0.shape) @ P_0_sqr

        def step(x, inputs):
            h, k = inputs
            noise = jax.random.normal(k, x.shape) @ self.Q_sqr(h)
            x_next = self.A(h) @ x + self.b(h) + noise
            return x_next, x_next

        _, path = jax.lax.scan(step, x_0, (np.diff(ts), keys[1:]))
        return np.concatenate([x_0[None], path], axis=0)
