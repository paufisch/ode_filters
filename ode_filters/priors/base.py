"""Base class of the Gauss-Markov priors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _cholesky_upper, _safe_sqrt

MatrixFunction = Callable[[ArrayLike], Array]
# vf is called positionally with 1 (order 1) or 2 (order 2) derivatives plus a
# keyword-only ``t``; ``Callable[..., Array]`` captures both arities.
VectorField = Callable[..., Array]


class BasePrior(ABC):
    """Base class of the Gauss-Markov priors.

    A prior models a ``d``-dimensional solution ``x(t)`` together with its first
    ``q`` derivatives, stacked derivative-major into the state
    ``Y = [x, x', ..., x^(q)]`` of length ``(q+1)*d``. Over a step of size ``h`` it
    is the linear-Gaussian transition ``Y(t+h) = A(h) Y(t) + b(h) + noise`` with
    process noise covariance ``Q(h) = Q_sqr(h).T @ Q_sqr(h)``. Subclasses define
    ``A``, ``b`` and ``Q``; the filters consume ``Q_sqr``.

    Args:
        q: Number of modelled derivatives (the smoothness order).
        d: Dimension of the solution ``x``.
        Xi: Component-correlation matrix of the driving noise, shape ``[d, d]``;
            defaults to the identity. It sets the scale of the prior uncertainty
            (together with the calibrated ``sigma^2``).
    """

    def __init__(self, q: int, d: int, Xi: ArrayLike | None = None):
        if not isinstance(q, int):
            raise TypeError("q must be an integer.")
        if q < 0:
            raise ValueError("q must be non-negative.")

        if not isinstance(d, int):
            raise TypeError("d must be an integer.")
        if d <= 0:
            raise ValueError("d must be positive.")

        xi = np.eye(d, dtype=float) if Xi is None else np.asarray(Xi, dtype=float)
        if xi.shape != (d, d):
            raise ValueError(f"Xi must have shape ({d}, {d}), got {xi.shape}.")

        self.q = q
        self._dim = d
        self.xi = xi
        self._id = np.eye(d, dtype=xi.dtype)
        self._b = np.zeros(d * (q + 1))
        eye_d = np.eye(d)
        basis = np.eye(q + 1)
        self._E0 = np.kron(basis[0:1], eye_d)
        self._E1 = np.kron(basis[1:2], eye_d)
        self._E2 = np.kron(basis[2:3], eye_d) if q >= 2 else None

    @property
    def E0(self) -> Array:
        """State extraction matrix (shape ``[d, (q+1)*d]``)."""
        return self._E0

    @property
    def E1(self) -> Array:
        """First derivative extraction matrix (shape ``[d, (q+1)*d]``)."""
        return self._E1

    @property
    def E2(self) -> Array | None:
        """Second derivative extraction matrix (shape ``[d, (q+1)*d]``), or None if q < 2."""
        return self._E2

    @staticmethod
    def _validate_h(h: ArrayLike) -> ArrayLike:
        return h

    @abstractmethod
    def A(self, h: ArrayLike) -> Array:
        pass  # pragma: no cover

    @abstractmethod
    def b(self, h: ArrayLike) -> Array:
        pass  # pragma: no cover

    @abstractmethod
    def Q(self, h: ArrayLike) -> Array:
        pass  # pragma: no cover

    def Q_sqr(self, h: ArrayLike) -> Array:
        """Upper-triangular square root of ``Q(h)``: ``Q_sqr.T @ Q_sqr == Q(h)``.

        This is the quantity the square-root filter actually consumes. The
        default factorizes the dense ``Q(h)`` with a (symmetrized) Cholesky;
        subclasses with a closed-form factor (``IWP`` / ``PrecondIWP``) or a
        block structure (``JointPrior``) override this to avoid factorizing a
        dense -- and, for the integrated Wiener process, severely
        ill-conditioned -- matrix at runtime.
        """
        return _cholesky_upper(self.Q(h))

    # ------------------------------------------------------------------
    # Block-aware calibration hooks. Defaults treat the whole state as a
    # single block; ``JointPrior`` / ``PrecondJointPrior`` override.
    # ------------------------------------------------------------------

    @property
    def E0_state(self) -> Array:
        """State-only value projection used by the step-size error norm.

        Bosch, Tronarp, Hennig (2022) sec. 3 recommends that the local
        error vector have the dimension of the ODE solution. For non-joint
        priors the state is everything, so ``E0_state == E0``. Joint priors
        override to return the state-only extractor.
        """
        return self._E0

    @property
    def xi_state(self) -> Array:
        """State-block component-correlation matrix ``Xi`` (shape ``[d, d]``).

        For non-joint priors this is just ``self.xi``. Joint priors override
        to return the state-prior's ``Xi`` — diagonal-mode calibration only
        needs the state block to be (per-component) diagonal; the input
        block can have arbitrary structure.
        """
        return self.xi

    def apply_state_sigma_sqr(self, Q_sqr: Array, sigma_sqr: ArrayLike) -> Array:
        """Apply diffusion ``sigma_sqr`` to the state block of ``Q_sqr``.

        Accepts a scalar (uniform scaling) or a length-``d`` vector
        (per-component scaling, used by ``calibration="diagonal"``). The
        vector path scales each component's columns by ``sqrt(sigma_sqr[i])``
        across all ``q + 1`` derivative orders.

        For non-joint priors the whole ``Q`` is the state block, so the
        scalar path is ``sqrt(sigma_sqr) * Q_sqr``. Joint priors override to
        leave the input block untouched (Schmidt et al. 2021 convention).
        """
        sigma_sqr = np.asarray(sigma_sqr)
        if sigma_sqr.ndim == 0:
            return _safe_sqrt(sigma_sqr) * Q_sqr
        if sigma_sqr.shape != (self._dim,):
            raise ValueError(
                f"sigma_sqr must be scalar or length-{self._dim} vector; "
                f"got shape {sigma_sqr.shape}."
            )
        tiled = np.tile(sigma_sqr, self.q + 1)
        return Q_sqr * _safe_sqrt(tiled)[None, :]

    def apply_state_sigma_to_cov_sqr(self, P_sqr: Array, sigma_sqr: ArrayLike) -> Array:
        """Apply ``sigma_sqr`` to the state block of a full-state covariance sqrt.

        A post-hoc covariance-rescaling utility (multiplies the stored square-root
        covariance by ``sqrt(sigma_sqr)``). Default scales everything; joint priors
        override to scale only the state block.
        """
        return _safe_sqrt(sigma_sqr) * P_sqr
