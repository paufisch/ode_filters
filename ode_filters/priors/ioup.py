"""Integrated Ornstein-Uhlenbeck prior: ``IOUPPrior``."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as np
from jax import Array
from jax.scipy.linalg import expm
from jax.typing import ArrayLike

from ..gaussian.algebra import _cholesky_upper
from ._lti import QUAD_NODES, _gauss_legendre, _sqr_noise_via_quadrature
from .base import BasePrior, check_order_and_dimension, component_matrix, diagonal_flag


class IOUPPrior(BasePrior):
    """Integrated Ornstein-Uhlenbeck Process prior (probabilistic exponential integrator).

    Like ``IWP``, IOUP is a ``q``-times integrated Gauss-Markov prior, but its highest
    derivative follows linear (Ornstein-Uhlenbeck) dynamics ``dY^(q) = R Y^(q) dt + dW``
    with a *rate* ``R``. Choosing ``R`` to be (part of) the ODE's linear dynamics bakes
    those dynamics into the prior, turning the solver into a probabilistic exponential
    integrator: the prior mean integrates the linear part exactly, so the filter only
    has to correct the nonlinear remainder (Bosch, Hennig & Tronarp, NeurIPS 2023).
    ``IWP`` is the ``R = 0`` special case.

    IOUP is a *drop-in prior*: the measurement model (the standard ODE information
    operator on the full vector field) and the filter/smoother recursion are unchanged.
    For a constant rate, ``A(h) = exp(F h)`` and the square-root process noise
    ``Q_sqr(h)`` are computed per step size like any other prior, the latter via the
    same square-root matrix-fraction decomposition as ``MaternPrior``
    (``_sqr_noise_via_quadrature``), never forming the dense ``Q(h)``.

    Note:
        The diffusion fit by ``calibration="dynamic"/"diagonal"`` is estimated on the
        IOUP residual, which is the nonlinear remainder and not the full vector field
        as under ``IWP``. The calibrated diffusion therefore has a different scale
        than under ``IWP``, which matters only when a calibration is compared or
        transferred across priors. A full matrix
        rate couples the output dimensions, so ``A``/``Q`` are not Kronecker-separable
        (a dense ``(q+1)*d`` matrix exponential); scalar/diagonal rates are separable in
        principle but are assembled densely here for a single uniform implementation.

    Args:
        q: Number of derivatives (smoothness order), as for ``IWP``.
        d: State dimension.
        rate: OU rate of the highest derivative: a scalar (uniform), a length-``d``
            vector (a diagonal rate per dimension), or a ``d x d`` matrix that couples
            the dimensions (e.g. the linear part or a reference Jacobian of the ODE).
        Xi: Optional component-correlation matrix of the driving noise (shape [d, d]).
        n_quad: Number of Gauss-Legendre nodes for the square-root matrix-fraction
            decomposition used by ``Q_sqr``.
    """

    q: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    xi: Array
    rate: Array
    n_quad: int = eqx.field(static=True)

    def __init__(
        self,
        q: int,
        d: int,
        rate: ArrayLike,
        Xi: ArrayLike | None = None,
        *,
        n_quad: int = QUAD_NODES,
    ):
        check_order_and_dimension(q, d)
        _gauss_legendre(n_quad, q)  # validates n_quad
        self.q, self.d, self.n_quad = q, d, n_quad
        self.xi = component_matrix(Xi, d)
        self.xi_diagonal = diagonal_flag(self.xi)
        self.rate = self._rate_matrix(rate, d)

    @staticmethod
    def _rate_matrix(rate: ArrayLike, d: int) -> Array:
        """Normalize ``rate`` to a ``d x d`` matrix (scalar -> r I, vector -> diag)."""
        r = np.asarray(rate, dtype=float)
        if r.ndim == 0:
            return r * np.eye(d)
        if r.ndim == 1:
            if r.shape != (d,):
                raise ValueError(f"vector rate must have shape ({d},), got {r.shape}.")
            return np.diag(r)
        if r.ndim == 2:
            if r.shape != (d, d):
                raise ValueError(
                    f"matrix rate must have shape ({d}, {d}), got {r.shape}."
                )
            return r
        raise ValueError("rate must be a scalar, length-d vector, or d x d matrix.")

    def _drift_and_dispersion(self) -> tuple[Array, Array]:
        """Companion drift ``F`` and dispersion ``G`` on the derivative-major state.

        Identity shift blocks on the super-diagonal (``dY^(i) = Y^(i+1)``), the
        rate in the bottom-right block (``dY^(q) = R Y^(q) + noise``), and the
        driving noise, correlated by ``Xi``, entering the ``q``-th block only.
        """
        q, d = self.q, self.d
        shift = np.kron(np.eye(q + 1, k=1), np.eye(d))
        F = shift.at[q * d :, q * d :].set(self.rate)
        G = np.zeros((self.state_dim, d)).at[q * d :, :].set(_cholesky_upper(self.xi).T)
        return F, G

    def A(self, h: ArrayLike) -> Array:
        """Transition matrix ``expm(F h)``."""
        F, _ = self._drift_and_dispersion()
        return expm(F * h)

    def Q_sqr(self, h: ArrayLike) -> Array:
        """Square-root process noise by the square-root matrix-fraction decomposition."""
        F, G = self._drift_and_dispersion()
        nodes, weights = _gauss_legendre(self.n_quad, self.q)
        return _sqr_noise_via_quadrature(F, G, nodes, weights, h)
