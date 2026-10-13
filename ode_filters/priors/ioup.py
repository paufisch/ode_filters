"""Integrated Ornstein-Uhlenbeck prior: ``IOUPPrior``."""

from __future__ import annotations

import jax.numpy as np
from jax import Array
from jax.scipy.linalg import expm
from jax.typing import ArrayLike

from ..gaussian.algebra import _cholesky_upper
from ._lti import _MATERN_QUAD_NODES, _gauss_legendre, _sqr_noise_via_quadrature
from .base import BasePrior


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
        IOUP residual -- the *nonlinear remainder*, not the full vector field as under
        ``IWP`` -- so the calibrated diffusion has a different scale than under ``IWP``
        (matters only when comparing/porting calibration across priors). A full matrix
        rate couples the output dimensions, so ``A``/``Q`` are not Kronecker-separable
        (a dense ``(q+1)*d`` matrix exponential); scalar/diagonal rates are separable in
        principle but are assembled densely here for a single uniform implementation.

    Args:
        q: Number of derivatives (smoothness order), as for ``IWP``.
        d: State dimension.
        rate: OU rate of the highest derivative -- scalar (uniform), length-``d`` vector
            (per-dimension diagonal rate), or ``d x d`` matrix (couples dimensions, e.g.
            the linear part / a reference Jacobian of the ODE).
        Xi: Optional component-correlation matrix of the driving noise (shape [d, d]).
        n_quad: Number of Gauss-Legendre nodes for the square-root matrix-fraction
            decomposition used by ``Q_sqr``.
    """

    def __init__(
        self,
        q: int,
        d: int,
        rate: ArrayLike,
        Xi: ArrayLike | None = None,
        *,
        n_quad: int = _MATERN_QUAD_NODES,
    ):
        super().__init__(q, d, Xi)
        self._R = self._build_rate(rate, d)
        self._xi_sqr = _cholesky_upper(self.xi)
        # Companion drift F and dispersion G on the derivative-major state
        # [Y^(0), ..., Y^(q)] (each block d-dimensional): identity shift blocks on the
        # super-diagonal (dY^(i) = Y^(i+1)), the rate R in the bottom-right (q-th
        # derivative) block (dY^(q) = R Y^(q) + noise), and the driving noise -- with
        # component-correlation Xi -- injected only into the q-th block.
        dim = (q + 1) * d
        F = np.zeros((dim, dim))
        for i in range(q):
            F = F.at[i * d : (i + 1) * d, (i + 1) * d : (i + 2) * d].set(self._id)
        F = F.at[q * d :, q * d :].set(self._R)
        self._F = F
        self._G = np.zeros((dim, d)).at[q * d :, :].set(self._xi_sqr.T)
        self._gl_nodes, self._gl_weights = _gauss_legendre(n_quad, q)

    @staticmethod
    def _build_rate(rate: ArrayLike, d: int) -> Array:
        """Normalize ``rate`` to a ``d x d`` matrix (scalar -> r*I, vector -> diag)."""
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

    def A(self, h: ArrayLike) -> Array:
        """Return the state transition matrix ``A(h) = exp(F h)``.

        Args:
            h: Step size.

        Returns:
            State transition matrix (shape ``[(q+1)*d, (q+1)*d]``).
        """
        return expm(self._F * self._validate_h(h))

    def b(self, h: ArrayLike) -> Array:
        """Return the zero drift vector (shape ``[(q+1)*d]``)."""
        return self._b

    def Q_sqr(self, h: ArrayLike) -> Array:
        """Upper-triangular square root of ``Q(h)``: ``Q_sqr.T @ Q_sqr == Q(h)``.

        Square-root matrix-fraction decomposition (see
        ``_sqr_noise_via_quadrature``); never forms the dense ``Q(h)``.
        """
        return _sqr_noise_via_quadrature(
            self._F, self._G, self._gl_nodes, self._gl_weights, self._validate_h(h)
        )

    def Q(self, h: ArrayLike) -> Array:
        """Return the dense diffusion matrix ``Q(h)`` (shape ``[(q+1)*d, (q+1)*d]``)."""
        Q_sqr = self.Q_sqr(h)
        return Q_sqr.T @ Q_sqr
