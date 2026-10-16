"""Matern priors: ``MaternPrior`` and ``PrecondMaternPrior``.

A Matern process of smoothness ``nu = q + 1/2`` and length scale ``l`` has the
textbook covariance ``k(r) = 2^(1-nu) / Gamma(nu) (sqrt(2 nu) r / l)^nu
K_nu(sqrt(2 nu) r / l)`` (Rasmussen & Williams 2006, Sec. 4.2.1, eq. 4.14),
e.g. ``k(r) = (1 + sqrt(3) r / l) exp(-sqrt(3) r / l)`` for ``q = 1`` (eq.
4.17). It is the stationary solution of the SDE ``(d/dt + lambda)^(q+1) u = w``
with ``lambda = sqrt(2 nu) / l = sqrt(2q+1) / l`` (Saerkkae, Solin & Hartikainen
2013, Example 1): on ``[u, u', ..., u^(q)]`` a companion drift ``F`` whose last
row holds ``-C(q+1, j) lambda^(q+1-j)``. In the coordinates ``D_lam^-1 Y`` with
``D_lam = diag(1, lambda, ..., lambda^q)`` the drift becomes ``lambda M`` with
the fixed binomial companion ``M`` and the diffusion ``lambda c_q e_q e_q.T``:
all of the length scale's dynamic range sits in the diagonal ``D_lam``. Every
quantity here is computed in those coordinates and mapped back, which keeps it
finite at short length scales and high order.

References:
    C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine
    Learning. MIT Press, 2006, Sec. 4.2.1.

    S. Saerkkae, A. Solin and J. Hartikainen. Spatiotemporal learning via
    infinite-dimensional Bayesian filtering and smoothing. IEEE Signal
    Processing Magazine 30(4), 2013.
"""

from __future__ import annotations

from math import comb, factorial

import equinox as eqx
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.scipy.linalg import expm
from jax.typing import ArrayLike

from ..gaussian.algebra import _cholesky_upper
from ._constants import cached_constants
from ._lti import QUAD_NODES, _gauss_legendre, _sqr_noise_via_quadrature
from .base import (
    BasePrior,
    Transition,
    check_order_and_dimension,
    component_matrix,
    diagonal_flag,
)
from .iwp import _iwp_T_diag


@cached_constants
def _matern_constants(q: int) -> tuple[onp.ndarray, float, onp.ndarray]:
    """Length-scale-free pieces of the scalar Matern prior, built once per ``q``.

    Returns ``(M, c, P_n_sqr)``: the binomial companion ``M`` (drift
    ``lambda M`` in normalized coordinates), the diffusion constant ``c``
    (diffusion ``lambda c e_q e_q.T``, which normalizes the stationary variance
    of ``u`` to one), and the upper-triangular square root of the stationary covariance in
    normalized coordinates, which solves ``M P + P M.T + c e_q e_q.T = 0`` and so
    does not depend on the length scale.
    """
    dim = q + 1
    M = onp.zeros((dim, dim))
    M[onp.arange(q), onp.arange(1, dim)] = 1.0
    M[q, :] = [-comb(dim, j) for j in range(dim)]
    c = factorial(q) ** 2 / factorial(2 * q) * 2.0 ** (2 * q + 1)
    C = onp.zeros((dim, dim))
    C[q, q] = c
    # Lyapunov equation by vectorization (row-major vec): (M kron I + I kron M).
    eye = onp.eye(dim)
    P_n = onp.linalg.solve(onp.kron(M, eye) + onp.kron(eye, M), -C.reshape(-1))
    P_n = P_n.reshape(dim, dim)
    P_n = 0.5 * (P_n + P_n.T)
    return M, c, onp.linalg.cholesky(P_n).T


def _rate(length_scale: ArrayLike, q: int) -> Array:
    """The SDE rate ``lambda = sqrt(2 nu) / length_scale``, ``nu = q + 1/2``."""
    return np.sqrt(2.0 * q + 1.0) / np.asarray(length_scale, dtype=float)


class MaternPrior(BasePrior):
    """Matern Gauss-Markov prior: a stationary process with a length scale.

    Each of the ``d`` components is a Matern process of smoothness
    ``nu = q + 1/2`` with unit marginal variance; ``Xi`` correlates them and sets
    their scale, ``Q(h) = kron(Q_scalar(h), Xi)``. Unlike the IWP it reverts to
    zero over the length scale. The square-root process noise is a square-root
    matrix-fraction decomposition (Bosch, Hennig & Tronarp 2023), which stays
    accurate up to the float64 order ceiling (``q ~ 18``).

    The length scale ``l`` is the textbook one (Rasmussen & Williams 2006, Sec.
    4.2.1, eq. 4.14): the correlation of a component at lag ``tau`` is the Matern
    kernel in ``r = sqrt(2 nu) tau / l``, i.e. ``exp(-tau / l)`` for ``q = 0``,
    ``(1 + r) exp(-r)`` for ``q = 1`` and ``(1 + r + r^2 / 3) exp(-r)`` for
    ``q = 2`` (eq. 4.17). The SDE rate is ``lambda = sqrt(2 nu) / l`` (Saerkkae,
    Solin & Hartikainen 2013, Example 1).

    Args:
        q: Smoothness order (``nu = q + 1/2``).
        d: Dimension of the solution.
        length_scale: Length scale ``l`` of the process, in the textbook
            convention above (SDE rate ``lambda = sqrt(2q+1) / l``).
        Xi: Component-correlation matrix, shape ``[d, d]``; defaults to the
            identity. Per-component calibration requires it diagonal.
        n_quad: Gauss-Legendre nodes of the square-root decomposition (must
            exceed ``q``; the default is robust to the order ceiling).
    """

    q: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    xi: Array
    length_scale: Array
    n_quad: int = eqx.field(static=True)

    def __init__(
        self,
        q: int,
        d: int,
        length_scale: ArrayLike,
        Xi: ArrayLike | None = None,
        *,
        n_quad: int = QUAD_NODES,
    ):
        check_order_and_dimension(q, d)
        _gauss_legendre(n_quad, q)  # validates n_quad
        self.q, self.d, self.n_quad = q, d, n_quad
        self.xi = component_matrix(Xi, d)
        self.xi_diagonal = diagonal_flag(self.xi)
        self.length_scale = np.asarray(length_scale, dtype=float)

    def _normalized(self) -> tuple[Array, Array, Array]:
        """``(F_bar, g_bar, D_lam diagonal)`` of the normalized coordinates."""
        M, c, _ = _matern_constants(self.q)
        lam = _rate(self.length_scale, self.q)
        g_bar = np.zeros((self.q + 1, 1)).at[self.q, 0].set(np.sqrt(c * lam))
        return lam * np.asarray(M), g_bar, lam ** np.arange(self.q + 1, dtype=float)

    def _scalar_A(self, h: ArrayLike) -> Array:
        F_bar, _, scale = self._normalized()
        return scale[:, None] * expm(F_bar * h) / scale[None, :]

    def _scalar_Q_sqr(self, h: ArrayLike) -> Array:
        F_bar, g_bar, scale = self._normalized()
        nodes, weights = _gauss_legendre(self.n_quad, self.q)
        R_bar = _sqr_noise_via_quadrature(F_bar, g_bar, nodes, weights, h)
        return R_bar * scale[None, :]

    def A(self, h: ArrayLike) -> Array:
        """Transition matrix ``kron(expm(F h), I_d)``."""
        return np.kron(self._scalar_A(h), np.eye(self.d))

    def Q_sqr(self, h: ArrayLike) -> Array:
        """``kron(R(h), chol(Xi))`` with ``R`` the scalar noise factor."""
        return np.kron(self._scalar_Q_sqr(h), _cholesky_upper(self.xi))

    def stationary_cov_sqr(self) -> Array:
        """Upper-triangular square root of the stationary covariance.

        The covariance of ``[u, u', ..., u^(q)]`` under the stationary law, for
        all ``d`` components: ``kron(P_scalar, Xi)``, whose top-left entry is
        ``Xi`` (unit marginal variance per component). Computed in normalized
        coordinates, where it does not depend on the length scale, so it is
        finite at short length scales and differentiable in ``length_scale``
        and ``Xi``.
        """
        _, _, P_n_sqr = _matern_constants(self.q)
        _, _, scale = self._normalized()
        scalar = np.asarray(P_n_sqr) * scale[None, :]
        return np.kron(scalar, _cholesky_upper(self.xi))


class PrecondMaternPrior(MaternPrior):
    """``MaternPrior`` computed in step-size-preconditioned coordinates.

    The process is the same as ``MaternPrior``, and ``A(h)``, ``Q(h)``,
    ``Q_sqr(h)`` and the stationary law are those of ``MaternPrior``. The
    filters, however, compute each step in the coordinates
    ``x_bar = T(h)^-1 x``, with the diagonal preconditioner of ``PrecondIWP``.
    Unlike for the IWP, the preconditioned transition and noise still depend on
    ``h`` (they tend to the IWP constants as ``h -> 0``).

    Args:
        q: Smoothness order (``nu = q + 1/2``).
        d: Dimension of the solution.
        length_scale: Length scale ``l`` of the process, as for ``MaternPrior``.
        Xi: Component-correlation matrix, shape ``[d, d]``.
        n_quad: Gauss-Legendre nodes of the square-root decomposition.
    """

    def T(self, h: ArrayLike) -> Array:
        """The diagonal preconditioner ``T(h)`` of ``PrecondIWP``, ``[D, D]``."""
        return np.diag(np.repeat(_iwp_T_diag(self.q, h), self.d))

    def transition(self, h: ArrayLike) -> Transition:
        """``T^-1 A T`` and ``Q_sqr T^-T`` with the preconditioner ``T(h)``.

        Scaling the columns of the upper-triangular square-root factor by the
        diagonal ``T^-1`` keeps it upper triangular.
        """
        t = _iwp_T_diag(self.q, h)
        A_bar = self._scalar_A(h) * t[None, :] / t[:, None]
        R_bar = self._scalar_Q_sqr(h) / t[None, :]
        return Transition(
            np.kron(A_bar, np.eye(self.d)),
            self.b(h),
            np.kron(R_bar, _cholesky_upper(self.xi)),
            np.repeat(t, self.d),
        )
