"""Integrated Wiener process priors: ``IWP`` and ``PrecondIWP``."""

from __future__ import annotations

import functools
from math import comb, factorial

import equinox as eqx
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _cholesky_upper
from .base import (
    BasePrior,
    Transition,
    check_order_and_dimension,
    component_matrix,
    diagonal_flag,
)


@functools.lru_cache
def _iwp_constants(q: int) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Step-size-independent pieces of the scalar IWP, built once per ``q``.

    Returns ``(A_bar, Q_bar_sqr, inv_factorials)``: the preconditioned
    transition ``A_bar[i, j] = C(q-i, q-j)``, the upper-triangular square root of
    the preconditioned noise ``Q_bar[i, j] = 1 / (2q+1-i-j)``, and
    ``1 / k!`` for ``k = 0..q``.

    ``Q_bar`` is the Gram matrix of the monomials ``s^(q-i)`` on ``[0, 1]``, a
    Hilbert-type matrix whose numerical Cholesky loses positive-definiteness in
    float64 around ``q >= 13``. Its factor is built instead from the exact
    closed-form lower factor (integer factorials) and re-triangularized once by
    a QR; this stays accurate well past ``q = 20`` (as in probdiffeq and
    ProbNumDiffEq.jl).
    """
    dim = q + 1
    A_bar = onp.zeros((dim, dim))
    for i in range(dim):
        for j in range(i, dim):
            A_bar[i, j] = comb(q - i, q - j)
    lower = onp.zeros((dim, dim))
    for m in range(dim):
        for n in range(m + 1):
            lower[m, n] = onp.sqrt(2 * q - 2 * m + 1) * (
                factorial(q - n) ** 2
                / (factorial(m - n) * factorial(2 * q - n - m + 1))
            )
    Q_bar_sqr = onp.linalg.qr(lower)[1]
    inv_factorials = onp.array([1.0 / factorial(k) for k in range(dim)])
    return A_bar, Q_bar_sqr, inv_factorials


def _iwp_scalar_A(q: int, h: ArrayLike) -> Array:
    """Scalar transition ``A(h)[i, j] = h^(j-i) / (j-i)!`` for ``j >= i``."""
    _, _, inv_factorials = _iwp_constants(q)
    i, j = onp.indices((q + 1, q + 1))
    power = onp.maximum(j - i, 0)
    return np.asarray(
        np.where(j >= i, np.asarray(h) ** power * inv_factorials[power], 0.0)
    )


def _iwp_T_diag(q: int, h: ArrayLike) -> Array:
    """Diagonal of the preconditioner ``T(h)[i, i] = sqrt(h) h^(q-i) / (q-i)!``."""
    _, _, inv_factorials = _iwp_constants(q)
    powers = q - onp.arange(q + 1)
    return np.sqrt(h) * np.asarray(h) ** powers * inv_factorials[powers]


class IWP(BasePrior):
    """Integrated Wiener process prior: the default choice.

    The ``q``-th derivative of the solution is Brownian motion, so between grid
    points the prior behaves like a polynomial of degree ``q`` whose top
    coefficient is driven by noise. ``A(h)`` and the square-root process noise
    ``Q_sqr(h)`` are closed-form.

    Args:
        q: Number of modelled derivatives (the order of the method).
        d: Dimension of the solution ``x``.
        Xi: Component-correlation matrix of the driving noise, shape ``[d, d]``;
            defaults to the identity.
    """

    q: int = eqx.field(static=True)
    d: int = eqx.field(static=True)
    xi: Array

    def __init__(self, q: int, d: int, Xi: ArrayLike | None = None):
        check_order_and_dimension(q, d)
        self.q, self.d = q, d
        self.xi = component_matrix(Xi, d)
        self.xi_diagonal = diagonal_flag(self.xi)

    def A(self, h: ArrayLike) -> Array:
        """Transition matrix ``kron(A_scalar(h), I_d)``."""
        return np.kron(_iwp_scalar_A(self.q, h), np.eye(self.d))

    def Q_sqr(self, h: ArrayLike) -> Array:
        """Closed-form ``kron(Q_bar_sqr @ T(h), chol(Xi))``.

        ``Q(h) = T(h) Q_bar T(h)`` with the diagonal ``T(h)`` and the constant
        Hilbert-type ``Q_bar``, so the step size enters only through the
        well-behaved ``T(h)``; the ill-conditioned dense ``Q(h)`` is never formed.
        """
        _, Q_bar_sqr, _ = _iwp_constants(self.q)
        scalar = np.asarray(Q_bar_sqr) * _iwp_T_diag(self.q, h)[None, :]
        return np.kron(scalar, _cholesky_upper(self.xi))


class PrecondIWP(IWP):
    """``IWP`` computed in step-size-preconditioned coordinates.

    The same process as ``IWP`` -- ``A(h)``, ``Q(h)`` and ``Q_sqr(h)`` are
    ``IWP``'s -- but the filters compute each step in the coordinates
    ``x_bar = T(h)^-1 x``, in which the transition and the process noise no
    longer depend on the step size; all of the ``h`` dependence sits in the
    diagonal ``T(h)`` (Kraemer & Hennig 2020). Results are reported in the
    original coordinates.

    Args:
        q: Number of modelled derivatives (the order of the method).
        d: Dimension of the solution ``x``.
        Xi: Component-correlation matrix of the driving noise, shape ``[d, d]``;
            defaults to the identity.
    """

    def T(self, h: ArrayLike) -> Array:
        """The diagonal preconditioner ``T(h)`` (``x = T x_bar``), ``[D, D]``."""
        return np.diag(np.repeat(_iwp_T_diag(self.q, h), self.d))

    def transition(self, h: ArrayLike) -> Transition:
        """Step-size independent ``A`` and ``Q_sqr`` with the preconditioner ``T(h)``."""
        A_bar, Q_bar_sqr, _ = _iwp_constants(self.q)
        return Transition(
            np.kron(np.asarray(A_bar), np.eye(self.d)),
            self.b(h),
            np.kron(np.asarray(Q_bar_sqr), _cholesky_upper(self.xi)),
            np.repeat(_iwp_T_diag(self.q, h), self.d),
        )
