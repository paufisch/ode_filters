"""Integrated Wiener process priors: ``IWP`` and ``PrecondIWP``."""

from __future__ import annotations

from math import comb, factorial
from operator import index

import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _cholesky_upper
from .base import BasePrior, MatrixFunction, Transition


def _make_iwp_state_matrices(q: int) -> tuple[MatrixFunction, MatrixFunction]:
    """Return callables producing the transition A(h) and diffusion Q(h) matrices.

    Parameters
    ----------
    q : int
        Smoothness order of the integrated Wiener process.

    Returns
    -------
    tuple[callable, callable]
        Functions A(h) and Q(h) that accept a positive scalar h and return
        square numpy arrays of shape (q + 1, q + 1).
    """
    q = index(q)
    if q < 0:
        raise ValueError("q must be a non-negative integer.")

    dim = q + 1

    def A(h: ArrayLike) -> Array:
        mat = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(i, dim):
                mat = mat.at[i, j].set(h ** (j - i) / factorial(j - i))
        return mat

    def Q(h: ArrayLike) -> Array:
        mat = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(dim):
                power = 2 * q + 1 - i - j
                denom = (2 * q + 1 - i - j) * factorial(q - i) * factorial(q - j)
                mat = mat.at[i, j].set((h**power) / denom)
        return mat

    return A, Q


def _make_iwp_precond_state_matrices(
    q: int,
) -> tuple[Array, Array, MatrixFunction]:
    """Return callables producing the transition, diffusion, and scaling matrices."""
    q = index(q)
    if q < 0:
        raise ValueError("q must be a non-negative integer.")

    dim = q + 1

    A_bar = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            n = q - i
            k = q - j
            if 0 <= k <= n:
                A_bar = A_bar.at[i, j].set(comb(n, k))

    Q_bar = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            Q_bar = Q_bar.at[i, j].set(1.0 / (2 * q + 1 - i - j))

    factorials = np.array([float(factorial(q - idx)) for idx in range(dim)])

    def T(h: ArrayLike) -> Array:
        sqrt_h = np.sqrt(h)
        powers = q - np.arange(dim)
        diag_entries = sqrt_h * (h**powers) / factorials
        return np.diag(diag_entries)

    return A_bar, Q_bar, T


def _iwp_precond_q_bar_sqr(q: int) -> Array:
    """Closed-form upper-triangular square root of the preconditioned IWP ``Q_bar``.

    ``Q_bar[i, j] = 1 / (2q+1-i-j)`` is the Gram matrix of the monomials ``s^(q-i)``
    on ``[0, 1]``; a numerical Cholesky of it loses positive-definiteness in float64
    around ``q >= 13`` and returns ``NaN``. Instead build the exact closed-form lower
    factor ``L`` (``L.T @ L == Q_bar``) from integer factorials and re-triangularize
    once with a QR into the library's upper-triangular convention. This stays finite
    and reconstructs ``Q_bar`` to machine precision well past ``q = 20`` (matching
    ``probdiffeq`` -- W. Kahan's Hilbert recurrence -- and ``ProbNumDiffEq.jl``, which
    both use a closed-form IWP factor). Ref: ProbNumDiffEq.jl
    ``make_preconditioned_iwp_transition_cov_lsqrt_1d``.
    """
    dim = q + 1
    factor = onp.zeros((dim, dim))
    for m in range(dim):
        for n in range(m + 1):
            factor[m, n] = onp.sqrt(2 * q - 2 * m + 1) * (
                factorial(q - n) ** 2
                / (factorial(m - n) * factorial(2 * q - n - m + 1))
            )
    return np.linalg.qr(np.asarray(factor), mode="reduced")[1]


def _check_iwp_q_bar_factor(factor: Array, q: int) -> None:
    """Defensive backstop: raise if the IWP ``Q_bar`` square-root factor is non-finite.

    With the closed-form factor (``_iwp_precond_q_bar_sqr``) this stays finite and
    accurate well past ``q = 20``, so this guard is not expected to fire in any
    practical range; it remains a cheap construction-time sanity check so that a
    pathological non-finite factor fails loudly instead of silently propagating a
    ``NaN`` into a solve.

    **Skipped under tracing.** The check needs a concrete value, and a prior built
    inside ``jax.jit`` / ``jax.grad`` has none -- so as an unconditional Python
    ``bool()`` this raised ``TracerBoolConversionError`` for *every* prior
    constructed from traced hyperparameters, which is what gradient-based
    hyperparameter inference does on each objective evaluation. Eager
    construction is where a pathological ``q`` gets introduced in the first
    place, and that path still fails loudly.
    """
    try:
        all_finite = bool(np.all(np.isfinite(factor)))
    except jax.errors.TracerBoolConversionError:
        return  # inside a JIT trace; factor is abstract, skip the check
    if not all_finite:  # pragma: no cover - defensive backstop
        raise ValueError(
            f"IWP order q={q} produced a non-finite square-root factor for the "
            "integrated-Wiener process noise. Reduce q (q <= 20 is reliable in "
            "float64), or use a MaternPrior."
        )


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

    def __init__(self, q: int, d: int, Xi: ArrayLike | None = None):
        super().__init__(q, d, Xi)
        self._A, self._Q = _make_iwp_state_matrices(q)
        # Closed-form square root of Q(h). The dense Hilbert-like Q(h) is
        # severely ill-conditioned for moderate q (entries scale as
        # h^(2q+1-i-j)); factorizing it directly loses precision. Instead use
        # Q(h) = T(h) Q_bar T(h) with T(h) diagonal and Q_bar the constant
        # (h-independent) Hilbert matrix, so its closed-form square-root factor is
        # computed once and the stepsize dependence enters only through the
        # well-behaved diagonal T(h): Q(h)_sqr = kron(Q_bar_sqr @ T(h), chol(xi).T).
        _, _, self._T_scalar = _make_iwp_precond_state_matrices(q)
        self._Q_bar_sqr_scalar = _iwp_precond_q_bar_sqr(q)
        _check_iwp_q_bar_factor(self._Q_bar_sqr_scalar, q)
        self._xi_sqr = _cholesky_upper(self.xi)

    def A(self, h: ArrayLike) -> Array:
        """Return the state transition matrix for step size h.

        Args:
            h: Step size.

        Returns:
            State transition matrix (shape ``[(q+1)*d, (q+1)*d]``).
        """
        return np.kron(self._A(self._validate_h(h)), self._id)

    def b(self, h: ArrayLike) -> Array:
        """Return the drift vector for step size h.

        Args:
            h: Step size.

        Returns:
            Zero drift vector (shape ``[(q+1)*d]``).
        """
        return self._b

    def Q(self, h: ArrayLike) -> Array:
        """Return the diffusion matrix for step size h.

        Args:
            h: Step size.

        Returns:
            Diffusion matrix (shape ``[(q+1)*d, (q+1)*d]``).
        """
        return np.kron(self._Q(self._validate_h(h)), self.xi)

    def Q_sqr(self, h: ArrayLike) -> Array:
        """Closed-form upper-triangular square root of ``Q(h)``.

        ``Q(h)_sqr = kron(chol(Q_bar).T @ T(h), chol(xi).T)`` (see
        ``__init__``); never forms or factorizes the dense ill-conditioned
        ``Q(h)``.
        """
        Q_scalar_sqr = self._Q_bar_sqr_scalar @ self._T_scalar(self._validate_h(h))
        return np.kron(Q_scalar_sqr, self._xi_sqr)


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

    def __init__(self, q: int, d: int, Xi: ArrayLike | None = None):
        super().__init__(q, d, Xi)
        A_bar, _, _ = _make_iwp_precond_state_matrices(q)
        self._A_bar = np.kron(A_bar, self._id)
        # Q is step-size independent in preconditioned coordinates, so its square
        # root is one constant: kron(Q_bar_sqr, chol(Xi)).
        self._Q_bar_sqr = np.kron(self._Q_bar_sqr_scalar, self._xi_sqr)

    def T(self, h: ArrayLike) -> Array:
        """The diagonal preconditioner ``T(h)`` (``x = T x_bar``).

        Args:
            h: Step size.

        Returns:
            Preconditioning matrix, shape ``[(q+1)*d, (q+1)*d]``.
        """
        return np.kron(self._T_scalar(self._validate_h(h)), self._id)

    def transition(self, h: ArrayLike) -> Transition:
        """Step-size independent ``A`` and ``Q_sqr`` with the preconditioner ``T(h)``."""
        T_diag = np.repeat(np.diag(self._T_scalar(self._validate_h(h))), self._dim)
        return Transition(self._A_bar, self._b, self._Q_bar_sqr, T_diag)
