"""Matern priors: ``MaternPrior`` and ``PrecondMaternPrior``."""

from __future__ import annotations

from math import comb, factorial

import jax.numpy as np
from jax import Array
from jax.scipy.linalg import expm
from jax.typing import ArrayLike

from ..gaussian.algebra import _cholesky_upper
from ._lti import _MATERN_QUAD_NODES, _gauss_legendre, _sqr_noise_via_quadrature
from .base import BasePrior
from .iwp import _make_iwp_precond_state_matrices


def _matern_companion_form(length_scale: float, q: int) -> tuple[Array, Array, Array]:
    """Construct the companion form matrices for a Matern GP prior.

    Parameters
    ----------
    length_scale : float
        Length scale parameter.
    q : int
        Smoothness parameter exponent (nu = q + 1/2, q in 1,2,3,...).

    Returns
    -------
    F : Array
        Drift matrix (shape [D, D]) - companion form matrix.
    L : Array
        Diffusion vector (shape [D]).
    q_coeff : float
        Diffusion coefficient.
    """
    if not isinstance(q, int) or q < 0:
        raise ValueError("Smoothness exponent q must be a non-negative integer.")

    # Compute nu = q + 1/2 and state dimensionality D = nu + 1/2 = q + 1
    D = q + 1

    # Compute lambda = sqrt(2*nu / length_scale) = sqrt((2q + 1) / length_scale)
    lam = np.sqrt((2.0 * q + 1.0) / length_scale)

    # Construct F matrix (companion form)
    F = np.zeros((D, D), dtype=float)

    # Super-diagonal: ones
    for i in range(D - 1):
        F = F.at[i, i + 1].set(1.0)

    # Last row: [-a_j * λ^(D-j) for j = 0, ..., D-1]
    # where a_j = C(D, j) are binomial coefficients
    for j in range(D):
        a_j = comb(D, j)  # Binomial coefficient C(D, j)
        power = D - j  # Power goes from D down to 1
        F = F.at[D - 1, j].set(-a_j * (lam**power))

    # Construct L vector: [0, 0, ..., 0, 1]ᵀ
    L = np.zeros((D, 1), dtype=float).at[-1, 0].set(1.0)

    # Diffusion coefficient: q = sigma^2 [(D-1)!]^2 / (2D-2)! * (2*lambda)^(2D-1)
    # assuming sigma = 1
    numerator = float(factorial(D - 1) ** 2)
    denominator = float(factorial(2 * D - 2))
    q_coeff = (numerator / denominator) * ((2.0 * lam) ** (2 * D - 1))

    return F, L, q_coeff


def _make_matern_sqr_noise(
    length_scale: float, q: int, n_quad: int
) -> tuple[Array, Array, Array, Array, Array]:
    """Precompute the step-size-independent pieces of the square-root matrix-
    fraction decomposition of the (scalar) Matern process noise.

    In the length-scale-normalized coordinates ``D_lam = diag(1, lam, ..., lam**q)``
    (``lam = sqrt((2q+1)/length_scale)``) the companion drift becomes the fixed
    binomial companion ``F_bar = lam * M`` -- independent of ``length_scale`` -- and
    the rank-1 diffusion is ``g_bar @ g_bar.T``. This normalization removes the
    ``lam**(q-j)`` dynamic range of the raw companion form (the dominant driver of
    the high-order conditioning blow-up), the analogue along the length-scale axis
    of the IWP step-size preconditioner.

    Returns ``(F_bar, g_bar, D_lam, gl_nodes, gl_weights)`` with ``gl_nodes`` /
    ``gl_weights`` the Gauss-Legendre nodes/weights on ``[-1, 1]``.

    Raises:
        ValueError: If ``n_quad <= q`` (see ``_gauss_legendre``).
    """
    gl_nodes, gl_weights = _gauss_legendre(n_quad, q)
    D = q + 1
    lam = np.sqrt((2.0 * q + 1.0) / length_scale)

    M = np.zeros((D, D))
    for i in range(D - 1):
        M = M.at[i, i + 1].set(1.0)
    for j in range(D):
        M = M.at[D - 1, j].set(-comb(D, j))
    F_bar = lam * M

    q_coeff = (factorial(D - 1) ** 2 / factorial(2 * D - 2)) * (2.0 * lam) ** (
        2 * D - 1
    )
    s_bar = q_coeff / lam ** (2 * q)
    g_bar = np.zeros((D, 1)).at[D - 1, 0].set(np.sqrt(s_bar))

    D_lam = np.diag(lam ** np.arange(D, dtype=float))

    return F_bar, g_bar, D_lam, gl_nodes, gl_weights


def _matern_scalar_Q_sqr(
    consts: tuple[Array, Array, Array, Array, Array], h: ArrayLike
) -> Array:
    """Upper-triangular square root ``R`` of the scalar Matern process noise
    ``Q(h)`` (the ``[q+1, q+1]`` block, before the ``kron`` with ``Xi``):
    ``R.T @ R == Q(h)``.

    Square-root matrix-fraction decomposition: ``Q(h) = int_0^h e^{F u} S e^{F.T u} du``
    with ``S`` rank one, so a square-root factor is the ``R`` of a QR of the
    Gauss-Legendre-propagated diffusion columns ``sqrt(w_k) e^{F_bar u_k} g_bar``
    (Bosch, Hennig & Tronarp 2023, used there for the IOUP prior). The dense,
    severely ill-conditioned ``Q(h)`` is never formed or factorized, and the
    matrix exponentials are evaluated in the well-scaled normalized coordinates of
    ``_make_matern_sqr_noise``; the factor is mapped back with ``D_lam``. This
    is jit/vmap/grad-safe (reduced-mode QR).
    """
    F_bar, g_bar, D_lam, gl_nodes, gl_weights = consts
    R_bar = _sqr_noise_via_quadrature(F_bar, g_bar, gl_nodes, gl_weights, h)
    return R_bar @ D_lam


def _matern_scalar_A(
    consts: tuple[Array, Array, Array, Array, Array], h: ArrayLike
) -> Array:
    """Scalar Matern transition ``A(h) = expm(F h)`` (the ``[q+1, q+1]`` block).

    Evaluated as ``D_lam expm(lam M h) D_lam^-1`` in the length-scale-normalized
    coordinates of ``_make_matern_sqr_noise``: the companion drift is
    ``F = D_lam (lam M) D_lam^-1``, so the exponential is taken of a matrix whose
    entries are ``O(lam h)`` instead of ``O(lam^(q+1) h)``. This stays finite at
    short length scales, where the ``expm`` of the ``[[F, S], [0, -F.T]]``
    Hamiltonian block (``S ~ lam^(2q+1)``) overflows to NaN.
    """
    F_bar, _, D_lam, _, _ = consts
    D_inv = np.diag(1.0 / np.diag(D_lam))
    return D_lam @ expm(F_bar * h) @ D_inv


class MaternPrior(BasePrior):
    """Matern Gaussian process prior model using block matrix exponential.

    Note:
        ``Q(h)`` is the dense block-matrix-exponential (matrix-fraction) noise,
        which is severely ill-conditioned at high smoothness. The square-root
        factor ``Q_sqr`` that the filter consumes is therefore *not* a
        Cholesky of that dense ``Q``; it is built directly by a square-root
        matrix-fraction decomposition in length-scale-normalized coordinates
        (see ``_matern_scalar_Q_sqr``), which stays finite and accurate up to
        the float64 order ceiling (roughly ``q ~ 18``). The number of quadrature
        nodes is configurable via ``n_quad``. (``Q(h)`` itself still loses
        positive-definiteness around ``q >= 6``; prefer ``Q_sqr`` downstream.)
    """

    def __init__(
        self,
        q: int,
        d: int,
        length_scale: float,
        Xi: ArrayLike | None = None,
        *,
        n_quad: int = _MATERN_QUAD_NODES,
    ):
        """Initialize the Matern prior.

        This creates a Matern process prior for a d-dimensional process whose
        dimensions share the same smoothness ``q`` and length scale and are
        coupled by the component-correlation matrix ``Xi``: the process noise is
        ``kron(Q_scalar(h), Xi)``. ``Xi`` may be any positive-(semi)definite
        matrix -- a diagonal ``Xi`` gives independent per-dimension output
        scales, off-diagonal entries add cross-dimension correlation. (The
        per-component ``"diagonal"`` calibration modes additionally require
        ``Xi`` diagonal; that is checked at solve time.)

        Args:
            q: Smoothness order.
            d: State dimension.
            length_scale: Length scale of the process.
            Xi: Optional component-correlation matrix (shape [d, d]).
            n_quad: Number of Gauss-Legendre nodes for the square-root
                matrix-fraction decomposition used by ``Q_sqr`` (must exceed
                ``q``; the default is robust to the float64 order ceiling).
        """
        super().__init__(q, d, Xi)
        self._F, self._L, self._q = _matern_companion_form(length_scale, q)
        self.S = self._q * self._L @ self._L.T  # Precompute S = L @ Q @ L.T
        self.n = self._F.shape[0]
        # Square-root process noise consumed by the filter: a stable square-root
        # matrix-fraction decomposition (never the Cholesky of the dense, severely
        # ill-conditioned Q(h)). See ``Q_sqr`` / ``_matern_scalar_Q_sqr``.
        self._xi_sqr = _cholesky_upper(self.xi)
        self._sqr_noise = _make_matern_sqr_noise(length_scale, q, n_quad)

        # Defensive checks - _matern_companion_form always returns valid shapes
        if self._F.shape != (self.n, self.n):  # pragma: no cover
            raise ValueError(f"F must be square, got shape {self._F.shape}")
        if self._L.shape[0] != self.n:  # pragma: no cover
            raise ValueError(
                f"L first dimension must match F: {self._L.shape[0]} != {self.n}"
            )
        if self.S.shape != (self.n, self.n):  # pragma: no cover
            raise ValueError(
                f"L @ Q @ L.T must be square with shape ({self.n}, {self.n}), "
                f"got {self.S.shape}"
            )

    def A_and_Q(self, h: ArrayLike) -> tuple[Array, Array]:
        """Scalar ``(A(h), Q(h))``: the ``[q+1, q+1]`` blocks before the ``kron``.

        Both are in the original coordinates; they are *computed* via the
        length-scale-normalized form for stability.

        ``A(h) = D_lam expm(lam M h) D_lam^-1`` -- the companion drift is
        ``F = D_lam (lam M) D_lam^-1`` with ``M`` the fixed binomial companion of
        ``_make_matern_sqr_noise``, so the exponential is taken of a matrix
        whose entries are ``O(lam h)`` instead of ``O(lam^(q+1) h)``.  ``Q(h)`` is
        ``R.T @ R`` from the square-root matrix-fraction decomposition that
        ``Q_sqr`` already uses.

        Both used to come out of one ``expm`` of the ``[[F, S], [0, -F.T]]``
        Hamiltonian block (matrix fraction decomposition).  That
        block carries ``S ~ lam^(2q+1)``, and the scaling-and-squaring exponential
        of it returns NaN once the length scale is short relative to the step --
        measured: ``q = 2``, ``length_scale = 2.6e-3`` (``lam ~ 44``), ``h = 1e-2``,
        i.e. ``lam h = 0.44`` -- which is exactly what a problem re-expressed on a
        unit time interval looks like.  The normalized form is finite there and
        agrees with the block form to ~1e-12 wherever the block form is finite.

        Args:
            h: Step size.

        Returns:
            Tuple ``(A_h, Q_h)`` of the scalar transition matrix and process-noise
            covariance, each of shape ``[q+1, q+1]``.
        """
        h = self._validate_h(h)
        A_h = self._A_scalar(h)
        R = _matern_scalar_Q_sqr(self._sqr_noise, h)
        return A_h, R.T @ R

    def _A_scalar(self, h: ArrayLike) -> Array:
        """``expm(F h)`` for the scalar companion drift, via the normalized form."""
        return _matern_scalar_A(self._sqr_noise, h)

    def A(self, h: ArrayLike) -> Array:
        """Return the state transition matrix for step size h.

        Args:
            h: Step size.

        Returns:
            State transition matrix (shape ``[(q+1)*d, (q+1)*d]``).
        """
        return np.kron(self._A_scalar(self._validate_h(h)), self._id)

    def b(self, h: ArrayLike) -> Array:
        """Return the drift vector for step size h.

        Args:
            h: Step size.

        Returns:
            Zero drift vector (shape ``[(q+1)*d]``).
        """
        # ``self._b`` is the Kron-lifted zero drift of length ``(q+1)*d`` set by
        # ``BasePrior.__init__`` -- using it (rather than ``np.zeros(self.n)``,
        # which has the un-lifted length ``q+1``) keeps ``b`` consistent with the
        # ``(q+1)*d`` state for ``d > 1``.
        return self._b

    def Q(self, h: ArrayLike) -> Array:
        """Return the diffusion matrix for step size h.

        Args:
            h: Step size.

        Returns:
            Diffusion matrix (shape ``[(q+1)*d, (q+1)*d]``).
        """
        _, Q_h = self.A_and_Q(h)
        Q_h = 0.5 * (Q_h + Q_h.T)
        return np.kron(Q_h, self.xi)

    def Q_sqr(self, h: ArrayLike) -> Array:
        """Upper-triangular square root of ``Q(h)``: ``Q_sqr.T @ Q_sqr == Q(h)``.

        ``Q(h)_sqr = kron(R(h), chol(Xi))`` where ``R`` is the scalar Matern noise
        factor from ``_matern_scalar_Q_sqr`` (a square-root matrix-fraction
        decomposition in length-scale-normalized coordinates). Unlike the
        dense-Cholesky default this never forms the severely ill-conditioned
        ``Q(h)``, so it stays finite and accurate up to the float64 order ceiling
        (``q ~ 18``) rather than NaN-ing around ``q >= 6``.
        """
        R = _matern_scalar_Q_sqr(self._sqr_noise, self._validate_h(h))
        return np.kron(R, self._xi_sqr)


class PrecondMaternPrior(BasePrior):
    """Preconditioned Matern Gaussian process prior model.

    Uses the same diagonal preconditioner D(h) as PrecondIWP, but the
    preconditioned transition A_bar(h) and diffusion Q_bar(h) are
    stepsize-dependent (they converge to the IWP constants as h -> 0).

    Note:
        The IWP diagonal preconditioner here only removes the step-size scaling;
        it does *not* address the length-scale/order conditioning of the dense
        ``expm``-derived ``Q``. The square-root factor ``Q_sqr`` therefore uses
        the same stable square-root matrix-fraction decomposition as
        ``MaternPrior`` (see ``_matern_scalar_Q_sqr``), staying finite up to the
        float64 order ceiling (``q ~ 18``) rather than NaN-ing around ``q >= 6``.

    Args:
        q: Smoothness order (nu = q + 1/2).
        d: State dimension.
        length_scale: Length scale of the Matern process.
        Xi: Optional scaling matrix (shape [d, d]).
        n_quad: Number of Gauss-Legendre nodes for the square-root
            matrix-fraction decomposition used by ``Q_sqr`` (must exceed
            ``q``; the default is robust to the float64 order ceiling).
    """

    def __init__(
        self,
        q: int,
        d: int,
        length_scale: float,
        Xi: ArrayLike | None = None,
        *,
        n_quad: int = _MATERN_QUAD_NODES,
    ):
        super().__init__(q, d, Xi)
        self._F, self._L, self._q_coeff = _matern_companion_form(length_scale, q)
        self.S = self._q_coeff * self._L @ self._L.T
        self.n = self._F.shape[0]
        _, _, self._T = _make_iwp_precond_state_matrices(q)
        self._xi_sqr = _cholesky_upper(self.xi)
        self._sqr_noise = _make_matern_sqr_noise(length_scale, q, n_quad)

    def _raw_A_and_Q(self, h: ArrayLike) -> tuple[Array, Array]:
        """Compute raw (non-preconditioned) scalar ``A(h)`` and ``Q(h)``.

        Same stable construction as ``MaternPrior.A_and_Q``: ``A`` from the
        length-scale-normalized exponential (``_matern_scalar_A``) and
        ``Q = R.T @ R`` from the square-root MFD factor. The ``expm`` of the
        ``[[F, S], [0, -F.T]]`` Hamiltonian block that used to be taken here
        overflows to NaN at short length scales (e.g. ``q = 2``,
        ``length_scale = 2.6e-3``, ``h = 1e-2``).

        Args:
            h: Step size.

        Returns:
            Tuple of (A_h, Q_h) scalar matrices (shape [q+1, q+1] each).
        """
        A_h = _matern_scalar_A(self._sqr_noise, h)
        R = _matern_scalar_Q_sqr(self._sqr_noise, h)
        return A_h, R.T @ R

    def _precond_discretise(self, h: ArrayLike) -> tuple[Array, Array]:
        """Compute preconditioned A_bar(h) and Q_bar(h).

        Applies the IWP diagonal preconditioner: A_bar = T^{-1} A T,
        Q_bar = T^{-1} Q T^{-T}.

        Args:
            h: Step size.

        Returns:
            Tuple of (A_bar, Q_bar) scalar matrices (shape [n, n] each).
        """
        h = self._validate_h(h)
        A_h, Q_h = self._raw_A_and_Q(h)
        T_h = self._T(h)
        T_inv = np.diag(1.0 / np.diag(T_h))

        A_bar = T_inv @ A_h @ T_h
        Q_bar = T_inv @ Q_h @ T_inv.T
        return A_bar, Q_bar

    def A(self, h: ArrayLike | None = None) -> Array:
        """Return the preconditioned transition matrix for step size h.

        Args:
            h: Step size (required; Matern preconditioning is stepsize-dependent).

        Returns:
            Preconditioned transition matrix (shape ``[(q+1)*d, (q+1)*d]``).
        """
        if h is None:
            raise ValueError("PrecondMaternPrior.A requires a step size h.")
        A_bar, _ = self._precond_discretise(h)
        return np.kron(A_bar, self._id)

    def Q(self, h: ArrayLike | None = None) -> Array:
        """Return the preconditioned diffusion matrix for step size h.

        Args:
            h: Step size (required; Matern preconditioning is stepsize-dependent).

        Returns:
            Preconditioned diffusion matrix (shape ``[(q+1)*d, (q+1)*d]``).
        """
        if h is None:
            raise ValueError("PrecondMaternPrior.Q requires a step size h.")
        _, Q_bar = self._precond_discretise(h)
        return np.kron(Q_bar, self.xi)

    def Q_sqr(self, h: ArrayLike | None = None) -> Array:
        """Upper-triangular square root of the preconditioned ``Q(h)``.

        Built from the stable scalar Matern noise factor (square-root MFD, see
        ``_matern_scalar_Q_sqr``) rather than a Cholesky of the dense
        ``expm``-derived ``Q``: ``Q(h)_sqr = kron(R(h) @ T_inv, chol(Xi))`` where
        ``R.T @ R`` is the raw scalar ``Q(h)`` and ``T_inv`` applies the IWP
        diagonal preconditioner (scaling the columns of the upper-triangular ``R``
        preserves upper-triangularity). Requires ``h`` (Matern preconditioning is
        step-size-dependent).
        """
        if h is None:
            raise ValueError("PrecondMaternPrior.Q_sqr requires a step size h.")
        h = self._validate_h(h)
        R_raw = _matern_scalar_Q_sqr(self._sqr_noise, h)
        T_inv = np.diag(1.0 / np.diag(self._T(h)))
        return np.kron(R_raw @ T_inv, self._xi_sqr)

    def b(self, h: ArrayLike | None = None) -> Array:
        """Return the zero drift vector.

        Args:
            h: Step size (unused, accepted for interface compatibility).

        Returns:
            Zero drift vector (shape ``[(q+1)*d]``).
        """
        return self._b

    def T(self, h: ArrayLike) -> Array:
        """Return the stepsize-dependent preconditioning transformation.

        Uses the same preconditioner as PrecondIWP.

        Args:
            h: Step size.

        Returns:
            Preconditioning transformation matrix (shape ``[(q+1)*d, (q+1)*d]``).
        """
        return np.kron(self._T(self._validate_h(h)), self._id)
