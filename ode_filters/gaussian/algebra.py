"""Square-root Gaussian algebra: marginalize, invert, update, compose.

Every covariance is held as an upper-triangular factor, ``P = P_sqr.T @ P_sqr``,
and never formed. All operations rest on one identity: for factors stacked
vertically,

    [X; Y].T @ [X; Y] = X.T @ X + Y.T @ Y,

so a *sum* of covariances is the Gram matrix of the stacked factors, and a QR
decomposition ``[X; Y] = Q R`` returns its upper-triangular factor ``R`` (the
orthogonal ``Q`` cancels in ``R.T @ R``). For the marginal of ``z = A x + b + w``
with ``x ~ N(m, P)`` and ``w ~ N(0, Q)``,

    Cov(z) = A P A.T + Q = [P_sqr A.T; Q_sqr].T @ [P_sqr A.T; Q_sqr],

which is ``marginalize``. The other operations stack their terms the same way:
``invert`` the Joseph form of the conditional covariance, ``compose`` the
covariance of a chain of two conditionals. The result is symmetric and
positive-semidefinite by construction, whatever the round-off.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import jax.scipy.linalg
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from .types import AffineModel, Conditional, Gaussian

# Jitter that lets the Cholesky through noise-free channels: a zero row/column
# of R (e.g. the noise-free ODE-defect rows) becomes a ~1e-16 diagonal entry.
# It is far below float64 resolution for any nonzero variance, so it does not
# rescue a singular R whose zero-variance direction mixes channels -- that is
# rejected up front by ``_check_noise_covariance``.
_JITTER = 1e-32


def _safe_qr(C: Array) -> Array:
    """Upper-triangular ``R`` factor of ``C``, differentiable at ``C = 0``.

    ``np.linalg.qr`` has an *undefined derivative* at a rank-deficient input: its
    JVP solves with ``R``, whose diagonal has a zero. The value is fine (``R = 0``
    for ``C = 0``) but reverse mode returns ``NaN``, and since a zero cotangent
    times ``NaN`` is still ``NaN``, one degenerate factor poisons the gradient of
    everything downstream -- even quantities that do not depend on it.

    An exactly-zero ``C`` is not a corner case here: it is what the first backward
    conditional of a filter started from a deterministic initial condition *is*.
    ``taylor_mode_initialization`` returns a zero
    ``P_0_sqr``, so at the first step ``invert`` sees a zero cross-term, a
    zero gain, and hence a zero stacked factor.

    Defining the derivative to be **zero** there is not a fudge, it is the correct
    value. Covariances are what the caller ultimately consumes, and
    ``P = P_sqr.T @ P_sqr`` is smooth even where ``P_sqr`` is not: ``P`` is
    quadratic in the factor, so ``dP = 2 P_sqr.T d(P_sqr)`` vanishes at
    ``P_sqr = 0`` whatever finite value ``d(P_sqr)`` takes. The ``NaN`` is the
    ``0 * inf`` of a cancellation that AD cannot see through; substituting a
    well-conditioned stand-in on the degenerate branch (the standard double-``where``
    idiom) lets the cancellation happen.

    Two boundaries. Only an *exactly* zero ``C`` is handled -- a merely
    rank-deficient one (a structurally singular ``Q``, say) still has an undefined
    QR derivative, which needs a rank-revealing factorization. And this repairs a
    degenerate *output* factor, not a degenerate *innovation* covariance:
    ``invert`` divides by the innovation factor in a triangular solve, so a
    singular innovation stays ``NaN`` -- correctly, since conditioning a
    zero-variance state on a zero-variance observation is ill-posed.

    Args:
        C: Stacked square-root factor, shape ``[M, N]`` with ``M >= N`` (both call
            sites stack a state-sized block on top of a noise-sized one).

    Returns:
        Upper-triangular ``R`` of shape ``[N, N]`` with ``C.T @ C = R.T @ R``.
    """
    degenerate = np.all(C == 0.0)
    # Full-column-rank stand-in, taken only on the degenerate branch. The outer
    # `where` zeroes the cotangent reaching the QR there, so nothing about the
    # stand-in leaks into either the value or the derivative.
    stand_in = np.eye(C.shape[0], C.shape[1])
    _, R = np.linalg.qr(np.where(degenerate, stand_in, C))
    return np.where(degenerate, np.zeros_like(R), R)


def _cholesky_upper(M: Array) -> Array:
    """Upper-triangular factor ``U`` with ``U.T @ U == M`` (square-root form).

    Symmetrizes ``M`` first so a tiny non-symmetric residual (e.g. from
    ``expm``) does not break the Cholesky. This is the library's square-root
    convention (``A = A_sqr.T @ A_sqr`` with ``A_sqr`` upper-triangular).
    """
    M = 0.5 * (M + M.T)
    return np.linalg.cholesky(M).T


def _safe_sqrt(x: ArrayLike) -> Array:
    """Elementwise ``sqrt`` whose reverse-mode derivative is finite at zero.

    The derivative of ``sqrt`` is infinite at ``0``. Wherever a diffusion
    estimate is *identically* zero (a residual that vanishes at an equilibrium,
    or a system started at rest) the backward pass would then evaluate
    ``0 * inf = NaN`` and poison every gradient, although the forward value is
    an exact ``0``. The double ``where`` keeps the ``sqrt`` off zero on both the
    forward and the reverse pass, so such entries get the zero gradient they
    should have; positive entries are unchanged, and negative entries still give
    ``NaN`` as ``np.sqrt`` does.
    """
    x = np.asarray(x)
    positive = x > 0.0
    root = np.sqrt(np.where(positive, x, 1.0))
    return np.where(positive, root, np.where(x == 0.0, 0.0, np.nan))


def _safe_cholesky_sqr(R: Array, dim: int) -> Array:
    """Upper-triangular Cholesky of a noise covariance with noise-free channels.

    Returns the upper-triangular square root L such that L.T @ L = R + jitter*I.
    Works for positive-definite ``R`` and for ``R`` with zero rows/columns
    (noise-free channels); see ``_check_noise_covariance`` for the rest.
    """
    return np.linalg.cholesky(R + _JITTER * np.eye(dim)).T


def _check_noise_covariance(R: ArrayLike, name: str) -> None:
    """Reject a concrete noise covariance the solvers cannot factorize.

    ``_safe_cholesky_sqr`` handles positive-definite matrices and noise-free
    channels (zero rows and columns), but a singular matrix whose zero-variance
    direction mixes channels (e.g. ``[[1, 1], [1, 1]]``) or an indefinite one
    would give NaN inside the solve. Skipped for traced values (e.g. when the
    noise is a differentiated parameter).
    """
    try:
        R_host = onp.asarray(R, dtype=float)
    except jax.errors.TracerArrayConversionError:
        return
    jittered = R_host + _JITTER * onp.eye(R_host.shape[0])
    try:
        onp.linalg.cholesky(0.5 * (jittered + jittered.T))
    except onp.linalg.LinAlgError:
        raise ValueError(
            f"{name} must be a symmetric positive-definite covariance; noise-free "
            "channels (zero rows and columns) are allowed, but a singular or "
            "indefinite covariance that mixes channels is not."
        ) from None


def _check_square(name: str, M: Array) -> None:
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {M.shape}.")


def marginalize(A: Array, b: Array, Q_sqr: Array, x: Gaussian) -> Gaussian:
    """Marginal of ``z = A x + b + w`` with ``x ~ N(m, P)`` and ``w ~ N(0, Q)``.

    ``z ~ N(A m + b, A P A.T + Q)``, with the covariance factor from one QR of
    ``[P_sqr A.T; Q_sqr]`` (see the module docstring).

    Args:
        A: Linear map, shape ``[k, n]``.
        b: Offset, shape ``[k]``.
        Q_sqr: Square-root noise covariance, shape ``[k, k]``.
        x: The Gaussian over ``x``, mean shape ``[n]``.

    Returns:
        The Gaussian over ``z``.

    Raises:
        ValueError: If the shapes do not match.
    """
    Q_sqr = np.atleast_2d(Q_sqr)
    P_sqr = np.atleast_2d(x.P_sqr)
    _check_square("Q_sqr", Q_sqr)
    _check_square("P_sqr", P_sqr)
    if A.shape[0] != b.shape[0] or A.shape[0] != Q_sqr.shape[0]:
        raise ValueError(
            f"A has {A.shape[0]} rows, but b has shape {b.shape} and Q_sqr has "
            f"shape {Q_sqr.shape}; all three describe the same output."
        )
    if A.shape[1] != P_sqr.shape[0]:
        raise ValueError(
            f"A has {A.shape[1]} columns but the state has dimension "
            f"{P_sqr.shape[0]} (A: {A.shape}, P_sqr: {P_sqr.shape})."
        )
    mz = A @ x.m + b
    Pz_sqr = _safe_qr(np.concatenate([P_sqr @ A.T, Q_sqr], axis=0))
    return Gaussian(mz, Pz_sqr)


def invert(A: Array, Q_sqr: Array, x: Gaussian, z: Gaussian) -> Conditional:
    """Bayes' rule for ``z = A x + b + w``: the conditional ``p(x | z)``.

    Given the prior ``x`` and the marginal ``z`` (from ``marginalize`` with the
    same ``A``, ``b``, ``Q_sqr``), returns ``x | z ~ N(G z + d, Lambda)`` with the
    gain ``G = P A.T Cov(z)^-1``, ``d = m - G E[z]`` and the Joseph form
    ``Lambda = (I - G A) P (I - G A).T + G Q G.T``. Observing ``z = 0`` gives the
    posterior ``N(d, Lambda)``; in the filters' prediction it is the backward
    conditional ``p(x_n | x_{n+1})``.

    Args:
        A: Linear map, shape ``[k, n]``.
        Q_sqr: Square-root noise covariance of ``w``, shape ``[k, k]``. A
            noise-free map needs an explicit zero factor.
        x: The prior over ``x``.
        z: The marginal over ``z``.

    Returns:
        The conditional ``p(x | z)``.
    """
    n_state = A.shape[1]
    P_sqr = np.atleast_2d(x.P_sqr)
    Pz_sqr = np.atleast_2d(z.P_sqr)
    # G = P A^T Pz^-1 by two triangular solves with the factor of Pz.
    cross = (A @ P_sqr.T) @ P_sqr
    Z = jax.scipy.linalg.solve_triangular(Pz_sqr.T, cross, lower=True)
    G = jax.scipy.linalg.solve_triangular(Pz_sqr, Z, lower=False).T
    d = x.m - G @ z.m
    B = np.eye(n_state) - G @ A
    stacked = np.concatenate(
        [P_sqr @ B.T, (np.atleast_2d(Q_sqr) @ G.T).reshape(-1, n_state)], axis=0
    )
    return Conditional(G, d, _safe_qr(stacked))


def update(model: AffineModel, x: Gaussian) -> tuple[Gaussian, Gaussian]:
    """Condition ``x`` on observing ``z = 0`` under ``z = H x + c + v``.

    Args:
        model: The observation model; the data enter through ``c`` (observing
            ``y`` is ``c = -y``).
        x: The prior.

    Returns:
        ``(posterior, innovation)``: the posterior over ``x`` and the
        predictive distribution of ``z`` (mean ``H m + c``, the residual at the
        prior mean), whose density at zero is the step's evidence.
    """
    innovation = marginalize(model.H, model.c, model.R_sqr, x)
    conditional = invert(model.H, model.R_sqr, x, innovation)
    return Gaussian(conditional.d, conditional.P_sqr), innovation


def compose(outer: Conditional, inner: Conditional) -> Conditional:
    """Compose ``p(x | y)`` (outer) and ``p(y | z)`` (inner) into ``p(x | z)``.

    ``G = G1 G2``, ``d = G1 d2 + d1``, ``P = G1 P2 G1.T + P1``: the marginal of
    the inner conditional's noise pushed through the outer one. Used by the
    adaptive fixed-point smoother to merge the per-step backward conditionals
    of a save interval into one (Kraemer 2025, "Adaptive Probabilistic ODE
    Solvers Without Adaptive Memory Requirements", Eq. 18/19).

    Args:
        outer: ``p(x | y) = N(G1 y + d1, P1)``.
        inner: ``p(y | z) = N(G2 z + d2, P2)``.

    Returns:
        ``p(x | z)``.
    """
    noise = marginalize(outer.G, outer.d, outer.P_sqr, Gaussian(inner.d, inner.P_sqr))
    return Conditional(outer.G @ inner.G, noise.m, noise.P_sqr)


def log_likelihood(innovation: Gaussian) -> Array:
    """Log-density at zero of the innovation ``N(mz, Pz)``: ``log N(0; mz, Pz)``.

    The evidence an update contributes when it observes ``z = 0``.
    """
    Pz_sqr = innovation.P_sqr
    log_det = 2.0 * np.sum(np.log(np.abs(np.diag(Pz_sqr))))
    v = jax.scipy.linalg.solve_triangular(Pz_sqr.T, innovation.m, lower=True)
    k = innovation.m.shape[0]
    return -0.5 * (k * np.log(2 * np.pi) + log_det + v @ v)
