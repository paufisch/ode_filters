from __future__ import annotations

import jax.numpy as np
import jax.scipy.linalg
from jax import Array


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
    ``P_0_sqr``, so at the first step ``sqr_inversion`` sees a zero cross-term, a
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
    ``sqr_inversion`` divides by ``Sigma_z_sqr`` in a triangular solve, so a
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


def sqr_marginalization(
    A: Array,
    b: Array,
    Q_sqr: Array,
    mu: Array,
    Sigma_sqr: Array,
) -> tuple[Array, Array]:
    """Marginalize out the linear transformation in a Gaussian model using square-root form.

    Computes the marginal distribution of z = Ax + b given p(x) ~ N(mu, Sigma)
    and p(z|x) ~ N(Ax + b, Q). The result is p(z) = N(mu_z, Sigma_z) where:
    - mu_z = A @ mu + b
    - Sigma_z = A @ Sigma @ A.T + Q

    The square-root form is preserved to maintain numerical stability.

    Args:
        A: Linear transformation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q_sqr: Square root of observation noise covariance, shape
            ``[n_obs, n_obs]`` (for a single observation a scalar or length-1
            array also works).
        mu: Prior mean (shape [n_state]).
        Sigma_sqr: Square root of prior covariance (shape [n_state, n_state]).

    Returns:
        Tuple of (mu_z, Sigma_z_sqr) where:
        - mu_z is the marginal mean of z (shape [n_obs])
        - Sigma_z_sqr is the square root of marginal covariance of z (shape [n_obs, n_obs])

    Raises:
        ValueError: If input shapes are incompatible or invalid.
    """
    Q_sqr = np.atleast_2d(Q_sqr)
    Sigma_sqr = np.atleast_2d(Sigma_sqr)

    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[0]} rows but b has shape {b.shape[0]}. "
            "b must have the same number of elements as A has rows."
        )

    if Q_sqr.shape[0] != Q_sqr.shape[1]:
        raise ValueError(
            f"Shape mismatch: Q is expected to be of square shape but has shape {Q_sqr.shape}"
        )

    if Sigma_sqr.shape[0] != Sigma_sqr.shape[1]:
        raise ValueError(
            f"Shape mismatch: Sigma_sqr is expected to be of square shape but has shape {Sigma_sqr.shape}"
        )

    if A.shape[1] != Sigma_sqr.shape[0]:
        raise ValueError(
            f"Shape mismatch: A has {A.shape[1]} columns but Sigma_sqr has "
            f"{Sigma_sqr.shape[0]} rows (A={A.shape}, Sigma_sqr={Sigma_sqr.shape})."
        )

    # Compute marginal statistics
    mu_z = A @ mu + b
    C = np.concatenate([Sigma_sqr @ A.T, Q_sqr], axis=0)
    Sigma_z_sqr = _safe_qr(C)

    return mu_z, Sigma_z_sqr


def sqr_inversion(
    A: Array,
    mu: Array,
    Sigma_sqr: Array,
    mu_z: Array,
    Sigma_z_sqr: Array,
    Q_sqr: Array | None = None,
) -> tuple[Array, Array, Array]:
    """Numerically stable Bayesian inference using square-root representations.

    Performs Bayesian update of a Gaussian given a linear observation model, using
    Cholesky factors and QR decomposition to maintain numerical stability.

    Given p(x) ~ N(mu, Sigma) and p(z|x) ~ N(Ax + b, Q), with the marginal
    p(z) ~ N(mu_z, Sigma_z) from ``sqr_marginalization``, returns the conditional
    p(x | z) = N(K z + d, Lambda) in square-root form. The filters observe the
    residual value ``z = 0``, so there the posterior mean is just ``d``.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        mu: Prior mean (shape [n_state]).
        Sigma_sqr: Square root of prior covariance (shape [n_state, n_state]).
        mu_z: Marginal observation mean (shape [n_obs]).
        Sigma_z_sqr: Square root of marginal observation covariance (shape [n_obs, n_obs]).
        Q_sqr: Square root of the noise covariance ``Q`` that went into
            ``Sigma_z_sqr``. ``None`` means noise-free (``Q = 0``); passing
            ``None`` for a noisy observation gives a wrong (too small) ``Lambda``.

    Returns:
        Tuple ``(K, d, Lambda_sqr)``: the Kalman gain ``K`` (shape
        ``[n_state, n_obs]``), the offset ``d`` of the conditional mean
        ``K z + d`` (shape ``[n_state]``; the posterior mean when ``z = 0`` is
        observed), and the conditional covariance square root ``Lambda_sqr``
        (shape ``[n_state, n_state]``).
    """
    if Q_sqr is None:
        Q_sqr = np.zeros_like(Sigma_z_sqr)

    n_state = A.shape[1]

    Sigma_z_sqr = np.atleast_2d(Sigma_z_sqr)
    # K = Sigma A^T Sigma_z^-1 by two triangular solves with the factor of
    # Sigma_z; the covariance uses the Joseph form (I - K A) Sigma (.)^T + K Q K^T,
    # stacked as factors so a single QR returns its upper-triangular root.
    cross = (A @ Sigma_sqr.T) @ Sigma_sqr
    Z = jax.scipy.linalg.solve_triangular(Sigma_z_sqr.T, cross, lower=True)
    K = jax.scipy.linalg.solve_triangular(Sigma_z_sqr, Z, lower=False).T
    d = mu - K @ mu_z
    B = np.eye(n_state) - K @ A
    C = np.concatenate([Sigma_sqr @ B.T, (Q_sqr @ K.T).reshape(-1, n_state)], axis=0)
    Lambda_sqr = _safe_qr(C)

    return K, d, Lambda_sqr


def compose_backward_conditionals(
    cond_outer: tuple[Array, Array, Array],
    cond_inner: tuple[Array, Array, Array],
) -> tuple[Array, Array, Array]:
    """Compose two affine Gaussian (backward) conditionals into one.

    Given an outer conditional ``p(x | y) = N(G1 y + d1, P1)`` and an inner
    conditional ``p(y | z) = N(G2 z + d2, P2)``, returns the marginalised
    conditional ``p(x | z) = N(G z + d, P)`` with::

        G = G1 @ G2,   d = G1 @ d2 + d1,   P = G1 @ P2 @ G1.T + P1,

    in square-root form (``P = P_sqr.T @ P_sqr``). This is the fixed-point
    smoothing "merge" of two conditionals (Kraemer 2025, "Adaptive Probabilistic
    ODE Solvers Without Adaptive Memory Requirements", Eq. 18/19): composing the
    per-step backward conditionals of an interval into a single conditional lets
    an adaptive smoother store one conditional per *save* interval instead of one
    per (data-dependent) sub-step.

    The mean offset and noise covariance are exactly the marginal of
    ``N(d2, P2)`` pushed through the affine map ``x = G1 (.) + d1``, so the noise
    reuses ``sqr_marginalization`` (its QR keeps the square-root form); only
    the composed linear operator ``G1 @ G2`` is added.

    Args:
        cond_outer: ``(G1, d1, P1_sqr)`` parametrising ``p(x | y)``.
        cond_inner: ``(G2, d2, P2_sqr)`` parametrising ``p(y | z)``.

    Returns:
        ``(G, d, P_sqr)`` parametrising the composed conditional ``p(x | z)``.
    """
    G1, d1, P1_sqr = cond_outer
    G2, d2, P2_sqr = cond_inner
    d, P_sqr = sqr_marginalization(G1, d1, P1_sqr, d2, P2_sqr)
    return G1 @ G2, d, P_sqr
