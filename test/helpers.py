"""Shared test helpers: canonical problems, a dense Kalman oracle, gradient checks.

The dense oracle is the moment-form (``P``, not ``P_sqr``) textbook Kalman
filter, RTS smoother and log-likelihood for a *linear* ODE, where the EK1
linearization is exact. Comparing the square-root solvers against it end to end
pins the actual numbers, not only structural properties (positive
definiteness, symmetry) that a wrong-but-plausible recursion would also have.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import comb, factorial

import jax
import jax.numpy as np
import numpy as onp
from jax import Array

# ---------------------------------------------------------------------------
# Small array helpers
# ---------------------------------------------------------------------------


def cov(P_sqr: Array) -> Array:
    """Covariance ``P = P_sqr.T @ P_sqr``, batched over leading axes."""
    return np.swapaxes(P_sqr, -1, -2) @ P_sqr


def assert_close(actual, expected, *, rtol: float = 1e-9, atol: float = 0.0) -> None:
    """``|actual - expected| <= atol + rtol * max|expected|`` elementwise.

    The tolerance is relative to the largest entry of ``expected``, so arrays
    whose entries span many orders of magnitude (covariances of high
    derivatives) are compared on the scale of the array, not entry by entry.
    """
    # Booleans (e.g. a success flag) compare as 0/1.
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    scale = float(np.max(np.abs(expected))) if expected.size else 0.0
    err = float(np.max(np.abs(actual - expected))) if expected.size else 0.0
    assert err <= atol + rtol * scale, f"max abs error {err:.3e} (scale {scale:.3e})"


def assert_grad_matches_fd(
    f: Callable[[Array], Array],
    theta: Array,
    *,
    eps: float = 1e-5,
    rtol: float = 2e-3,
    atol: float = 1e-7,
) -> None:
    """Assert ``jax.grad(f)(theta)`` matches a central finite difference.

    ``f`` maps a scalar to a scalar; it is jitted once and reused for both the
    gradient and the two finite-difference evaluations.
    """
    f_jit = jax.jit(f)
    g = float(jax.jit(jax.grad(f))(theta))
    fd = float((f_jit(theta + eps) - f_jit(theta - eps)) / (2 * eps))
    assert np.isfinite(g), f"non-finite gradient {g}"
    assert abs(g - fd) <= atol + rtol * abs(fd), f"grad={g:.6e} vs fd={fd:.6e}"


# ---------------------------------------------------------------------------
# Canonical problems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """An ODE ``x' = vf(x, t)`` with an initial value and, if known, a solution."""

    vf: Callable[..., Array]
    x0: Array
    tspan: tuple[float, float]
    solution: Callable[[Array], Array] | None = None

    @property
    def d(self) -> int:
        return int(self.x0.shape[0])


def _logistic_vf(x, *, t):
    return x * (1.0 - x)


def _decay_vf(x, *, t):
    return -x


_LINEAR_2D = np.array([[-0.5, 1.0], [-1.0, -0.5]])


def _linear_2d_vf(x, *, t):
    return _LINEAR_2D @ x


def _linear_2d_solution(ts):
    x0 = np.array([1.0, 0.0])
    return jax.vmap(lambda t: jax.scipy.linalg.expm(_LINEAR_2D * t) @ x0)(ts)


def _sir_vf(x, *, t, beta=0.5, gamma=0.1):
    s, i, _ = x
    return np.array([-beta * s * i, beta * s * i - gamma * i, gamma * i])


def _lotka_volterra_vf(x, *, t, a=1.5, b=1.0, c=3.0, d=1.0):
    return np.array([a * x[0] - b * x[0] * x[1], -c * x[1] + d * x[0] * x[1]])


LOGISTIC = Problem(
    _logistic_vf,
    np.array([0.5]),
    (0.0, 2.0),
    solution=lambda ts: (1.0 / (1.0 + np.exp(-ts)))[:, None],
)
DECAY = Problem(
    _decay_vf, np.array([1.0]), (0.0, 2.0), solution=lambda ts: np.exp(-ts)[:, None]
)
LINEAR_2D = Problem(
    _linear_2d_vf, np.array([1.0, 0.0]), (0.0, 3.0), solution=_linear_2d_solution
)
SIR = Problem(_sir_vf, np.array([0.99, 0.01, 0.0]), (0.0, 10.0))
LOTKA_VOLTERRA = Problem(_lotka_volterra_vf, np.array([1.0, 1.0]), (0.0, 5.0))


# ---------------------------------------------------------------------------
# Dense (moment-form) Kalman oracle for linear models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DenseStep:
    """The linear-Gaussian pieces of one filter step, in moment form.

    The process noise of the step is ``Q`` scaled by the diffusion, which the
    oracle computes itself for ``calibration="dynamic"`` (through
    ``scale_noise``) and leaves at one for ``calibration="none"``.
    """

    A: Array
    Q: Array
    H: Array
    c: Array
    R: Array | None = None
    H_obs: Array | None = None
    c_obs: Array | None = None
    R_obs: Array | None = None


@dataclass(frozen=True)
class DenseResult:
    m: Array
    P: Array
    log_likelihood: Array
    log_likelihood_obs: Array
    sigma_sqr: Array
    m_smooth: Array
    P_smooth: Array


def _gauss_logpdf_zero_mean_residual(r: Array, S: Array) -> Array:
    """``log N(r; 0, S)``."""
    k = r.shape[0]
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * (k * np.log(2 * np.pi) + logdet + r @ np.linalg.solve(S, r))


def _joseph(P: Array, K: Array, H: Array, R: Array | None) -> Array:
    """Joseph-form covariance update ``(I - K H) P (I - K H).T + K R K.T``.

    The short form ``P - K S K.T`` is not symmetric in floating point. Under a
    per-step diffusion that spans orders of magnitude the asymmetry compounds
    over the steps: at ``q = 3`` it cost the oracle seven digits.
    """
    I_KH = np.eye(P.shape[0]) - K @ H
    P = I_KH @ P @ I_KH.T
    return P if R is None else P + K @ R @ K.T


def dense_kalman(
    m0: Array,
    P0: Array,
    steps: list[DenseStep],
    *,
    calibration: str = "none",
    scale_noise: Callable[[Array, Array], Array] | None = None,
    ode_dim: int | None = None,
) -> DenseResult:
    """Moment-form Kalman filter + RTS smoother over the given steps.

    Each step predicts with ``(A, Q)``, conditions on the ODE information
    ``H x + c = 0`` with noise ``R`` (none if ``R`` is ``None``), and then, if the step carries observations,
    on ``H_obs x + c_obs = 0`` with noise ``R_obs`` (so ``c_obs = -y``).

    Args:
        m0: Initial mean.
        P0: Initial covariance.
        steps: One ``DenseStep`` per filter step.
        calibration: ``"none"`` (unit diffusion) or ``"dynamic"`` (the
            per-step quasi-MLE ``r.T (H Q H.T)^-1 r / d`` of the ODE-information
            rows, applied to the step's ``Q`` before predicting).
        scale_noise: ``(Q, sigma_sqr) -> scaled Q``; defaults to ``sigma_sqr * Q``.
            Joint priors scale the state block only.
        ode_dim: Number of leading ODE-information rows that drive the
            calibration; defaults to all rows.
    """
    if scale_noise is None:
        scale_noise = lambda Q, s: s * Q  # noqa: E731
    m, P = m0, P0
    ll = 0.0
    ll_obs = 0.0
    ms, Ps, sigmas, backward = [m0], [P0], [], []
    for step in steps:
        A, H, c = step.A, step.H, step.c
        m_pred = A @ m
        r_pred = H @ m_pred + c
        d = H.shape[0] if ode_dim is None else ode_dim
        HQH = H[:d] @ step.Q @ H[:d].T
        sigma_sqr = r_pred[:d] @ np.linalg.solve(HQH, r_pred[:d]) / d
        Q = scale_noise(step.Q, sigma_sqr) if calibration == "dynamic" else step.Q
        P_pred = A @ P @ A.T + Q

        # A solve, not pinv: pinv's SVD cutoff costs the smoother ~1e-11.
        G = np.linalg.solve(P_pred, A @ P).T
        backward.append((G, m - G @ m_pred, P - G @ P_pred @ G.T))

        S = H @ P_pred @ H.T + (0.0 if step.R is None else step.R)
        K = P_pred @ H.T @ np.linalg.inv(S)
        m = m_pred - K @ r_pred
        P = _joseph(P_pred, K, H, step.R)
        ll = ll + _gauss_logpdf_zero_mean_residual(r_pred, S)

        if step.H_obs is not None:
            r_obs = step.H_obs @ m + step.c_obs
            S_obs = step.H_obs @ P @ step.H_obs.T + step.R_obs
            K_obs = P @ step.H_obs.T @ np.linalg.inv(S_obs)
            m = m - K_obs @ r_obs
            P = _joseph(P, K_obs, step.H_obs, step.R_obs)
            ll_obs = ll_obs + _gauss_logpdf_zero_mean_residual(r_obs, S_obs)

        ms.append(m)
        Ps.append(P)
        sigmas.append(sigma_sqr)

    m_s, P_s = ms[-1], Ps[-1]
    m_smooth, P_smooth = [m_s], [P_s]
    for G, dvec, Pb in reversed(backward):
        m_s = G @ m_s + dvec
        P_s = G @ P_s @ G.T + Pb
        m_smooth.append(m_s)
        P_smooth.append(P_s)

    return DenseResult(
        m=np.stack(ms),
        P=np.stack(Ps),
        log_likelihood=np.asarray(ll),
        log_likelihood_obs=np.asarray(ll_obs),
        sigma_sqr=np.stack(sigmas),
        m_smooth=np.stack(m_smooth[::-1]),
        P_smooth=np.stack(P_smooth[::-1]),
    )


# ---------------------------------------------------------------------------
# The textbook Matern kernel: an independent oracle for the Matern prior
# ---------------------------------------------------------------------------


def matern_kernel_derivatives(
    q: int, length_scale: float, tau: float, n: int
) -> onp.ndarray:
    """``[k(tau), k'(tau), ..., k^(n)(tau)]`` of the unit-variance Matern kernel.

    The textbook closed form for the half-integer smoothness ``nu = q + 1/2``
    (Rasmussen & Williams 2006, eq. 4.16), valid for ``tau >= 0``::

        k(tau) = exp(-r) q! / (2q)! sum_i (q+i)! / (i! (q-i)!) (2 r)^(q-i)

    with ``r = sqrt(2 nu) tau / length_scale``. It is written as
    ``exp(-a tau) P(tau)`` with a polynomial ``P``, whose derivatives follow
    from Leibniz' rule. At ``tau = 0`` they are the right-hand limits, which are
    the derivatives of ``k`` up to the order ``2q`` where ``k`` is smooth.
    """
    a = (2.0 * q + 1.0) ** 0.5 / length_scale
    coef = [
        factorial(q)
        / factorial(2 * q)
        * factorial(q + i)
        / (factorial(i) * factorial(q - i))
        * (2.0 * a) ** (q - i)
        for i in range(q + 1)
    ]
    P = onp.polynomial.Polynomial(coef[::-1])  # coefficient of tau^(q-i)
    return onp.array(
        [
            onp.exp(-a * tau)
            * sum(comb(m, k) * (-a) ** (m - k) * P.deriv(k)(tau) for k in range(m + 1))
            for m in range(n + 1)
        ]
    )


def matern_lag_covariance(q: int, length_scale: float, h: float) -> onp.ndarray:
    """``Cov(Y(t + h), Y(t))`` of the stationary state ``Y = [u, u', ..., u^(q)]``.

    Entry ``[i, j]`` is ``d^i/ds^i d^j/dt^j k(s - t) = (-1)^j k^(i+j)(h)`` at
    ``s - t = h >= 0``; at ``h = 0`` it is the stationary covariance.
    """
    k = matern_kernel_derivatives(q, length_scale, h, 2 * q)
    i, j = onp.indices((q + 1, q + 1))
    return (-1.0) ** j * k[i + j]


def matern_sde(q: int, length_scale: float) -> tuple[Array, Array]:
    """Drift ``F`` and diffusion ``S`` of the Matern state, from the kernel alone.

    For a stationary Gauss-Markov state, ``Cov(Y(t + h), Y(t)) = e^{F h} Sigma``,
    so the derivative at ``h = 0+`` is ``F Sigma``. Its first ``q`` rows are the
    shift ``Y^(i)' = Y^(i+1)``; the last row ``(-1)^j k^(q+j+1)(0+)`` gives the
    last row of ``F``, and the stationary Lyapunov equation
    ``F Sigma + Sigma F.T + S = 0`` the diffusion ``S = s e_q e_q.T`` with
    ``s = -2 (F Sigma)[q, q]``. No companion form or rate formula is assumed.
    """
    D = q + 1
    Sigma = matern_lag_covariance(q, length_scale, 0.0)
    k = matern_kernel_derivatives(q, length_scale, 0.0, 2 * q + 1)
    last = onp.array([(-1.0) ** j * k[q + j + 1] for j in range(D)])  # (F Sigma)[q]
    F = onp.eye(D, k=1)
    F[q] = onp.linalg.solve(Sigma, last)  # Sigma is symmetric
    S = onp.zeros((D, D))
    S[q, q] = -2.0 * last[q]
    return np.asarray(F), np.asarray(S)


def van_loan(F: Array, S: Array, h: float) -> tuple[Array, Array]:
    """``(expm(F h), int_0^h e^{F u} S e^{F.T u} du)`` by Van Loan's block exponential."""
    n = F.shape[0]
    block = np.block([[F, S], [np.zeros_like(F), -F.T]])
    E = jax.scipy.linalg.expm(block * h)
    A = E[:n, :n]
    return A, E[:n, n:] @ A.T


# ---------------------------------------------------------------------------
# Gradient check of a function of several parameters
# ---------------------------------------------------------------------------


def assert_grad_matches_fd_vector(
    f: Callable[[Array], Array],
    theta: Array,
    *,
    eps: float = 1e-5,
    rtol: float = 2e-3,
    atol: float = 1e-7,
) -> None:
    """Assert ``jax.grad(f)(theta)`` matches central finite differences.

    The vector variant of ``assert_grad_matches_fd``: ``f`` maps a vector to a
    scalar, and every component of the gradient must satisfy
    ``|g_i - fd_i| <= atol + rtol |fd_i|``.
    """
    theta = np.asarray(theta, dtype=float)
    f_jit = jax.jit(f)
    g = jax.jit(jax.grad(f))(theta)
    steps = eps * np.eye(theta.shape[0])
    fd = np.array([(f_jit(theta + e) - f_jit(theta - e)) / (2 * eps) for e in steps])
    assert bool(np.all(np.isfinite(g))), f"non-finite gradient {g}"
    err = np.abs(g - fd)
    assert bool(np.all(err <= atol + rtol * np.abs(fd))), f"grad={g} vs fd={fd}"


# ---------------------------------------------------------------------------
# A linear ODE driven by a hidden input, with its solution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiddenProblem:
    """An ODE ``x' = vf(x, u, t)`` driven by a known input ``u(t)``.

    The filters treat ``u`` as hidden (a ``JointPrior`` block); ``hidden``
    and ``solution`` give the true input and solution at given times, shapes
    ``[K, d_u]`` and ``[K, d_x]``.
    """

    vf: Callable[..., Array]
    x0: Array
    tspan: tuple[float, float]
    hidden: Callable[[Array], Array]
    solution: Callable[[Array], Array]


_HIDDEN_B = np.array([[0.0], [1.0]])
_HIDDEN_OMEGA = 1.5


def _hidden_linear_vf(x, u, *, t):
    return _LINEAR_2D @ x + _HIDDEN_B @ u


def _hidden_linear_input(ts):
    return np.sin(_HIDDEN_OMEGA * ts)[:, None]


def _hidden_linear_solution(ts):
    """``x(t)`` from ``expm(M t)``: with the oscillator ``z = (sin, cos)(omega t)``
    that generates the input, ``[x, z]`` solves the linear ODE ``y' = M y``."""
    W = np.array([[0.0, _HIDDEN_OMEGA], [-_HIDDEN_OMEGA, 0.0]])
    M = np.block(
        [[_LINEAR_2D, _HIDDEN_B @ np.array([[1.0, 0.0]])], [np.zeros((2, 2)), W]]
    )
    y0 = np.array([1.0, 0.0, 0.0, 1.0])
    return jax.vmap(lambda t: jax.scipy.linalg.expm(M * t) @ y0)(ts)[:, :2]


# ``x' = L x + B u`` with ``u(t) = sin(1.5 t)`` entering the second component.
HIDDEN_LINEAR = HiddenProblem(
    _hidden_linear_vf,
    np.array([1.0, 0.0]),
    (0.0, 3.0),
    hidden=_hidden_linear_input,
    solution=_hidden_linear_solution,
)
