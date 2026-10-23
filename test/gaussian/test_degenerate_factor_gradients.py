"""Gradients through a square-root factor that is exactly zero.

``np.linalg.qr`` has an undefined derivative at a rank-deficient input, and an
exactly-zero stacked factor is not exotic here: it is what the first backward
conditional *is* when the filter starts from a deterministic initial condition,
which is exactly what :func:`taylor_mode_initialization` produces. Before
``_safe_qr``, that single degenerate factor returned ``NaN`` cotangents, and
because ``0 * NaN`` is ``NaN`` it poisoned the gradient of every downstream
quantity -- including ones that do not depend on it at all.

These tests pin both halves of the contract: the values are untouched, and the
gradients are not merely finite but *correct*, checked against central finite
differences.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp

from ode_filters import (
    IWP,
    ODEInformation,
    TaylorCorrection,
    gaussian_filter,
    rts_smoother,
)
from ode_filters.gaussian import Gaussian, invert, marginalize
from ode_filters.gaussian.algebra import _safe_qr
from ode_filters.priors import taylor_mode_initialization
from test.helpers import assert_close, assert_grad_matches_fd_vector

TSPAN = (0.0, 2.0)
N_STEPS = 20


def logistic(x, *, t):
    return x * (1.0 - x)


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


def test_safe_qr_matches_plain_qr_away_from_zero():
    C = jax.random.normal(jax.random.PRNGKey(0), (7, 4))
    _, R_ref = np.linalg.qr(C)
    assert onp.allclose(_safe_qr(C), R_ref, atol=1e-14)


def test_safe_qr_returns_zero_at_zero():
    C = np.zeros((7, 4))
    R = _safe_qr(C)
    assert R.shape == (4, 4)
    assert onp.allclose(R, 0.0)


def test_safe_qr_gradient_is_finite_and_zero_at_zero():
    """The derivative at a zero factor is zero, which is the correct value.

    ``P = P_sqr.T @ P_sqr`` is quadratic in the factor, so ``dP`` vanishes at
    ``P_sqr = 0`` whatever finite value ``d(P_sqr)`` takes -- the plain QR's
    ``NaN`` is a ``0 * inf`` that never gets to cancel.
    """

    def through_zero(scale):
        return np.sum(_safe_qr(scale * np.zeros((6, 3))))

    grad = jax.grad(through_zero)(1.0)
    assert onp.isfinite(float(grad))
    assert float(grad) == 0.0

    # The plain QR is the thing that fails, so the test is meaningful.
    def plain(scale):
        return np.sum(np.linalg.qr(scale * np.zeros((6, 3)))[1])

    assert onp.isnan(float(jax.grad(plain)(1.0)))


def test_safe_qr_gradient_of_the_reconstructed_covariance_is_correct():
    """Differentiating ``P = R.T @ R`` through a shrinking factor must match FD."""

    def cov_sum(theta):
        C = theta[0] * np.eye(5, 3) + theta[1] * np.ones((5, 3))
        R = _safe_qr(C)
        return np.sum(R.T @ R)

    theta = np.array([0.3, -0.15])
    assert_grad_matches_fd_vector(cov_sum, theta, eps=1e-6, rtol=1e-6, atol=1e-8)


def test_marginalization_takes_a_zero_factor_without_nan_gradients():
    """A deterministic state pushed through a noiseless map: the factor is zero."""
    n = 3
    A = jax.random.normal(jax.random.PRNGKey(1), (n, n))

    def total(mu):
        z = marginalize(
            A, np.zeros(n), np.zeros((n, n)), Gaussian(mu, np.zeros((n, n)))
        )
        return np.sum(z.m) + np.sum(z.P_sqr.T @ z.P_sqr)

    assert bool(np.all(np.isfinite(jax.grad(total)(np.array([0.4, -0.2, 0.7])))))


def test_inversion_takes_a_zero_prior_factor_without_nan_gradients():
    """The filter's step-0 backward pass: zero prior factor, nonsingular innovation.

    A zero ``Sigma_sqr`` zeroes the cross-term, hence the gain, hence the whole
    stacked factor -- while the innovation covariance stays nonsingular because
    the process noise does not vanish. This is the exact configuration the ODE
    filter hits at its first step.
    """
    n = 3
    A = jax.random.normal(jax.random.PRNGKey(1), (n, n))
    Q_sqr = np.eye(n)  # nonsingular innovation, as in a real prediction

    def total(mu):
        x = Gaussian(mu, np.zeros((n, n)))
        z = marginalize(A, np.zeros(n), Q_sqr, x)
        cond = invert(A, Q_sqr, x, z)
        assert onp.allclose(cond.G, 0.0)  # the premise: the gain really is zero
        return np.sum(cond.d) + np.sum(cond.P_sqr.T @ cond.P_sqr)

    assert bool(np.all(np.isfinite(jax.grad(total)(np.array([0.4, -0.2, 0.7])))))


def test_a_zero_innovation_covariance_gives_a_zero_gain():
    """``_safe_qr`` repairs a degenerate *output* factor; ``invert`` handles a
    zero-variance *innovation* separately: such a component carries no
    information, so its gain is zero (the pseudo-inverse). The conditional is
    then the prior, ``d = m`` with a zero covariance, and the gradient is that
    of ``sum(m)``.
    """
    n = 3
    A = jax.random.normal(jax.random.PRNGKey(1), (n, n))
    zero = np.zeros((n, n))

    def total(mu):
        x = Gaussian(mu, zero)
        cond = invert(A, zero, x, marginalize(A, np.zeros(n), zero, x))
        return np.sum(cond.d) + np.sum(cond.P_sqr.T @ cond.P_sqr)

    assert_close(jax.grad(total)(np.array([0.4, -0.2, 0.7])), np.ones(n))


def test_a_singular_innovation_mixing_components_is_unhandled():
    """Documents the boundary: only components with exactly zero variance are
    repaired. A ``Cov(z)`` that is singular in a direction mixing components
    still divides by a zero pivot.
    """
    x = Gaussian(np.zeros(2), np.array([[1.0, 1.0], [0.0, 0.0]]))
    A, zero = np.eye(2), np.zeros((2, 2))
    z = marginalize(A, np.zeros(2), zero, x)
    assert float(np.min(np.sum(z.P_sqr**2, axis=0))) > 0.0  # no zero column
    cond = invert(A, zero, x, z)
    assert not bool(np.all(np.isfinite(cond.G)))


# ---------------------------------------------------------------------------
# The failure this actually fixes: smoothed covariances from a Taylor-mode start
# ---------------------------------------------------------------------------


def _setup():
    prior = IWP(q=3, d=1)
    measure = ODEInformation(logistic, prior.E0, prior.E1)
    m_0, P_0_sqr = taylor_mode_initialization(logistic, np.array([0.1]), q=3)
    return prior, measure, m_0, P_0_sqr


def test_taylor_mode_initialization_is_exactly_singular():
    """Guard the premise: the trigger is a genuinely zero initial factor."""
    _, _, _, P_0_sqr = _setup()
    assert onp.allclose(P_0_sqr, 0.0)
    assert int(onp.linalg.matrix_rank(onp.asarray(P_0_sqr))) == 0


def test_first_backward_conditional_is_exactly_zero():
    """And that it propagates into a rank-0 backward conditional at step 0."""
    prior, measure, m_0, P_0_sqr = _setup()
    res = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*TSPAN, N_STEPS + 1),
        calibration="none",
    )
    assert res.P_back_sqr is not None
    assert onp.allclose(res.P_back_sqr[0], 0.0)
    assert not onp.allclose(res.P_back_sqr[1], 0.0)  # only the first step


def _smoothed(mu, prior, measure, P_0_sqr):
    res = gaussian_filter(
        mu,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*TSPAN, N_STEPS + 1),
        correction=TaylorCorrection(order=1),
        calibration="none",
    )
    return rts_smoother(res)


def test_smoothed_covariance_gradient_is_finite_from_a_singular_start():
    prior, measure, m_0, P_0_sqr = _setup()

    def cov_sum(mu):
        _, P_sqr = _smoothed(mu, prior, measure, P_0_sqr)
        return np.sum(np.einsum("kij,kil->kjl", P_sqr, P_sqr))

    assert bool(np.all(np.isfinite(jax.grad(cov_sum)(m_0))))


def test_smoothed_mean_gradient_is_not_poisoned_by_the_covariance_branch():
    """The mean does not depend on the degenerate factor, and must not inherit its NaN."""
    prior, measure, m_0, P_0_sqr = _setup()

    def mean_out(mu):
        m, _ = _smoothed(mu, prior, measure, P_0_sqr)
        return m[-1, 0]

    assert bool(np.all(np.isfinite(jax.grad(mean_out)(m_0))))


def test_smoothed_covariance_gradient_matches_finite_differences():
    """Finiteness is not enough -- the derivative has to be the right one."""
    prior, measure, m_0, P_0_sqr = _setup()

    def cov_sum(mu):
        _, P_sqr = _smoothed(mu, prior, measure, P_0_sqr)
        return np.sum(np.einsum("kij,kil->kjl", P_sqr, P_sqr))

    # The trailing components are ~1e-8 against an objective of ~0.2: at small
    # steps round-off in f, divided by 2*eps, swamps them. f is smooth, so a larger
    # step costs almost no truncation error and keeps the round-off ~1e-12.
    assert_grad_matches_fd_vector(cov_sum, m_0, eps=1e-4, atol=1e-10)


def test_values_are_unchanged_and_continuous_in_a_regularizing_jitter():
    """The zero-factor result is the limit of the jittered one, so no value moved."""
    prior, measure, m_0, P_0_sqr = _setup()

    def cov_sum(P0):
        _, P_sqr = _smoothed(m_0, prior, measure, P0)
        return float(np.sum(np.einsum("kij,kil->kjl", P_sqr, P_sqr)))

    exact = cov_sum(P_0_sqr)
    errs = [abs(exact - cov_sum(P_0_sqr + j * np.eye(4))) for j in (1e-5, 1e-7)]
    assert errs[1] < errs[0]  # converges as the jitter shrinks
    assert errs[1] < 1e-8
