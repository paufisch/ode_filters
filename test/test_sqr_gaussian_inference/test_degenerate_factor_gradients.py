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
from ode_filters.inference.sqr_gaussian_inference import (
    _safe_qr,
    sqr_inversion,
    sqr_marginalization,
)
from ode_filters.priors import taylor_mode_initialization

TSPAN = (0.0, 2.0)
N_STEPS = 20


def logistic(x, *, t):
    return x * (1.0 - x)


def _central_diff(f, x, h=1e-6):
    return np.array(
        [
            (
                f(x + np.zeros_like(x).at[i].set(h))
                - f(x - np.zeros_like(x).at[i].set(h))
            )
            / (2 * h)
            for i in range(x.shape[0])
        ]
    )


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
    ad = jax.grad(cov_sum)(theta)
    fd = _central_diff(cov_sum, theta)
    assert onp.allclose(ad, fd, rtol=1e-6)


def test_marginalization_takes_a_zero_factor_without_nan_gradients():
    """A deterministic state pushed through a noiseless map: the factor is zero."""
    n = 3
    A = jax.random.normal(jax.random.PRNGKey(1), (n, n))

    def total(mu):
        mu_z, S_sqr = sqr_marginalization(
            A, np.zeros(n), np.zeros((n, n)), mu, np.zeros((n, n))
        )
        return np.sum(mu_z) + np.sum(S_sqr.T @ S_sqr)

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
        Sigma_sqr = np.zeros((n, n))
        mu_z, S_sqr = sqr_marginalization(A, np.zeros(n), Q_sqr, mu, Sigma_sqr)
        K, d, Lam_sqr = sqr_inversion(A, mu, Sigma_sqr, mu_z, S_sqr, Q_sqr)
        assert onp.allclose(K, 0.0)  # the premise: the gain really is zero
        return np.sum(d) + np.sum(Lam_sqr.T @ Lam_sqr)

    assert bool(np.all(np.isfinite(jax.grad(total)(np.array([0.4, -0.2, 0.7])))))


def test_singular_innovation_covariance_is_a_separate_unhandled_degeneracy():
    """Documents the boundary of the fix.

    ``_safe_qr`` repairs a degenerate *output* factor. It does not repair a
    degenerate *innovation* covariance: ``sqr_inversion`` divides by
    ``Sigma_z_sqr`` in a triangular solve, so a zero innovation covariance is
    ill-posed (conditioning a zero-variance state on a zero-variance
    observation) and stays ``NaN``. Recorded so the boundary is a known
    limitation rather than a later surprise.
    """
    n = 3
    A = jax.random.normal(jax.random.PRNGKey(1), (n, n))
    zero = np.zeros((n, n))

    def total(mu):
        mu_z, S_sqr = sqr_marginalization(A, np.zeros(n), zero, mu, zero)
        _, d, Lam_sqr = sqr_inversion(A, mu, zero, mu_z, S_sqr, zero)
        return np.sum(d) + np.sum(Lam_sqr.T @ Lam_sqr)

    assert bool(np.all(np.isnan(jax.grad(total)(np.array([0.4, -0.2, 0.7])))))


# ---------------------------------------------------------------------------
# The failure this actually fixes: smoothed covariances from a Taylor-mode start
# ---------------------------------------------------------------------------


def _setup():
    prior = IWP(q=3, d=1)
    measure = ODEInformation(logistic, prior.E0, prior.E1)
    mu_0, P0_sqr = taylor_mode_initialization(logistic, np.array([0.1]), q=3)
    return prior, measure, mu_0, P0_sqr


def test_taylor_mode_initialization_is_exactly_singular():
    """Guard the premise: the trigger is a genuinely zero initial factor."""
    _, _, _, P0_sqr = _setup()
    assert onp.allclose(P0_sqr, 0.0)
    assert int(onp.linalg.matrix_rank(onp.asarray(P0_sqr))) == 0


def test_first_backward_conditional_is_exactly_zero():
    """And that it propagates into a rank-0 backward conditional at step 0."""
    prior, measure, mu_0, P0_sqr = _setup()
    res = gaussian_filter(
        mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, calibration="none"
    )
    assert res.P_back_sqr is not None
    assert onp.allclose(res.P_back_sqr[0], 0.0)
    assert not onp.allclose(res.P_back_sqr[1], 0.0)  # only the first step


def _smoothed(mu, prior, measure, P0_sqr):
    res = gaussian_filter(
        mu,
        P0_sqr,
        prior,
        measure,
        TSPAN,
        N_STEPS,
        correction=TaylorCorrection(order=1),
        calibration="none",
    )
    return rts_smoother(prior, res)


def test_smoothed_covariance_gradient_is_finite_from_a_singular_start():
    prior, measure, mu_0, P0_sqr = _setup()

    def cov_sum(mu):
        _, P_sqr = _smoothed(mu, prior, measure, P0_sqr)
        return np.sum(np.einsum("kij,kil->kjl", P_sqr, P_sqr))

    assert bool(np.all(np.isfinite(jax.grad(cov_sum)(mu_0))))


def test_smoothed_mean_gradient_is_not_poisoned_by_the_covariance_branch():
    """The mean does not depend on the degenerate factor, and must not inherit its NaN."""
    prior, measure, mu_0, P0_sqr = _setup()

    def mean_out(mu):
        m, _ = _smoothed(mu, prior, measure, P0_sqr)
        return m[-1, 0]

    assert bool(np.all(np.isfinite(jax.grad(mean_out)(mu_0))))


def test_smoothed_covariance_gradient_matches_finite_differences():
    """Finiteness is not enough -- the derivative has to be the right one."""
    prior, measure, mu_0, P0_sqr = _setup()

    def cov_sum(mu):
        _, P_sqr = _smoothed(mu, prior, measure, P0_sqr)
        return np.sum(np.einsum("kij,kil->kjl", P_sqr, P_sqr))

    ad = jax.grad(cov_sum)(mu_0)
    fd = _central_diff(cov_sum, mu_0)
    # The trailing components are ~1e-8 against an O(1) objective, where central
    # differences are dominated by cancellation; compare on the shared scale.
    assert onp.allclose(ad, fd, rtol=5e-3, atol=1e-10)


def test_values_are_unchanged_and_continuous_in_a_regularizing_jitter():
    """The zero-factor result is the limit of the jittered one, so no value moved."""
    prior, measure, mu_0, P0_sqr = _setup()

    def cov_sum(P0):
        _, P_sqr = _smoothed(mu_0, prior, measure, P0)
        return float(np.sum(np.einsum("kij,kil->kjl", P_sqr, P_sqr)))

    exact = cov_sum(P0_sqr)
    errs = [abs(exact - cov_sum(P0_sqr + j * np.eye(4))) for j in (1e-5, 1e-7)]
    assert errs[1] < errs[0]  # converges as the jitter shrinks
    assert errs[1] < 1e-8
