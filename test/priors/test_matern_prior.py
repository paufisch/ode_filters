"""Tests for MaternPrior, against the textbook Matern kernel and its stationary law.

The oracles are the closed-form kernels of Rasmussen & Williams (2006, eqs.
4.16-4.17), written out here and in ``test.helpers``; none of them reuses the
prior's companion form or its rate formula.
"""

import jax
import jax.numpy as np
import pytest

from ode_filters.priors import MaternPrior, PrecondMaternPrior
from test.helpers import (
    assert_close,
    assert_grad_matches_fd,
    matern_kernel_derivatives,
    matern_lag_covariance,
    matern_sde,
    van_loan,
)

SQRT3, SQRT5 = 3.0**0.5, 5.0**0.5


def _textbook_kernel(q, tau, ell):
    """Matern-1/2, -3/2 and -5/2 kernels (Rasmussen & Williams 2006, eq. 4.17)."""
    if q == 0:
        return np.exp(-tau / ell)
    if q == 1:
        r = SQRT3 * tau / ell
        return (1.0 + r) * np.exp(-r)
    r = SQRT5 * tau / ell
    return (1.0 + r + r**2 / 3.0) * np.exp(-r)


def _textbook_stationary_covariance(q, ell):
    """``Cov([u, u', ..., u^(q)])``: the spectral moments ``(-1)^j k^(i+j)(0)``."""
    if q == 0:
        return np.ones((1, 1))
    if q == 1:
        return np.diag(np.array([1.0, 3.0 / ell**2]))
    v1, v2 = 5.0 / (3.0 * ell**2), 25.0 / ell**4
    return np.array([[1.0, 0.0, -v1], [0.0, v1, 0.0], [-v1, 0.0, v2]])


# ---------------------------------------------------------------------------
# The length-scale convention
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("length_scale", [0.3, 1.0, 4.0])
@pytest.mark.parametrize("tau", [0.1, 0.7, 1.3, 3.0])
def test_correlation_of_the_solution_is_the_textbook_kernel(q, length_scale, tau):
    """``E0 A(tau) Sigma E0.T / (E0 Sigma E0.T)`` is ``k_{q+1/2}(tau)``."""
    prior = MaternPrior(q, 1, length_scale)
    S = prior.stationary_cov_sqr()
    Sigma = S.T @ S
    E0 = prior.E0
    rho = (E0 @ prior.A(tau) @ Sigma @ E0.T) / (E0 @ Sigma @ E0.T)
    assert_close(rho[0, 0], _textbook_kernel(q, tau, length_scale), atol=1e-12)


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("length_scale", [0.3, 1.0, 4.0])
def test_stationary_covariance_has_the_textbook_variances(q, length_scale):
    S = MaternPrior(q, 1, length_scale).stationary_cov_sqr()
    assert_close(S.T @ S, _textbook_stationary_covariance(q, length_scale), rtol=1e-12)


@pytest.mark.parametrize("q", [0, 1, 2])
@pytest.mark.parametrize("tau", [0.0, 0.4, 2.5])
def test_the_general_kernel_oracle_is_the_textbook_kernel(q, tau):
    """``matern_kernel_derivatives`` (eq. 4.16, any ``q``) matches eq. 4.17."""
    k = matern_kernel_derivatives(q, 0.8, tau, 0)
    assert_close(k[0], _textbook_kernel(q, tau, 0.8), rtol=1e-14)


@pytest.mark.parametrize("q", [0, 1, 2])
def test_the_kernel_oracle_has_the_textbook_variances(q):
    """Its derivatives at ``0+`` give the spectral moments written out above."""
    expected = _textbook_stationary_covariance(q, 0.8)
    assert_close(matern_lag_covariance(q, 0.8, 0.0), expected, rtol=1e-14)


# ---------------------------------------------------------------------------
# Transition and process noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0, 1, 2, 3])
@pytest.mark.parametrize("length_scale", [0.5, 2.0])
@pytest.mark.parametrize("h", [0.05, 0.5])
def test_transition_carries_the_kernel_lag_covariance(q, length_scale, h):
    """``A(h) Sigma = Cov(Y(t + h), Y(t))``, entrywise ``(-1)^j k^(i+j)(h)``."""
    prior = MaternPrior(q, 1, length_scale)
    Sigma = matern_lag_covariance(q, length_scale, 0.0)
    expected = matern_lag_covariance(q, length_scale, h)
    assert_close(prior.A(h) @ Sigma, expected, rtol=1e-12)


@pytest.mark.parametrize("q", [0, 1, 2, 3])
@pytest.mark.parametrize("length_scale", [0.5, 2.0])
@pytest.mark.parametrize("h", [0.05, 0.5])
def test_transition_and_noise_match_van_loan_of_the_kernel_sde(q, length_scale, h):
    """``A(h)`` and ``Q(h)`` against Van Loan's block exponential of the SDE
    that the kernel implies (``matern_sde``), not the square-root MFD."""
    A_ref, Q_ref = van_loan(*matern_sde(q, length_scale), h)
    prior = MaternPrior(q, 1, length_scale)
    assert_close(prior.A(h), A_ref, rtol=1e-10)
    assert_close(prior.Q(h), Q_ref, rtol=1e-9)


def test_kronecker_structure_and_shapes():
    xi = np.array([[2.0, 0.3], [0.3, 1.0]])
    prior, scalar = MaternPrior(2, 2, 1.5, Xi=xi), MaternPrior(2, 1, 1.5)
    h = 0.3
    assert (
        prior.state_dim == 6 and prior.E0.shape == (2, 6) and prior.E1.shape == (2, 6)
    )
    assert np.allclose(prior.A(h), np.kron(scalar.A(h), np.eye(2)))
    assert np.allclose(prior.Q(h), np.kron(scalar.Q(h), xi), rtol=1e-10)
    assert np.allclose(prior.b(h), np.zeros(6))
    assert np.allclose(prior.A(0.0), np.eye(6))


def test_n_quad_must_exceed_the_order():
    with pytest.raises(ValueError, match="n_quad must exceed q"):
        MaternPrior(q=4, d=1, length_scale=1.0, n_quad=4)


# ---------------------------------------------------------------------------
# The stationary law
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("q", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("length_scale", [0.3, 1.0, 4.0])
def test_stationary_covariance_matches_the_kernel(q, length_scale):
    S = MaternPrior(q, 1, length_scale).stationary_cov_sqr()
    assert_close(S.T @ S, matern_lag_covariance(q, length_scale, 0.0), rtol=1e-12)


@pytest.mark.parametrize("prior_cls", [MaternPrior, PrecondMaternPrior])
@pytest.mark.parametrize("q", [1, 2, 4])
@pytest.mark.parametrize("h", [0.01, 0.3, 3.0])
def test_stationary_covariance_is_invariant_under_the_transition(prior_cls, q, h):
    xi = np.diag(np.array([0.5, 2.0]))
    prior = prior_cls(q, 2, 0.8, Xi=xi)
    S = prior.stationary_cov_sqr()
    P = S.T @ S
    A = prior.A(h)
    assert np.allclose(A @ P @ A.T + prior.Q(h), P, atol=1e-10 * float(np.max(P)))
    assert np.allclose(np.tril(S, -1), 0.0)


@pytest.mark.parametrize("q", [0, 1, 2, 3])
def test_stationary_variance_of_the_solution_is_xi(q):
    xi = np.diag(np.array([0.5, 3.0]))
    S = MaternPrior(q, 2, 0.7, Xi=xi).stationary_cov_sqr()
    P = S.T @ S
    assert np.allclose(P[:2, :2], xi, rtol=1e-10)


def test_stationary_covariance_is_finite_at_short_length_scales():
    S = MaternPrior(2, 1, 0.051).stationary_cov_sqr()
    assert bool(np.all(np.isfinite(S)))


def test_stationary_covariance_is_differentiable_and_traceable():
    def entry(log_ell):
        S = MaternPrior(2, 1, np.exp(log_ell)).stationary_cov_sqr()
        return (S.T @ S)[1, 1]

    assert_grad_matches_fd(entry, np.array(0.3))

    def scaled(xi):
        S = MaternPrior(1, 1, 1.0, Xi=xi * np.eye(1)).stationary_cov_sqr()
        return (S.T @ S)[0, 0]

    assert_grad_matches_fd(scaled, np.array(1.7))
    jitted = jax.jit(lambda ell: MaternPrior(2, 1, ell).stationary_cov_sqr())(0.5)
    assert np.allclose(jitted, MaternPrior(2, 1, 0.5).stationary_cov_sqr())
