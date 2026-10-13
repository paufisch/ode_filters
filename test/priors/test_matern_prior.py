"""Tests for MaternPrior, against the companion form and its stationary law."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.priors import MaternPrior, PrecondMaternPrior
from test.helpers import assert_grad_matches_fd, matern_companion, van_loan


@pytest.mark.parametrize("q", [0, 1, 2, 3])
@pytest.mark.parametrize("length_scale", [0.5, 2.0])
@pytest.mark.parametrize("h", [0.05, 0.5])
def test_transition_and_noise_match_the_companion_form(q, length_scale, h):
    A_ref, Q_ref = van_loan(*matern_companion(length_scale, q), h)
    prior = MaternPrior(q, 1, length_scale)
    assert np.allclose(prior.A(h), A_ref, rtol=1e-10, atol=1e-12)
    assert np.allclose(
        prior.Q(h), Q_ref, rtol=1e-8, atol=1e-12 * float(np.max(np.abs(Q_ref)))
    )


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


def _lyapunov(F, S):
    """Stationary covariance from ``F P + P F.T + S = 0`` by vectorization."""
    F, S = onp.asarray(F), onp.asarray(S)
    n = F.shape[0]
    eye = onp.eye(n)
    P = onp.linalg.solve(onp.kron(F, eye) + onp.kron(eye, F), -S.reshape(-1))
    return P.reshape(n, n)


@pytest.mark.parametrize("q", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("length_scale", [0.3, 1.0, 4.0])
def test_stationary_covariance_matches_the_lyapunov_solution(q, length_scale):
    prior = MaternPrior(q, 1, length_scale)
    S = prior.stationary_cov_sqr()
    expected = _lyapunov(*matern_companion(length_scale, q))
    assert np.allclose(
        S.T @ S, expected, rtol=1e-9, atol=1e-12 * onp.abs(expected).max()
    )


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
    S = MaternPrior(2, 1, 2.6e-3).stationary_cov_sqr()
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
