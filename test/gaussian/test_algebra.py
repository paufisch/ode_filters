"""The square-root Gaussian algebra against dense (moment-form) formulas.

Each operation is checked on random inputs of random sizes against the textbook
formula it implements, which is what pins its values; triangularity and the
input checks are tested separately.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ode_filters.gaussian import (
    AffineModel,
    Conditional,
    Gaussian,
    compose,
    invert,
    log_likelihood,
    marginalize,
    update,
)
from test.helpers import assert_close, cov

seeds = st.integers(0, 2**31 - 1)
dims = st.integers(1, 5)


def _spd_sqr(key, n, scale=1.0):
    """Upper-triangular factor of a random SPD matrix."""
    M = jax.random.normal(key, (n, n))
    P = scale * (M @ M.T / n + 0.5 * np.eye(n))
    return np.linalg.cholesky(P).T


def _random_gaussian(key, n):
    k1, k2 = jax.random.split(key)
    return Gaussian(jax.random.normal(k1, (n,)), _spd_sqr(k2, n))


def _is_upper_triangular(M) -> bool:
    return bool(np.allclose(np.tril(M, -1), 0.0))


@given(seed=seeds, n=dims, k=dims)
def test_marginalize_matches_dense(seed, n, k):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = _random_gaussian(keys[0], n)
    A = jax.random.normal(keys[1], (k, n))
    b = jax.random.normal(keys[2], (k,))
    Q_sqr = _spd_sqr(keys[3], k)

    z = marginalize(A, b, Q_sqr, x)

    assert_close(z.m, A @ x.m + b)
    assert_close(cov(z.P_sqr), A @ cov(x.P_sqr) @ A.T + cov(Q_sqr), rtol=1e-10)
    assert _is_upper_triangular(z.P_sqr)


@given(seed=seeds, n=dims, k=dims, noise_free=st.booleans())
def test_invert_matches_dense_bayes_rule(seed, n, k, noise_free):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = _random_gaussian(keys[0], n)
    A = jax.random.normal(keys[1], (k, n))
    b = jax.random.normal(keys[2], (k,))
    Q_sqr = np.zeros((k, k)) if noise_free else _spd_sqr(keys[3], k)
    if noise_free and k > n:
        return  # z would be degenerate: more noise-free rows than states

    z = marginalize(A, b, Q_sqr, x)
    conditional = invert(A, Q_sqr, x, z)

    P, S = cov(x.P_sqr), cov(z.P_sqr)
    G = P @ A.T @ np.linalg.inv(S)
    assert_close(conditional.G, G, rtol=1e-8)
    assert_close(conditional.d, x.m - G @ z.m, rtol=1e-8)
    # Noise-free rows leave an (exactly or nearly) zero conditional covariance
    # in their directions: compare on the scale of the prior.
    scale = float(np.max(np.abs(P)))
    assert_close(cov(conditional.P_sqr), P - G @ S @ G.T, atol=1e-10 * scale)
    assert _is_upper_triangular(conditional.P_sqr)


@given(seed=seeds, n=dims, k=dims)
def test_update_and_log_likelihood_match_dense_kalman(seed, n, k):
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = _random_gaussian(keys[0], n)
    H = jax.random.normal(keys[1], (k, n))
    c = jax.random.normal(keys[2], (k,))
    R_sqr = _spd_sqr(keys[3], k, scale=0.1)

    posterior, innovation = update(AffineModel(H, c, R_sqr), x)

    P = cov(x.P_sqr)
    r = H @ x.m + c
    S = H @ P @ H.T + cov(R_sqr)
    K = P @ H.T @ np.linalg.inv(S)
    assert_close(innovation.m, r)
    assert_close(cov(innovation.P_sqr), S, rtol=1e-10)
    assert_close(posterior.m, x.m - K @ r, rtol=1e-8)
    assert_close(cov(posterior.P_sqr), P - K @ S @ K.T, rtol=1e-8)
    expected = jax.scipy.stats.multivariate_normal.logpdf(np.zeros(k), r, S)
    assert_close(log_likelihood(innovation), expected, rtol=1e-10)


@given(seed=seeds, n=dims)
def test_compose_matches_dense_chain(seed, n):
    keys = jax.random.split(jax.random.PRNGKey(seed), 6)
    outer = Conditional(
        jax.random.normal(keys[0], (n, n)),
        jax.random.normal(keys[1], (n,)),
        _spd_sqr(keys[2], n),
    )
    inner = Conditional(
        jax.random.normal(keys[3], (n, n)),
        jax.random.normal(keys[4], (n,)),
        _spd_sqr(keys[5], n),
    )

    composed = compose(outer, inner)

    assert_close(composed.G, outer.G @ inner.G, rtol=1e-12)
    assert_close(composed.d, outer.G @ inner.d + outer.d, rtol=1e-12)
    expected = outer.G @ cov(inner.P_sqr) @ outer.G.T + cov(outer.P_sqr)
    assert_close(cov(composed.P_sqr), expected, rtol=1e-10)


def test_compose_with_the_identity_is_a_no_op():
    n = 3
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    c = Conditional(
        jax.random.normal(keys[0], (n, n)),
        jax.random.normal(keys[1], (n,)),
        _spd_sqr(keys[2], n),
    )
    identity = Conditional(np.eye(n), np.zeros(n), np.zeros((n, n)))
    composed = compose(identity, c)
    assert_close(composed.G, c.G)
    assert_close(composed.d, c.d)
    assert_close(cov(composed.P_sqr), cov(c.P_sqr))


def test_marginalize_accepts_a_scalar_noise_factor_for_one_output():
    x = Gaussian(np.array([1.0, 2.0]), np.eye(2))
    z = marginalize(np.array([[1.0, 1.0]]), np.zeros(1), np.array(0.5), x)
    assert_close(cov(z.P_sqr), np.array([[2.25]]))


@pytest.mark.parametrize(
    ("A", "b", "Q_sqr", "P_sqr", "match"),
    [
        (np.ones((2, 3)), np.zeros(3), np.eye(2), np.eye(3), "rows"),
        (np.ones((2, 3)), np.zeros(2), np.eye(3), np.eye(3), "rows"),
        (np.ones((2, 3)), np.zeros(2), np.eye(2), np.eye(2), "columns"),
        (np.ones((2, 3)), np.zeros(2), np.ones((2, 3)), np.eye(3), "square"),
        (np.ones((2, 3)), np.zeros(2), np.eye(2), np.ones((3, 2)), "square"),
    ],
)
def test_marginalize_rejects_mismatched_shapes(A, b, Q_sqr, P_sqr, match):
    x = Gaussian(np.zeros(P_sqr.shape[0]), P_sqr)
    with pytest.raises(ValueError, match=match):
        marginalize(A, b, Q_sqr, x)


def test_gaussian_cov_and_std_from_the_factor():
    P_sqr = jax.random.normal(jax.random.PRNGKey(7), (4, 3, 3))
    x = Gaussian(np.zeros((4, 3)), P_sqr)
    P = np.einsum("nij,nik->njk", P_sqr, P_sqr)
    assert np.allclose(x.cov, P, rtol=1e-12)
    assert x.std.shape == (4, 3)
    assert np.allclose(x.std, np.sqrt(jax.vmap(np.diag)(P)), rtol=1e-12)
