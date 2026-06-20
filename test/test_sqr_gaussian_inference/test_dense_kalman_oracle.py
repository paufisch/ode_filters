"""Square-root primitives vs a dense (moment-form) Kalman reference.

The existing property tests assert structural invariants (PD, symmetry,
uncertainty reduction) but never check that the square-root predict/update
reproduce the *textbook* moment-form covariance. A wrong-but-still-PD update
would pass those. These oracle tests pin the actual values against a dense
Kalman predict (``A Sigma A.T + Q``) and a dense Joseph-form update.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import IWP
from ode_filters.inference.sqr_gaussian_inference import (
    compose_backward_conditionals,
    sqr_inversion,
    sqr_marginalization,
)


def _random_spd_sqr(key, n):
    """Return (Sigma, Sigma_sqr) with Sigma SPD and Sigma == Sigma_sqr.T @ Sigma_sqr."""
    M = jax.random.normal(key, (n, n))
    Sigma = M @ M.T + n * np.eye(n)  # well-conditioned SPD
    Sigma_sqr = np.linalg.cholesky(Sigma).T
    return Sigma, Sigma_sqr


# (n_state, n_obs): include rectangular observation operators.
_SHAPES = [(3, 3), (4, 2), (2, 4)]
_SEEDS = [0, 7]


@pytest.mark.parametrize("n_state,n_obs", _SHAPES)
@pytest.mark.parametrize("seed", _SEEDS)
def test_marginalization_matches_dense_predict(n_state, n_obs, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed), 5)
    A = jax.random.normal(keys[0], (n_obs, n_state))
    b = jax.random.normal(keys[1], (n_obs,))
    mu = jax.random.normal(keys[2], (n_state,))
    _, Sigma_sqr = _random_spd_sqr(keys[3], n_state)
    Q, Q_sqr = _random_spd_sqr(keys[4], n_obs)
    Sigma = Sigma_sqr.T @ Sigma_sqr

    mu_z, Sigma_z_sqr = sqr_marginalization(A, b, Q_sqr, mu, Sigma_sqr)

    assert np.allclose(mu_z, A @ mu + b, atol=1e-10)
    Sigma_z = Sigma_z_sqr.T @ Sigma_z_sqr
    assert np.allclose(Sigma_z, A @ Sigma @ A.T + Q, atol=1e-9)


@pytest.mark.parametrize("n_state,n_obs", _SHAPES)
@pytest.mark.parametrize("seed", _SEEDS)
def test_inversion_matches_dense_joseph_update(n_state, n_obs, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed + 100), 5)
    A = jax.random.normal(keys[0], (n_obs, n_state))
    b = jax.random.normal(keys[1], (n_obs,))
    mu = jax.random.normal(keys[2], (n_state,))
    Sigma, Sigma_sqr = _random_spd_sqr(keys[3], n_state)
    R, R_sqr = _random_spd_sqr(keys[4], n_obs)

    # Predicted-observation marginal (residual against an implicit z = 0).
    mu_z, Sigma_z_sqr = sqr_marginalization(A, b, R_sqr, mu, Sigma_sqr)
    K, d, Lambda_sqr = sqr_inversion(A, mu, Sigma_sqr, mu_z, Sigma_z_sqr, R_sqr)

    # Dense Kalman reference.
    S = A @ Sigma @ A.T + R
    K_dense = Sigma @ A.T @ np.linalg.inv(S)
    m_post = mu - K_dense @ mu_z
    IKA = np.eye(n_state) - K_dense @ A
    P_post = IKA @ Sigma @ IKA.T + K_dense @ R @ K_dense.T  # Joseph form

    assert np.allclose(K, K_dense, atol=1e-8)
    assert np.allclose(d, m_post, atol=1e-8)
    assert np.allclose(Lambda_sqr.T @ Lambda_sqr, P_post, atol=1e-8)
    # Cross-check the simplified form Sigma - K S K.T == Joseph form here.
    assert np.allclose(P_post, Sigma - K_dense @ S @ K_dense.T, atol=1e-8)


@pytest.mark.parametrize("n_state,n_obs", [(3, 3), (4, 2)])
def test_inversion_no_noise_matches_dense(n_state, n_obs):
    """The noise-free inversion (Q_sqr=None, used by the smoother backward pass)
    must equal the dense update with R = 0.

    Restricted to ``n_obs <= n_state``: with R = 0 and more observations than
    states the innovation covariance ``A Sigma A.T`` is rank-deficient and the
    noiseless conditioning is ill-posed (which is never the smoother's case,
    where ``A`` is the square transition matrix)."""
    keys = jax.random.split(jax.random.PRNGKey(13), 4)
    A = jax.random.normal(keys[0], (n_obs, n_state))
    b = jax.random.normal(keys[1], (n_obs,))
    mu = jax.random.normal(keys[2], (n_state,))
    Sigma, Sigma_sqr = _random_spd_sqr(keys[3], n_state)

    R_sqr = np.zeros((n_obs, n_obs))
    mu_z, Sigma_z_sqr = sqr_marginalization(A, b, R_sqr, mu, Sigma_sqr)
    K, d, Lambda_sqr = sqr_inversion(A, mu, Sigma_sqr, mu_z, Sigma_z_sqr, None)

    S = A @ Sigma @ A.T
    K_dense = Sigma @ A.T @ np.linalg.inv(S)
    IKA = np.eye(n_state) - K_dense @ A
    P_post = IKA @ Sigma @ IKA.T  # R = 0

    assert np.allclose(K, K_dense, atol=1e-7)
    assert np.allclose(d, mu - K_dense @ mu_z, atol=1e-7)
    assert np.allclose(Lambda_sqr.T @ Lambda_sqr, P_post, atol=1e-7)


# --------------------------------------------------------------------------- #
# Diagonal-calibration denominator: the row-norm form used by
# _calibrate_diffusion must equal the dense diag(H Q H.T), so the filter never
# needs to form the dense Q (change #2).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n_state,n_obs", _SHAPES)
@pytest.mark.parametrize("seed", _SEEDS)
def test_diag_hqht_row_norms_match_dense(n_state, n_obs, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed + 200), 2)
    H = jax.random.normal(keys[0], (n_obs, n_state))
    Q, Q_sqr = _random_spd_sqr(keys[1], n_state)

    dense = np.einsum("ij,jk,ik->i", H, Q, H)  # diag(H Q H.T)
    row_norms = np.sum((H @ Q_sqr.T) ** 2, axis=1)  # _calibrate_diffusion form
    assert np.allclose(row_norms, dense, rtol=1e-9, atol=1e-12)


def test_diag_hqht_row_norms_match_dense_for_real_prior():
    """Tie the identity to the actual closed-form prior square root."""
    prior = IWP(q=3, d=2)
    h = 0.2
    H = prior.E1
    Q = prior.Q(h)
    Q_sqr = prior.Q_sqr(h)
    dense = np.einsum("ij,jk,ik->i", H, Q, H)
    row_norms = np.sum((H @ Q_sqr.T) ** 2, axis=1)
    assert np.allclose(row_norms, dense, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- #
# Composition of two affine Gaussian (backward) conditionals -- the fixed-point
# smoothing "merge" used by the adaptive smoother.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [2, 3, 5])
@pytest.mark.parametrize("seed", _SEEDS)
def test_compose_backward_conditionals_matches_dense(n, seed):
    keys = jax.random.split(jax.random.PRNGKey(seed + 300), 6)
    G1 = jax.random.normal(keys[0], (n, n))
    d1 = jax.random.normal(keys[1], (n,))
    _, P1_sqr = _random_spd_sqr(keys[2], n)
    G2 = jax.random.normal(keys[3], (n, n))
    d2 = jax.random.normal(keys[4], (n,))
    _, P2_sqr = _random_spd_sqr(keys[5], n)

    G, d, P_sqr = compose_backward_conditionals((G1, d1, P1_sqr), (G2, d2, P2_sqr))

    P1 = P1_sqr.T @ P1_sqr
    P2 = P2_sqr.T @ P2_sqr
    assert np.allclose(G, G1 @ G2, atol=1e-10)
    assert np.allclose(d, G1 @ d2 + d1, atol=1e-10)
    assert np.allclose(P_sqr.T @ P_sqr, G1 @ P2 @ G1.T + P1, atol=1e-9)


def test_compose_with_identity_is_noop():
    """Composing with the identity conditional (the per-interval reset) is a no-op."""
    n = 4
    keys = jax.random.split(jax.random.PRNGKey(99), 2)
    G2 = jax.random.normal(keys[0], (n, n))
    d2 = jax.random.normal(keys[1], (n,))
    _, P2_sqr = _random_spd_sqr(jax.random.PRNGKey(100), n)
    ident = (np.eye(n), np.zeros(n), np.zeros((n, n)))

    G, d, P_sqr = compose_backward_conditionals(ident, (G2, d2, P2_sqr))
    assert np.allclose(G, G2, atol=1e-12)
    assert np.allclose(d, d2, atol=1e-12)
    assert np.allclose(P_sqr.T @ P_sqr, P2_sqr.T @ P2_sqr, atol=1e-12)
