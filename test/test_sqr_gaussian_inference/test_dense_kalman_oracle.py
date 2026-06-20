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

from ode_filters.inference.sqr_gaussian_inference import (
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
