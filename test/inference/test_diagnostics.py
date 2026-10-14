"""Innovation diagnostics and observation predictives."""

from __future__ import annotations

import jax
import jax.numpy as np

from ode_filters import (
    IWP,
    affine_filter,
    nis,
    observation_predictive,
    residual_autocorrelation,
    rts_smoother,
    whitened_residuals,
)
from ode_filters.gaussian import Gaussian
from ode_filters.information import Measurement, prepare_observations
from ode_filters.linearization.surrogate import AffineSurrogate
from test.helpers import assert_close, cov

PRIOR = IWP(q=1, d=1)
M_0 = np.zeros(2)
P_0_SQR = np.eye(2)
N = 4000


def _filter_on_simulated_data(noise_std: float, assumed_std: float, every: int = 1):
    """A prior path observed in ``x`` with noise, filtered with no ODE information."""
    ts = np.linspace(0.0, 40.0, N + 1)
    k_path, k_noise = jax.random.split(jax.random.PRNGKey(3))
    path = PRIOR.sample(k_path, ts, m_0=M_0, P_0_sqr=P_0_SQR)
    times = ts[every::every]
    y = path[every::every, :1] + noise_std * jax.random.normal(
        k_noise, (times.shape[0], 1)
    )
    obs = prepare_observations(
        [Measurement(np.eye(1), y, times, noise=assumed_std**2)], PRIOR.E0, ts
    )
    blank = AffineSurrogate(np.zeros((N, 1, 2)), np.zeros((N, 1)), np.ones((N, 1, 1)))
    return affine_filter(M_0, P_0_SQR, PRIOR, blank, ts, obs_model=obs)


def test_whitening_matches_the_dense_formula():
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    mz = jax.random.normal(keys[0], (5, 3))
    M = jax.random.normal(keys[1], (5, 3, 3))
    Pz_sqr = jax.vmap(lambda a: np.linalg.qr(a)[1])(M)
    expected = jax.vmap(lambda m, S: m @ np.linalg.solve(S.T @ S, m))(mz, Pz_sqr)
    assert_close(nis(mz, Pz_sqr), expected, rtol=1e-10)
    v = whitened_residuals(mz, Pz_sqr)
    # The factor with a non-negative diagonal maps the residuals back to mz.
    canonical = jax.vmap(lambda S: np.sign(np.diag(S))[:, None] * S)(Pz_sqr)
    assert_close(jax.vmap(lambda S, w: S.T @ w)(canonical, v), mz, rtol=1e-10)


def test_whitening_does_not_depend_on_the_signs_of_the_factor():
    """QR leaves the signs of a factor's rows arbitrary; the residuals must not."""
    keys = jax.random.split(jax.random.PRNGKey(2), 3)
    mz = jax.random.normal(keys[0], (6, 3))
    Pz_sqr = jax.vmap(lambda a: np.linalg.cholesky(a @ a.T + np.eye(3)).T)(
        jax.random.normal(keys[1], (6, 3, 3))
    )
    signs = np.where(jax.random.bernoulli(keys[2], 0.5, (6, 3)), -1.0, 1.0)
    flipped = signs[:, :, None] * Pz_sqr  # the same covariances
    assert_close(whitened_residuals(mz, flipped), whitened_residuals(mz, Pz_sqr))
    # With a non-negative diagonal the whitening is the Cholesky one.
    expected = jax.vmap(lambda m, S: np.linalg.solve(S.T, m))(mz, Pz_sqr)
    assert_close(whitened_residuals(mz, Pz_sqr), expected, rtol=1e-10)


def test_a_slow_drift_is_positively_autocorrelated():
    """A drifting misfit shows as positive autocorrelation, whatever the signs."""
    n = np.arange(200)
    mz = np.sin(2.0 * np.pi * n / 50.0)[:, None]
    Pz_sqr = np.where(n % 2 == 0, 1.0, -1.0)[:, None, None]  # alternating QR signs
    rho = residual_autocorrelation(whitened_residuals(mz, Pz_sqr), max_lag=1)
    assert float(rho[0, 0]) > 0.9


def test_a_consistent_model_has_white_unit_innovations():
    result = _filter_on_simulated_data(noise_std=0.1, assumed_std=0.1)
    values = nis(result.mz_obs, result.Pz_obs_sqr)
    # NIS ~ chi^2_1: mean 1, standard deviation sqrt(2 / N) of the average.
    assert abs(float(values.mean()) - 1.0) < 4.0 * np.sqrt(2.0 / N)
    rho = residual_autocorrelation(
        whitened_residuals(result.mz_obs, result.Pz_obs_sqr), max_lag=5
    )
    assert rho.shape == (5, 1)
    assert np.all(np.abs(rho) < 4.0 / np.sqrt(N))


def test_underestimated_noise_shows_in_the_nis():
    result = _filter_on_simulated_data(noise_std=0.5, assumed_std=0.1)
    assert float(nis(result.mz_obs, result.Pz_obs_sqr).mean()) > 3.0


def test_steps_without_data_contribute_nothing():
    result = _filter_on_simulated_data(noise_std=0.1, assumed_std=0.1, every=2)
    values = nis(result.mz_obs, result.Pz_obs_sqr)
    assert np.all(values[0::2] == 0.0)
    assert abs(float(values[1::2].mean()) - 1.0) < 4.0 * np.sqrt(4.0 / N)


def test_autocorrelation_of_an_alternating_sequence():
    v = np.tile(np.array([1.0, -1.0]), 50)[:, None]
    rho = residual_autocorrelation(v, max_lag=2)
    assert_close(rho[:, 0], np.array([-0.99, 0.98]), rtol=1e-12)


def test_observation_predictive_is_the_pushforward():
    keys = jax.random.split(jax.random.PRNGKey(1), 3)
    m = jax.random.normal(keys[0], (4, 3))
    P_sqr = jax.vmap(lambda a: np.linalg.qr(a)[1])(
        jax.random.normal(keys[1], (4, 3, 3))
    )
    H = jax.random.normal(keys[2], (2, 3))
    R_sqr = np.diag(np.array([0.1, 0.3]))
    marginals = Gaussian(m, P_sqr)
    noisy = observation_predictive(marginals, H, R_sqr)
    assert_close(noisy.m, m @ H.T, rtol=1e-12)
    assert_close(cov(noisy.P_sqr), H @ cov(P_sqr) @ H.T + R_sqr.T @ R_sqr, rtol=1e-10)
    clean = observation_predictive(marginals, H)
    assert_close(cov(clean.P_sqr), H @ cov(P_sqr) @ H.T, rtol=1e-10)


def test_posterior_predictive_of_a_filter_result():
    result = _filter_on_simulated_data(noise_std=0.1, assumed_std=0.1)
    smoothed = rts_smoother(result)
    predictive = observation_predictive(smoothed, PRIOR.E0)
    assert predictive.m.shape == (N + 1, 1)
    assert predictive.P_sqr.shape == (N + 1, 1, 1)
    # Without noise, the pushforward through E0: the first state component.
    assert_close(predictive.m, smoothed.m[:, :1], rtol=1e-14)
    assert_close(cov(predictive.P_sqr), cov(smoothed.P_sqr)[:, :1, :1], rtol=1e-12)
