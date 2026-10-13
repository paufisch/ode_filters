"""Sample paths of the priors."""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import IWP, IOUPPrior, JointPrior, MaternPrior
from test.helpers import cov

N_SAMPLES = 4000


def _assert_covariance_close(P_hat, P):
    """Within four Monte Carlo standard errors of each entry, sqrt((P_ii P_jj + P_ij^2) / N)."""
    d = np.diag(P)
    se = np.sqrt((d[:, None] * d[None, :] + P**2) / N_SAMPLES)
    assert bool(np.all(np.abs(P_hat - P) < 4 * se)), (P_hat, P)


def _empirical(samples):
    centered = samples - samples.mean(axis=0)
    return samples.mean(axis=0), centered.T @ centered / samples.shape[0]


def test_matern_samples_follow_the_stationary_law():
    prior = MaternPrior(q=1, d=1, length_scale=0.5, Xi=2.0 * np.eye(1))
    ts = np.linspace(0.0, 2.0, 9)
    keys = jax.random.split(jax.random.PRNGKey(0), N_SAMPLES)
    paths = jax.vmap(lambda k: prior.sample(k, ts))(keys)
    mean, P = _empirical(paths[:, -1])
    stationary = cov(prior.stationary_cov_sqr())
    standard_error = np.sqrt(np.diag(stationary) / N_SAMPLES)
    assert bool(np.all(np.abs(mean) < 4 * standard_error))
    _assert_covariance_close(P, stationary)


def test_iwp_samples_follow_the_propagated_initial_law_on_a_non_uniform_grid():
    prior = IWP(q=1, d=1)
    ts = np.array([0.0, 0.1, 0.35, 0.4, 1.0])
    m_0 = np.array([1.0, -0.5])
    P_0_sqr = 0.3 * np.eye(2)
    keys = jax.random.split(jax.random.PRNGKey(1), N_SAMPLES)
    paths = jax.vmap(lambda k: prior.sample(k, ts, m_0=m_0, P_0_sqr=P_0_sqr))(keys)

    m, P = m_0, cov(P_0_sqr)
    for h in np.diff(ts):
        A = prior.A(h)
        m, P = A @ m, A @ P @ A.T + prior.Q(h)
    mean, P_hat = _empirical(paths[:, -1])
    assert bool(np.all(np.abs(mean - m) < 4 * np.sqrt(np.diag(P) / N_SAMPLES)))
    _assert_covariance_close(P_hat, P)


def test_sampling_is_reproducible_and_jittable():
    prior = JointPrior(IWP(q=1, d=1), MaternPrior(q=0, d=1, length_scale=1.0))
    ts = np.linspace(0.0, 1.0, 5)
    m_0, P_0_sqr = np.zeros(3), 0.1 * np.eye(3)
    key = jax.random.PRNGKey(3)
    a = prior.sample(key, ts, m_0=m_0, P_0_sqr=P_0_sqr)
    b = jax.jit(lambda k: prior.sample(k, ts, m_0=m_0, P_0_sqr=P_0_sqr))(key)
    assert a.shape == (5, 3)
    assert np.allclose(a, b)
    assert not np.allclose(
        a, prior.sample(jax.random.PRNGKey(4), ts, m_0=m_0, P_0_sqr=P_0_sqr)
    )


@pytest.mark.parametrize(
    "prior", [IWP(q=1, d=1), IOUPPrior(q=1, d=1, rate=-1.0)], ids=["iwp", "ioup"]
)
def test_a_prior_without_a_stationary_law_needs_an_initial_law(prior):
    with pytest.raises(ValueError, match="no stationary law"):
        prior.sample(jax.random.PRNGKey(0), np.linspace(0.0, 1.0, 3))


def test_initial_mean_and_covariance_come_together():
    with pytest.raises(ValueError, match="both"):
        MaternPrior(q=1, d=1, length_scale=1.0).sample(
            jax.random.PRNGKey(0), np.linspace(0.0, 1.0, 3), m_0=np.zeros(2)
        )
