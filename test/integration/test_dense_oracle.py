"""End-to-end comparison of the square-root solvers with a dense Kalman oracle.

For a linear ODE the EK1 linearization is exact, so the filter is an ordinary
Kalman filter on the prior's state space. ``test.helpers.dense_kalman`` runs that
filter (and the RTS smoother) in moment form with textbook formulas; the
solvers must reproduce its means, covariances, log-likelihoods and diffusion
estimates over whole runs.

The ODE information carries a small noise ``R_ODE``. Noise-free conditioning
drives the predicted residual covariance towards singularity, where the
moment-form oracle (not the square-root solver) loses digits; the noise keeps
the oracle exact to the tolerance used here.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    JointPrior,
    MaternPrior,
    ODEInformation,
    PrecondIWP,
    gaussian_filter,
    prepare_observations,
    rts_smoother,
)
from ode_filters.measurement import Measurement, ODEInformationWithHidden
from test.helpers import DenseStep, assert_close, cov, dense_kalman

L = np.array([[-0.5, 1.0], [-1.0, -0.5]])
B = np.array([[0.0], [1.0]])
TSPAN = (0.0, 3.0)
N = 24
OBS_EVERY = 4
OBS_NOISE = 0.05**2
R_ODE = 1e-4
# The moment-form oracle (inverses, slogdet) is the less accurate side.
RTOL = 1e-7


def _vf(x, *, t):
    return L @ x


def _vf_hidden(x, u, *, t):
    return L @ x + B @ u


def _spd(key, n, scale=0.1):
    M = jax.random.normal(key, (n, n))
    return scale * (M @ M.T / n + np.eye(n))


def _upper_chol(P):
    return np.linalg.cholesky(P).T


def _transition(prior, h):
    """Original-coordinate ``(A, Q)`` of one step, for plain and preconditioned priors."""
    A = prior.A(h)
    Q_sqr = prior.Q_sqr(h)
    if hasattr(prior, "T"):
        T = prior.T(h)
        T_inv = np.linalg.inv(T)
        return T @ A @ T_inv, T @ (Q_sqr.T @ Q_sqr) @ T.T
    return A, Q_sqr.T @ Q_sqr


def _observations(E0, ts, H_obs_x, key):
    """Noisy observations of the first solution component every ``OBS_EVERY`` steps."""
    t_obs = ts[OBS_EVERY::OBS_EVERY]
    z = jax.random.normal(key, (t_obs.shape[0], 1)) * 0.3 + 0.5
    meas = Measurement(H_obs_x, z, t_obs, noise=OBS_NOISE)
    return prepare_observations([meas], E0, ts), dict(
        zip(t_obs.tolist(), z, strict=True)
    )


def _steps(prior, measure_H, ts, obs=None, H_obs=None):
    h = ts[1] - ts[0]
    A, Q = _transition(prior, h)
    steps = []
    for t in ts[1:]:
        kwargs = {}
        if obs is not None and float(t) in obs:
            kwargs = dict(
                H_obs=H_obs,
                c_obs=-obs[float(t)],
                R_obs=OBS_NOISE * np.eye(1),
            )
        steps.append(
            DenseStep(
                A=A, Q=Q, H=measure_H, c=np.zeros(2), R=R_ODE * np.eye(2), **kwargs
            )
        )
    return steps


PLAIN_PRIORS = {
    "iwp": lambda: IWP(q=2, d=2),
    "precond_iwp": lambda: PrecondIWP(q=3, d=2),
    "matern": lambda: MaternPrior(q=2, d=2, length_scale=2.0),
}


@pytest.mark.parametrize("prior_name", sorted(PLAIN_PRIORS))
@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_linear_ode_matches_dense_kalman(prior_name, calibration):
    prior = PLAIN_PRIORS[prior_name]()
    measure = ODEInformation(_vf, prior.E0, prior.E1)
    measure.R = R_ODE
    D = prior.E0.shape[1]
    m0 = np.zeros(D).at[:2].set(np.array([1.0, 0.0]))
    P0 = _spd(jax.random.PRNGKey(0), D)
    ts = np.linspace(*TSPAN, N + 1)

    res = gaussian_filter(
        m0, _upper_chol(P0), prior, measure, TSPAN, N, calibration=calibration
    )
    H = prior.E1 - L @ prior.E0
    ref = dense_kalman(m0, P0, _steps(prior, H, ts), calibration=calibration)

    assert_close(res.m, ref.m, rtol=RTOL)
    assert_close(cov(res.P_sqr), ref.P, rtol=RTOL)
    assert_close(res.log_likelihood, ref.log_likelihood, rtol=RTOL)
    assert_close(res.sigma_sqr, ref.sigma_sqr, rtol=RTOL)
    m_s, P_s_sqr = rts_smoother(prior, res)
    assert_close(m_s, ref.m_smooth, rtol=RTOL)
    assert_close(cov(P_s_sqr), ref.P_smooth, rtol=RTOL)


@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_linear_ode_with_observations_matches_dense_kalman(calibration):
    prior = IWP(q=2, d=2)
    measure = ODEInformation(_vf, prior.E0, prior.E1)
    measure.R = R_ODE
    D = prior.E0.shape[1]
    m0 = np.zeros(D).at[:2].set(np.array([1.0, 0.0]))
    P0 = _spd(jax.random.PRNGKey(1), D)
    ts = np.linspace(*TSPAN, N + 1)
    H_obs_x = np.array([[1.0, 0.0]])
    obs_model, obs = _observations(prior.E0, ts, H_obs_x, jax.random.PRNGKey(2))

    res = gaussian_filter(
        m0,
        _upper_chol(P0),
        prior,
        measure,
        TSPAN,
        N,
        calibration=calibration,
        obs_model=obs_model,
    )
    H = prior.E1 - L @ prior.E0
    steps = _steps(prior, H, ts, obs=obs, H_obs=H_obs_x @ prior.E0)
    ref = dense_kalman(m0, P0, steps, calibration=calibration)

    assert_close(res.m, ref.m, rtol=RTOL)
    assert_close(cov(res.P_sqr), ref.P, rtol=RTOL)
    assert_close(res.log_likelihood, ref.log_likelihood, rtol=RTOL)
    assert res.log_likelihood_obs is not None
    assert_close(res.log_likelihood_obs, ref.log_likelihood_obs, rtol=RTOL)
    m_s, P_s_sqr = rts_smoother(prior, res)
    assert_close(m_s, ref.m_smooth, rtol=RTOL)
    assert_close(cov(P_s_sqr), ref.P_smooth, rtol=RTOL)


@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_hidden_input_with_non_diagonal_xi_matches_dense_kalman(calibration):
    """A joint prior whose state block has a non-diagonal ``Xi``.

    ``x' = L x + B u`` with ``u`` a once-integrated Wiener process. Only the
    state block of the process noise is calibrated, so the oracle scales that
    block alone.
    """
    Xi_x = np.array([[1.0, 0.6], [0.6, 2.0]])
    prior = JointPrior(IWP(q=2, d=2, Xi=Xi_x), IWP(q=1, d=1))
    measure = ODEInformationWithHidden(
        _vf_hidden, prior.E0_x, prior.E1, prior.E0_hidden
    )
    measure.R = R_ODE
    D = prior.E0.shape[1]
    D_x = 6
    m0 = np.zeros(D).at[:2].set(np.array([1.0, 0.0]))
    P0 = _spd(jax.random.PRNGKey(3), D)
    ts = np.linspace(*TSPAN, N + 1)
    H_obs_x = np.array([[1.0, 0.0, 0.0]])
    obs_model, obs = _observations(prior.E0, ts, H_obs_x, jax.random.PRNGKey(4))

    res = gaussian_filter(
        m0,
        _upper_chol(P0),
        prior,
        measure,
        TSPAN,
        N,
        calibration=calibration,
        obs_model=obs_model,
    )
    H = prior.E1 - L @ prior.E0_x - B @ prior.E0_hidden
    steps = _steps(prior, H, ts, obs=obs, H_obs=H_obs_x @ prior.E0)
    ref = dense_kalman(
        m0,
        P0,
        steps,
        calibration=calibration,
        scale_noise=lambda Q, s: Q.at[:D_x, :D_x].multiply(s),
    )

    assert_close(res.m, ref.m, rtol=RTOL)
    assert_close(cov(res.P_sqr), ref.P, rtol=RTOL)
    assert_close(res.log_likelihood, ref.log_likelihood, rtol=RTOL)
    assert res.log_likelihood_obs is not None
    assert_close(res.log_likelihood_obs, ref.log_likelihood_obs, rtol=RTOL)
    assert_close(res.sigma_sqr, ref.sigma_sqr, rtol=RTOL)
    m_s, P_s_sqr = rts_smoother(prior, res)
    assert_close(m_s, ref.m_smooth, rtol=RTOL)
    assert_close(cov(P_s_sqr), ref.P_smooth, rtol=RTOL)
