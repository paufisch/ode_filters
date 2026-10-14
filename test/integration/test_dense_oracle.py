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
    interpolate,
    prepare_observations,
    rts_smoother,
)
from ode_filters.information import Measurement, ODEInformationWithHidden
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
    """Original-coordinate ``(A, Q)`` of one step (also for preconditioned priors)."""
    Q_sqr = prior.Q_sqr(h)
    return prior.A(h), Q_sqr.T @ Q_sqr


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
    measure = measure.with_noise(R_ODE)
    D = prior.E0.shape[1]
    m0 = np.zeros(D).at[:2].set(np.array([1.0, 0.0]))
    P0 = _spd(jax.random.PRNGKey(0), D)
    ts = np.linspace(*TSPAN, N + 1)

    res = gaussian_filter(
        m0,
        _upper_chol(P0),
        prior,
        measure,
        np.linspace(*TSPAN, N + 1),
        calibration=calibration,
    )
    H = prior.E1 - L @ prior.E0
    ref = dense_kalman(m0, P0, _steps(prior, H, ts), calibration=calibration)

    assert_close(res.m, ref.m, rtol=RTOL)
    assert_close(cov(res.P_sqr), ref.P, rtol=RTOL)
    assert_close(res.log_likelihood, ref.log_likelihood, rtol=RTOL)
    assert_close(res.sigma_sqr, ref.sigma_sqr, rtol=RTOL)
    m_s, P_s_sqr = rts_smoother(res)
    assert_close(m_s, ref.m_smooth, rtol=RTOL)
    assert_close(cov(P_s_sqr), ref.P_smooth, rtol=RTOL)


@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_linear_ode_with_observations_matches_dense_kalman(calibration):
    prior = IWP(q=2, d=2)
    measure = ODEInformation(_vf, prior.E0, prior.E1)
    measure = measure.with_noise(R_ODE)
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
        np.linspace(*TSPAN, N + 1),
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
    m_s, P_s_sqr = rts_smoother(res)
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
    measure = measure.with_noise(R_ODE)
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
        np.linspace(*TSPAN, N + 1),
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
    m_s, P_s_sqr = rts_smoother(res)
    assert_close(m_s, ref.m_smooth, rtol=RTOL)
    assert_close(cov(P_s_sqr), ref.P_smooth, rtol=RTOL)


def _bridge(prior, ref, ts, n, tau):
    """Dense filtered and smoothed marginals at ``ts[n] + tau`` (moment form).

    The prior bridges node ``n`` and node ``n + 1`` with the diffusion of step
    ``n`` (``ref.sigma_sqr[n]``): predict from the filtered state to the query
    time, then condition on the smoothed state at the next node through the
    joint Gaussian of the two.
    """
    h = ts[n + 1] - ts[n]
    s = ref.sigma_sqr[n]
    A1, Q1 = prior.A(tau), s * prior.Q(tau)
    A2, Q2 = prior.A(h - tau), s * prior.Q(h - tau)
    m_t = A1 @ ref.m[n]
    P_t = A1 @ ref.P[n] @ A1.T + Q1
    m_ahead = A2 @ m_t
    P_ahead = A2 @ P_t @ A2.T + Q2
    G = P_t @ A2.T @ np.linalg.inv(P_ahead)
    m_s = m_t + G @ (ref.m_smooth[n + 1] - m_ahead)
    P_s = P_t + G @ (ref.P_smooth[n + 1] - P_ahead) @ G.T
    return m_t, P_t, m_s, P_s


def test_interpolation_under_dynamic_calibration_matches_the_dense_bridge():
    """Between nodes, each step's own diffusion bridges it (not a neighbour's).

    The per-step diffusions of this run span five orders of magnitude and
    change by up to a factor of 100 from one step to the next. The query times
    are off the midpoints, where bridging forward over ``tau`` and back over
    ``h - tau`` would not tell the two sub-steps apart.
    """
    prior = IWP(q=2, d=2)
    measure = ODEInformation(_vf, prior.E0, prior.E1).with_noise(R_ODE)
    D = prior.E0.shape[1]
    m0 = np.zeros(D).at[:2].set(np.array([1.0, 0.0]))
    P0 = _spd(jax.random.PRNGKey(5), D)
    ts = np.linspace(*TSPAN, N + 1)
    res = gaussian_filter(m0, _upper_chol(P0), prior, measure, ts)
    ref = dense_kalman(
        m0, P0, _steps(prior, prior.E1 - L @ prior.E0, ts), calibration="dynamic"
    )
    assert_close(res.sigma_sqr, ref.sigma_sqr, rtol=RTOL)

    h = ts[1] - ts[0]
    queries = [(n, frac * h) for n in (0, 2, 9, 17, N - 1) for frac in (0.17, 0.83)]
    expected = [_bridge(prior, ref, ts, n, tau) for n, tau in queries]
    tq = np.array([ts[n] + tau for n, tau in queries])
    filtered = interpolate(res, prior, tq, smoothed=False)
    smoothed = interpolate(res, prior, tq)
    assert_close(filtered.m, np.stack([e[0] for e in expected]), rtol=RTOL)
    assert_close(cov(filtered.P_sqr), np.stack([e[1] for e in expected]), rtol=RTOL)
    assert_close(smoothed.m, np.stack([e[2] for e in expected]), rtol=RTOL)
    assert_close(cov(smoothed.P_sqr), np.stack([e[3] for e in expected]), rtol=RTOL)


@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_sensors_with_different_noises_and_times_match_dense_kalman(calibration):
    """Three sensors, each with its own noise, rows and times, partly coinciding.

    At each step the oracle conditions on the rows of the sensors that have
    data then, with their block of the noise covariance (sensor ``b``'s two rows
    are correlated).
    """
    prior = IWP(q=2, d=2)
    measure = ODEInformation(_vf, prior.E0, prior.E1).with_noise(R_ODE)
    D = prior.E0.shape[1]
    m0 = np.zeros(D).at[:2].set(np.array([1.0, 0.0]))
    P0 = _spd(jax.random.PRNGKey(6), D)
    ts = np.linspace(*TSPAN, N + 1)
    R_b = np.array([[0.04, 0.01], [0.01, 0.09]])
    # (A on x, noise covariance, the nodes with data)
    sensors = [
        (np.array([[1.0, 0.0]]), np.array([[0.0025]]), np.arange(3, N + 1, 3)),
        (np.array([[0.0, 1.0], [1.0, 1.0]]), R_b, np.arange(2, N + 1, 4)),
        (np.array([[0.5, -1.0]]), np.array([[0.3]]), np.array([5, 15, N])),
    ]
    keys = jax.random.split(jax.random.PRNGKey(7), len(sensors))
    data = [
        jax.random.normal(key, (nodes.shape[0], A.shape[0]))
        for key, (A, _, nodes) in zip(keys, sensors, strict=True)
    ]
    obs_model = prepare_observations(
        [
            Measurement(A, z, ts[nodes], noise=R)
            for (A, R, nodes), z in zip(sensors, data, strict=True)
        ],
        prior.E0,
        ts,
    )

    res = gaussian_filter(
        m0,
        _upper_chol(P0),
        prior,
        measure,
        ts,
        calibration=calibration,
        obs_model=obs_model,
    )
    H = prior.E1 - L @ prior.E0
    A_step, Q_step = _transition(prior, ts[1] - ts[0])
    steps = []
    for node in range(1, N + 1):
        rows, offsets, noises = [], [], []
        for (A, R, nodes), z in zip(sensors, data, strict=True):
            hit = np.flatnonzero(nodes == node)
            if hit.shape[0]:
                rows.append(A @ prior.E0)
                offsets.append(-z[int(hit[0])])
                noises.append(R)
        observed = {}
        if rows:
            observed = dict(
                H_obs=np.concatenate(rows),
                c_obs=np.concatenate(offsets),
                R_obs=jax.scipy.linalg.block_diag(*noises),
            )
        steps.append(
            DenseStep(
                A=A_step,
                Q=Q_step,
                H=H,
                c=np.zeros(2),
                R=R_ODE * np.eye(2),
                **observed,
            )
        )
    ref = dense_kalman(m0, P0, steps, calibration=calibration)

    assert_close(res.m, ref.m, rtol=RTOL)
    assert_close(cov(res.P_sqr), ref.P, rtol=RTOL)
    assert_close(res.log_likelihood, ref.log_likelihood, rtol=RTOL)
    assert res.log_likelihood_obs is not None
    assert_close(res.log_likelihood_obs, ref.log_likelihood_obs, rtol=RTOL)
    m_s, P_s_sqr = rts_smoother(res)
    assert_close(m_s, ref.m_smooth, rtol=RTOL)
    assert_close(cov(P_s_sqr), ref.P_smooth, rtol=RTOL)
