"""Filtering with data: the ``obs_model`` path of the filters.

Reference problem: scalar exponential decay ``x' = -lam x`` with the rate ``lam``
a hidden state, observed noisily in ``x`` only.
"""

from __future__ import annotations

import jax.numpy as np
import jax.random as jrandom
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    JointPrior,
    Measurement,
    ODEInformation,
    ODEInformationWithHidden,
    gaussian_filter,
    gaussian_filter_adaptive,
    prepare_observations,
    taylor_mode_initialization,
)
from test.helpers import assert_close, cov

LAM_TRUE = 0.8
LAM_PRIOR = 0.4
X0 = 2.0
TSPAN = (0.0, 6.0)
N = 300
NOISE_STD = 0.01


def _decay_setup():
    ts = np.linspace(TSPAN[0], TSPAN[1], N + 1)
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    z = (x_true + NOISE_STD * jrandom.normal(jrandom.PRNGKey(0), (N,))).reshape(-1, 1)
    measurement = Measurement(np.eye(1), z, ts[1:], noise=NOISE_STD**2)

    joint = JointPrior(IWP(q=2, d=1), IWP(q=0, d=1, Xi=1e-10 * np.eye(1)))

    def vf(x, u, *, t):
        return -u * x

    measure = ODEInformationWithHidden(vf, joint.E0_x, joint.E1, joint.E0_hidden)
    obs_model = prepare_observations([measurement], joint.E0_x, ts)
    m_x = np.array([X0, -LAM_PRIOR * X0, LAM_PRIOR**2 * X0])
    m_0 = np.concatenate([m_x, np.array([LAM_PRIOR])])
    P_0_sqr = np.diag(np.array([1e-3, 1e-1, 1.0, 0.5]))
    return ts, joint, measure, obs_model, m_0, P_0_sqr


@pytest.mark.parametrize("calibration", ["dynamic", "none"])
def test_hidden_parameter_is_identified(calibration):
    ts, joint, measure, obs_model, m_0, P_0_sqr = _decay_setup()
    res = gaussian_filter(
        m_0, P_0_sqr, joint, measure, ts, calibration=calibration, obs_model=obs_model
    )
    i = int(joint.E0_hidden.argmax())
    lam_hat = float(res.m[-1, i])
    lam_std = float(np.sqrt(cov(res.P_sqr[-1])[i, i]))
    assert abs(lam_hat - LAM_TRUE) < 0.05, f"lam_hat={lam_hat}"
    assert lam_std < 0.1  # tightened from the 0.5 prior std
    assert abs(lam_hat - LAM_TRUE) < 4 * lam_std + 0.02


def _innovation_loglik(mz, Pz_sqr):
    """``sum_n log N(mz_n; 0, Pz_n)`` of scalar innovations."""
    variance = cov(Pz_sqr)[:, 0, 0]
    return np.sum(-0.5 * (np.log(2 * np.pi * variance) + mz[:, 0] ** 2 / variance))


def test_result_layout_and_likelihoods():
    ts, joint, measure, obs_model, m_0, P_0_sqr = _decay_setup()
    res = gaussian_filter(m_0, P_0_sqr, joint, measure, ts, obs_model=obs_model)
    D = m_0.shape[0]
    assert res.m.shape == (N + 1, D)
    assert res.P_sqr.shape == (N + 1, D, D)
    assert res.sigma_sqr.shape == (N,)
    assert res.mz_obs.shape == (N, 1)
    assert res.Pz_obs_sqr.shape == (N, 1, 1)
    # Every step has data, so each evidence is the density of its innovations.
    assert_close(res.log_likelihood, _innovation_loglik(res.mz, res.Pz_sqr), rtol=1e-12)
    expected = _innovation_loglik(res.mz_obs, res.Pz_obs_sqr)
    assert_close(res.log_likelihood_obs, expected, rtol=1e-12)


def test_without_data_the_observation_fields_are_none():
    ts, joint, measure, _, m_0, P_0_sqr = _decay_setup()
    res = gaussian_filter(m_0, P_0_sqr, joint, measure, ts)
    assert res.log_likelihood_obs is None
    assert res.mz_obs is None and res.Pz_obs_sqr is None


def _two_sensors(times_b):
    def vf(x, *, t):
        return np.array([-0.5 * x[0] + x[1], -x[0] - 0.5 * x[1]])

    prior = IWP(q=2, d=2)
    ts = np.linspace(0.0, 2.0, 21)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([1.0, 0.0]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    times_a = ts[4::4]
    sensor_a = Measurement(
        np.array([[1.0, 0.0]]),
        0.5 * np.ones((times_a.shape[0], 1)),
        times_a,
        noise=1e-3,
    )
    sensor_b = Measurement(
        np.array([[0.0, 1.0]]),
        -0.2 * np.ones((times_b.shape[0], 1)),
        times_b,
        noise=1e-3,
    )
    return prior, measure, ts, m_0, P_0_sqr, sensor_a, sensor_b


def test_asynchronous_sensors_give_partial_mask_rows():
    """Sensors at different times: each step conditions on the rows with data."""
    ts = np.linspace(0.0, 2.0, 21)
    prior, measure, ts, m_0, P_0_sqr, sensor_a, sensor_b = _two_sensors(ts[2::4])
    both = prepare_observations([sensor_a, sensor_b], prior.E0, ts)
    assert both is not None
    assert set(onp.asarray(both.mask).sum(axis=1).tolist()) == {0, 1}
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, obs_model=both)
    assert bool(np.all(np.isfinite(res.m))) and np.isfinite(res.log_likelihood_obs)


def test_a_sensor_without_data_on_the_grid_changes_nothing():
    ts = np.linspace(0.0, 2.0, 21)
    prior, measure, ts, m_0, P_0_sqr, sensor_a, sensor_b = _two_sensors(ts[3:4])
    # Sensor b's single time is on the grid but outside the steps compared.
    both = prepare_observations([sensor_a, sensor_b], prior.E0, ts)
    a_only = prepare_observations([sensor_a], prior.E0, ts)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, obs_model=both)
    res_a = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, obs_model=a_only)
    # Before sensor b fires (node 3), the states agree exactly.
    assert_close(res.m[:3], res_a.m[:3], rtol=1e-12)
    assert_close(cov(res.P_sqr[:3]), cov(res_a.P_sqr[:3]), rtol=1e-10)
    # Afterwards the evidence differs by sensor b's single observation only.
    ll_b = res.log_likelihood_obs - res_a.log_likelihood_obs
    assert np.isfinite(ll_b) and float(ll_b) != 0.0


def test_measurement_times_outside_the_grid_raise():
    prior = IWP(q=2, d=1)
    ts = np.linspace(0.0, 2.0, 11)
    for times, match in [([0.0], "at or before"), ([2.5], "after the final")]:
        data = Measurement(np.eye(1), np.ones((1, 1)), np.array(times))
        with pytest.raises(ValueError, match=match):
            prepare_observations([data], prior.E0, ts)


@pytest.mark.parametrize("merged", [False, True])
def test_off_grid_data_on_data_only_nodes_or_a_merged_grid(merged):
    def vf(x, *, t):
        return -x

    prior = IWP(q=2, d=1)
    t_obs = np.array([0.33, 1.71])
    ts = onp.linspace(0.0, 2.0, 11)
    if merged:
        ts = onp.union1d(ts, onp.asarray(t_obs))
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
    data = Measurement(np.eye(1), np.exp(-t_obs)[:, None], t_obs, noise=1e-4)
    obs_model = prepare_observations([data], prior.E0, ts)
    assert obs_model is not None and (obs_model.t is None) == merged
    res = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        ODEInformation(vf, prior.E0, prior.E1),
        ts,
        obs_model=obs_model,
    )
    assert int(onp.asarray(obs_model.mask).sum()) == 2
    assert res.t.shape == (13,)
    assert_close(res.m[:, 0], np.exp(-res.t), atol=1e-3)


def test_adaptive_path_reports_the_data_evidence_separately():
    def vf(x, *, t):
        return -x

    prior = IWP(q=2, d=1)
    save_at = np.linspace(0.0, 2.0, 5)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    data = Measurement(
        np.eye(1), np.exp(-save_at[1:])[:, None], save_at[1:], noise=1e-4
    )
    obs_model = prepare_observations([data], prior.E0, save_at)
    with_data = gaussian_filter_adaptive(
        m_0, P_0_sqr, prior, measure, save_at, obs_model=obs_model
    )
    without = gaussian_filter_adaptive(m_0, P_0_sqr, prior, measure, save_at)
    assert with_data.log_likelihood_obs is not None
    assert without.log_likelihood_obs is None
    # Data at every save time: the evidence is the density of its innovations.
    expected = _innovation_loglik(with_data.mz_obs, with_data.Pz_obs_sqr)
    assert_close(with_data.log_likelihood_obs, expected, rtol=1e-12)
