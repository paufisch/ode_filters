"""Tests for ekf1_sqr_loop_dynamic_scan with external observations (obs_model).

The obs_model branch combines online sigma^2 calibration with masked
observation updates -- the construction used for joint state-parameter
estimation from sensor data. Reference problem: scalar exponential decay
dx/dt = -lam * x with the rate lam as a hidden parameter, observed
noisily in x only.
"""

from __future__ import annotations

import jax.numpy as np
import jax.random as jrandom
import numpy as onp
import pytest

from ode_filters.filters import (
    ekf1_sqr_loop_dynamic_scan,
    ekf1_sqr_loop_sequential_scan,
)
from ode_filters.measurement import (
    Measurement,
    ODEInformationWithHidden,
    prepare_observations,
)
from ode_filters.priors import IWP, JointPrior

LAM_TRUE = 0.8
LAM_PRIOR = 0.4
X0 = 2.0
TSPAN = (0.0, 6.0)
N = 300
NOISE_STD = 0.01


def _decay_setup():
    """Joint prior, measurement model, obs_model, and init for the decay ODE."""
    ts = np.linspace(TSPAN[0], TSPAN[1], N + 1)

    # Noisy observations of x(t) = X0 * exp(-lam * t) at every step.
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    key = jrandom.PRNGKey(0)
    z = (x_true + NOISE_STD * jrandom.normal(key, shape=(N,))).reshape(-1, 1)
    measurement = Measurement(np.eye(1), z, ts[1:], noise=NOISE_STD**2)

    prior_x = IWP(q=2, d=1, Xi=np.eye(1))
    prior_lam = IWP(q=0, d=1, Xi=1e-10 * np.eye(1))
    joint = JointPrior(prior_x, prior_lam)

    def vf(x, u, *, t):
        return -u * x

    measure = ODEInformationWithHidden(
        vf=vf,
        E0=joint.E0_x,
        E1=joint.E1,
        E0_hidden=joint.E0_hidden,
    )
    obs_model = prepare_observations([measurement], joint.E0_x, ts)

    # Init: x-block from the prior-mean rate, lam-block at the (wrong) prior.
    mu_x = np.array([X0, -LAM_PRIOR * X0, LAM_PRIOR**2 * X0])
    mu_0 = np.concatenate([mu_x, np.array([LAM_PRIOR])])
    S_0 = np.diag(np.array([1e-3, 1e-1, 1.0, 0.5]))
    return joint, measure, obs_model, mu_0, S_0


def test_calibration_none_matches_sequential_scan():
    """With calibration='none', the obs branch is the sequential scan."""
    joint, measure, obs_model, mu_0, S_0 = _decay_setup()

    m_dyn, P_dyn_sqr, *_rest_dyn = ekf1_sqr_loop_dynamic_scan(
        mu_0,
        S_0,
        joint,
        measure,
        TSPAN,
        N,
        calibration="none",
        obs_model=obs_model,
    )
    m_seq, P_seq_sqr, *_rest_seq = ekf1_sqr_loop_sequential_scan(
        mu_0, S_0, joint, measure, TSPAN, N, obs_model=obs_model
    )

    onp.testing.assert_allclose(onp.asarray(m_dyn), onp.asarray(m_seq), atol=1e-10)
    onp.testing.assert_allclose(
        onp.asarray(P_dyn_sqr.transpose(0, 2, 1) @ P_dyn_sqr),
        onp.asarray(P_seq_sqr.transpose(0, 2, 1) @ P_seq_sqr),
        atol=1e-10,
    )


@pytest.mark.parametrize("calibration", ["dynamic", "none"])
def test_hidden_parameter_converges(calibration):
    """The hidden decay rate is identified from noisy x-observations."""
    joint, measure, obs_model, mu_0, S_0 = _decay_setup()

    result = ekf1_sqr_loop_dynamic_scan(
        mu_0,
        S_0,
        joint,
        measure,
        TSPAN,
        N,
        calibration=calibration,
        obs_model=obs_model,
    )
    m_seq, P_seq_sqr = result[0], result[1]

    lam_idx = joint.E0_hidden.argmax()
    lam_hat = float(m_seq[-1, lam_idx])
    P_final = P_seq_sqr[-1].T @ P_seq_sqr[-1]
    lam_std = float(np.sqrt(P_final[lam_idx, lam_idx]))

    assert abs(lam_hat - LAM_TRUE) < 0.05, f"lam_hat={lam_hat}"
    assert lam_std < 0.1  # posterior tightened from the 0.5 prior std
    assert abs(lam_hat - LAM_TRUE) < 4 * lam_std + 0.02  # roughly calibrated


def test_obs_result_layout_and_likelihoods():
    """Obs branch returns the 14-element layout with finite likelihoods."""
    joint, measure, obs_model, mu_0, S_0 = _decay_setup()

    result = ekf1_sqr_loop_dynamic_scan(
        mu_0,
        S_0,
        joint,
        measure,
        TSPAN,
        N,
        calibration="dynamic",
        obs_model=obs_model,
    )
    assert len(result) == 14
    m_seq, P_seq_sqr = result[0], result[1]
    sigma_sqr_seq, ll_ode, ll_obs = result[11], result[12], result[13]
    assert m_seq.shape == (N + 1, mu_0.shape[0])
    assert P_seq_sqr.shape == (N + 1, mu_0.shape[0], mu_0.shape[0])
    assert sigma_sqr_seq.shape == (N,)
    assert np.isfinite(ll_ode) and np.isfinite(ll_obs)


def test_no_obs_backward_compatible():
    """obs_model=None keeps the original 11-element DynamicScanLoopResult."""
    joint, measure, _obs, mu_0, S_0 = _decay_setup()

    result = ekf1_sqr_loop_dynamic_scan(
        mu_0, S_0, joint, measure, TSPAN, N, calibration="dynamic"
    )
    assert len(result) == 11
    assert result[0].shape == (N + 1, mu_0.shape[0])
