"""``marginal_loglik`` with adaptive steps (``InferenceProblem.adaptive``).

Reference problem: exponential decay ``x' = -lam x`` from ``x(0) = 2``, observed
with noise at the ten nodes of the save grid ``ts`` after ``ts[0]``. At tight
tolerances the adaptive values agree with a fixed grid two hundred times finer
than ``ts``.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    AdaptiveSettings,
    InferenceProblem,
    Measurement,
    ODEInformation,
    gaussian_filter,
    gaussian_filter_adaptive,
    marginal_loglik,
    prepare_observations,
    taylor_mode_initialization,
)
from test.helpers import assert_close, assert_grad_matches_fd

PRIOR = IWP(q=2, d=1)
TS = onp.linspace(0.0, 5.0, 11)
TIGHT = AdaptiveSettings(atol=1e-8, rtol=1e-8)
THETA = np.array(0.7)


def _measurement() -> Measurement:
    t_obs = TS[1:]
    noise = onp.random.default_rng(1).standard_normal(t_obs.shape)
    y = 2.0 * onp.exp(-0.8 * t_obs) + 0.02 * noise
    return Measurement(np.eye(1), y[:, None], t_obs, noise=0.02**2)


def _build(theta):
    def vf(x, *, t):
        return -theta * x

    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([2.0]), q=2)
    return m_0, P_0_sqr, ODEInformation(vf, PRIOR.E0, PRIOR.E1)


def _problem(ts, adaptive=None):
    data = prepare_observations([_measurement()], PRIOR.E0, ts)
    return InferenceProblem(_build, PRIOR, ts, adaptive=adaptive), data


def test_the_default_is_the_fixed_grid():
    problem = InferenceProblem(_build, PRIOR, TS, "none", None, None)
    assert problem.adaptive is None


@pytest.mark.parametrize("channel", ["obs", "fenrir"])
def test_values_agree_with_a_fine_fixed_grid(channel):
    adaptive, data = _problem(TS, TIGHT)
    fine, fine_data = _problem(onp.linspace(0.0, 5.0, 2001))
    value = marginal_loglik(THETA, data, model=adaptive, channel=channel)
    expected = marginal_loglik(THETA, fine_data, model=fine, channel=channel)
    # The differences are about 5e-7 on values of about -24.
    assert_close(value, expected, rtol=1e-7)


def test_the_channels_are_the_evidences_of_the_adaptive_solve():
    """``"ode"`` sums over the solver's own steps; on a fixed grid of exactly
    those steps the filter reproduces both evidences."""
    problem, data = _problem(TS, TIGHT)
    ll_ode, ll_obs = marginal_loglik(THETA, data, model=problem, channel="both")
    assert_close(
        marginal_loglik(THETA, data, model=problem, channel="ode"), ll_ode, rtol=0.0
    )
    assert_close(marginal_loglik(THETA, data, model=problem), ll_obs, rtol=0.0)

    m_0, P_0_sqr, measure = _build(THETA)
    solve = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        PRIOR,
        measure,
        TS,
        calibration="none",
        obs_model=data,
        atol=TIGHT.atol,
        rtol=TIGHT.rtol,
        record_steps=4096,
    )
    assert solve.steps is not None
    steps = onp.asarray(solve.steps.t)[: int(solve.steps.count)]
    grid = onp.concatenate([TS[:1], steps])
    on_grid = gaussian_filter(
        m_0,
        P_0_sqr,
        PRIOR,
        measure,
        grid,
        calibration="none",
        obs_model=prepare_observations([_measurement()], PRIOR.E0, grid),
    )
    assert on_grid.log_likelihood_obs is not None
    assert_close(ll_ode, on_grid.log_likelihood, rtol=1e-11)
    assert_close(ll_obs, on_grid.log_likelihood_obs, rtol=1e-11)


@pytest.mark.parametrize("channel", ["obs", "fenrir"])
def test_grad_matches_finite_differences(channel):
    problem, data = _problem(TS, TIGHT)
    assert_grad_matches_fd(
        lambda th: marginal_loglik(th, data, model=problem, channel=channel),
        THETA,
        rtol=1e-4,
    )


def test_jit_and_vmap():
    problem, data = _problem(TS, AdaptiveSettings(atol=1e-6, rtol=1e-6))

    def loglik(theta):
        return marginal_loglik(theta, data, model=problem)

    thetas = np.array([0.6, 0.9])
    eager = np.stack([loglik(theta) for theta in thetas])
    assert np.all(np.isfinite(eager))
    assert_close(jax.jit(loglik)(thetas[0]), eager[0], rtol=1e-12)
    assert_close(jax.vmap(loglik)(thetas), eager, rtol=1e-12)


def test_a_failed_solve_returns_nan():
    problem, data = _problem(TS, AdaptiveSettings(atol=1e-10, rtol=1e-10, max_steps=3))
    assert np.isnan(marginal_loglik(THETA, data, model=problem))
    assert np.isnan(marginal_loglik(THETA, data, model=problem, channel="fenrir"))
    ll_ode, ll_obs = marginal_loglik(THETA, data, model=problem, channel="both")
    assert np.isnan(ll_ode) and np.isnan(ll_obs)


def test_dalton_is_not_available_with_adaptive_steps():
    problem, data = _problem(TS, TIGHT)
    with pytest.raises(ValueError, match="not available with adaptive steps"):
        marginal_loglik(THETA, data, model=problem, channel="dalton")
