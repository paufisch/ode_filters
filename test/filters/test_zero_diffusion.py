"""Smoothing through a step whose calibrated diffusion is zero.

From an exact (Taylor-mode) initial state, a calibrated diffusion of exactly
zero -- the scalar one at an equilibrium, or the component of the diagonal one
whose residual vanishes, as for a pendulum released at rest -- leaves the first
step's predicted covariance singular. The filter itself is fine, but the
backward conditional divides by that covariance in a triangular solve
(``invert``), so it is NaN, and so is the smoothed state at the start. A
positive ``min_sigma_sqr`` floor avoids it.
"""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    DiagonalCalibration,
    DynamicCalibration,
    ODEInformation,
    gaussian_filter,
    rts_smoother,
    taylor_mode_initialization,
)

TS = np.linspace(0.0, 2.0, 11)


def _logistic(x, *, t):
    return x * (1.0 - x)


def _pendulum(x, *, t):
    return np.array([x[1], -np.sin(x[0])])


# (vector field, initial value, calibration policy without a floor)
ZERO_DIFFUSION = {
    "equilibrium-dynamic": (_logistic, [1.0], DynamicCalibration),
    "at-rest-diagonal": (_pendulum, [1.0, 0.0], DiagonalCalibration),
}


def _solve(name, min_sigma_sqr):
    vf, x0, policy = ZERO_DIFFUSION[name]
    x0 = np.array(x0)
    prior = IWP(q=2, d=x0.shape[0])
    m_0, P_0_sqr = taylor_mode_initialization(vf, x0, q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    calibration = policy(min_sigma_sqr=min_sigma_sqr)
    return gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration=calibration)


@pytest.mark.parametrize("name", list(ZERO_DIFFUSION))
def test_the_first_diffusion_is_exactly_zero(name):
    """Guard the premise, and that the filter itself stays finite."""
    result = _solve(name, 0.0)
    assert float(np.min(result.sigma_sqr[0])) == 0.0
    assert bool(np.all(np.isfinite(result.m)))
    assert bool(np.all(np.isfinite(result.P_sqr)))


@pytest.mark.xfail(
    strict=True, reason="invert divides by the singular predicted covariance"
)
@pytest.mark.parametrize("name", list(ZERO_DIFFUSION))
def test_smoothing_through_a_zero_diffusion_is_finite(name):
    smoothed = rts_smoother(_solve(name, 0.0))
    assert bool(np.all(np.isfinite(smoothed.m)))


@pytest.mark.parametrize("name", list(ZERO_DIFFUSION))
def test_a_diffusion_floor_keeps_the_smoother_finite(name):
    result = _solve(name, 1e-12)
    smoothed = rts_smoother(result)
    assert bool(np.all(np.isfinite(smoothed.m)))
    assert bool(np.all(np.isfinite(smoothed.P_sqr)))
    # The smoother ends where the filter does, and starts at the exact value.
    assert np.allclose(smoothed.m[-1], result.m[-1], rtol=1e-14)
    assert np.allclose(smoothed.m[0], result.m[0], atol=1e-12)
