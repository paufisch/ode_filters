"""Tests for the ``min_sigma_sqr`` clamp on per-step diffusion estimates.

A trivial fixed point of the ODE (vector field vanishing at zero, ``x0=0``)
gives an identically zero ODE residual on every step. The per-step quasi-MLE
of sigma^2 then collapses to exactly 0; baking that into Q_step zeroes the
state-block diffusion; the next prediction is again zero-residual; and the
trajectory propagates with zero state covariance, ending in NaN once the
Cholesky / log-likelihood touches a singular Pz_sqr.

The ``min_sigma_sqr`` keyword on the dynamic loops clamps the per-step
estimate from below before it enters ``apply_state_sigma_sqr``. Default is
``0.0`` (backward-compatible); users on collapse-prone problems opt in to a
small positive value.
"""

from __future__ import annotations

import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.filters import (
    ekf1_sqr_adaptive_loop,
    ekf1_sqr_loop_dynamic,
    ekf1_sqr_loop_dynamic_scan,
)
from ode_filters.measurement.measurement_models import ODEInformation
from ode_filters.priors.gmp_priors import IWP, taylor_mode_initialization


def _vf_trivial_zero(x, *, t):
    """Vector field that vanishes at the origin; x0=0 is a fixed point."""
    return -x


@pytest.fixture
def trivial_zero_setup():
    prior = IWP(q=2, d=1)
    x0 = np.array([0.0])
    mu_0, S0 = taylor_mode_initialization(_vf_trivial_zero, x0, q=2)
    measure = ODEInformation(_vf_trivial_zero, prior.E0, prior.E1)
    return prior, mu_0, S0, measure


class TestSigmaFloorFixedStep:
    """The fixed-step dynamic loop collapses without a floor and recovers
    with one."""

    def test_default_no_clamp_collapses_sigma_to_zero(self, trivial_zero_setup):
        prior, mu_0, S0, measure = trivial_zero_setup
        r = ekf1_sqr_loop_dynamic(
            mu_0, S0, prior, measure, (0.0, 1.0), 10, calibration="dynamic"
        )
        sigmas = onp.asarray(r[-2], dtype=float)
        # With default min_sigma_sqr=0, the trivial-IC trajectory collapses:
        # every per-step sigma is exactly 0 (residual is identically 0 and
        # the math has no defensive floor).
        assert onp.all(sigmas == 0.0)

    def test_floor_keeps_sigma_above_threshold(self, trivial_zero_setup):
        prior, mu_0, S0, measure = trivial_zero_setup
        floor = 1e-30
        r = ekf1_sqr_loop_dynamic(
            mu_0,
            S0,
            prior,
            measure,
            (0.0, 1.0),
            10,
            calibration="dynamic",
            min_sigma_sqr=floor,
        )
        sigmas = onp.asarray(r[-2], dtype=float)
        assert onp.all(sigmas >= floor)
        # Finite final mean / covariance with the floor active.
        assert onp.all(onp.isfinite(onp.asarray(r[0][-1])))
        assert onp.all(onp.isfinite(onp.asarray(r[1][-1])))


class TestSigmaFloorScan:
    """Same behaviour in the JIT-scan loop."""

    def test_floor_keeps_sigma_above_threshold(self, trivial_zero_setup):
        prior, mu_0, S0, measure = trivial_zero_setup
        floor = 1e-30
        r = ekf1_sqr_loop_dynamic_scan(
            mu_0,
            S0,
            prior,
            measure,
            (0.0, 1.0),
            10,
            calibration="dynamic",
            min_sigma_sqr=floor,
        )
        sigmas = onp.asarray(r[-2], dtype=float)
        assert onp.all(sigmas >= floor)
        assert onp.all(onp.isfinite(onp.asarray(r[0][-1])))


class TestSigmaFloorAdaptive:
    """The adaptive loop accumulates per-step sigma into a running mean used
    by the error norm. A single zero-sigma step would poison the running
    mean; the floor prevents this."""

    def test_floor_keeps_sigma_above_threshold(self, trivial_zero_setup):
        prior, mu_0, S0, measure = trivial_zero_setup
        floor = 1e-30
        r = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            (0.0, 1.0),
            calibration="dynamic",
            min_sigma_sqr=floor,
        )
        sigmas = onp.asarray([float(onp.asarray(s).flat[0]) for s in r.sigma_sqr_seq])
        assert onp.all(sigmas >= floor)
        assert onp.all(onp.isfinite(onp.asarray(r.m_seq[-1])))


class TestSigmaFloorIsOptIn:
    """``min_sigma_sqr=0`` (default) must reproduce the unclamped result on
    non-collapsing problems exactly."""

    def test_default_matches_zero_explicit(self):
        prior = IWP(q=2, d=1)

        def vf(x, *, t):
            return x * (1 - x)

        x0 = np.array([0.1])
        mu_0, S0 = taylor_mode_initialization(vf, x0, q=2)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        r_default = ekf1_sqr_loop_dynamic(
            mu_0, S0, prior, measure, (0.0, 1.0), 20, calibration="dynamic"
        )
        r_zero = ekf1_sqr_loop_dynamic(
            mu_0,
            S0,
            prior,
            measure,
            (0.0, 1.0),
            20,
            calibration="dynamic",
            min_sigma_sqr=0.0,
        )
        assert np.allclose(r_default[0][-1], r_zero[0][-1], atol=0.0)
        assert np.allclose(
            onp.asarray(r_default[-2], dtype=float),
            onp.asarray(r_zero[-2], dtype=float),
            atol=0.0,
        )
