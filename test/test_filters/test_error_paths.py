"""Error-path coverage for the filter loops, the adaptive driver, and corrections."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters import IteratedTaylorCorrection, gaussian_filter
from ode_filters.filters.ode_filter_adaptive import sqr_adaptive_loop
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, PrecondIWP, taylor_mode_initialization


def _logistic(prior_cls=IWP):
    def vf(x, *, t):
        return x * (1.0 - x)

    prior = prior_cls(q=2, d=1)
    mu_0, P0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return prior, measure, mu_0, P0_sqr


def test_iterated_correction_rejects_bad_max_iters():
    with pytest.raises(ValueError, match="max_iters must be >= 1"):
        IteratedTaylorCorrection(max_iters=0)


def test_gaussian_filter_rejects_unknown_calibration():
    prior, measure, mu_0, P0_sqr = _logistic()
    with pytest.raises(ValueError, match="calibration must be one of"):
        gaussian_filter(
            mu_0, P0_sqr, prior, measure, (0.0, 1.0), N=5, calibration="bogus"
        )


def test_gaussian_filter_preconditioned_rejects_unknown_calibration():
    prior, measure, mu_0, P0_sqr = _logistic(PrecondIWP)
    with pytest.raises(ValueError, match="calibration must be one of"):
        gaussian_filter(
            mu_0, P0_sqr, prior, measure, (0.0, 1.0), N=5, calibration="bogus"
        )


def test_adaptive_loop_rejects_nonincreasing_tspan():
    prior, measure, mu_0, P0_sqr = _logistic()
    with pytest.raises(ValueError, match="tspan must be increasing"):
        sqr_adaptive_loop(mu_0, P0_sqr, prior, measure, (5.0, 0.0))


def test_adaptive_loop_exceeds_max_steps():
    prior, measure, mu_0, P0_sqr = _logistic()
    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        sqr_adaptive_loop(mu_0, P0_sqr, prior, measure, (0.0, 5.0), max_steps=1)
