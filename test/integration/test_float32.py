"""A solve in single precision.

The suite runs in float64 (``conftest.py``), which the docs recommend. This
checks that the solvers still run in JAX's default float32: no dtype errors, a
finite posterior, and an accuracy at the float32 round-off level.
"""

from __future__ import annotations

import jax
import jax.numpy as np

from ode_filters import (
    IWP,
    ODEInformation,
    gaussian_filter,
    gaussian_filter_adaptive,
    rts_smoother,
    taylor_mode_initialization,
)


def _logistic(x, *, t):
    return x * (1.0 - x)


def _exact(t):
    return 1.0 / (1.0 + 9.0 * np.exp(-t))


def test_fixed_grid_and_adaptive_solves_run_in_float32():
    with jax.enable_x64(False):
        prior = IWP(q=2, d=1)
        m_0, P_0_sqr = taylor_mode_initialization(_logistic, np.array([0.1]), q=2)
        measure = ODEInformation(_logistic, prior.E0, prior.E1)
        ts = np.linspace(0.0, 5.0, 51)
        result = gaussian_filter(m_0, P_0_sqr, prior, measure, ts)
        smoothed = rts_smoother(result)
        adaptive = gaussian_filter_adaptive(
            m_0, P_0_sqr, prior, measure, np.array([0.0, 2.5, 5.0])
        )
        assert smoothed.m.dtype == np.float32
        assert bool(np.all(np.isfinite(smoothed.m)))
        assert bool(np.all(np.isfinite(smoothed.P_sqr)))
        assert float(np.max(np.abs(smoothed.m[:, 0] - _exact(ts)))) < 1e-3
        assert bool(adaptive.success)
        assert float(np.max(np.abs(adaptive.m[:, 0] - _exact(adaptive.t)))) < 1e-2
