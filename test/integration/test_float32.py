"""A solve in single precision.

The suite runs in float64 (``conftest.py``), which the docs recommend. This
checks that the solvers still run in JAX's default float32: no dtype errors, a
finite posterior, and an accuracy at the float32 round-off level.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    ODEInformation,
    gaussian_filter,
    gaussian_filter_adaptive,
    rts_smoother,
    taylor_mode_initialization,
)


@pytest.fixture(autouse=True)
def _fresh_jax_caches():
    """Keep the float32 solve from leaking into later float64 tests.

    From JAX 0.9.2 on (0.9.1 is fine; still in 0.11.2), a NumPy constant used
    under ``jit`` keeps its device copy across a change of ``jax_enable_x64``,
    keyed by the array object: after a float32 solve, the cached prior
    constants (``_iwp_constants``) came back as float32 in the float64 tests
    that ran next in the same process, eight digits short. A JAX bug.
    """
    jax.clear_caches()
    yield
    jax.clear_caches()


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
