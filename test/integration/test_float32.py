"""A solve in single precision.

The suite runs in float64 (``conftest.py``), which the docs recommend. This
checks that the solvers still run in JAX's default float32: no dtype errors, a
finite posterior, and an accuracy at the float32 round-off level.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
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
    keyed by the array object. The library's cached constants are fresh copies
    for that reason (``priors/_constants.py``, tested below); clearing the
    caches keeps any other constant from leaking between tests.
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


# A NumPy grid: it takes the precision of the run it enters.
GRID = onp.linspace(0.0, 2.0, 11)


def _solve():
    """``eqx.filter_jit`` of a solve that builds its prior inside the trace."""

    def run(x0, s):
        prior = IWP(q=2, d=1, Xi=np.exp(s) * np.eye(1))
        m_0, P_0_sqr = taylor_mode_initialization(_logistic, x0, q=2)
        measure = ODEInformation(_logistic, prior.E0, prior.E1)
        return gaussian_filter(m_0, P_0_sqr, prior, measure, GRID).m

    return eqx.filter_jit(run)(onp.array([0.1]), 0.3)


def test_switching_precision_in_one_process_leaves_both_solves_intact():
    """The JAX bug above, without clearing the caches in between.

    With the cached IWP constants shared between traces, a float32 solve after
    a float64 one warned about truncating the float64 constants (an error in
    this suite). This is the ``eqx.filter_jit`` pattern the docs recommend; a
    plain ``jax.jit`` of the same solve did not show it.
    """
    reference = _solve()
    with jax.enable_x64(False):
        single = _solve()
    again = _solve()
    assert single.dtype == np.float32
    assert again.dtype == np.float64
    assert bool(np.all(again == reference))
