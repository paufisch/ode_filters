"""Non-autonomous ODEs: every solver must evaluate the vector field at the right time.

A step from ``t_n`` to ``t_{n+1}`` conditions on the ODE at ``t_{n+1}``. For an
autonomous vector field the time is never read, so a solver that passed
``t_n`` (or any other time) would go unnoticed; here the solutions depend on
it. Evaluating one step early moves the solution of ``x' = cos(t) x`` on the
grid below by about ``1e-1``; the solvers are accurate to better than ``1e-6``
there, and the tolerances below leave a margin of ten.

The initial states are the exact derivatives of the closed-form solutions.
(``taylor_mode_initialization`` holds the time fixed at ``t0`` while it
differentiates, so for a non-autonomous vector field its higher derivatives
are those of the frozen-time ODE.)
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    ODEInformation,
    PrecondIWP,
    SecondOrderODEInformation,
    affine_filter,
    gaussian_filter,
    gaussian_filter_adaptive,
    iterated_smoother,
    rts_smoother,
    taylor_surrogate,
)
from test.helpers import assert_close, cov

TSPAN = (0.0, 5.0)
N = 200


def _cos_vf(x, *, t):
    return np.cos(t) * x


def _cos_solution(t):
    """``x(t) = exp(sin t)`` solves ``x' = cos(t) x``, ``x(0) = 1``."""
    return np.exp(np.sin(t))


def _forced_vf(x, v, *, t):
    return -x + np.cos(2.0 * t)


def _forced_solution(t):
    """``x'' = -x + cos(2 t)``, ``x(0) = 1``, ``x'(0) = 0``."""
    return 4.0 / 3.0 * np.cos(t) - np.cos(2.0 * t) / 3.0


def _exact_state(solution, q: int) -> tuple[jax.Array, jax.Array]:
    """``[x, x', ..., x^(q)]`` of a scalar closed-form solution at ``t = 0``."""
    derivatives, f = [], solution
    for _ in range(q + 1):
        derivatives.append(f(0.0))
        f = jax.grad(f)
    return np.array(derivatives), np.zeros((q + 1, q + 1))


def _non_uniform(n: int) -> jax.Array:
    """A grid on ``TSPAN`` whose steps shrink by a factor of four along it."""
    u = np.linspace(0.0, 1.0, n + 1)
    return TSPAN[1] * (2.0 * u - 0.75 * u**2) / 1.25


@pytest.mark.parametrize("prior_cls", [IWP, PrecondIWP], ids=["iwp", "precond"])
@pytest.mark.parametrize("grid", ["uniform", "non_uniform"])
@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_first_order_on_a_fixed_grid(prior_cls, grid, calibration):
    prior = prior_cls(q=3, d=1)
    m_0, P_0_sqr = _exact_state(_cos_solution, q=3)
    ts = np.linspace(*TSPAN, N + 1) if grid == "uniform" else _non_uniform(N)
    measure = ODEInformation(_cos_vf, prior.E0, prior.E1)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration=calibration)
    assert float(np.max(np.abs(res.m[:, 0] - _cos_solution(ts)))) < 5e-6
    m_s, _ = rts_smoother(res)
    assert float(np.max(np.abs(m_s[:, 0] - _cos_solution(ts)))) < 5e-6


def test_first_order_with_the_adaptive_solver():
    prior = IWP(q=3, d=1)
    m_0, P_0_sqr = _exact_state(_cos_solution, q=3)
    save_at = np.linspace(*TSPAN, 6)
    measure = ODEInformation(_cos_vf, prior.E0, prior.E1)
    res = gaussian_filter_adaptive(
        m_0, P_0_sqr, prior, measure, save_at, atol=1e-9, rtol=1e-9, smoother=True
    )
    assert bool(res.success)
    assert float(np.max(np.abs(res.m[:, 0] - _cos_solution(save_at)))) < 1e-8
    m_s, _ = rts_smoother(res)
    assert float(np.max(np.abs(m_s[:, 0] - _cos_solution(save_at)))) < 1e-8


def test_a_surrogate_at_the_predictions_reproduces_ek1():
    """``taylor_surrogate`` linearizes step ``i`` at its update time ``ts[i + 1]``.

    At the update times, linearizing at the EK1 predictions reproduces the EK1
    filter; at the step's start times it does not.
    """
    prior = IWP(q=3, d=1)
    m_0, P_0_sqr = _exact_state(_cos_solution, q=3)
    ts = np.linspace(*TSPAN, N + 1)
    measure = ODEInformation(_cos_vf, prior.E0, prior.E1)
    ek1 = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration="none")
    assert ek1.m_pred is not None
    surrogate = taylor_surrogate(measure, ek1.m_pred, ts=ts[1:])
    frozen = affine_filter(m_0, P_0_sqr, prior, surrogate, ts)
    assert_close(frozen.m, ek1.m, rtol=1e-10)
    assert_close(cov(frozen.P_sqr), cov(ek1.P_sqr), rtol=1e-10)
    assert_close(frozen.log_likelihood, ek1.log_likelihood, rtol=1e-10)
    assert float(np.max(np.abs(frozen.m[:, 0] - _cos_solution(ts)))) < 1e-6

    early = taylor_surrogate(measure, ek1.m_pred, ts=ts[:-1])
    shifted = affine_filter(m_0, P_0_sqr, prior, early, ts)
    assert float(np.max(np.abs(shifted.m[:, 0] - _cos_solution(ts)))) > 1e-2


@pytest.mark.parametrize("linearization", ["taylor", "slr"])
def test_iterated_smoother_passes_keep_the_time(linearization):
    """The ODE is linear in ``x``: every pass reproduces the one-shot smoother."""
    prior = IWP(q=3, d=1)
    m_0, P_0_sqr = _exact_state(_cos_solution, q=3)
    ts = np.linspace(*TSPAN, N + 1)
    measure = ODEInformation(_cos_vf, prior.E0, prior.E1)
    one_shot = iterated_smoother(
        m_0, P_0_sqr, prior, measure, ts, linearization=linearization, n_iters=0
    )
    iterated = iterated_smoother(
        m_0, P_0_sqr, prior, measure, ts, linearization=linearization, n_iters=2
    )
    assert_close(iterated.m, one_shot.m, rtol=1e-8)
    assert_close(cov(iterated.P_sqr), cov(one_shot.P_sqr), rtol=1e-8)
    assert float(np.max(np.abs(iterated.m[:, 0] - _cos_solution(ts)))) < 1e-6


@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_second_order_with_a_time_dependent_forcing(calibration):
    prior = IWP(q=4, d=1)
    m_0, P_0_sqr = _exact_state(_forced_solution, q=4)
    ts = np.linspace(*TSPAN, N + 1)
    measure = SecondOrderODEInformation(_forced_vf, prior.E0, prior.E1, prior.E2)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration=calibration)
    assert float(np.max(np.abs(res.m[:, 0] - _forced_solution(ts)))) < 1e-6
    # The derivative block follows the solution's derivative.
    velocity = jax.vmap(jax.grad(_forced_solution))(ts)
    assert float(np.max(np.abs(res.m[:, 1] - velocity))) < 1e-5
