"""The steps of ``gaussian_filter_adaptive``: ``max_step`` and the controller.

The controller tests rebuild every accepted step of a solve from its record
(``record_steps=``) and the documented formulas alone. With ``rtol=0`` the
error estimate of a step of size ``h`` depends on ``h`` and the diffusion
only; for an ``IWP(q)`` prior with unit ``Xi`` it is

    err = sqrt(mean_i(sigma_i^2 * h^(2q+1) / ((2q+1) q!^2))) / atol,

so the recorded diffusions determine it: per step, it is the step's own; with
``sigma_in_error="running_mean"`` the mean over the accepted steps before it,
which also fixes the error of every rejected attempt. The replay continues
from each recorded step, so round-off cannot compound along the sequence.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    ODEInformation,
    PController,
    PIController,
    QuadratureCorrection,
    gaussian_filter,
    gaussian_filter_adaptive,
    taylor_mode_initialization,
)
from test.helpers import assert_close, assert_grad_matches_fd, cov

Q = 2
SAVE = np.linspace(0.0, 5.0, 4)


def lotka_volterra(x, *, t):
    return np.array([1.5 * x[0] - x[0] * x[1], -3.0 * x[1] + x[0] * x[1]])


def logistic(x, *, t):
    return x * (1.0 - x)


def _problem(vf, x0, q=Q):
    prior = IWP(q=q, d=len(x0))
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array(x0), q=q)
    return prior, m_0, P_0_sqr, ODEInformation(vf, prior.E0, prior.E1)


# --------------------------------------------------------------------------- #
# Replay of the step-size arithmetic
# --------------------------------------------------------------------------- #


class _Controller(NamedTuple):
    """The documented parameters of a controller (``beta = 0``: no memory)."""

    alpha: float
    beta: float = 0.0
    safety: float = 0.9
    min_factor: float = 0.2
    max_factor: float = 5.0


class _Replay(NamedTuple):
    h: onp.ndarray  # predicted size of every accepted step
    n_accepted: onp.ndarray  # per save interval
    n_rejected: onp.ndarray
    clipped_low: int  # proposals clipped at min_factor
    clipped_high: int  # proposals clipped at max_factor
    accepted_after_rejection: int


def _error(h, sigma_sqr, *, atol, q=Q):
    """The documented error estimate with ``rtol=0`` (IWP(q), unit Xi)."""
    variance = h ** (2 * q + 1) / ((2 * q + 1) * math.factorial(q) ** 2)
    return math.sqrt(onp.mean(sigma_sqr * variance)) / atol


def _replay(
    res, save_at, ctrl, *, atol, h_init, running_mean, max_step=math.inf
) -> _Replay:
    """Rebuild the accepted steps of ``res`` from its record and the formulas."""
    steps = res.steps
    n = int(steps.count)
    assert n <= steps.h.shape[0], "the record must hold every accepted step"
    h_rec, t_rec, sigma = (
        onp.asarray(a[:n]) for a in (steps.h, steps.t, steps.sigma_sqr)
    )
    edges = onp.asarray(save_at)
    clipped = [0, 0]

    def propose(h, err, err_prev):
        factor = ctrl.safety * err ** (-ctrl.alpha)
        if err_prev is not None:
            factor *= (err_prev / err) ** ctrl.beta
        clipped[0] += factor < ctrl.min_factor
        clipped[1] += factor > ctrl.max_factor
        return min(h * min(max(factor, ctrl.min_factor), ctrl.max_factor), max_step)

    t, h, err_prev = float(edges[0]), min(h_init, max_step), None
    predicted, interval = [], []
    n_rejected = onp.zeros(len(edges) - 1, dtype=int)
    after_rejection = 0
    for k in range(n):
        i = next(i for i, s in enumerate(edges) if t < s - (1e-10 * abs(s) + 1e-12))
        sigma_err = sigma[:k].mean(axis=0) if running_mean and k else sigma[k]
        rejections = 0
        while True:
            h_try = min(h, edges[i] - t)  # the last step lands on the save time
            err = _error(h_try, sigma_err, atol=atol)
            if err <= 1.0:
                break
            rejections += 1
            h = propose(h_try, err, None)  # after a rejection: no memory
        n_rejected[i - 1] += rejections
        after_rejection += bool(rejections) and k > 0
        predicted.append(h_try)
        interval.append(i - 1)
        # Continue from the recorded step; the PI memory is the accepted error.
        err = _error(h_rec[k], sigma_err, atol=atol)
        h = propose(h_rec[k], err, err_prev)
        t, err_prev = t_rec[k], err
    return _Replay(
        onp.array(predicted),
        onp.bincount(interval, minlength=len(edges) - 1),
        n_rejected,
        *clipped,
        after_rejection,
    )


def _solve_and_replay(controller, documented, *, sigma_in_error, atol=1e-4, **kwargs):
    prior, m_0, P_0_sqr, measure = _problem(lotka_volterra, [1.0, 1.0])
    h_init = kwargs.pop("h_init", 1e-3)
    res = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        SAVE,
        atol=atol,
        rtol=0.0,
        h_init=h_init,
        controller=controller,
        sigma_in_error=sigma_in_error,
        calibration="diagonal",  # per-component diffusions: the mean matters
        record_steps=1000,
        **kwargs,
    )
    assert bool(res.success)
    replay = _replay(
        res,
        SAVE,
        documented,
        atol=atol,
        h_init=h_init,
        running_mean=sigma_in_error == "running_mean",
        max_step=kwargs.get("max_step") or math.inf,
    )
    n = int(res.steps.count)
    assert replay.h.shape == (n,)
    # Step by step, to round-off: the steps span orders of magnitude.
    assert onp.allclose(replay.h, res.steps.h[:n], rtol=1e-12, atol=0.0)
    assert onp.array_equal(replay.n_accepted, res.n_accepted)
    assert onp.array_equal(replay.n_rejected, res.n_rejected)
    return res, replay


# The per-step diffusion of a rejected attempt is not recorded, so the per-step
# cases use a safety factor at which no step is rejected.
@pytest.mark.parametrize(
    ("controller", "documented", "sigma_in_error"),
    [
        (PController(order=Q, safety=0.5), _Controller(1 / Q, safety=0.5), "per_step"),
        (
            PIController(order=Q, safety=0.5),
            _Controller(0.7 / Q, 0.4 / Q, safety=0.5),
            "per_step",
        ),
        (PController(order=Q), _Controller(1 / Q), "running_mean"),
        (PIController(order=Q), _Controller(0.7 / Q, 0.4 / Q), "running_mean"),
    ],
    ids=["P-per_step", "PI-per_step", "P-running_mean", "PI-running_mean"],
)
def test_every_accepted_step_follows_from_the_documented_formulas(
    controller, documented, sigma_in_error
):
    res, replay = _solve_and_replay(
        controller, documented, sigma_in_error=sigma_in_error
    )
    assert int(res.steps.count) > 100
    if sigma_in_error == "per_step":
        assert int(np.sum(res.n_rejected)) == 0
    # From h_init = 1e-3 the step grows at the maximum factor.
    assert replay.clipped_high > 0
    assert float(res.steps.h[1] / res.steps.h[0]) == pytest.approx(5.0, rel=1e-12)


def test_rejected_steps_do_not_update_the_pi_memory():
    """After a rejection the next accepted step's PI term uses the last accepted error.

    An over-eager gain (alpha = 1, above the stable 2 / (q + 1/2)) makes the
    error oscillate around the tolerance, so many accepted steps directly follow
    a rejection.
    """
    controller = PIController(order=Q, alpha=1.0, beta=0.3)
    _, replay = _solve_and_replay(
        controller, _Controller(1.0, 0.3), sigma_in_error="running_mean"
    )
    assert replay.accepted_after_rejection > 10


def test_the_step_factor_is_clipped_to_its_bounds():
    """Narrow bounds bind often, at both ends; the replay clips as documented."""
    controller = PIController(
        order=Q, alpha=1.0, beta=0.3, min_factor=0.5, max_factor=1.5
    )
    documented = _Controller(1.0, 0.3, min_factor=0.5, max_factor=1.5)
    _, replay = _solve_and_replay(controller, documented, sigma_in_error="running_mean")
    assert replay.clipped_low > 0 and replay.clipped_high > 0


def test_each_rejection_of_an_enormous_error_shrinks_by_min_factor():
    """A first step across the whole span is rejected with an enormous error.

    At safety 0.5 every factor after a rejection is below min_factor = 0.5, so
    each rejection halves the step exactly (and none follows the first
    accepted step); unclipped, the first shrink alone would be far larger.
    """
    prior, m_0, P_0_sqr, measure = _problem(lotka_volterra, [1.0, 1.0])
    span = float(SAVE[-1])
    res = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.array([0.0, span]),
        atol=1e-4,
        rtol=0.0,
        h_init=span,
        controller=PController(order=Q, safety=0.5, min_factor=0.5),
        record_steps=1,
    )
    n_rej = int(res.n_rejected[0])
    assert n_rej >= 3
    assert float(res.steps.h[0]) == pytest.approx(span * 0.5**n_rej, rel=1e-12)


def test_max_step_caps_every_proposal_in_the_replay():
    res, _ = _solve_and_replay(
        PIController(order=Q),
        _Controller(0.7 / Q, 0.4 / Q),
        sigma_in_error="running_mean",
        max_step=0.02,
    )
    h = res.steps.h[: int(res.steps.count)]
    assert float(np.max(h)) == 0.02
    assert 0 < int(np.sum(h == 0.02)) < h.shape[0]  # the cap binds, not always


# --------------------------------------------------------------------------- #
# max_step
# --------------------------------------------------------------------------- #

PULSE_WIDTH = 0.05


def pulse(x, *, t):
    """Slow decay hit by a pulse of width 0.05 at t = 5."""
    return -0.3 * x + 5.0 * np.exp(-(((t - 5.0) / PULSE_WIDTH) ** 2))


def _pulse_solution_at_10():
    """x(10) for x(0) = 1: the pulse integral in closed form (erf(100) = 1)."""
    a, w = 0.3, PULSE_WIDTH
    return math.exp(-10 * a) * (
        1 + 5 * w * math.sqrt(math.pi) * math.exp(5 * a + (a * w) ** 2 / 4)
    )


def _solve_pulse(**kwargs):
    prior, m_0, P_0_sqr, measure = _problem(pulse, [1.0])
    return gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.array([0.0, 10.0]),
        atol=1e-3,
        rtol=1e-2,
        record_steps=1000,
        **kwargs,
    )


def test_max_step_resolves_a_pulse_that_a_loose_tolerance_steps_over():
    exact = _pulse_solution_at_10()
    free = _solve_pulse()
    bounded = _solve_pulse(max_step=PULSE_WIDTH)
    assert bool(free.success) and bool(bounded.success)
    # Without a bound the steps grow far past the pulse, which goes unseen ...
    assert float(np.max(free.steps.h[: int(free.steps.count)])) > 10 * PULSE_WIDTH
    assert abs(float(free.m[-1, 0]) - exact) > 0.5 * exact
    # ... with one, no step exceeds it and the pulse is resolved.
    h = bounded.steps.h[: int(bounded.steps.count)]
    assert float(np.max(h)) <= PULSE_WIDTH
    assert abs(float(bounded.m[-1, 0]) - exact) < 1e-3 * exact


def test_max_step_caps_the_first_step_too():
    """At a tolerance that accepts any step, every step is max_step (bar the last)."""
    prior, m_0, P_0_sqr, measure = _problem(logistic, [0.5])
    res = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.array([0.0, 2.0]),
        atol=1e3,
        rtol=1e3,
        h_init=1.0,
        max_step=0.3,
        record_steps=10,
    )
    n = int(res.steps.count)
    assert n == 7 and int(res.n_rejected[0]) == 0
    assert bool(np.all(res.steps.h[: n - 1] == 0.3))
    assert float(res.steps.h[n - 1]) == pytest.approx(0.2, rel=1e-12)


def test_no_max_step_is_the_default():
    prior, m_0, P_0_sqr, measure = _problem(logistic, [0.5])
    args = (m_0, P_0_sqr, prior, measure, SAVE)
    default = gaussian_filter_adaptive(*args)
    unbounded = gaussian_filter_adaptive(*args, max_step=math.inf)
    assert np.array_equal(default.m, unbounded.m)
    assert np.array_equal(default.n_accepted, unbounded.n_accepted)


def test_max_step_under_jit_and_vmap_matches_eager():
    prior, _, _, measure = _problem(logistic, [0.5])

    def solve(x0):
        m_0, P_0_sqr = taylor_mode_initialization(logistic, x0, q=Q)
        return gaussian_filter_adaptive(
            m_0, P_0_sqr, prior, measure, SAVE, max_step=0.1
        )

    x0s = np.array([[0.3], [0.5]])
    eager = [solve(x0) for x0 in x0s]
    batched = eqx.filter_jit(jax.vmap(solve))(x0s)
    for i, res in enumerate(eager):
        assert np.allclose(batched.m[i], res.m, rtol=1e-10)
        assert np.array_equal(batched.n_accepted[i], res.n_accepted)
    assert int(np.sum(eager[0].n_accepted)) >= 50  # 5 / 0.1: the cap binds


@pytest.mark.wrong_input_types  # a traced max_step is not a float
def test_a_traced_max_step_is_accepted():
    prior, m_0, P_0_sqr, measure = _problem(logistic, [0.5])

    @jax.jit
    def final_mean(max_step):
        return gaussian_filter_adaptive(
            m_0, P_0_sqr, prior, measure, SAVE, max_step=max_step
        ).m[-1]

    eager = gaussian_filter_adaptive(m_0, P_0_sqr, prior, measure, SAVE, max_step=0.1)
    assert np.allclose(final_mean(0.1), eager.m[-1], rtol=1e-10)


def test_grad_with_max_step_matches_finite_difference():
    prior = IWP(q=Q, d=1)

    def endpoint(theta):
        def vf(x, *, t):
            return theta * x * (1.0 - x)

        m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=Q)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        res = gaussian_filter_adaptive(
            m_0, P_0_sqr, prior, measure, SAVE, atol=1e-7, rtol=1e-7, max_step=0.05
        )
        return res.m[-1, 0]

    assert_grad_matches_fd(endpoint, np.array(1.3))


@pytest.mark.parametrize("max_step", [0.0, -0.1, math.nan])
def test_a_non_positive_max_step_raises(max_step):
    prior, m_0, P_0_sqr, measure = _problem(logistic, [0.5])
    with pytest.raises(ValueError, match="max_step must be > 0"):
        gaussian_filter_adaptive(m_0, P_0_sqr, prior, measure, SAVE, max_step=max_step)


@pytest.mark.wrong_input_types
def test_a_non_scalar_max_step_raises():
    prior, m_0, P_0_sqr, measure = _problem(logistic, [0.5])
    with pytest.raises(ValueError, match="max_step must be a scalar"):
        gaussian_filter_adaptive(
            m_0, P_0_sqr, prior, measure, SAVE, max_step=np.array([0.1, 0.2])
        )


# --------------------------------------------------------------------------- #
# QuadratureCorrection on the adaptive path
# --------------------------------------------------------------------------- #


def test_quadrature_correction_solves_adaptively_and_matches_the_fixed_grid():
    """Accurate against the exact solution, and the same filter as the fixed grid.

    Run on the accepted steps of the adaptive solve, the fixed-grid filter with
    the same correction reproduces every recorded step.
    """
    prior, m_0, P_0_sqr, measure = _problem(logistic, [0.1], q=3)
    correction = QuadratureCorrection()
    save_at = np.linspace(0.0, 5.0, 6)
    res = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        save_at,
        correction=correction,
        atol=1e-6,
        rtol=1e-4,
        record_steps=500,
    )
    assert bool(res.success)
    exact = 1.0 / (1.0 + 9.0 * np.exp(-save_at))
    assert float(np.max(np.abs(res.m[:, 0] - exact))) < 2e-5

    n = int(res.steps.count)
    grid = np.concatenate([save_at[:1], res.steps.t[:n]])
    fixed = gaussian_filter(m_0, P_0_sqr, prior, measure, grid, correction=correction)
    assert_close(fixed.m[1:], res.steps.m[:n], rtol=1e-9)
    assert_close(fixed.sigma_sqr, res.steps.sigma_sqr[:n, 0], rtol=1e-8)
    assert_close(cov(fixed.P_sqr[-1]), cov(res.P_sqr[-1]), rtol=1e-8)
    assert_close(fixed.log_likelihood, res.log_likelihood, rtol=1e-9)

    # The quadrature update is not EK1's, so the agreement pins the correction.
    ek1 = gaussian_filter(m_0, P_0_sqr, prior, measure, grid)
    scale = float(np.max(np.abs(fixed.m[1:])))
    assert float(np.max(np.abs(ek1.m[1:] - fixed.m[1:]))) > 1e-6 * scale
