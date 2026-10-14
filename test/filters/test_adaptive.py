"""Adaptive stepping: the steps ``gaussian_filter_adaptive`` takes.

Uses ``record_steps`` to see every accepted step, not only the save grid.
"""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    ODEInformation,
    PController,
    gaussian_filter_adaptive,
    taylor_mode_initialization,
)
from ode_filters.calibration.sigma import quasi_mle_sigma_sqr_from_Q

X0 = np.array([0.1])
SAVE = np.array([0.0, 5.0])


def logistic_vf(x, *, t):
    return x * (1 - x)


def logistic_analytic(t, x0=0.1):
    return 1.0 / (1.0 + (1.0 / x0 - 1.0) * np.exp(-t))


def _setup(q=2):
    prior = IWP(q=q, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(logistic_vf, X0, q=q)
    return prior, m_0, P_0_sqr, ODEInformation(logistic_vf, prior.E0, prior.E1)


def _solve(save_at=SAVE, record_steps=500, **kwargs):
    prior, m_0, P_0_sqr, measure = _setup()
    kwargs = {"atol": 1e-4, "rtol": 1e-2, **kwargs}
    res = gaussian_filter_adaptive(
        m_0, P_0_sqr, prior, measure, save_at, record_steps=record_steps, **kwargs
    )
    return prior, m_0, measure, res


def test_the_record_holds_every_accepted_step():
    _, _, _, res = _solve()
    steps = res.steps
    n = int(steps.count)
    assert bool(res.success)
    assert n == int(np.sum(res.n_accepted))
    assert float(steps.t[n - 1]) == pytest.approx(5.0, rel=1e-12)
    assert float(np.sum(steps.h[:n])) == pytest.approx(5.0, rel=1e-12)
    assert np.allclose(steps.m[n - 1], res.m[-1])
    assert bool(np.all(np.isnan(steps.t[n:])))


def test_a_short_record_keeps_the_first_steps_and_the_full_count():
    _, _, _, full = _solve()
    _, _, _, short = _solve(record_steps=3)
    assert int(short.steps.count) == int(full.steps.count)
    assert np.allclose(short.steps.t, full.steps.t[:3])


def test_tighter_tolerances_reduce_the_error_and_the_step_size():
    _, _, _, loose = _solve(atol=1e-2, rtol=1e-1)
    _, _, _, tight = _solve(atol=1e-7, rtol=1e-5)
    x_true = float(logistic_analytic(5.0))
    assert abs(float(tight.m[-1, 0]) - x_true) < abs(float(loose.m[-1, 0]) - x_true)
    mean_h = [float(np.mean(r.steps.h[: int(r.steps.count)])) for r in (tight, loose)]
    assert mean_h[0] < mean_h[1]


def test_p_controller_reaches_the_end():
    _, _, _, res = _solve(controller=PController(order=2), atol=1e-5, rtol=1e-3)
    assert bool(res.success) and int(res.steps.count) > 0


def _previous_means(m_0, steps, n):
    return np.concatenate([m_0[None], steps.m[: n - 1]], axis=0)


@pytest.mark.parametrize("calibration", ["dynamic", "diagonal"])
def test_each_steps_diffusion_is_the_quasi_mle_at_its_prediction(calibration):
    prior, m_0, measure, res = _solve(calibration=calibration)
    steps = res.steps
    n = int(steps.count)
    for i, m_prev in enumerate(_previous_means(m_0, steps, n)):
        h, t = float(steps.h[i]), float(steps.t[i])
        m_pred = prior.A(h) @ m_prev + prior.b(h)
        H, c = measure.linearize(m_pred, t=t)
        residual = H @ m_pred + c
        if calibration == "dynamic":
            expected = quasi_mle_sigma_sqr_from_Q(residual, H, prior.Q_sqr(h))
        else:
            expected = residual**2 / np.einsum("ij,jk,ik->i", H, prior.Q(h), H)
        assert np.allclose(steps.sigma_sqr[i], expected, rtol=1e-9)


def test_no_calibration_records_the_estimate_but_propagates_unscaled():
    _, _, _, dynamic = _solve(calibration="dynamic")
    _, _, _, none = _solve(calibration="none")
    n = int(none.steps.count)
    assert bool(np.all(none.steps.sigma_sqr[:n] > 0))
    sigmas = dynamic.steps.sigma_sqr[: int(dynamic.steps.count)]
    if float(np.max(sigmas)) < 1.0:
        cov_dyn = dynamic.P_sqr[-1].T @ dynamic.P_sqr[-1]
        cov_none = none.P_sqr[-1].T @ none.P_sqr[-1]
        assert float(np.trace(cov_dyn)) < float(np.trace(cov_none))


def test_running_mean_diffusion_in_the_error_rejects_fewer_steps():
    _, _, _, per_step = _solve(atol=1e-5, rtol=1e-3, sigma_in_error="per_step")
    _, _, _, running = _solve(atol=1e-5, rtol=1e-3, sigma_in_error="running_mean")
    assert bool(per_step.success) and bool(running.success)
    assert int(np.sum(running.n_rejected)) <= int(np.sum(per_step.n_rejected))


def test_diagonal_calibration_resolves_a_multiscale_problem():
    rate = 2.0
    x0 = np.asarray([1e-5, 1e-10])

    def vf(x, *, t):
        return rate * x * (1 - x)

    prior = IWP(q=3, d=2)
    m_0, P_0_sqr = taylor_mode_initialization(vf, x0, q=3)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    res = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.array([0.0, 15.0]),
        atol=1e-5,
        rtol=1e-3,
        calibration="diagonal",
        max_steps=20000,
    )
    x2_true = 1.0 / (1.0 + (1.0 / float(x0[1]) - 1.0) * np.exp(-rate * 15.0))
    assert bool(res.success)
    assert abs(float(res.m[-1, 1]) - float(x2_true)) < 1e-3


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma_in_error": "ema"}, "sigma_in_error"),
        ({"record_steps": -1}, "record_steps"),
        ({"calibration": "bogus"}, "calibration must be"),
    ],
)
@pytest.mark.wrong_input_types
def test_invalid_arguments_raise(kwargs, match):
    prior, m_0, P_0_sqr, measure = _setup()
    with pytest.raises(ValueError, match=match):
        gaussian_filter_adaptive(m_0, P_0_sqr, prior, measure, SAVE, **kwargs)
