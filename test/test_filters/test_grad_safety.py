"""Reverse-mode autodiff (jax.grad) regression coverage for every inference path.

The library's value proposition (gradient-based ODE-parameter inference) rests on
every solver loop being reverse-mode differentiable. That property was previously
unguarded: the suite checked *values* but almost never *gradients* across the
{plain, preconditioned} x {EK0, EK1, IEKF} x {fixed, adaptive, smoother} matrix.

These tests pin grad == central-finite-difference (in float64, via conftest) so a
future refactor that breaks reverse-AD (e.g. a plain lax.while_loop, a Python
branch on a traced value, or an IEKF convergence while-loop) fails loudly. They
also pin the one *intentionally* non-differentiable entry point -- the Python
while-driver ``sqr_adaptive_loop`` -- as not traceable.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    InferenceProblem,
    IteratedTaylorCorrection,
    ODEInformation,
    PrecondIWP,
    TaylorCorrection,
    gaussian_filter,
    gaussian_filter_adaptive,
    marginal_loglik,
    prepare_observations,
    rts_smoother,
    taylor_mode_initialization,
)
from ode_filters.filters.ode_filter_adaptive import sqr_adaptive_loop
from ode_filters.filters.ode_filter_loop import _calibrate_diffusion
from ode_filters.measurement.measurement_models import Measurement

TSPAN = (0.0, 2.0)
N = 30
SAVE = np.linspace(0.0, 2.0, 6)
THETA = np.array(1.3)

CORRECTIONS = {
    "EK1": None,  # loop/solver default
    "EK0": TaylorCorrection(order=0),
    "IEKF": IteratedTaylorCorrection(max_iters=3),
}


def _central_fd(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def _assert_grad_matches_fd(f, theta=THETA, rtol=2e-3, atol=1e-7):
    g = float(jax.grad(f)(theta))
    d = float(_central_fd(f, theta))
    assert np.isfinite(g), f"non-finite gradient {g}"
    assert abs(g - d) <= atol + rtol * abs(d), f"grad={g:.6e} vs fd={d:.6e}"


def _logistic_inputs(prior, theta):
    """Build (mu_0, Sigma_0_sqr, measure) for dx/dt = theta x (1-x), x0=0.5."""

    def vf(x, *, t):
        return theta * x * (1.0 - x)

    mu_0, S0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=prior.q)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return mu_0, S0_sqr, measure


# --------------------------------------------------------------------------- #
# Fixed-grid filter: plain & preconditioned x EK0/EK1/IEKF
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prior_cls", [IWP, PrecondIWP])
@pytest.mark.parametrize("corr_name", list(CORRECTIONS))
def test_grad_fixed_filter(prior_cls, corr_name):
    prior = prior_cls(q=2, d=1)

    def f(theta):
        mu_0, S0_sqr, measure = _logistic_inputs(prior, theta)
        res = gaussian_filter(
            mu_0,
            S0_sqr,
            prior,
            measure,
            TSPAN,
            N,
            correction=CORRECTIONS[corr_name],
            calibration="none",
        )
        return res.m[-1, 0]

    _assert_grad_matches_fd(f)


@pytest.mark.parametrize("corr_name", list(CORRECTIONS))
def test_grad_fixed_filter_dynamic_calibration(corr_name):
    """Calibration baked into Q each step must stay differentiable too."""
    prior = IWP(q=2, d=1)

    def f(theta):
        mu_0, S0_sqr, measure = _logistic_inputs(prior, theta)
        res = gaussian_filter(
            mu_0,
            S0_sqr,
            prior,
            measure,
            TSPAN,
            N,
            correction=CORRECTIONS[corr_name],
            calibration="dynamic",
        )
        return res.log_likelihood

    _assert_grad_matches_fd(f)


def test_grad_fixed_filter_with_obs():
    prior = IWP(q=2, d=1)
    ts = np.linspace(*TSPAN, N + 1)
    obs = prepare_observations(
        [Measurement(np.eye(1), np.array([[0.7]]), np.array([ts[15]]), noise=1e-3)],
        prior.E0,
        ts,
    )

    def f(theta):
        mu_0, S0_sqr, measure = _logistic_inputs(prior, theta)
        res = gaussian_filter(
            mu_0, S0_sqr, prior, measure, TSPAN, N, calibration="none", obs_model=obs
        )
        return res.log_likelihood_obs

    _assert_grad_matches_fd(f)


# --------------------------------------------------------------------------- #
# RTS smoother (plain & preconditioned) -- uses the noise-free sqr_inversion +
# reverse lax.scan, a distinct backward-AD path from the forward filter.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("prior_cls", [IWP, PrecondIWP])
def test_grad_rts_smoother(prior_cls):
    prior = prior_cls(q=2, d=1)

    def f(theta):
        mu_0, S0_sqr, measure = _logistic_inputs(prior, theta)
        res = gaussian_filter(
            mu_0, S0_sqr, prior, measure, TSPAN, N, calibration="none"
        )
        m_s, _ = rts_smoother(prior, res)
        return np.sum(m_s[:, 0])  # whole smoothed trajectory (theta-sensitive)

    _assert_grad_matches_fd(f)


# --------------------------------------------------------------------------- #
# Adaptive save-at solver (checkpointed while-loop) x EK0/EK1/IEKF
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("corr_name", list(CORRECTIONS))
def test_grad_adaptive(corr_name):
    prior = IWP(q=2, d=1)

    def f(theta):
        mu_0, S0_sqr, measure = _logistic_inputs(prior, theta)
        res = gaussian_filter_adaptive(
            mu_0,
            S0_sqr,
            prior,
            measure,
            SAVE,
            correction=CORRECTIONS[corr_name],
            atol=1e-7,
            rtol=1e-7,
        )
        return res.m[-1, 0]

    _assert_grad_matches_fd(f)


def test_grad_adaptive_with_obs():
    prior = IWP(q=2, d=1)
    obs = prepare_observations(
        [Measurement(np.eye(1), np.array([[0.7]]), np.array([SAVE[3]]), noise=1e-3)],
        prior.E0,
        SAVE,
    )

    def f(theta):
        mu_0, S0_sqr, measure = _logistic_inputs(prior, theta)
        res = gaussian_filter_adaptive(
            mu_0,
            S0_sqr,
            prior,
            measure,
            SAVE,
            obs_model=obs,
            calibration="none",
            atol=1e-7,
            rtol=1e-7,
        )
        return res.log_likelihood

    _assert_grad_matches_fd(f, rtol=5e-3)


# --------------------------------------------------------------------------- #
# Public inference objective
# --------------------------------------------------------------------------- #


def test_grad_marginal_loglik():
    prior = IWP(q=2, d=1)
    obs = prepare_observations(
        [
            Measurement(
                np.eye(1),
                np.array([[0.7]]),
                np.array([np.linspace(*TSPAN, N + 1)[15]]),
                noise=1e-2,
            )
        ],
        prior.E0,
        np.linspace(*TSPAN, N + 1),
    )

    def build(theta):
        return _logistic_inputs(prior, theta)

    problem = InferenceProblem(build=build, prior=prior, tspan=TSPAN, N=N)

    def f(theta):
        return marginal_loglik(theta, obs, model=problem)

    _assert_grad_matches_fd(f)


# --------------------------------------------------------------------------- #
# Robustness: grad stays finite through a recoverable guard-firing solve, and
# the diagonal-calibration zero-denominator guard does not poison gradients.
# --------------------------------------------------------------------------- #


def test_grad_finite_through_recoverable_guard_firing():
    """A near-blow-up solve that still reaches the end (success=True) must yield a
    finite gradient -- the NaN/inf step-rejection guard must not poison cotangents."""
    prior = IWP(q=2, d=1)
    save = np.linspace(0.0, 2.0, 6)  # dx/dt = x^2, x0=0.49 -> blow-up at ~2.04 > 2.0

    def solve(x0):
        def vf(x, *, t):
            return x * x

        mu_0, S0_sqr = taylor_mode_initialization(vf, np.array([x0]), q=2)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        return gaussian_filter_adaptive(
            mu_0, S0_sqr, prior, measure, save, atol=1e-6, rtol=1e-6
        )

    x0 = np.array(0.49)
    res = solve(x0)
    assert bool(res.success)  # forces rejections near the end but completes
    g = jax.grad(lambda x: solve(x).m[-1, 0])(x0)
    assert np.isfinite(g)


@pytest.mark.parametrize("mode", ["diagonal", "diagonal_ekf0"])
def test_diagonal_calibration_zero_denominator_floors_without_nan(mode):
    """A zero diagonal-calibration denominator must floor to min_sigma_sqr (not
    NaN/inf), forward and under grad."""
    prior = IWP(q=1, d=1)  # state dim 2
    H = prior.E1  # [1, 2]
    E1 = prior.E1
    Q_sqr_zero = np.zeros((2, 2))  # forces denom == 0
    min_sigma_sqr = 1e-30

    def f(mz_ode):
        sigma_sqr, _ = _calibrate_diffusion(
            mode, mz_ode, H, E1, Q_sqr_zero, prior, min_sigma_sqr
        )
        return np.sum(sigma_sqr)

    mz = np.array([1.0])
    sigma_sum = f(mz)
    assert np.isfinite(sigma_sum)
    assert np.allclose(sigma_sum, min_sigma_sqr)  # floored, not NaN/inf
    assert np.isfinite(jax.grad(f)(mz))  # no NaN cotangent from 0/0


# --------------------------------------------------------------------------- #
# The Python while-driver is intentionally NOT traceable / differentiable.
# --------------------------------------------------------------------------- #


def test_python_while_driver_is_not_traceable():
    """sqr_adaptive_loop has data-dependent Python control flow + variable
    output length, so it cannot be jit/grad'd. Pin the limitation (and the fact
    that sqr_adaptive_solve / gaussian_filter_adaptive is the AD-safe
    alternative) so a refactor cannot silently change this contract."""
    prior = IWP(q=2, d=1)

    def vf(x, *, t):
        return x * (1.0 - x)

    mu_0, S0_sqr = taylor_mode_initialization(vf, np.array([0.5]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)

    jitted = jax.jit(sqr_adaptive_loop, static_argnums=(2, 3, 4))
    with pytest.raises(
        (jax.errors.ConcretizationTypeError, jax.errors.TracerBoolConversionError)
    ):
        jitted(mu_0, S0_sqr, prior, measure, TSPAN)


# --------------------------------------------------------------------------- #
# Adaptive fixed-point smoother (smoother=True) -- backward pass through the
# composed conditionals + reverse-scan smoother must stay reverse-AD safe.
# --------------------------------------------------------------------------- #


def test_grad_adaptive_smoother():
    prior = IWP(q=2, d=1)
    save = np.linspace(0.0, 2.0, 5)

    def f(theta):
        mu_0, S0_sqr, measure = _logistic_inputs(prior, theta)
        res = gaussian_filter_adaptive(
            mu_0, S0_sqr, prior, measure, save, smoother=True, atol=1e-7, rtol=1e-7
        )
        m_s, _ = rts_smoother(prior, res)
        return np.sum(m_s[:, 0])

    _assert_grad_matches_fd(f)
