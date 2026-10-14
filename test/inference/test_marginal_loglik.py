"""The differentiable marginal log-likelihood of the data.

Reference problem: scalar exponential decay ``dx/dt = -lam * x`` with ``x0`` known
and ``lam`` the unknown parameter, observed noisily in ``x``. The marginal
log-likelihood of the data is differentiable / jittable / vmappable over ``lam``
and identifies the true rate.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import jax.random as jrandom
import pytest

from ode_filters import InferenceProblem, gaussian_filter, marginal_loglik
from ode_filters.information import Measurement, ODEInformation, prepare_observations
from ode_filters.priors import IWP
from test.helpers import assert_close, assert_grad_matches_fd

X0 = 2.0
LAM_TRUE = 0.8
TSPAN = (0.0, 5.0)
N = 200
NOISE_STD = 0.02


def _problem_and_data():
    ts = np.linspace(TSPAN[0], TSPAN[1], N + 1)
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    z = (x_true + NOISE_STD * jrandom.normal(jrandom.PRNGKey(0), (N,))).reshape(-1, 1)

    prior = IWP(q=2, d=1, Xi=np.eye(1))
    e0, e1 = prior.E0, prior.E1
    measurement = Measurement(np.eye(1), z, ts[1:], noise=NOISE_STD**2)
    obs_model = prepare_observations([measurement], e0, ts)

    def build(theta):
        lam = theta["lam"]

        def vf(x, *, t):
            return -lam * x

        measure = ODEInformation(vf, e0, e1)
        # theta-independent diffuse init: x0 known, derivatives loosely prior'd.
        m_0 = np.array([X0, 0.0, 0.0])
        sigma_0_sqr = np.diag(np.array([1e-2, 1.0, 1.0]))  # sqrt of the covariance
        return m_0, sigma_0_sqr, measure

    problem = InferenceProblem(
        build=build, prior=prior, ts=np.linspace(*TSPAN, N + 1), calibration="none"
    )
    return problem, obs_model


def _loglik(lam, problem, data):
    return marginal_loglik({"lam": lam}, data, model=problem)


def _direct(lam, problem, data):
    """The filter run ``marginal_loglik`` stands for, spelled out."""
    m_0, P_0_sqr, measure = problem.build({"lam": lam})
    return gaussian_filter(
        m_0,
        P_0_sqr,
        problem.prior,
        measure,
        problem.ts,
        calibration=problem.calibration,
        obs_model=data,
    )


def test_loglik_is_the_data_evidence_of_the_filter():
    problem, data = _problem_and_data()
    ll = marginal_loglik({"lam": np.array(LAM_TRUE)}, data, model=problem)
    assert ll.shape == ()
    expected = _direct(np.array(LAM_TRUE), problem, data).log_likelihood_obs
    assert_close(ll, expected, rtol=1e-13)


def test_loglik_jittable():
    problem, data = _problem_and_data()
    f = jax.jit(lambda lam: _loglik(lam, problem, data))
    lam = np.array(LAM_TRUE)
    assert_close(f(lam), _loglik(lam, problem, data), rtol=1e-12)


def test_loglik_peaks_near_true():
    problem, data = _problem_and_data()
    at_true = _loglik(np.array(LAM_TRUE), problem, data)
    assert at_true > _loglik(np.array(LAM_TRUE + 0.3), problem, data)
    assert at_true > _loglik(np.array(LAM_TRUE - 0.3), problem, data)


def test_grad_matches_finite_difference():
    problem, data = _problem_and_data()
    f = jax.jit(lambda lam: _loglik(lam, problem, data))
    assert_grad_matches_fd(f, np.array(0.6), eps=1e-4, rtol=1e-3, atol=1e-3)


def test_vmap_over_theta():
    problem, data = _problem_and_data()
    lams = np.linspace(0.4, 1.2, 9)
    lls = jax.vmap(lambda lam: _loglik(lam, problem, data))(lams)
    assert lls.shape == (9,)
    assert np.all(np.isfinite(lls))
    assert abs(lams[np.argmax(lls)] - LAM_TRUE) <= 0.15


def test_gradient_descent_recovers_parameter():
    """Adam on the negative log-likelihood recovers the true rate from a wrong start."""
    problem, data = _problem_and_data()
    neg_grad = jax.jit(jax.grad(lambda lam: -_loglik(lam, problem, data)))

    lam = np.array(0.4)  # wrong starting guess
    m, v = np.array(0.0), np.array(0.0)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for i in range(1, 401):
        grad = neg_grad(lam)
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * grad**2
        mhat = m / (1 - b1**i)
        vhat = v / (1 - b2**i)
        lam = lam - lr * mhat / (np.sqrt(vhat) + eps)

    assert abs(float(lam) - LAM_TRUE) < 0.1, f"recovered lam={float(lam)}"


# --------------------------------------------------------------------------
# channel selection and prior_fn
# --------------------------------------------------------------------------


def test_channel_both_matches_scalar_channels():
    """``channel="both"`` returns exactly the pair the scalar modes return."""
    problem, obs_model = _problem_and_data()
    theta = {"lam": np.array(LAM_TRUE)}

    ll_obs = marginal_loglik(theta, obs_model, model=problem)
    ll_ode_only = marginal_loglik(theta, obs_model, model=problem, channel="ode")
    ll_ode, ll_obs_both = marginal_loglik(
        theta, obs_model, model=problem, channel="both"
    )

    assert ll_obs_both == ll_obs
    assert ll_ode == ll_ode_only
    # The two channels are genuinely different quantities.
    assert not np.allclose(ll_ode, ll_obs)


def test_channel_ode_needs_no_observations():
    """The residual channel is defined without data."""
    problem, _ = _problem_and_data()
    ll_ode = marginal_loglik(
        {"lam": np.array(LAM_TRUE)}, None, model=problem, channel="ode"
    )
    expected = _direct(np.array(LAM_TRUE), problem, None).log_likelihood
    assert_close(ll_ode, expected, rtol=1e-13)


def test_channel_validation():
    problem, obs_model = _problem_and_data()
    theta = {"lam": np.array(LAM_TRUE)}
    for bad in ("joint", "OBS", ""):
        try:
            marginal_loglik(theta, obs_model, model=problem, channel=bad)
        except ValueError as exc:
            assert "channel" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"channel={bad!r} should have raised")
    # obs channel still requires data
    try:
        marginal_loglik(theta, None, model=problem)
    except ValueError as exc:
        assert "requires observations" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("channel='obs' with data=None should have raised")


def test_channel_both_is_grad_safe():
    """Both channels are differentiable w.r.t. theta."""
    problem, obs_model = _problem_and_data()

    def loss(lam):
        ll_ode, ll_obs = marginal_loglik(
            {"lam": lam}, obs_model, model=problem, channel="both"
        )
        return -(ll_ode + ll_obs)

    assert_grad_matches_fd(loss, np.array(LAM_TRUE), eps=1e-4, rtol=1e-3, atol=1e-3)


def test_prior_fn_fits_prior_hyperparameters():
    """``prior_fn`` puts the prior's diffusion scale in the traced region."""
    problem, obs_model = _problem_and_data()

    def prior_fn(theta):
        return IWP(q=2, d=1, Xi=np.exp(theta["log_sigma"]) * np.eye(1))

    hyper = problem._replace(prior=None, prior_fn=prior_fn)

    def build(theta):
        m_0, P_0_sqr, measure = problem.build(theta)
        return m_0, P_0_sqr, measure

    hyper = hyper._replace(build=build)
    theta = {"lam": np.array(LAM_TRUE), "log_sigma": np.array(0.0)}

    # The static `prior=None` is never touched when prior_fn is set.
    ll = marginal_loglik(theta, obs_model, model=hyper)
    assert np.isfinite(ll)

    # ...and the hyperparameter carries gradient.
    g = jax.jit(jax.grad(lambda th: -marginal_loglik(th, obs_model, model=hyper)))(
        theta
    )
    assert np.isfinite(g["log_sigma"])
    assert not np.allclose(g["log_sigma"], 0.0)


def test_prior_fn_absent_uses_static_prior():
    """Default path is unchanged: prior_fn=None uses the static prior."""
    problem, obs_model = _problem_and_data()
    theta = {"lam": np.array(LAM_TRUE)}
    baseline = marginal_loglik(theta, obs_model, model=problem)
    explicit = marginal_loglik(theta, obs_model, model=problem._replace(prior_fn=None))
    assert baseline == explicit


def test_marginal_loglik_requires_data():
    prior = IWP(q=2, d=1)

    def build(theta):
        measure = ODEInformation(lambda x, *, t: -x, prior.E0, prior.E1)
        return np.array([1.0, 0.0, 0.0]), np.eye(3), measure

    problem = InferenceProblem(
        build=build, prior=prior, ts=np.linspace(*(0.0, 1.0), 10 + 1)
    )
    with pytest.raises(ValueError, match="requires observations"):
        marginal_loglik({}, None, model=problem)
