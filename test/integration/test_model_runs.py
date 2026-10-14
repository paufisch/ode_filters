"""Measurement models end to end: whole solver runs, not only the model's algebra.

``TransformedMeasurement`` and ``SecondOrderODEInformationWithHidden`` are
otherwise tested as models (residuals, Jacobians, linearizations). Here they
drive filters, and the runs are compared with an equivalent plain model and
with closed-form solutions.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    JointPrior,
    MaternPrior,
    Measurement,
    ODEInformationWithHidden,
    QuadratureCorrection,
    SecondOrderODEInformation,
    SecondOrderODEInformationWithHidden,
    TransformedMeasurement,
    gaussian_filter,
    prepare_observations,
    rts_smoother,
)
from test.helpers import assert_close, cov

# ---------------------------------------------------------------------------
# TransformedMeasurement: a positivity link on a hidden decay rate
# ---------------------------------------------------------------------------


def _decay(x, u, *, t):
    return -u * x


def _decay_softplus(x, u, *, t):
    return -jax.nn.softplus(u) * x


def _rate_link(prior):
    """``sigma``: the softplus of the input ``u`` in the joint state."""
    i_u = int(np.argmax(prior.E0_hidden[0]))
    return lambda state: state.at[i_u].set(jax.nn.softplus(state[i_u]))


def _linked_models(prior):
    """The linked model and the plain model with the link inside the vector field."""
    linked = TransformedMeasurement(
        ODEInformationWithHidden(_decay, prior.E0_x, prior.E1, prior.E0_hidden),
        _rate_link(prior),
    )
    plain = ODEInformationWithHidden(
        _decay_softplus, prior.E0_x, prior.E1, prior.E0_hidden
    )
    return linked, plain


@pytest.mark.parametrize(
    "correction", [None, QuadratureCorrection(n_nodes=3)], ids=["ek1", "slr"]
)
@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_a_linked_input_runs_like_the_link_inside_the_vector_field(
    correction, calibration
):
    """An uncertain rate, inferred from data: both models give the same run."""
    prior = JointPrior(IWP(q=2, d=1), MaternPrior(q=1, d=1, length_scale=1.0))
    linked, plain = _linked_models(prior)
    ts = np.linspace(0.0, 3.0, 31)
    t_obs = ts[3::3]
    rate = 1.5
    z = (2.0 * np.exp(-rate * t_obs) + 0.01 * np.sin(11.0 * t_obs))[:, None]
    obs = prepare_observations(
        [Measurement(np.eye(1), z, t_obs, noise=1e-4)], prior.E0_x, ts
    )
    m_0 = np.array([2.0, -0.6, 0.18, 0.0, 0.0])
    P_0_sqr = np.diag(np.array([1e-3, 0.3, 0.3, 1.0, 0.5]))
    runs = [
        gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            ts,
            correction=correction,
            calibration=calibration,
            obs_model=obs,
        )
        for measure in (linked, plain)
    ]
    a, b = runs
    assert_close(a.m, b.m, rtol=1e-10)
    assert_close(cov(a.P_sqr), cov(b.P_sqr), rtol=1e-10)
    assert_close(a.log_likelihood, b.log_likelihood, rtol=1e-10)
    assert_close(a.log_likelihood_obs, b.log_likelihood_obs, rtol=1e-10)
    assert_close(a.sigma_sqr, b.sigma_sqr, rtol=1e-10)
    m_s = rts_smoother(a).m
    assert_close(m_s, rts_smoother(b).m, rtol=1e-10)
    # Not a trivial run: the data move the input from the prior mean 0 to
    # softplus^-1(1.5) = 1.25.
    assert abs(float(m_s[15, 3]) - float(np.log(np.expm1(rate)))) < 0.1


def test_a_linked_constant_rate_gives_the_exponential_decay():
    """A known rate ``softplus(u0)``: ``x(t) = x0 exp(-softplus(u0) t)``."""
    prior = JointPrior(IWP(q=3, d=1), IWP(q=0, d=1, Xi=1e-14 * np.eye(1)))
    linked, _ = _linked_models(prior)
    u0, x0 = 0.4, 1.5
    rate = jax.nn.softplus(u0)
    m_0 = np.array([x0, -rate * x0, rate**2 * x0, -(rate**3) * x0, u0])
    ts = np.linspace(0.0, 4.0, 81)
    res = gaussian_filter(m_0, np.zeros((5, 5)), prior, linked, ts)
    assert float(np.max(np.abs(res.m[:, 0] - x0 * np.exp(-rate * ts)))) < 1e-6
    assert_close(res.m[:, 4], np.full(ts.shape, u0), rtol=1e-9)


# ---------------------------------------------------------------------------
# SecondOrderODEInformationWithHidden: an oscillator driven by a hidden input
# ---------------------------------------------------------------------------

# x'' = -x + u with the input u(t) = A_U + B_U t, x(0) = X0, x'(0) = V0.
A_U, B_U, X0, V0 = 0.5, -0.3, 1.0, 0.2


def _driven(x, v, u, *, t):
    return -x + u


def _driven_solution(t):
    return A_U + B_U * t + (X0 - A_U) * np.cos(t) + (V0 - B_U) * np.sin(t)


def _forced(x, v, *, t):
    return -x + A_U + B_U * t


def _exact_derivatives(q: int) -> jax.Array:
    derivatives, f = [], _driven_solution
    for _ in range(q + 1):
        derivatives.append(f(0.0))
        f = jax.grad(f)
    return np.array(derivatives)


# Under dynamic calibration the diffusion is estimated from a residual of about
# 1e-8 that is a difference of O(1) numbers; the joint model forms the input as
# u_n + h u'_n, the plain one as A_U + B_U t, and that rounding moves the
# estimate (and so the run) by about 1e-8 relative.
@pytest.mark.parametrize(("calibration", "rtol"), [("none", 1e-10), ("dynamic", 1e-6)])
def test_a_driven_oscillator_with_a_known_input(calibration, rtol):
    """The input is a line known exactly (a once-integrated prior without noise).

    The run must match the closed-form solution, and equal the plain
    second-order model whose vector field carries the same input as an explicit
    function of time.
    """
    q = 4
    # A noise-free input up to a variance far below the state's.
    prior = JointPrior(IWP(q=q, d=1), IWP(q=1, d=1, Xi=1e-18 * np.eye(1)))
    measure = SecondOrderODEInformationWithHidden(
        _driven, prior.E0_x, prior.E1, prior.E2, prior.E0_hidden
    )
    m_0 = np.concatenate([_exact_derivatives(q), np.array([A_U, B_U])])
    ts = np.linspace(0.0, 6.0, 121)
    res = gaussian_filter(
        m_0, np.zeros((q + 3, q + 3)), prior, measure, ts, calibration=calibration
    )
    assert float(np.max(np.abs(res.m[:, 0] - _driven_solution(ts)))) < 1e-6
    velocity = jax.vmap(jax.grad(_driven_solution))(ts)
    assert float(np.max(np.abs(res.m[:, 1] - velocity))) < 1e-5
    assert_close(res.m[:, q + 1], A_U + B_U * ts, rtol=1e-9)

    plain_prior = IWP(q=q, d=1)
    plain = gaussian_filter(
        _exact_derivatives(q),
        np.zeros((q + 1, q + 1)),
        plain_prior,
        SecondOrderODEInformation(
            _forced, plain_prior.E0, plain_prior.E1, plain_prior.E2
        ),
        ts,
        calibration=calibration,
    )
    assert_close(res.m[:, : q + 1], plain.m, rtol=rtol)
    assert_close(cov(res.P_sqr)[:, : q + 1, : q + 1], cov(plain.P_sqr), rtol=rtol)
    assert_close(res.log_likelihood, plain.log_likelihood, rtol=rtol)


def test_a_driven_oscillator_infers_its_input_from_data():
    """An unknown constant input, identified from noisy positions."""
    q = 3
    prior = JointPrior(IWP(q=q, d=1), MaternPrior(q=1, d=1, length_scale=1.0))
    measure = SecondOrderODEInformationWithHidden(
        _driven, prior.E0_x, prior.E1, prior.E2, prior.E0_hidden
    )
    u_true = 0.8
    ts = np.linspace(0.0, 6.0, 61)
    x_true = u_true + (X0 - u_true) * np.cos(ts) + V0 * np.sin(ts)
    noise = 0.01 * jax.random.normal(jax.random.PRNGKey(0), ts[1:].shape)
    obs = prepare_observations(
        [Measurement(np.eye(1), (x_true[1:] + noise)[:, None], ts[1:], noise=1e-4)],
        prior.E0_x,
        ts,
    )
    m_0 = np.array([X0, V0, -X0, -V0, 0.0, 0.0])
    P_0_sqr = np.diag(np.array([1e-3, 1e-3, 1.0, 1.0, 1.0, 0.5]))
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, obs_model=obs)
    m_s, P_s_sqr = rts_smoother(res)
    u_std = np.sqrt(cov(P_s_sqr)[:, q + 1, q + 1])
    # Away from the ends the smoothed input is within three standard deviations.
    inner = slice(10, -10)
    assert bool(np.all(np.abs(m_s[inner, q + 1] - u_true) < 3.0 * u_std[inner]))
    assert float(np.max(np.abs(m_s[inner, q + 1] - u_true))) < 0.1
