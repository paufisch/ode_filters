"""``iterated_smoother``: the IEKS (Taylor) and the IPLS (statistical linearization)."""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    JointPrior,
    MaternPrior,
    Measurement,
    ODEInformation,
    ODEInformationWithHidden,
    PrecondIWP,
    QuadratureCorrection,
    affine_filter,
    gaussian_filter,
    iterated_smoother,
    prepare_observations,
    rts_smoother,
    taylor_mode_initialization,
    taylor_surrogate,
)
from test.helpers import assert_close, assert_grad_matches_fd

TS = np.linspace(0.0, 2.0, 33)


def _logistic(q=3, x0=0.1):
    def vf(x, *, t):
        return x * (1.0 - x)

    prior = IWP(q=q, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([x0]), q=q)
    return prior, ODEInformation(vf, prior.E0, prior.E1), m_0, P_0_sqr


def _hidden():
    def vf(x, u, *, t):
        return -0.8 * x + x * u

    prior = JointPrior(IWP(q=2, d=1), MaternPrior(q=1, d=1, length_scale=1.0))
    measure = ODEInformationWithHidden(vf, prior.E0_x, prior.E1, prior.E0_hidden)
    return prior, measure, np.array([1.0, -0.8, 0.64, 0.2, 0.0]), 0.1 * np.eye(5)


def _data(E0, noise=1e-2):
    t_obs = TS[4::4]
    z = 0.1 * np.exp(t_obs) / (0.9 + 0.1 * np.exp(t_obs)) + 0.02 * np.sin(37 * t_obs)
    return prepare_observations(
        [Measurement(np.eye(1), z[:, None], t_obs, noise=noise)], E0, TS
    )


@pytest.mark.parametrize("with_data", [False, True])
def test_zero_iterations_are_the_one_shot_filter_and_smoother(with_data):
    prior, measure, m_0, P_0_sqr = _logistic()
    obs = _data(prior.E0) if with_data else None
    for linearization, correction in [
        ("taylor", None),
        ("slr", QuadratureCorrection(n_nodes=3)),
    ]:
        res = iterated_smoother(
            m_0,
            P_0_sqr,
            prior,
            measure,
            TS,
            linearization=linearization,
            n_iters=0,
            n_nodes=3,
            obs_model=obs,
        )
        one_shot = gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            TS,
            correction=correction,
            calibration="none",
            obs_model=obs,
        )
        assert_close(res.m, rts_smoother(one_shot).m, rtol=1e-12)
        assert_close(res.filter.m, one_shot.m, rtol=1e-12)
        assert res.surrogate is None and res.step.shape == (0,)


def test_a_linear_ode_is_solved_exactly_by_every_pass():
    def vf(x, *, t):
        return -0.7 * x

    prior = IWP(q=2, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    res = iterated_smoother(m_0, P_0_sqr, prior, measure, TS, n_iters=3)
    one_shot = rts_smoother(
        gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="none")
    )
    assert_close(res.m, one_shot.m, rtol=1e-10)
    assert float(np.max(res.step)) < 1e-10


@pytest.mark.parametrize("linearization", ["taylor", "slr"])
def test_nonlinear_iterations_converge_to_a_fixed_point(linearization):
    prior, measure, m_0, P_0_sqr = _hidden()
    obs = _data(prior.E0_x, noise=1e-3)
    res = iterated_smoother(
        m_0,
        P_0_sqr,
        prior,
        measure,
        TS,
        linearization=linearization,
        n_iters=8,
        n_nodes=3,
        obs_model=obs,
    )
    steps = res.step
    assert float(steps[-1]) < 1e-3 * float(steps[0])
    if linearization == "taylor":
        # Fixed point: relinearizing at the final mean reproduces it.
        again = affine_filter(
            m_0,
            P_0_sqr,
            prior,
            taylor_surrogate(measure, res.m[1:], ts=TS[1:]),
            TS,
            obs_model=obs,
        )
        assert_close(rts_smoother(again).m, res.m, atol=1e-5)


def test_warm_start_from_a_converged_point_does_not_move():
    prior, measure, m_0, P_0_sqr = _logistic()
    obs = _data(prior.E0)
    converged = iterated_smoother(
        m_0, P_0_sqr, prior, measure, TS, n_iters=8, obs_model=obs
    )
    warm = iterated_smoother(
        m_0, P_0_sqr, prior, measure, TS, n_iters=1, init=converged.m[1:], obs_model=obs
    )
    assert float(warm.step[0]) < 1e-8
    assert_close(warm.m, converged.m, atol=1e-8)
    slr = iterated_smoother(
        m_0,
        P_0_sqr,
        prior,
        measure,
        TS,
        linearization="slr",
        n_iters=1,
        n_nodes=3,
        init=converged.m[1:],
        init_P_sqr=converged.P_sqr[1:],
        obs_model=obs,
    )
    assert bool(np.all(np.isfinite(slr.m)))


def test_diagnostics_are_stacked_per_pass():
    prior, measure, m_0, P_0_sqr = _logistic()
    obs = _data(prior.E0)

    def diagnostics(state):
        return {
            "ll_obs": state.filter_result.log_likelihood_obs,
            "end": state.m_smooth[-1],
            "point": state.previous_point[0],
        }

    res = iterated_smoother(
        m_0,
        P_0_sqr,
        prior,
        measure,
        TS,
        n_iters=3,
        obs_model=obs,
        diagnostics=diagnostics,
    )
    assert res.diagnostics["ll_obs"].shape == (3,)
    assert res.diagnostics["end"].shape == (3, 4)
    assert_close(
        res.diagnostics["ll_obs"][-1], res.filter.log_likelihood_obs, rtol=1e-12
    )
    assert_close(res.diagnostics["end"][-1], res.m[-1], rtol=1e-12)
    assert res.surrogate is not None and res.surrogate.H.shape == (32, 1, 4)


def test_preconditioned_prior_gives_the_same_smoother():
    _, measure, m_0, P_0_sqr = _logistic(q=2)
    plain = iterated_smoother(m_0, P_0_sqr, IWP(q=2, d=1), measure, TS, n_iters=2)
    precond = iterated_smoother(
        m_0, P_0_sqr, PrecondIWP(q=2, d=1), measure, TS, n_iters=2
    )
    assert_close(precond.m, plain.m, rtol=1e-9)


def test_jit_vmap_and_grad():
    _, measure, m_0, P_0_sqr = _logistic(q=2)
    ts = TS[:9]
    obs = prepare_observations(
        [Measurement(np.eye(1), np.array([[0.12], [0.14]]), ts[4::4], noise=1e-3)],
        IWP(q=2, d=1).E0,
        ts,
    )

    def ll_obs(log_xi):
        prior = IWP(q=2, d=1, Xi=np.exp(log_xi) * np.eye(1))
        res = iterated_smoother(
            m_0, P_0_sqr, prior, measure, ts, n_iters=2, obs_model=obs
        )
        return res.filter.log_likelihood_obs

    assert np.isfinite(jax.jit(ll_obs)(np.array(0.0)))
    assert jax.vmap(ll_obs)(np.array([0.0, 0.5])).shape == (2,)
    assert_grad_matches_fd(ll_obs, np.array(0.2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_iters": -1}, "n_iters"),
        ({"n_iters": 0, "init": np.zeros((32, 4))}, "warm start"),
        ({"linearization": "slr", "init": np.zeros((32, 4))}, "init_P_sqr"),
        ({"init": np.zeros((32, 4)), "init_P_sqr": np.zeros((32, 4, 4))}, "Taylor"),
        ({"init": np.zeros((31, 4))}, "init must have shape"),
        ({"linearization": "newton"}, "linearization"),
    ],
)
@pytest.mark.wrong_input_types
def test_invalid_arguments_raise(kwargs, match):
    prior, measure, m_0, P_0_sqr = _logistic()
    with pytest.raises(ValueError, match=match):
        iterated_smoother(m_0, P_0_sqr, prior, measure, TS, **kwargs)
