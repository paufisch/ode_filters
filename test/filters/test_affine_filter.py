"""``affine_filter`` and the trajectory surrogates.

An ODE filter linearized along a fixed trajectory is a linear-Gaussian model.
Linearized at a filter's own predictions it must reproduce that filter; on a
random linear-Gaussian model it must reproduce the dense Kalman filter.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    AffineSurrogate,
    JointPrior,
    MaternPrior,
    Measurement,
    ODEInformation,
    ODEInformationWithHidden,
    PrecondIWP,
    affine_filter,
    gaussian_filter,
    prepare_observations,
    rts_smoother,
    slr_linearize,
    slr_surrogate,
    taylor_mode_initialization,
    taylor_surrogate,
)
from test.helpers import (
    DenseStep,
    assert_close,
    assert_grad_matches_fd,
    cov,
    dense_kalman,
)

TS = np.linspace(0.0, 2.0, 21)


def _logistic(prior=None):
    def vf(x, *, t):
        return x * (1.0 - x)

    prior = IWP(q=2, d=1) if prior is None else prior
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.3]), q=prior.q)
    return prior, ODEInformation(vf, prior.E0, prior.E1), m_0, P_0_sqr


def _hidden():
    def vf(x, u, *, t):
        return -0.8 * x + x * u

    prior = JointPrior(IWP(q=2, d=1), MaternPrior(q=1, d=1, length_scale=1.0))
    measure = ODEInformationWithHidden(vf, prior.E0_x, prior.E1, prior.E0_hidden)
    m_0 = np.array([1.0, -0.8, 0.64, 0.2, 0.0])
    return prior, measure, m_0, 0.1 * np.eye(5)


def _data(E0):
    t_obs = TS[3::3]
    z = (0.5 + 0.1 * np.cos(4.0 * t_obs))[:, None]
    return prepare_observations([Measurement(np.eye(1), z, t_obs, noise=1e-3)], E0, TS)


@pytest.mark.parametrize("setup", [_logistic, _hidden], ids=["logistic", "hidden"])
def test_linearizing_at_the_predictions_reproduces_ek1(setup):
    prior, measure, m_0, P_0_sqr = setup()
    E0 = prior.E0_x if hasattr(prior, "E0_x") else prior.E0
    obs = _data(E0)
    ek1 = gaussian_filter(
        m_0, P_0_sqr, prior, measure, TS, calibration="none", obs_model=obs
    )
    surrogate = taylor_surrogate(measure, ek1.m_pred, ts=TS[1:])
    frozen = affine_filter(m_0, P_0_sqr, prior, surrogate, TS, obs_model=obs)
    assert_close(frozen.m, ek1.m, rtol=1e-10)
    assert_close(cov(frozen.P_sqr), cov(ek1.P_sqr), rtol=1e-10)
    assert_close(frozen.log_likelihood, ek1.log_likelihood, rtol=1e-10)
    assert_close(frozen.log_likelihood_obs, ek1.log_likelihood_obs, rtol=1e-10)
    assert_close(frozen.mz, ek1.mz, rtol=1e-10)
    assert_close(cov(frozen.Pz_sqr), cov(ek1.Pz_sqr), rtol=1e-10)
    assert_close(cov(frozen.Pz_obs_sqr), cov(ek1.Pz_obs_sqr), rtol=1e-10)
    assert_close(frozen.sigma_sqr, ek1.sigma_sqr, rtol=1e-10)
    assert_close(rts_smoother(frozen).m, rts_smoother(ek1).m, rtol=1e-10)


def test_random_linear_gaussian_model_matches_dense_kalman():
    prior = IWP(q=1, d=2)
    D, N, k = 4, TS.shape[0] - 1, 2
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    H = jax.random.normal(keys[0], (N, k, D))
    c = jax.random.normal(keys[1], (N, k))
    R_sqr = 0.3 * np.broadcast_to(np.eye(k), (N, k, k))
    m_0 = jax.random.normal(keys[2], (D,))
    P_0_sqr = 0.5 * np.eye(D)
    res = affine_filter(m_0, P_0_sqr, prior, AffineSurrogate(H, c, R_sqr), TS)

    h = TS[1] - TS[0]
    Q_sqr = prior.Q_sqr(h)
    steps = [
        DenseStep(A=prior.A(h), Q=Q_sqr.T @ Q_sqr, H=H[i], c=c[i], R=cov(R_sqr[i]))
        for i in range(N)
    ]
    ref = dense_kalman(m_0, cov(P_0_sqr), steps)
    assert_close(res.m, ref.m, rtol=1e-8)
    assert_close(cov(res.P_sqr), ref.P, rtol=1e-8)
    assert_close(res.log_likelihood, ref.log_likelihood, rtol=1e-8)
    smoothed = rts_smoother(res)
    assert_close(smoothed.m, ref.m_smooth, rtol=1e-8)
    assert_close(cov(smoothed.P_sqr), ref.P_smooth, rtol=1e-8)


def test_preconditioned_prior_gives_the_same_frozen_filter():
    prior, measure, m_0, P_0_sqr = _logistic()
    ek1 = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="none")
    surrogate = taylor_surrogate(measure, ek1.m_pred, ts=TS[1:])
    plain = affine_filter(m_0, P_0_sqr, prior, surrogate, TS)
    precond = affine_filter(m_0, P_0_sqr, PrecondIWP(q=2, d=1), surrogate, TS)
    assert_close(precond.m, plain.m, rtol=1e-9)
    assert_close(precond.log_likelihood, plain.log_likelihood, rtol=1e-9)


def test_slr_surrogate_is_the_stepwise_slr():
    prior, measure, m_0, P_0_sqr = _logistic()
    ek1 = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="none")
    surrogate = slr_surrogate(measure, ek1.m[1:], ek1.P_sqr[1:], ts=TS[1:], n_nodes=3)
    i = 7
    step = slr_linearize(
        measure, ek1.m[i + 1], ek1.P_sqr[i + 1], t=TS[i + 1], n_nodes=3
    )
    assert_close(surrogate.H[i], step.H)
    assert_close(surrogate.c[i], step.c)
    assert_close(cov(surrogate.R_sqr[i]), cov(step.R_eff_sqr))
    assert_close(surrogate.model(i).H, step.H)


def test_without_data_the_observation_fields_are_none():
    prior, measure, m_0, P_0_sqr = _logistic()
    surrogate = taylor_surrogate(measure, np.tile(m_0, (20, 1)), ts=TS[1:])
    res = affine_filter(m_0, P_0_sqr, prior, surrogate, TS)
    assert res.log_likelihood_obs is None and res.mz_obs is None


def test_jit_vmap_and_gradients():
    prior, measure, m_0, P_0_sqr = _logistic()
    obs = _data(prior.E0)
    surrogate = taylor_surrogate(measure, np.tile(m_0, (20, 1)), ts=TS[1:])

    def ll_obs(s):
        return affine_filter(
            m_0, P_0_sqr, prior, s, TS, obs_model=obs
        ).log_likelihood_obs

    assert_close(jax.jit(ll_obs)(surrogate), ll_obs(surrogate), rtol=1e-12)
    batch = jax.tree.map(lambda a: np.stack([a, 1.01 * a]), surrogate)
    assert jax.vmap(ll_obs)(batch).shape == (2,)
    grad = jax.grad(ll_obs)(surrogate)
    assert bool(np.all(np.isfinite(grad.H))) and bool(np.all(np.isfinite(grad.c)))

    _, hidden_measure, hidden_m_0, hidden_P_0_sqr = _hidden()
    hidden_points = np.tile(hidden_m_0, (20, 1))
    # The data must be prepared outside the trace: their times are matched on
    # the host.
    hidden_obs = _data(_hidden()[0].E0_x)

    def by_length_scale(ell):
        joint = JointPrior(IWP(q=2, d=1), MaternPrior(q=1, d=1, length_scale=ell))
        s = taylor_surrogate(hidden_measure, hidden_points, ts=TS[1:])
        res = affine_filter(
            hidden_m_0, hidden_P_0_sqr, joint, s, TS, obs_model=hidden_obs
        )
        return res.log_likelihood_obs

    assert_grad_matches_fd(by_length_scale, np.array(1.0))

    def by_scale(log_xi):
        iwp = IWP(q=2, d=1, Xi=np.exp(log_xi) * np.eye(1))
        return affine_filter(
            m_0, P_0_sqr, iwp, surrogate, TS, obs_model=obs
        ).log_likelihood_obs

    assert_grad_matches_fd(by_scale, np.array(0.3))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda s: s._replace(H=s.H[:-1], c=s.c[:-1], R_sqr=s.R_sqr[:-1]), "N = 20"),
        (lambda s: s._replace(H=s.H[:, :, :2]), "dimension"),
        (lambda s: s._replace(c=s.c[:, :0]), "shapes"),
    ],
    ids=["length", "state-dimension", "rows"],
)
def test_inconsistent_surrogates_raise(change, match):
    prior, measure, m_0, P_0_sqr = _logistic()
    surrogate = taylor_surrogate(measure, np.tile(m_0, (20, 1)), ts=TS[1:])
    with pytest.raises(ValueError, match=match):
        affine_filter(m_0, P_0_sqr, prior, change(surrogate), TS)
