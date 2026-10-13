"""Joint Taylor-mode initialization of an ODE driven by a hidden input."""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    JointPrior,
    MaternPrior,
    ODEInformationWithHidden,
    gaussian_filter,
    joint_taylor_initialization,
)
from test.helpers import assert_close, assert_grad_matches_fd, cov

A_DECAY = 0.7


def _additive(x, u, *, t):
    return -A_DECAY * x + u


def test_linear_ode_with_an_additive_input_has_the_analytic_cross_term():
    prior = JointPrior(IWP(q=2, d=1), MaternPrior(q=1, d=1, length_scale=1.0))
    m, P_sqr = joint_taylor_initialization(
        _additive, np.array([1.5]), prior=prior, u0=np.array([0.4])
    )
    # x = x0, x' = -a x0 + u, x'' = -a x' + u' = a^2 x0 - a u + u'.
    assert_close(
        m[:3], np.array([1.5, -A_DECAY * 1.5 + 0.4, A_DECAY**2 * 1.5 - A_DECAY * 0.4])
    )
    assert_close(m[3:], np.array([0.4, 0.0]))
    J = np.array([[0.0, 0.0], [1.0, 0.0], [-A_DECAY, 1.0]])
    S_u = cov(prior.prior_u.stationary_cov_sqr())
    P = cov(P_sqr)
    assert_close(P[:3, :3], J @ S_u @ J.T, rtol=1e-12)
    assert_close(P[:3, 3:], J @ S_u, rtol=1e-12)
    assert_close(P[3:, 3:], S_u, rtol=1e-12)


def test_second_order_with_additive_input():
    def vf(x, v, u, *, t):
        return -x - 0.2 * v + u

    prior = JointPrior(IWP(q=3, d=1), MaternPrior(q=2, d=1, length_scale=0.5))
    m, P_sqr = joint_taylor_initialization(
        vf, (np.array([1.0]), np.array([0.0])), prior=prior, order=2
    )
    # x'' = -x - 0.2 v + u, x''' = -v - 0.2 x'' + u'.
    assert_close(m[:4], np.array([1.0, 0.0, -1.0, 0.2]))
    P = cov(P_sqr)
    assert np.allclose(P, P.T)
    assert float(np.min(np.linalg.eigvalsh(P))) > -1e-10
    # The position and velocity are pinned; the acceleration carries Var(u).
    assert_close(P[0, 0], 0.0, atol=1e-14)
    assert_close(P[2, 2], cov(prior.prior_u.stationary_cov_sqr())[0, 0], rtol=1e-12)


@pytest.mark.parametrize(("d_x", "d_u"), [(2, 1), (1, 2), (2, 2)])
def test_shapes_for_several_dimensions(d_x, d_u):
    def vf(x, u, *, t):
        return -x + np.sum(u) * np.ones(d_x)

    prior = JointPrior(IWP(q=2, d=d_x), MaternPrior(q=1, d=d_u, length_scale=1.0))
    m, P_sqr = joint_taylor_initialization(vf, np.ones(d_x), prior=prior)
    D = prior.state_dim
    assert m.shape == (D,) and P_sqr.shape == (D, D)
    assert_close(
        cov(P_sqr)[3 * d_x :, 3 * d_x :], cov(prior.prior_u.stationary_cov_sqr())
    )


def test_explicit_covariances_and_a_non_stationary_input():
    prior = JointPrior(IWP(q=1, d=1), IWP(q=0, d=1))
    _, P_sqr = joint_taylor_initialization(
        _additive,
        np.array([1.0]),
        prior=prior,
        input_cov_sqr=0.5 * np.eye(1),
        state_cov_sqr=0.1 * np.eye(2),
    )
    P = cov(P_sqr)
    assert_close(P[2, 2], 0.25)
    assert_close(P[0, 0], 0.01)  # x0 does not depend on u
    assert_close(P[1, 1], 0.01 + 0.25)  # x' = -a x0 + u


def test_the_initial_law_feeds_a_differentiable_filter():
    ts = np.linspace(0.0, 1.0, 11)

    def log_likelihood(log_ell):
        prior = JointPrior(
            IWP(q=2, d=1), MaternPrior(q=1, d=1, length_scale=np.exp(log_ell))
        )
        measure = ODEInformationWithHidden(
            _additive, prior.E0_x, prior.E1, prior.E0_hidden
        )
        m, P_sqr = joint_taylor_initialization(_additive, np.array([1.0]), prior=prior)
        return gaussian_filter(
            m, P_sqr, prior, measure, ts, calibration="none"
        ).log_likelihood

    assert np.isfinite(log_likelihood(np.array(0.0)))
    assert_grad_matches_fd(log_likelihood, np.array(0.1))
    assert np.isfinite(jax.jit(log_likelihood)(np.array(0.0)))


def test_invalid_arguments_raise():
    joint = JointPrior(IWP(q=1, d=1), IWP(q=0, d=1))
    with pytest.raises(TypeError, match="JointPrior"):
        joint_taylor_initialization(_additive, np.ones(1), prior=IWP(q=1, d=1))
    with pytest.raises(ValueError, match="no stationary law"):
        joint_taylor_initialization(_additive, np.ones(1), prior=joint)
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        joint_taylor_initialization(_additive, np.ones(1), prior=joint, order=3)
    with pytest.raises(ValueError, match="tuple"):
        joint_taylor_initialization(
            _additive, np.ones(1), prior=joint, order=2, input_cov_sqr=np.eye(1)
        )
