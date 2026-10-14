"""Joint Taylor-mode initialization of an ODE driven by a hidden input."""

from __future__ import annotations

from math import factorial

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


def _recursive_derivatives(vf, initial, u_block, *, d_u, q_u, q_x, order):
    """``[x, x', ..., x^(q_x)]`` by differentiating the ODE along its own Taylor polynomial.

    An oracle independent of Taylor-mode (``jet``): with the derivatives
    ``x^(0..k)`` known, ``x^(k+1)`` is the ``(k + 1 - order)``-th time derivative
    of ``vf`` along the polynomial they define, driven by the input's Taylor
    polynomial with coefficients ``u_block``. Nested ``jax.jacfwd`` in ``t``
    takes the derivatives. The time argument of ``vf`` is held at ``0``.
    """
    u_coefficients = u_block.reshape(q_u + 1, d_u)
    derivatives = list(initial)

    def polynomial(coefficients, t):
        return sum(c * t**j / factorial(j) for j, c in enumerate(coefficients))

    while len(derivatives) < q_x + 1:
        known = list(derivatives)

        def rhs(t, known=known):
            x = polynomial(known, t)
            u = polynomial(u_coefficients, t)
            if order == 1:
                return vf(x, u, t=0.0)
            v = jax.jacfwd(lambda s: polynomial(known, s))(t)
            return vf(x, v, u, t=0.0)

        f = rhs
        for _ in range(len(known) - order):
            f = jax.jacfwd(f)
        derivatives.append(f(np.array(0.0)))
    return np.concatenate(derivatives)


def _nonlinear_first_order(x, u, *, t):
    return np.array([-x[0] * np.exp(u[0]) + x[1], np.sin(x[0]) - 0.5 * x[1] * u[0]])


def _nonlinear_second_order(x, v, u, *, t):
    return -np.sin(x) - 0.3 * v * (1.0 + u**2) + x * u


@pytest.mark.parametrize(
    ("order", "vf", "inits", "q_x"),
    [
        (1, _nonlinear_first_order, np.array([0.7, -0.4]), 3),
        (2, _nonlinear_second_order, (np.array([0.6]), np.array([-0.2])), 4),
    ],
    ids=["first-order", "second-order"],
)
def test_matches_an_independent_oracle(order, vf, inits, q_x):
    """Mean and covariance against the recursive derivatives and their Jacobian.

    The input ``u`` enters nonlinearly, so ``J`` depends on where it is taken
    (the input mean), and both blocks of the initial covariance are non-zero.
    """
    initial = [inits] if order == 1 else list(inits)
    d_x = initial[0].shape[0]
    prior = JointPrior(IWP(q=q_x, d=d_x), MaternPrior(q=2, d=1, length_scale=1.0))
    D_x, q_u = prior.prior_x.state_dim, prior.prior_u.q
    keys = jax.random.split(jax.random.PRNGKey(4), 2)
    state_cov_sqr = np.triu(0.1 * jax.random.normal(keys[0], (D_x, D_x)))
    input_cov_sqr = np.triu(jax.random.normal(keys[1], (q_u + 1, q_u + 1)))
    u0 = 0.3
    m, P_sqr = joint_taylor_initialization(
        vf,
        inits,
        prior=prior,
        order=order,
        u0=np.array([u0]),
        input_cov_sqr=input_cov_sqr,
        state_cov_sqr=state_cov_sqr,
    )

    u_mean = np.zeros(q_u + 1).at[0].set(u0)

    def oracle(u_block):
        return _recursive_derivatives(
            vf, initial, u_block, d_u=1, q_u=q_u, q_x=q_x, order=order
        )

    m_x = oracle(u_mean)
    J = jax.jacfwd(oracle)(u_mean)
    S_x, S_u = cov(state_cov_sqr), cov(input_cov_sqr)
    P = np.block([[S_x + J @ S_u @ J.T, J @ S_u], [S_u @ J.T, S_u]])
    assert_close(m, np.concatenate([m_x, u_mean]), rtol=1e-12)
    assert_close(cov(P_sqr), P, rtol=1e-12)
    # The input reaches the state block beyond its first derivative.
    assert float(np.max(np.abs(J[(order + 1) * d_x :, 1:]))) > 0.1


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
