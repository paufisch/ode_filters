"""A hidden input recovered from data of the state it drives.

``HIDDEN_LINEAR`` (``test.helpers``) is the linear ODE ``x' = L x + B u``
driven by ``u(t) = sin(1.5 t)``, with its solution in closed form. The filter
sees the input only as a ``JointPrior`` block and the data only of ``x``; the
smoother must recover both.
"""

from __future__ import annotations

import jax
import jax.numpy as np

from ode_filters import (
    IWP,
    JointPrior,
    MaternPrior,
    Measurement,
    ODEInformationWithHidden,
    gaussian_filter,
    joint_taylor_initialization,
    prepare_observations,
    rts_smoother,
)
from test.helpers import HIDDEN_LINEAR, assert_close

PROBLEM = HIDDEN_LINEAR
TS = np.linspace(*PROBLEM.tspan, 61)


def test_the_closed_form_solves_the_problem():
    ts = np.array([0.0, 0.7, 2.9])
    x = PROBLEM.solution(ts)
    dx = jax.vmap(jax.jacfwd(lambda t: PROBLEM.solution(t[None])[0]))(ts)
    u = PROBLEM.hidden(ts)
    assert_close(x[0], PROBLEM.x0, rtol=1e-14)
    vf = jax.vmap(lambda x, u, t: PROBLEM.vf(x, u, t=t))(x, u, ts)
    assert_close(dx, vf, rtol=1e-12)


def test_the_smoother_recovers_the_solution_and_the_hidden_input():
    prior = JointPrior(IWP(q=2, d=2), MaternPrior(q=1, d=1, length_scale=1.0))
    m_0, P_0_sqr = joint_taylor_initialization(PROBLEM.vf, PROBLEM.x0, prior=prior)
    measure = ODEInformationWithHidden(
        PROBLEM.vf, prior.E0_x, prior.E1, prior.E0_hidden
    )
    # Both solution components at every node, with a standard deviation 1e-3.
    data = Measurement(np.eye(2), PROBLEM.solution(TS[1:]), TS[1:], noise=1e-6)
    obs_model = prepare_observations([data], prior.E0_x, TS)
    result = gaussian_filter(
        m_0, P_0_sqr, prior, measure, TS, calibration="none", obs_model=obs_model
    )
    smoothed = rts_smoother(result)

    x = smoothed.m @ prior.E0_x.T
    assert float(np.max(np.abs(x - PROBLEM.solution(TS)))) < 1e-3
    # The input is seen only through the derivative of the data, which is
    # least determined at the ends of the interval.
    u_error = np.abs(smoothed.m @ prior.E0_hidden.T - PROBLEM.hidden(TS))[:, 0]
    assert float(np.max(u_error[5:-5])) < 2e-3
    assert float(np.max(u_error)) < 5e-2
