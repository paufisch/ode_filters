"""Coverage for state-block diffusion scaling validation + accessors, and the
PrecondMatern step-size-required guards."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters.priors.gmp_priors import (
    IWP,
    JointPrior,
    PrecondIWP,
    PrecondJointPrior,
    PrecondMaternPrior,
)


def test_iwp_apply_state_sigma_sqr_bad_shape_raises():
    prior = IWP(q=2, d=2)  # _dim = 2
    with pytest.raises(ValueError, match="must be scalar or length"):
        prior.apply_state_sigma_sqr(np.eye(6), np.zeros(3))


def test_joint_apply_state_sigma_sqr_bad_shape_raises():
    prior = JointPrior(IWP(q=2, d=1), IWP(q=1, d=1))  # state _dim = 1
    with pytest.raises(ValueError, match="must be scalar or length"):
        prior.apply_state_sigma_sqr(np.eye(5), np.zeros(2))


def test_precond_joint_state_accessor_and_scaling():
    pj = PrecondJointPrior(PrecondIWP(q=2, d=1), PrecondIWP(q=1, d=1))

    # State-only value projection accessor.
    assert pj.E0_state.shape[0] == 1

    # Bad-shape per-component sigma -> ValueError.
    with pytest.raises(ValueError, match="must be scalar or length"):
        pj.apply_state_sigma_sqr(np.eye(5), np.zeros(2))

    # Scalar scaling of a full-state covariance sqrt touches only the state cols.
    D_x = pj._D_x  # 3
    P_sqr = np.arange(1.0, 26.0).reshape(5, 5)
    out = pj.apply_state_sigma_to_cov_sqr(P_sqr, 4.0)
    assert np.allclose(out[:, :D_x], 2.0 * P_sqr[:, :D_x])
    assert np.allclose(out[:, D_x:], P_sqr[:, D_x:])


def test_precond_matern_A_and_Q_require_step_size():
    p = PrecondMaternPrior(q=2, d=1, length_scale=1.0)
    with pytest.raises(ValueError, match="requires a step size"):
        p.A()
    with pytest.raises(ValueError, match="requires a step size"):
        p.Q()
    with pytest.raises(ValueError, match="requires a step size"):
        p.Q_sqr()
