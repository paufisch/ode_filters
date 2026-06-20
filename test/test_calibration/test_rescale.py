"""Tests for ode_filters.calibration.rescale."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters.calibration.rescale import rescale_sqr, rescale_sqr_seq
from ode_filters.priors.gmp_priors import IWP, JointPrior


def test_rescale_sqr_scales_covariance():
    P_sqr = np.asarray([[1.0, 0.5], [0.0, 2.0]])
    sigma_sqr = 4.0
    out = rescale_sqr(P_sqr, sigma_sqr)
    # P_out = out.T @ out should equal sigma_sqr * (P_sqr.T @ P_sqr).
    P_in = P_sqr.T @ P_sqr
    P_out = out.T @ out
    assert np.allclose(P_out, sigma_sqr * P_in)


def test_rescale_sqr_seq_scalar():
    P_seq = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [0.0, 1.5]],
        ]
    )
    sigma_sqr = 9.0
    out = rescale_sqr_seq(P_seq, sigma_sqr)
    assert np.allclose(out, 3.0 * P_seq)


def test_rescale_sqr_seq_per_step():
    P_seq = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 1.0], [0.0, 1.5]],
        ]
    )
    sigma_sqr = np.asarray([4.0, 9.0])
    out = rescale_sqr_seq(P_seq, sigma_sqr)
    assert np.allclose(out[0], 2.0 * P_seq[0])
    assert np.allclose(out[1], 3.0 * P_seq[1])


def test_rescale_sqr_seq_shape_mismatch_raises():
    P_seq = np.zeros((3, 2, 2))
    with pytest.raises(ValueError, match="sigma_sqr must be scalar"):
        rescale_sqr_seq(P_seq, np.asarray([1.0, 2.0]))


def test_prior_aware_rescale_matches_default_for_non_joint():
    """For a non-joint prior, prior= must give the same result as the default."""
    prior = IWP(q=2, d=1)
    P_sqr = np.asarray([[1.0, 0.5, 0.2], [0.0, 2.0, 0.1], [0.0, 0.0, 1.5]])
    assert np.allclose(rescale_sqr(P_sqr, 4.0, prior=prior), rescale_sqr(P_sqr, 4.0))
    P_seq = np.stack([P_sqr, 2.0 * P_sqr])
    assert np.allclose(
        rescale_sqr_seq(P_seq, 9.0, prior=prior), rescale_sqr_seq(P_seq, 9.0)
    )


def _joint_prior():
    # state x: IWP(q=2, d=1) -> D_x = 3 columns; input u: IWP(q=1, d=1) -> 2.
    return JointPrior(IWP(q=2, d=1), IWP(q=1, d=1))


def test_joint_rescale_leaves_input_block_unchanged():
    """Block-aware post-hoc rescale must touch only the ODE-state block.

    Regression for the latent-force correctness gap: the default (non-block-aware)
    path wrongly scales the hidden/input block; passing prior= fixes it.
    """
    prior = _joint_prior()
    D_x = prior._D_x  # 3
    P_sqr = np.arange(1.0, 26.0).reshape(5, 5)  # full-state sqrt, D = D_x + D_u = 5
    sigma_sqr = 4.0

    out = rescale_sqr(P_sqr, sigma_sqr, prior=prior)

    # State columns are scaled by sqrt(sigma_sqr); input columns are untouched.
    assert np.allclose(out[:, :D_x], 2.0 * P_sqr[:, :D_x])
    assert np.allclose(out[:, D_x:], P_sqr[:, D_x:])
    # The naive path (no prior) would have scaled the input block too -> differs.
    naive = rescale_sqr(P_sqr, sigma_sqr)
    assert not np.allclose(naive[:, D_x:], out[:, D_x:])


def test_joint_rescale_seq_per_step():
    prior = _joint_prior()
    D_x = prior._D_x
    P0 = np.arange(1.0, 26.0).reshape(5, 5)
    P_seq = np.stack([P0, 2.0 * P0])
    sigma_seq = np.asarray([4.0, 9.0])

    out = rescale_sqr_seq(P_seq, sigma_seq, prior=prior)

    assert np.allclose(out[0, :, :D_x], 2.0 * P_seq[0, :, :D_x])
    assert np.allclose(out[0, :, D_x:], P_seq[0, :, D_x:])
    assert np.allclose(out[1, :, :D_x], 3.0 * P_seq[1, :, :D_x])
    assert np.allclose(out[1, :, D_x:], P_seq[1, :, D_x:])
