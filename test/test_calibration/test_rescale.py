"""Tests for ode_filters.calibration.rescale."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters.calibration.rescale import rescale_sqr, rescale_sqr_seq


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
