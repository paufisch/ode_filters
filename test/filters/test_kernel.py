"""The filter step: prediction under a change of variables, masked data."""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import IWP, PrecondIWP
from ode_filters.calibration import DiagonalCalibration
from ode_filters.filters.kernel import (
    FrozenInformation,
    filter_step,
    observe,
    predict,
)
from ode_filters.gaussian import AffineModel, Gaussian, log_likelihood, update
from test.helpers import assert_close, cov


def _random_gaussian(n, seed=0):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    M = jax.random.normal(k2, (n, n))
    return Gaussian(
        jax.random.normal(k1, (n,)), np.linalg.cholesky(M @ M.T + np.eye(n)).T
    )


def test_predicting_in_preconditioned_coordinates_changes_nothing():
    x = _random_gaussian(6)
    h = 0.05
    plain_pred, plain_back = predict(x, IWP(q=2, d=2).transition(h))
    pre_pred, pre_back = predict(x, PrecondIWP(q=2, d=2).transition(h))
    assert_close(pre_pred.m, plain_pred.m, rtol=1e-12)
    assert_close(cov(pre_pred.P_sqr), cov(plain_pred.P_sqr), rtol=1e-10)
    assert_close(pre_back.G, plain_back.G, rtol=1e-9)
    assert_close(pre_back.d, plain_back.d, rtol=1e-9)
    assert_close(cov(pre_back.P_sqr), cov(plain_back.P_sqr), rtol=1e-9)


def test_rows_without_data_are_inert():
    """Observing with a mask equals observing only the rows that have data."""
    x = _random_gaussian(4)
    keys = jax.random.split(jax.random.PRNGKey(1), 2)
    H = jax.random.normal(keys[0], (3, 4))
    c = jax.random.normal(keys[1], (3,))
    R_sqr = np.diag(np.array([0.1, 0.2, 0.3]))
    mask = np.array([True, False, True])
    rows = np.array([0, 2])

    posterior, innovation, ll = observe(x, AffineModel(H, c, R_sqr), mask)
    expected, expected_innovation = update(
        AffineModel(H[rows], c[rows], R_sqr[rows][:, rows]), x
    )
    assert_close(posterior.m, expected.m, rtol=1e-12)
    assert_close(cov(posterior.P_sqr), cov(expected.P_sqr), rtol=1e-12)
    assert_close(innovation.m[rows], expected_innovation.m, rtol=1e-12)
    assert float(innovation.m[1]) == 0.0
    assert_close(ll, log_likelihood(expected_innovation), rtol=1e-12)


def test_a_step_without_any_data_leaves_the_state_unchanged():
    x = _random_gaussian(3)
    model = AffineModel(np.ones((2, 3)), np.ones(2), np.eye(2))
    posterior, _, ll = observe(x, model, np.zeros(2, dtype=bool))
    assert_close(posterior.m, x.m, rtol=1e-14)
    assert_close(cov(posterior.P_sqr), cov(x.P_sqr), rtol=1e-12)
    assert float(ll) == pytest.approx(0.0, abs=1e-12)


def test_a_frozen_model_has_no_left_hand_side_selector():
    prior = IWP(q=1, d=1)
    model = AffineModel(prior.E1, np.zeros(1), np.zeros((1, 1)))
    with pytest.raises(NotImplementedError, match="left-hand-side"):
        filter_step(
            _random_gaussian(2),
            prior.transition(0.1),
            FrozenInformation(model),
            t=0.1,
            calibration=DiagonalCalibration(jacobian="ek0"),
            scale_noise=prior.apply_state_sigma_sqr,
        )
