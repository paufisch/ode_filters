"""Tests for ``ode_filters.filters.smoothing``: the RTS smoother's outputs.

Posterior samples and dense output (``sample_posterior``, ``interpolate``), and
smoothing through a step whose calibrated diffusion is zero: from an exact
(Taylor-mode) initial state such a step leaves the predicted covariance
singular, and the backward conditional's triangular solve (``invert``) gives
NaN. A positive ``min_sigma_sqr`` floor avoids it; the failure is pinned below.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    DiagonalCalibration,
    DynamicCalibration,
    ODEInformation,
    affine_filter,
    gaussian_filter,
    interpolate,
    rts_smoother,
    sample_posterior,
    taylor_mode_initialization,
)
from ode_filters.linearization.surrogate import AffineSurrogate
from test.helpers import assert_close, cov

PRIOR = IWP(q=2, d=1)
M_0 = np.array([1.0, -1.0, 1.0])
P_0_SQR = 1e-2 * np.eye(3)
# The decay ODE x' + x = 0 as frozen information, observed with a little noise.
H_ODE = (PRIOR.E1 + PRIOR.E0)[None]
R_ODE = 1e-2 * np.ones((1, 1))


def _surrogate(n_steps: int) -> AffineSurrogate:
    return AffineSurrogate(
        np.tile(H_ODE, (n_steps, 1, 1)),
        np.zeros((n_steps, 1)),
        np.tile(R_ODE, (n_steps, 1, 1)),
    )


def _coarse(n_steps: int = 6):
    ts = np.linspace(0.0, 3.0, n_steps + 1)
    return ts, affine_filter(M_0, P_0_SQR, PRIOR, _surrogate(n_steps), ts)


def _refined(ts):
    """The same model on a grid with the midpoints added, uninformative there."""
    midpoints = 0.5 * (ts[:-1] + ts[1:])
    fine = np.sort(np.concatenate([ts, midpoints]))
    n = ts.shape[0] - 1
    coarse = _surrogate(n)
    # No information at a midpoint: zero rows with unit noise.
    blank = AffineSurrogate(
        np.zeros_like(coarse.H), coarse.c, np.ones_like(coarse.R_sqr)
    )
    interleaved = jax.tree.map(
        lambda b, c: np.stack([b, c], axis=1).reshape(2 * n, *b.shape[1:]),
        blank,
        coarse,
    )
    return midpoints, affine_filter(M_0, P_0_SQR, PRIOR, interleaved, fine)


def test_interpolation_equals_a_pass_on_the_refined_grid():
    """Bridging between nodes equals filtering through an uninformative midpoint."""
    ts, coarse = _coarse()
    midpoints, fine = _refined(ts)
    smoothed_fine = rts_smoother(fine)
    smoothed = interpolate(coarse, PRIOR, midpoints)
    assert_close(smoothed.m, smoothed_fine.m[1::2], rtol=1e-10)
    assert_close(cov(smoothed.P_sqr), cov(smoothed_fine.P_sqr[1::2]), rtol=1e-10)
    filtered = interpolate(coarse, PRIOR, midpoints, smoothed=False)
    assert_close(filtered.m, fine.m[1::2], rtol=1e-10)
    assert_close(cov(filtered.P_sqr), cov(fine.P_sqr[1::2]), rtol=1e-10)


@pytest.mark.parametrize("smoothed", [True, False])
def test_interpolation_at_the_nodes(smoothed):
    ts, result = _coarse()
    reference = rts_smoother(result) if smoothed else result
    at_nodes = interpolate(result, PRIOR, ts, smoothed=smoothed)
    assert_close(at_nodes.m, reference.m, rtol=1e-14)
    assert_close(at_nodes.P_sqr, reference.P_sqr, rtol=1e-14)


def test_limits_at_an_interior_node():
    """Smoothed marginals are continuous; filtered ones jump at the update."""
    ts, result = _coarse()
    node, eps = 3, 1e-7
    around = np.array([ts[node] - eps, ts[node] + eps])
    smoothed = interpolate(result, PRIOR, around)
    assert_close(smoothed.m, np.stack([rts_smoother(result).m[node]] * 2), rtol=1e-5)
    filtered = interpolate(result, PRIOR, around, smoothed=False)
    # From the left: the prediction of the step into the node; from the right: the
    # filtered state at the node.
    assert result.m_pred is not None
    expected = np.stack([result.m_pred[node - 1], result.m[node]])
    assert_close(filtered.m, expected, rtol=1e-5)


def test_interpolation_needs_a_per_step_diffusion():
    _, result = _coarse()
    with pytest.raises(ValueError, match="fixed-grid"):
        interpolate(result._replace(diffusion=None), PRIOR, np.array([0.5]))


def _cross_cov(a, b):
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    return np.einsum("si,sj->ij", a, b) / (a.shape[0] - 1)


def _assert_within_standard_errors(sample_cov, true_cov, var_a, var_b, n, k=5.0):
    """Entrywise ``|C_hat - C| <= k SE``, ``SE^2 = (var_a_i var_b_j + C_ij^2) / n``."""
    se = np.sqrt((np.outer(var_a, var_b) + true_cov**2) / n)
    assert np.all(np.abs(sample_cov - true_cov) <= k * se + 1e-15)


def test_samples_are_joint_draws_from_the_smoothing_posterior():
    """Mean, covariance and the covariance between neighbours match the posterior."""
    ts, result = _coarse()
    n = 20_000
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    paths = jax.vmap(sample_posterior, in_axes=(0, None))(keys, result)
    assert paths.shape == (n, ts.shape[0], 3)
    smoothed = rts_smoother(result)
    P = cov(smoothed.P_sqr)
    variances = jax.vmap(np.diag)(P)
    se_mean = np.sqrt(variances / n)
    assert np.all(np.abs(paths.mean(0) - smoothed.m) <= 5.0 * se_mean)
    backward = result.backward
    for k in (0, 3, ts.shape[0] - 1):
        _assert_within_standard_errors(
            _cross_cov(paths[:, k], paths[:, k]), P[k], variances[k], variances[k], n
        )
    # Cov(x_k, x_{k+1}) = G_k Cov(x_{k+1}): only a joint draw gets this right.
    k = 2
    _assert_within_standard_errors(
        _cross_cov(paths[:, k], paths[:, k + 1]),
        backward.G[k] @ P[k + 1],
        variances[k],
        variances[k + 1],
        n,
    )


def _logistic_dynamic():
    """A dynamically calibrated EK1 run: the diffusion changes from step to step."""

    def vf(x, *, t):
        return x * (1.0 - x)

    prior = IWP(q=2, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.1]), q=2)
    ts = np.linspace(0.0, 5.0, 11)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    return prior, ts, gaussian_filter(m_0, P_0_sqr, prior, measure, ts)


def test_filtered_limits_at_a_node_under_dynamic_calibration():
    """Left of node ``k + 1`` the filtered bridge is step ``k``'s prediction.

    The prediction carries step ``k``'s diffusion in its covariance; just right
    of the node the bridge starts from the filtered state there.
    """
    prior, ts, result = _logistic_dynamic()
    assert result.sigma_sqr is not None and result.P_pred_sqr is not None
    assert result.m_pred is not None
    k, eps = 4, 1e-9
    # The test can only see the diffusion index if neighbouring steps differ.
    assert abs(float(result.sigma_sqr[k] / result.sigma_sqr[k - 1]) - 1.0) > 0.1
    around = np.array([ts[k + 1] - eps, ts[k + 1] + eps])
    filtered = interpolate(result, prior, around, smoothed=False)
    assert_close(filtered.m[0], result.m_pred[k], rtol=1e-6)
    assert_close(cov(filtered.P_sqr[0]), cov(result.P_pred_sqr[k]), rtol=1e-6)
    assert_close(filtered.m[1], result.m[k + 1], rtol=1e-6)
    assert_close(cov(filtered.P_sqr[1]), cov(result.P_sqr[k + 1]), rtol=1e-6)


def test_smoothed_marginals_are_continuous_under_dynamic_calibration():
    """Smoothed covariances, not only means, are continuous across a node."""
    prior, ts, result = _logistic_dynamic()
    smooth = rts_smoother(result)
    node, eps = 6, 1e-9
    around = np.array([ts[node] - eps, ts[node] + eps])
    smoothed = interpolate(result, prior, around)
    assert_close(
        cov(smoothed.P_sqr), np.stack([cov(smooth.P_sqr[node])] * 2), rtol=1e-6
    )
    assert_close(smoothed.m, np.stack([smooth.m[node]] * 2), rtol=1e-6)


TS = np.linspace(0.0, 2.0, 11)


def _logistic(x, *, t):
    return x * (1.0 - x)


def _pendulum(x, *, t):
    return np.array([x[1], -np.sin(x[0])])


# (vector field, initial value, calibration policy without a floor)
ZERO_DIFFUSION = {
    "equilibrium-dynamic": (_logistic, [1.0], DynamicCalibration),
    "at-rest-diagonal": (_pendulum, [1.0, 0.0], DiagonalCalibration),
}


def _solve(name, min_sigma_sqr):
    vf, x0, policy = ZERO_DIFFUSION[name]
    x0 = np.array(x0)
    prior = IWP(q=2, d=x0.shape[0])
    m_0, P_0_sqr = taylor_mode_initialization(vf, x0, q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    calibration = policy(min_sigma_sqr=min_sigma_sqr)
    return gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration=calibration)


@pytest.mark.parametrize("name", list(ZERO_DIFFUSION))
def test_the_first_diffusion_is_exactly_zero(name):
    """Guard the premise, and that the filter itself stays finite."""
    result = _solve(name, 0.0)
    assert float(np.min(result.sigma_sqr[0])) == 0.0
    assert bool(np.all(np.isfinite(result.m)))
    assert bool(np.all(np.isfinite(result.P_sqr)))


@pytest.mark.xfail(
    strict=True, reason="invert divides by the singular predicted covariance"
)
@pytest.mark.parametrize("name", list(ZERO_DIFFUSION))
def test_smoothing_through_a_zero_diffusion_is_finite(name):
    smoothed = rts_smoother(_solve(name, 0.0))
    assert bool(np.all(np.isfinite(smoothed.m)))


@pytest.mark.parametrize("name", list(ZERO_DIFFUSION))
def test_a_diffusion_floor_keeps_the_smoother_finite(name):
    result = _solve(name, 1e-12)
    smoothed = rts_smoother(result)
    assert bool(np.all(np.isfinite(smoothed.m)))
    assert bool(np.all(np.isfinite(smoothed.P_sqr)))
    # The smoother ends where the filter does, and starts at the exact value.
    assert np.allclose(smoothed.m[-1], result.m[-1], rtol=1e-14)
    assert np.allclose(smoothed.m[0], result.m[0], atol=1e-12)
