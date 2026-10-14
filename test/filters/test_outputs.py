"""Posterior samples and interpolation between grid nodes."""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import IWP, affine_filter, interpolate, rts_smoother, sample_posterior
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
