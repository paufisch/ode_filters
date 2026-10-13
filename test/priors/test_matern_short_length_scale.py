"""`MaternPrior` transition and noise at SHORT length scales.

The Hamiltonian-block matrix fraction decomposition overflowed to NaN once
`lam = sqrt((2q+1)/length_scale)` was a few tens -- `lam h` still well below 1 --
which is what a problem re-expressed on a unit time interval looks like.  The
normalized form must stay finite there and agree with the block form where that
is finite.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ode_filters import (
    MaternPrior,
    ODEInformation,
    PrecondMaternPrior,
    gaussian_filter,
    taylor_mode_initialization,
)
from test.helpers import matern_companion, van_loan

jax.config.update("jax_enable_x64", True)


def _block_form(q, length_scale, h):
    """The former implementation: `expm` of `[[F, S], [0, -F.T]] h`."""
    return van_loan(*matern_companion(length_scale, q), h)


# (q, length_scale, h) cells where the block form is finite AND accurate; at
# q = 3, length_scale = 0.16, h = 0.5 it is finite but already losing digits.
@pytest.mark.parametrize(
    "q, length_scale, h",
    [
        (q, ell, h)
        for q in (1, 2, 3)
        for ell in (200.0, 10.0, 0.16)
        for h in (1e-2, 0.5)
        if not (q == 3 and ell == 0.16 and h == 0.5)
    ],
)
def test_matches_block_form_where_finite(q, length_scale, h):
    prior = MaternPrior(q, 1, length_scale=length_scale, Xi=np.eye(1))
    A_ref, Q_ref = (np.asarray(v) for v in _block_form(q, length_scale, h))
    assert np.isfinite(A_ref).all()
    A, Q = np.asarray(prior.A(h)), np.asarray(prior.Q(h))
    assert np.allclose(A, A_ref, rtol=1e-10, atol=1e-12 * np.abs(A_ref).max())
    assert np.allclose(Q, Q_ref, rtol=1e-8, atol=1e-10 * np.abs(Q_ref).max())


@pytest.mark.parametrize("length_scale", [2.56e-3, 1.2e-4])
@pytest.mark.parametrize("h", [1e-2, 1.0 / 512, 1.0 / 8640])
def test_finite_at_short_length_scale(length_scale, h):
    prior = MaternPrior(2, 1, length_scale=length_scale, Xi=np.eye(1))
    A = np.asarray(prior.A(h))
    Q = np.asarray(prior.Q(h))
    assert np.isfinite(A).all() and np.isfinite(Q).all()
    # Q(h) is a covariance: symmetric positive semi-definite.
    assert np.allclose(Q, Q.T)
    assert np.linalg.eigvalsh(Q).min() > -1e-12 * np.abs(Q).max()
    # and consistent with its own square-root factor
    R = np.asarray(prior.Q_sqr(h))
    assert np.allclose(R.T @ R, Q, rtol=1e-10, atol=1e-12 * np.abs(Q).max())


@pytest.mark.parametrize("length_scale", [2.56e-3, 1.2e-4])
@pytest.mark.parametrize("h", [1e-2, 1.0 / 512, 1.0 / 8640])
def test_precond_matern_is_the_plain_matern_in_preconditioned_coordinates(
    length_scale, h
):
    """PrecondMaternPrior shares MaternPrior's stable discretization.

    Its ``A`` used to come from the Hamiltonian-block ``expm`` that overflows at
    short length scales, so it returned NaN where MaternPrior was already fixed.
    In preconditioned coordinates ``A_bar = T^-1 A T`` and
    ``Q_bar = T^-1 Q T^-T``.
    """
    plain = MaternPrior(2, 1, length_scale=length_scale, Xi=np.eye(1))
    precond = PrecondMaternPrior(2, 1, length_scale=length_scale, Xi=np.eye(1))
    tr = precond.transition(h)
    T = np.diag(np.asarray(tr.T_diag))
    A_bar = np.asarray(tr.A)
    Q_bar = np.asarray(tr.Q_sqr.T @ tr.Q_sqr)
    assert np.isfinite(A_bar).all() and np.isfinite(Q_bar).all()
    A, Q = np.asarray(plain.A(h)), np.asarray(plain.Q(h))
    assert np.allclose(
        T @ A_bar @ np.linalg.inv(T), A, rtol=1e-10, atol=1e-12 * np.abs(A).max()
    )
    assert np.allclose(T @ Q_bar @ T.T, Q, rtol=1e-8, atol=1e-10 * np.abs(Q).max())


def test_precond_matern_filter_matches_plain_at_short_length_scale():
    """The configuration from the MaternPrior fix (q=2, length_scale=2.6e-3,
    N=100 on a unit interval) must also solve with the preconditioned prior."""

    def vf(x, *, t):
        return -x

    mu_0, P0_sqr = taylor_mode_initialization(vf, jnp.array([1.0]), q=2)
    means = []
    for cls in (MaternPrior, PrecondMaternPrior):
        prior = cls(2, 1, 2.6e-3)
        measure = ODEInformation(vf, prior.E0, prior.E1)
        res = gaussian_filter(mu_0, P0_sqr, prior, measure, np.linspace(0.0, 1.0, 101))
        means.append(np.asarray(res.m))
    assert np.isfinite(means[1]).all()
    assert np.allclose(means[1], means[0], atol=1e-10)
