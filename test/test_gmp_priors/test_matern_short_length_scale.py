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
from jax.scipy.linalg import expm

from ode_filters import MaternPrior

jax.config.update("jax_enable_x64", True)


def _block_form(prior, h):
    """The former implementation: `expm` of `[[F, S], [0, -F.T]] h`."""
    n = prior.n
    H = jnp.block([[prior._F, prior.S], [jnp.zeros_like(prior._F), -prior._F.T]])
    E = expm(H * h)
    A = E[:n, :n]
    return A, E[:n, n:] @ A.T


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
    A_ref, Q_ref = (np.asarray(v) for v in _block_form(prior, h))
    assert np.isfinite(A_ref).all()
    A, Q = (np.asarray(v) for v in prior.A_and_Q(h))
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
