"""Tests for the integrated Wiener process priors."""

from math import factorial

import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.priors import IWP, PrecondIWP
from ode_filters.priors.iwp import _iwp_constants, _iwp_scalar_A


@pytest.mark.parametrize("q,h", [(0, 0.7), (1, 0.3), (3, 0.05)])
def test_transition_and_noise_match_the_closed_forms(q, h):
    dim = q + 1
    A = onp.zeros((dim, dim))
    Q = onp.zeros((dim, dim))
    for i in range(dim):
        for j in range(dim):
            if j >= i:
                A[i, j] = h ** (j - i) / factorial(j - i)
            p = 2 * q + 1 - i - j
            Q[i, j] = h**p / (p * factorial(q - i) * factorial(q - j))
    prior = IWP(q=q, d=1)
    assert np.allclose(_iwp_scalar_A(q, h), A)
    assert np.allclose(prior.A(h), A)
    assert np.allclose(prior.Q(h), Q, rtol=1e-10)


def test_kronecker_structure_with_xi():
    xi = np.array([[2.0, 0.3], [0.3, 1.0]])
    prior, scalar = IWP(q=2, d=2, Xi=xi), IWP(q=2, d=1)
    h = 0.4
    assert np.allclose(prior.A(h), np.kron(scalar.A(h), np.eye(2)))
    assert np.allclose(prior.Q(h), np.kron(scalar.Q(h), xi), rtol=1e-10)
    assert np.allclose(np.tril(prior.Q_sqr(h), -1), 0.0)


@pytest.mark.parametrize("q", [1, 5, 13, 20])
def test_closed_form_noise_factor_reconstructs_the_hilbert_matrix(q):
    _, Q_bar_sqr, _ = _iwp_constants(q)
    i, j = onp.indices((q + 1, q + 1))
    hilbert = 1.0 / (2 * q + 1 - i - j)
    rel = onp.linalg.norm(Q_bar_sqr.T @ Q_bar_sqr - hilbert) / onp.linalg.norm(hilbert)
    assert rel < 1e-12


@pytest.mark.parametrize("prior_cls", [IWP, PrecondIWP])
@pytest.mark.parametrize("q", [10, 15, 20])
def test_high_order_noise_is_finite_and_triangular(prior_cls, q):
    Q_sqr = prior_cls(q=q, d=2).transition(0.01).Q_sqr
    assert bool(np.all(np.isfinite(Q_sqr)))
    assert np.allclose(np.tril(Q_sqr, -1), 0.0)


@pytest.mark.parametrize("prior_cls", [IWP, PrecondIWP])
@pytest.mark.parametrize(
    ("q", "d", "error"),
    [
        (1.5, 1, TypeError),
        (-1, 1, ValueError),
        (2, 0, ValueError),
        (2, 1.0, TypeError),
        (True, 1, TypeError),
    ],
)
def test_invalid_order_or_dimension_raises(prior_cls, q, d, error):
    with pytest.raises(error):
        prior_cls(q=q, d=d)


def test_xi_of_the_wrong_shape_raises():
    with pytest.raises(ValueError, match="Xi must have shape"):
        IWP(q=1, d=2, Xi=np.eye(3))


def test_selectors_and_dimensions():
    prior = IWP(q=2, d=2)
    state = np.arange(6.0)
    assert prior.state_dim == 6
    assert np.allclose(prior.E0 @ state, state[:2])
    assert np.allclose(prior.E1 @ state, state[2:4])
    assert np.allclose(prior.E2 @ state, state[4:])
    assert IWP(q=1, d=1).E2 is None
    assert np.allclose(prior.b(0.3), 0.0)
