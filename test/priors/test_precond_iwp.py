"""Tests for PrecondIWP: the IWP, computed in preconditioned coordinates."""

import jax.numpy as np
import pytest

from ode_filters.priors import IWP, PrecondIWP
from ode_filters.priors.iwp import _iwp_constants

H = 0.3


@pytest.mark.parametrize("q,d", [(1, 1), (2, 2), (4, 1)])
def test_process_is_the_iwp(q, d):
    """A(h), Q(h) and Q_sqr(h) are the IWP's: preconditioning is internal."""
    xi = np.diag(np.arange(1.0, d + 1.0))
    precond, iwp = PrecondIWP(q=q, d=d, Xi=xi), IWP(q=q, d=d, Xi=xi)
    assert np.allclose(precond.A(H), iwp.A(H))
    assert np.allclose(precond.Q(H), iwp.Q(H))
    assert np.allclose(precond.Q_sqr(H), iwp.Q_sqr(H))


def test_transition_is_step_size_independent_up_to_T():
    precond = PrecondIWP(q=3, d=2)
    small, large = precond.transition(0.01), precond.transition(0.7)
    assert np.allclose(small.A, large.A)
    assert np.allclose(small.Q_sqr, large.Q_sqr)
    assert not np.allclose(small.T_diag, large.T_diag)


@pytest.mark.parametrize("q,d", [(1, 1), (2, 2), (3, 3)])
def test_transition_reconstructs_the_original_coordinates(q, d):
    precond = PrecondIWP(q=q, d=d, Xi=0.5 * np.eye(d))
    tr = precond.transition(H)
    t = tr.T_diag
    assert t is not None
    assert np.allclose(np.diag(precond.T(H)), t)
    assert np.allclose(t[:, None] * tr.A / t[None, :], precond.A(H))
    Q_sqr = tr.Q_sqr * t[None, :]
    assert np.allclose(Q_sqr.T @ Q_sqr, precond.Q(H))


def test_transition_matrices_are_the_closed_form_constants():
    A_bar, Q_bar_sqr, _ = _iwp_constants(2)
    Q_bar = Q_bar_sqr.T @ Q_bar_sqr
    xi = np.array([[2.0, 0.0], [0.0, 3.0]])
    tr = PrecondIWP(q=2, d=2, Xi=xi).transition(H)
    assert np.allclose(tr.A, np.kron(A_bar, np.eye(2)))
    assert np.allclose(tr.Q_sqr.T @ tr.Q_sqr, np.kron(Q_bar, xi))
    assert np.allclose(tr.b, 0.0)


def test_T_is_diagonal_and_vanishes_at_zero():
    precond = PrecondIWP(q=2, d=2)
    T = precond.T(0.5)
    assert T.shape == (6, 6)
    assert np.allclose(T - np.diag(np.diag(T)), 0.0)
    assert np.allclose(precond.T(0.0), 0.0)


def test_high_order_stays_finite():
    tr = PrecondIWP(q=20, d=1).transition(1e-3)
    assert bool(np.all(np.isfinite(tr.Q_sqr)))
    assert bool(np.all(np.isfinite(tr.T_diag)))
