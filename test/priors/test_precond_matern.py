"""Tests for PrecondMaternPrior: the Matern prior, computed in preconditioned coordinates."""

import jax.numpy as np
import pytest

from ode_filters.priors import MaternPrior, PrecondIWP, PrecondMaternPrior


@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("h", [0.01, 0.1, 0.5])
def test_process_is_the_matern_prior(q, h):
    raw = MaternPrior(q=q, d=1, length_scale=1.5)
    precond = PrecondMaternPrior(q=q, d=1, length_scale=1.5)
    assert np.allclose(precond.A(h), raw.A(h))
    assert np.allclose(precond.Q(h), raw.Q(h))


@pytest.mark.parametrize("q", [1, 2, 3])
@pytest.mark.parametrize("h", [0.01, 0.1, 0.5])
def test_transition_reconstructs_the_original_coordinates(q, h):
    precond = PrecondMaternPrior(
        q=q, d=2, length_scale=1.5, Xi=np.diag(np.array([2.0, 3.0]))
    )
    tr = precond.transition(h)
    t = tr.T_diag
    assert t is not None
    assert np.allclose(t[:, None] * tr.A / t[None, :], precond.A(h), atol=1e-10)
    Q_sqr = tr.Q_sqr * t[None, :]
    assert np.allclose(Q_sqr.T @ Q_sqr, precond.Q(h), atol=1e-10)
    assert np.allclose(np.tril(tr.Q_sqr, -1), 0.0)


def test_T_matches_precond_iwp():
    matern = PrecondMaternPrior(q=2, d=2, length_scale=1.0)
    assert np.allclose(matern.T(0.1), PrecondIWP(q=2, d=2).T(0.1))


@pytest.mark.parametrize("q", [1, 2, 3])
def test_transition_converges_to_the_iwp_constants(q):
    """As the length scale grows, the preconditioned transition tends to the IWP's."""
    h = 0.1
    iwp = PrecondIWP(q=q, d=1).transition(h)
    matern = PrecondMaternPrior(q=q, d=1, length_scale=1e8).transition(h)
    assert np.allclose(matern.A, iwp.A, atol=1e-4)
    # The noise converges up to the Matern's (non-vanishing) diffusion constant.
    ratio = (matern.Q_sqr.T @ matern.Q_sqr) / (iwp.Q_sqr.T @ iwp.Q_sqr)
    assert np.allclose(ratio, ratio[0, 0], atol=1e-3)


def test_short_length_scale_stays_finite():
    tr = PrecondMaternPrior(q=2, d=1, length_scale=0.051).transition(1e-2)
    assert bool(np.all(np.isfinite(tr.A))) and bool(np.all(np.isfinite(tr.Q_sqr)))
