"""Tests for the square-root process noise ``prior.Q_sqr(h)``.

``Q_sqr`` is the quantity the square-root filter consumes; it must satisfy
``Q_sqr.T @ Q_sqr == Q(h)`` for every prior. For IWP / PrecondIWP it is a
closed form that avoids factorizing the dense, ill-conditioned Hilbert-like
``Q(h)`` at runtime (the motivation for exposing it).
"""

import jax.numpy as np
import pytest

from ode_filters.priors.gmp_priors import (
    IWP,
    JointPrior,
    MaternPrior,
    PrecondIWP,
    PrecondJointPrior,
    PrecondMaternPrior,
)


def _is_upper_triangular(M, atol=1e-10):
    return np.allclose(np.tril(M, k=-1), 0.0, atol=atol)


def _priors():
    """A representative spread of priors and the h at which to test them."""
    xi_full = np.array([[2.0, 0.3], [0.3, 1.5]])
    return [
        ("IWP q=2 d=1", IWP(q=2, d=1), 0.3),
        ("IWP q=3 d=2", IWP(q=3, d=2), 0.2),
        ("IWP q=2 d=2 Xi", IWP(q=2, d=2, Xi=xi_full), 0.25),
        ("PrecondIWP q=3 d=2", PrecondIWP(q=3, d=2), 0.2),
        ("Matern q=2 d=1", MaternPrior(q=2, d=1, length_scale=1.0), 0.3),
        ("Matern q=2 d=2", MaternPrior(q=2, d=2, length_scale=1.5), 0.25),
        ("PrecondMatern q=2 d=2", PrecondMaternPrior(q=2, d=2, length_scale=1.5), 0.2),
        (
            "JointPrior IWP+IWP",
            JointPrior(IWP(q=2, d=2), IWP(q=1, d=1)),
            0.25,
        ),
        (
            "PrecondJointPrior",
            PrecondJointPrior(PrecondIWP(q=2, d=2), PrecondIWP(q=1, d=1)),
            0.2,
        ),
    ]


@pytest.mark.parametrize("name,prior,h", _priors(), ids=[p[0] for p in _priors()])
def test_q_sqr_reconstructs_q(name, prior, h):
    """Q_sqr.T @ Q_sqr must reconstruct Q(h)."""
    Q = prior.Q(h)
    Q_sqr = prior.Q_sqr(h)
    assert Q_sqr.shape == Q.shape
    assert np.allclose(Q_sqr.T @ Q_sqr, Q, rtol=1e-9, atol=1e-12), name


@pytest.mark.parametrize("name,prior,h", _priors(), ids=[p[0] for p in _priors()])
def test_q_sqr_is_upper_triangular(name, prior, h):
    """The square-root factor is upper-triangular (the library convention)."""
    assert _is_upper_triangular(prior.Q_sqr(h)), name


def test_joint_q_sqr_is_block_diagonal():
    """JointPrior.Q_sqr keeps the state/input blocks decoupled (off-diagonal 0)."""
    prior = JointPrior(IWP(q=2, d=2), IWP(q=1, d=1))
    D_x = prior._D_x
    Q_sqr = prior.Q_sqr(0.3)
    # The off-diagonal blocks must be exactly zero.
    assert np.allclose(Q_sqr[:D_x, D_x:], 0.0)
    assert np.allclose(Q_sqr[D_x:, :D_x], 0.0)


def test_iwp_q_sqr_well_conditioned_vs_dense_cholesky():
    """The closed-form IWP factor stays accurate where the dense Q is ill-conditioned.

    For moderate q and small h the dense Hilbert-like Q(h) has a huge condition
    number, so factorizing it directly loses precision. The closed form factors
    the constant (h-independent) Q_bar instead, whose conditioning does not blow
    up with h -- so it reconstructs Q(h) to high accuracy even in that regime.
    """
    q, d, h = 5, 1, 0.05
    prior = IWP(q=q, d=d)
    Q = prior.Q(h)

    # We are genuinely in the ill-conditioned regime.
    cond_dense = np.linalg.cond(Q)
    assert cond_dense > 1e8, (
        f"expected ill-conditioned dense Q, got cond={cond_dense:g}"
    )

    # Closed-form factor reconstructs Q to high relative accuracy regardless.
    Q_sqr = prior.Q_sqr(h)
    rel_err = np.linalg.norm(Q_sqr.T @ Q_sqr - Q) / np.linalg.norm(Q)
    assert rel_err < 1e-10, f"closed-form reconstruction rel_err={rel_err:g}"


def test_precond_iwp_q_sqr_is_constant():
    """PrecondIWP.Q_sqr is stepsize-independent (matches its constant Q)."""
    prior = PrecondIWP(q=3, d=2)
    assert np.allclose(prior.Q_sqr(0.1), prior.Q_sqr(2.0))
    assert np.allclose(prior.Q_sqr().T @ prior.Q_sqr(), prior.Q())
