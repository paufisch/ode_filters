"""Tests for the square-root process noise ``prior.Q_sqr(h)``.

``Q_sqr`` is the quantity the square-root filter consumes, and ``Q(h)`` is
derived from it (``Q_sqr.T @ Q_sqr``). Both are checked against Van Loan's
block exponential of each prior's SDE, built here from the prior's definition
(for the Matern prior: from the textbook kernel), never from the prior itself.
For IWP / PrecondIWP the factor is a closed form that avoids factorizing the
dense, ill-conditioned Hilbert-like ``Q(h)`` at runtime (the motivation for
exposing it).
"""

import jax.numpy as np
import numpy as onp
import pytest
from scipy.linalg import block_diag

from ode_filters.priors import (
    IWP,
    IOUPPrior,
    JointPrior,
    MaternPrior,
    PrecondIWP,
    PrecondMaternPrior,
)
from test.helpers import assert_close, matern_sde, van_loan


def _is_upper_triangular(M, atol=1e-10):
    return np.allclose(np.tril(M, k=-1), 0.0, atol=atol)


def _integrated_sde(q, d, *, rate=None, Xi=None):
    """``(F, S)`` of a ``q``-times integrated process on ``[Y, Y', ..., Y^(q)]``.

    ``dY^(i) = Y^(i+1) dt`` for ``i < q`` and ``dY^(q) = R Y^(q) dt + dW`` with
    ``Cov(dW) = Xi dt``: the IWP for ``R = 0``, the IOUP otherwise.
    """
    F = onp.kron(onp.eye(q + 1, k=1), onp.eye(d))
    if rate is not None:
        F[q * d :, q * d :] = rate
    S = onp.zeros(((q + 1) * d, (q + 1) * d))
    S[q * d :, q * d :] = onp.eye(d) if Xi is None else Xi
    return F, S


def _matern_sde(q, d, length_scale, Xi=None):
    """``(F, S)`` of ``d`` Matern components, from the scalar kernel SDE."""
    F, S = (onp.asarray(M) for M in matern_sde(q, length_scale))
    return onp.kron(F, onp.eye(d)), onp.kron(S, onp.eye(d) if Xi is None else Xi)


def _joint_sde(sde_x, sde_u):
    return block_diag(sde_x[0], sde_u[0]), block_diag(sde_x[1], sde_u[1])


def _priors():
    """A representative spread of priors, the h to test them at, and their SDE."""
    xi_full = np.array([[2.0, 0.3], [0.3, 1.5]])
    rate_a = np.array([[-2.0, 1.0], [0.0, -3.0]])
    rate_b = np.array([[-2.0, 0.5], [0.0, -3.0]])
    return [
        ("IWP q=2 d=1", IWP(q=2, d=1), 0.3, _integrated_sde(2, 1)),
        ("IWP q=3 d=2", IWP(q=3, d=2), 0.2, _integrated_sde(3, 2)),
        (
            "IWP q=2 d=2 Xi",
            IWP(q=2, d=2, Xi=xi_full),
            0.25,
            _integrated_sde(2, 2, Xi=xi_full),
        ),
        ("PrecondIWP q=3 d=2", PrecondIWP(q=3, d=2), 0.2, _integrated_sde(3, 2)),
        (
            "Matern q=2 d=1",
            MaternPrior(q=2, d=1, length_scale=1.0),
            0.3,
            _matern_sde(2, 1, 1.0),
        ),
        (
            "Matern q=2 d=2",
            MaternPrior(q=2, d=2, length_scale=1.5),
            0.25,
            _matern_sde(2, 2, 1.5),
        ),
        (
            "PrecondMatern q=2 d=2 Xi",
            PrecondMaternPrior(q=2, d=2, length_scale=1.5, Xi=xi_full),
            0.2,
            _matern_sde(2, 2, 1.5, Xi=xi_full),
        ),
        (
            "IOUP q=2 d=1 scalar",
            IOUPPrior(q=2, d=1, rate=-2.0),
            0.25,
            _integrated_sde(2, 1, rate=-2.0),
        ),
        (
            "IOUP q=2 d=2 matrix Xi",
            IOUPPrior(q=2, d=2, rate=rate_a, Xi=xi_full),
            0.2,
            _integrated_sde(2, 2, rate=rate_a, Xi=xi_full),
        ),
        (
            "JointPrior IWP+IWP",
            JointPrior(IWP(q=2, d=2), IWP(q=1, d=1)),
            0.25,
            _joint_sde(_integrated_sde(2, 2), _integrated_sde(1, 1)),
        ),
        (
            "JointPrior IOUP+IWP",
            JointPrior(
                IOUPPrior(q=2, d=2, rate=rate_b),
                IWP(q=0, d=1, Xi=1e-3 * np.eye(1)),
            ),
            0.2,
            _joint_sde(
                _integrated_sde(2, 2, rate=rate_b),
                _integrated_sde(0, 1, Xi=1e-3 * onp.eye(1)),
            ),
        ),
        (
            "JointPrior PrecondIWP+PrecondIWP",
            JointPrior(PrecondIWP(q=2, d=2), PrecondIWP(q=1, d=1)),
            0.2,
            _joint_sde(_integrated_sde(2, 2), _integrated_sde(1, 1)),
        ),
    ]


@pytest.mark.parametrize("name,prior,h,sde", _priors(), ids=[p[0] for p in _priors()])
def test_transition_and_noise_match_van_loan(name, prior, h, sde):
    """``A(h)`` and ``Q(h) = Q_sqr.T @ Q_sqr`` are the SDE's exact discretization."""
    A_ref, Q_ref = van_loan(*(np.asarray(M) for M in sde), h)
    assert prior.Q_sqr(h).shape == Q_ref.shape, name
    assert_close(prior.A(h), A_ref, rtol=1e-10)
    assert_close(prior.Q(h), Q_ref, rtol=1e-9)


@pytest.mark.parametrize("name,prior,h,sde", _priors(), ids=[p[0] for p in _priors()])
def test_q_sqr_is_upper_triangular(name, prior, h, sde):
    """The square-root factor is upper-triangular (the library convention)."""
    assert _is_upper_triangular(prior.Q_sqr(h)), name


def test_joint_q_sqr_is_block_diagonal():
    """JointPrior.Q_sqr keeps the state/input blocks decoupled (off-diagonal 0)."""
    prior = JointPrior(IWP(q=2, d=2), IWP(q=1, d=1))
    D_x = prior.prior_x.state_dim
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


def test_precond_iwp_transition_noise_is_constant():
    """PrecondIWP's preconditioned noise factor is step-size independent."""
    prior = PrecondIWP(q=3, d=2)
    assert np.allclose(prior.transition(0.1).Q_sqr, prior.transition(2.0).Q_sqr)


@pytest.mark.parametrize("prior_cls", [MaternPrior, PrecondMaternPrior])
@pytest.mark.parametrize("q", [6, 8, 10, 12])
@pytest.mark.parametrize("d", [1, 2])
def test_matern_q_sqr_stable_at_high_order(prior_cls, q, d):
    """Matern Q_sqr stays finite + upper-triangular well past q = 6.

    The dense process-noise Q(h) loses positive-definiteness in float64 around
    q >= 6, so the old dense-Cholesky Q_sqr returned NaN there. The square-root
    matrix-fraction decomposition builds the factor directly (never factorizing
    the dense Q) and stays finite up to the float64 order ceiling (~q = 18).
    Regression for that NaN cliff.
    """
    prior = prior_cls(q=q, d=d, length_scale=1.0)
    Q_sqr = prior.Q_sqr(0.1)
    assert Q_sqr.shape == ((q + 1) * d, (q + 1) * d)
    assert np.all(np.isfinite(Q_sqr)), f"{prior_cls.__name__} q={q} d={d}"
    assert _is_upper_triangular(Q_sqr), f"{prior_cls.__name__} q={q} d={d}"


@pytest.mark.parametrize("q", [8, 12])
def test_matern_q_sqr_quadrature_converged(q):
    """The square-root MFD is converged in the quadrature node count ``n_quad``.

    At high order the dense Q(h) is NaN, so it cannot serve as a reference; we
    check self-consistency instead -- the reconstructed Q = Q_sqr.T @ Q_sqr must be
    insensitive to ``n_quad`` (the default vs a finer rule agree to ~1e-12),
    evidence that the quadrature has converged to the true integral. Also
    exercises the ``n_quad`` constructor argument.
    """
    h = 0.1
    coarse = MaternPrior(q=q, d=1, length_scale=1.0, n_quad=64).Q_sqr(h)
    fine = MaternPrior(q=q, d=1, length_scale=1.0, n_quad=100).Q_sqr(h)
    rel = np.linalg.norm(coarse.T @ coarse - fine.T @ fine) / np.linalg.norm(
        fine.T @ fine
    )
    assert rel < 1e-9


@pytest.mark.parametrize("prior_cls", [MaternPrior, PrecondMaternPrior])
@pytest.mark.parametrize("n_quad", [1, 5, 6])
def test_matern_rejects_too_few_quadrature_nodes(prior_cls, n_quad):
    """``n_quad <= q`` is rejected at construction, not deep inside a solve.

    The Gauss-Legendre rule contributes one row per node to the QR, so fewer than
    ``q + 1`` nodes silently produces a ``[n_quad, q + 1]`` factor instead of a
    square one. That is a wrong *shape*, not a coarser approximation, and it used
    to surface only later as an opaque "Q is expected to be of square shape"
    error from inside the filter.
    """
    with pytest.raises(ValueError, match="n_quad must exceed q"):
        prior_cls(q=6, d=1, length_scale=1.0, n_quad=n_quad)


@pytest.mark.parametrize("prior_cls", [MaternPrior, PrecondMaternPrior])
def test_matern_accepts_the_minimal_quadrature_node_count(prior_cls):
    """``n_quad == q + 1`` is the boundary case and must be allowed."""
    q = 6
    Q_sqr = prior_cls(q=q, d=1, length_scale=1.0, n_quad=q + 1).Q_sqr(0.1)
    assert Q_sqr.shape == (q + 1, q + 1)
    assert _is_upper_triangular(Q_sqr)
