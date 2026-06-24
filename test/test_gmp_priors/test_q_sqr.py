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
    BasePrior,
    IOUPPrior,
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
        ("IOUP q=2 d=1 scalar", IOUPPrior(q=2, d=1, rate=-2.0), 0.25),
        (
            "IOUP q=2 d=2 matrix Xi",
            IOUPPrior(q=2, d=2, rate=np.array([[-2.0, 1.0], [0.0, -3.0]]), Xi=xi_full),
            0.2,
        ),
        (
            "JointPrior IWP+IWP",
            JointPrior(IWP(q=2, d=2), IWP(q=1, d=1)),
            0.25,
        ),
        (
            "JointPrior IOUP+IWP",
            JointPrior(
                IOUPPrior(q=2, d=2, rate=np.array([[-2.0, 0.5], [0.0, -3.0]])),
                IWP(q=0, d=1, Xi=1e-3 * np.eye(1)),
            ),
            0.2,
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


class _DenseDefaultPrior(BasePrior):
    """Minimal prior that does NOT override ``Q_sqr``.

    Exercises the ``BasePrior.Q_sqr`` dense-Cholesky default, the documented
    fallback for custom priors that lack a closed-form / structured square root
    (every shipped prior overrides it).
    """

    def A(self, h):  # pragma: no cover - not exercised by the Q_sqr test
        n = (self.q + 1) * self._dim
        return np.eye(n)

    def b(self, h):  # pragma: no cover - not exercised by the Q_sqr test
        return self._b

    def Q(self, h):
        n = (self.q + 1) * self._dim
        return h * np.eye(n)


def test_base_prior_q_sqr_default_is_dense_cholesky():
    """BasePrior.Q_sqr factorizes the dense Q (fallback for custom priors)."""
    prior = _DenseDefaultPrior(q=2, d=1)
    h = 0.3
    Q_sqr = prior.Q_sqr(h)
    assert _is_upper_triangular(Q_sqr)
    assert np.allclose(Q_sqr.T @ Q_sqr, prior.Q(h))


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
