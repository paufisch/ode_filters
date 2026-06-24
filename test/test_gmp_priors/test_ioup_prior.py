"""Tests for IOUPPrior (integrated Ornstein-Uhlenbeck / probabilistic exp. integrator)."""

import jax.numpy as np
import numpy as onp
import pytest
from jax.scipy.linalg import expm

from ode_filters import IOUPPrior, ODEInformation, gaussian_filter
from ode_filters.priors.gmp_priors import IWP, taylor_mode_initialization


def _is_upper_triangular(M, atol=1e-9):
    return bool(np.allclose(np.tril(M, k=-1), 0.0, atol=atol))


def _vanloan_Q(F, G, h):
    """Independent dense van Loan MFD reference for Q(h) of the LTI (F, G)."""
    n = F.shape[0]
    S = onp.asarray(G) @ onp.asarray(G).T
    H = onp.block([[onp.asarray(F), S], [onp.zeros((n, n)), -onp.asarray(F).T]])
    E = onp.asarray(expm(np.asarray(H) * h))
    A = E[:n, :n]
    Q = E[:n, n:] @ A.T
    return 0.5 * (Q + Q.T)


class TestIOUPConstruction:
    def test_basic(self):
        prior = IOUPPrior(q=2, d=1, rate=-3.0)
        assert prior.q == 2
        assert prior._dim == 1

    def test_with_xi(self):
        xi = np.array([[2.0, 0.5], [0.5, 1.0]])
        prior = IOUPPrior(q=1, d=2, rate=-1.0, Xi=xi)
        assert np.allclose(prior.xi, xi)

    def test_rate_normalization_scalar_vector_matrix_consistent(self):
        """Scalar r, vector [r,r], and matrix r*I must yield identical A(h)/Q(h)."""
        q, d, h = 2, 2, 0.2
        r = -1.5
        a = IOUPPrior(q, d, rate=r)
        b = IOUPPrior(q, d, rate=np.array([r, r]))
        c = IOUPPrior(q, d, rate=r * np.eye(d))
        assert np.allclose(a.A(h), b.A(h)) and np.allclose(a.A(h), c.A(h))
        assert np.allclose(a.Q(h), b.Q(h)) and np.allclose(a.Q(h), c.Q(h))

    def test_rejects_wrong_vector_rate(self):
        with pytest.raises(ValueError, match=r"vector rate must have shape"):
            IOUPPrior(q=2, d=2, rate=np.zeros(3))

    def test_rejects_wrong_matrix_rate(self):
        with pytest.raises(ValueError, match=r"matrix rate must have shape"):
            IOUPPrior(q=2, d=2, rate=np.zeros((2, 3)))

    def test_rejects_high_rank_rate(self):
        with pytest.raises(ValueError, match=r"scalar, length-d vector, or"):
            IOUPPrior(q=2, d=2, rate=np.zeros((2, 2, 2)))


class TestIOUPShapes:
    @pytest.mark.parametrize(("q", "d"), [(2, 1), (2, 2), (3, 2)])
    def test_A_Q_b_shapes(self, q, d):
        prior = IOUPPrior(q, d, rate=-1.0)
        dim = (q + 1) * d
        assert prior.A(0.1).shape == (dim, dim)
        assert prior.Q(0.1).shape == (dim, dim)
        assert prior.b(0.1).shape == (dim,)

    def test_E0_E1(self):
        prior = IOUPPrior(q=2, d=1, rate=-1.0)
        assert prior.E0.shape == (1, 3)
        assert prior.E1.shape == (1, 3)


class TestIOUPProperties:
    def test_A_zero_is_identity(self):
        prior = IOUPPrior(q=2, d=2, rate=-2.0)
        assert np.allclose(prior.A(0.0), np.eye((prior.q + 1) * prior._dim))

    def test_Q_symmetric_psd(self):
        prior = IOUPPrior(q=2, d=1, rate=-2.0)
        Q = prior.Q(0.3)
        assert np.allclose(Q, Q.T)
        assert np.all(np.linalg.eigvalsh(Q) >= -1e-10)

    def test_b_is_zero(self):
        prior = IOUPPrior(q=2, d=1, rate=-1.0)
        assert np.allclose(prior.b(0.5), np.zeros(3))

    @pytest.mark.parametrize(
        "rate",
        [-3.0, onp.array([-1.0, -5.0]), onp.array([[-2.0, 1.0], [0.5, -3.0]])],
    )
    def test_Q_sqr_reconstructs_vanloan(self, rate):
        """Q_sqr is upper-triangular and reconstructs the independent dense MFD Q."""
        d = 1 if onp.ndim(rate) == 0 else 2
        prior = IOUPPrior(q=2, d=d, rate=rate)
        h = 0.15
        Q_sqr = prior.Q_sqr(h)
        assert _is_upper_triangular(Q_sqr)
        Q_ref = _vanloan_Q(prior._F, prior._G, h)
        rel = onp.linalg.norm(onp.asarray(Q_sqr.T @ Q_sqr) - Q_ref) / onp.linalg.norm(
            Q_ref
        )
        assert rel < 1e-9


class TestIOUPEqualsIWP:
    """IOUP with rate 0 is exactly the integrated Wiener process."""

    @pytest.mark.parametrize(("q", "d"), [(2, 1), (3, 2)])
    def test_zero_rate_matches_iwp(self, q, d):
        h = 0.2
        ioup = IOUPPrior(q, d, rate=0.0)
        iwp = IWP(q, d)
        assert np.allclose(ioup.A(h), iwp.A(h), atol=1e-12)
        assert np.allclose(ioup.Q(h), iwp.Q(h), atol=1e-12)


class TestIOUPSolve:
    """IOUP is drop-in: the unmodified gaussian_filter solves with the standard
    ODE-information residual on the full vector field."""

    def test_dropin_solve_linear_decay(self):
        def vf(x, *, t):
            return -x  # linear part L = -1; IOUP(rate=-1) integrates it exactly

        q = 2
        prior = IOUPPrior(q, d=1, rate=-1.0)
        x0 = np.array([1.0])
        mu0, P0 = taylor_mode_initialization(vf, x0, q)
        P0_sqr = np.linalg.cholesky(P0 + 1e-12 * np.eye(P0.shape[0])).T
        measure = ODEInformation(vf, prior.E0, prior.E1)
        out = gaussian_filter(mu0, P0_sqr, prior, measure, (0.0, 2.0), 50)
        ts = np.linspace(0.0, 2.0, 51)
        err = float(np.max(np.abs(out.m[:, 0] - np.exp(-ts))))
        assert np.all(np.isfinite(out.m))
        assert np.isfinite(out.log_likelihood)
        assert err < 1e-5
