"""Tests for PrecondMaternPrior class."""

import jax.numpy as np
import pytest

from ode_filters.priors.gmp_priors import (
    MaternPrior,
    PrecondIWP,
    PrecondMaternPrior,
)


class TestPrecondMaternPriorConstruction:
    """Tests for PrecondMaternPrior construction."""

    def test_basic_construction(self):
        prior = PrecondMaternPrior(q=2, d=1, length_scale=1.0)
        assert prior.q == 2
        assert prior._dim == 1
        assert prior.n == 3

    def test_construction_with_xi(self):
        xi = np.array([[2.0, 0.5], [0.5, 1.0]])
        prior = PrecondMaternPrior(q=1, d=2, length_scale=1.0, Xi=xi)
        assert np.allclose(prior.xi, xi)

    def test_rejects_negative_q(self):
        with pytest.raises(ValueError):
            PrecondMaternPrior(q=-1, d=1, length_scale=1.0)

    def test_rejects_zero_d(self):
        with pytest.raises(ValueError):
            PrecondMaternPrior(q=2, d=0, length_scale=1.0)


class TestPrecondMaternPriorShapes:
    """Tests for output shapes of PrecondMaternPrior methods."""

    @pytest.fixture
    def prior(self):
        return PrecondMaternPrior(q=2, d=2, length_scale=1.5)

    def test_A_shape(self, prior):
        A = prior.A(0.1)
        assert A.shape == (6, 6)

    def test_Q_shape(self, prior):
        Q = prior.Q(0.1)
        assert Q.shape == (6, 6)

    def test_b_shape(self, prior):
        b = prior.b()
        assert b.shape == (6,)

    def test_T_shape(self, prior):
        T = prior.T(0.1)
        assert T.shape == (6, 6)

    def test_E0_shape(self, prior):
        assert prior.E0.shape == (2, 6)

    def test_E1_shape(self, prior):
        assert prior.E1.shape == (2, 6)


class TestPrecondMaternPriorProperties:
    """Tests for mathematical properties of PrecondMaternPrior."""

    def test_Q_is_symmetric(self):
        prior = PrecondMaternPrior(q=2, d=1, length_scale=1.0)
        Q = prior.Q(0.1)
        assert np.allclose(Q, Q.T)

    def test_Q_is_positive_semidefinite(self):
        prior = PrecondMaternPrior(q=2, d=1, length_scale=1.0)
        Q = prior.Q(0.1)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-10)

    def test_b_is_zero(self):
        prior = PrecondMaternPrior(q=2, d=1, length_scale=1.0)
        b = prior.b()
        assert np.allclose(b, np.zeros(3))

    def test_T_is_diagonal(self):
        prior = PrecondMaternPrior(q=2, d=1, length_scale=1.0)
        T = prior.T(0.5)
        off_diag = T - np.diag(np.diag(T))
        assert np.allclose(off_diag, np.zeros_like(off_diag))

    def test_T_matches_precond_iwp(self):
        """T(h) should be identical to PrecondIWP's T(h)."""
        q, d, h = 2, 2, 0.1
        matern = PrecondMaternPrior(q=q, d=d, length_scale=1.0)
        iwp = PrecondIWP(q=q, d=d)
        assert np.allclose(matern.T(h), iwp.T(h))


class TestPrecondMaternConsistency:
    """Tests for consistency between preconditioned and raw Matern priors."""

    @pytest.mark.parametrize("q", [1, 2, 3])
    @pytest.mark.parametrize("h", [0.01, 0.1, 0.5])
    def test_preconditioned_matches_raw(self, q, h):
        """T(h) @ A_bar(h) @ T(h)^{-1} should equal the raw A(h)."""
        length_scale = 1.5
        raw = MaternPrior(q=q, d=1, length_scale=length_scale)
        precond = PrecondMaternPrior(q=q, d=1, length_scale=length_scale)

        T_h = precond.T(h)
        A_bar = precond.A(h)
        A_raw = raw.A(h)

        A_reconstructed = T_h @ A_bar @ np.linalg.inv(T_h)
        assert np.allclose(A_reconstructed, A_raw, atol=1e-10)

    @pytest.mark.parametrize("q", [1, 2, 3])
    @pytest.mark.parametrize("h", [0.01, 0.1, 0.5])
    def test_preconditioned_Q_matches_raw(self, q, h):
        """T(h) @ Q_bar(h) @ T(h)^T should equal the raw Q(h)."""
        length_scale = 1.5
        raw = MaternPrior(q=q, d=1, length_scale=length_scale)
        precond = PrecondMaternPrior(q=q, d=1, length_scale=length_scale)

        T_h = precond.T(h)
        Q_bar = precond.Q(h)
        Q_raw = raw.Q(h)

        Q_reconstructed = T_h @ Q_bar @ T_h.T
        assert np.allclose(Q_reconstructed, Q_raw, atol=1e-10)


class TestPrecondMaternConvergesToIWP:
    """Tests that Matern converges to IWP as length_scale -> inf."""

    @pytest.mark.parametrize("q", [1, 2, 3])
    def test_A_bar_converges_to_iwp(self, q):
        """As length_scale -> inf, A_bar(h) -> A_bar_iwp."""
        h = 0.1
        iwp = PrecondIWP(q=q, d=1)
        matern = PrecondMaternPrior(q=q, d=1, length_scale=1e8)

        A_iwp = iwp.A()
        A_matern = matern.A(h)
        assert np.allclose(A_iwp, A_matern, atol=1e-4)

    @pytest.mark.parametrize("q", [1, 2, 3])
    def test_Q_bar_converges_to_iwp(self, q):
        """As length_scale -> inf, Q_bar(h) converges to a scalar multiple of Q_bar_iwp."""
        h = 0.1
        iwp = PrecondIWP(q=q, d=1)
        matern = PrecondMaternPrior(q=q, d=1, length_scale=1e8)

        Q_iwp = iwp.Q()
        Q_matern = matern.Q(h)

        # The shapes of Q_bar should match and the ratio should be approximately constant
        # (scaled by the Matern's q_coeff which doesn't vanish)
        ratio = Q_matern / Q_iwp
        # All entries should have the same ratio (up to numerical noise)
        assert np.allclose(ratio, ratio[0, 0], atol=1e-3)


class TestPrecondMaternKronecker:
    """Tests for correct Kronecker structure with d > 1."""

    def test_A_kronecker_structure(self):
        """A(h) should be kron(A_scalar, I_d)."""
        q, d, h = 2, 3, 0.1
        prior = PrecondMaternPrior(q=q, d=d, length_scale=1.0)
        prior_1d = PrecondMaternPrior(q=q, d=1, length_scale=1.0)

        A_full = prior.A(h)
        A_1d = prior_1d.A(h)
        expected = np.kron(A_1d, np.eye(d))
        assert np.allclose(A_full, expected)

    def test_Q_kronecker_with_xi(self):
        """Q(h) should be kron(Q_scalar, Xi)."""
        q, d, h = 1, 2, 0.1
        xi = np.array([[2.0, 0.0], [0.0, 3.0]])
        prior = PrecondMaternPrior(q=q, d=d, length_scale=1.0, Xi=xi)
        prior_1d = PrecondMaternPrior(q=q, d=1, length_scale=1.0)

        Q_full = prior.Q(h)
        Q_1d = prior_1d.Q(h)
        expected = np.kron(Q_1d, xi)
        assert np.allclose(Q_full, expected)
