"""Tests for PrecondIWP class."""

import jax.numpy as np
import pytest

from ode_filters.priors.gmp_priors import PrecondIWP, _make_iwp_precond_state_matrices


class TestPrecondIWPMethods:
    """Tests for PrecondIWP matrix methods."""

    @pytest.fixture
    def precond_iwp(self):
        """Create a standard PrecondIWP for testing."""
        return PrecondIWP(q=2, d=2)

    def test_A_returns_correct_shape(self, precond_iwp):
        """Test that A() returns correct shape."""
        A = precond_iwp.A()
        expected_shape = (6, 6)  # (q+1)*d = 3*2 = 6
        assert A.shape == expected_shape

    def test_b_returns_correct_shape(self, precond_iwp):
        """Test that b() returns correct shape."""
        b = precond_iwp.b()
        expected_shape = (6,)  # (q+1)*d = 3*2 = 6
        assert b.shape == expected_shape

    def test_b_returns_zeros(self, precond_iwp):
        """Test that b() returns zero vector."""
        b = precond_iwp.b()
        assert np.allclose(b, np.zeros(6))

    def test_Q_returns_correct_shape(self, precond_iwp):
        """Test that Q() returns correct shape."""
        Q = precond_iwp.Q()
        expected_shape = (6, 6)
        assert Q.shape == expected_shape

    def test_T_returns_correct_shape(self, precond_iwp):
        """Test that T(h) returns correct shape."""
        T = precond_iwp.T(0.5)
        expected_shape = (6, 6)
        assert T.shape == expected_shape

    def test_T_is_diagonal(self, precond_iwp):
        """Test that T(h) is a diagonal matrix."""
        T = precond_iwp.T(0.5)
        # Check off-diagonal elements are zero
        off_diag = T - np.diag(np.diag(T))
        assert np.allclose(off_diag, np.zeros_like(off_diag))

    def test_T_at_h_zero_is_zero(self):
        """Test that T(0) is zero matrix (since sqrt(0) = 0)."""
        precond = PrecondIWP(q=2, d=1)
        T = precond.T(0.0)
        assert np.allclose(T, np.zeros_like(T))

    def test_T_rejects_negative_h(self, precond_iwp):
        """Test that T(h) rejects negative step size."""
        with pytest.raises(ValueError, match="h must be non-negative"):
            precond_iwp.T(-0.5)


class TestPrecondIWPProperties:
    """Tests for PrecondIWP mathematical properties."""

    def test_A_is_constant(self):
        """Test that A() is independent of step size."""
        precond = PrecondIWP(q=2, d=1)
        A1 = precond.A()
        A2 = precond.A()
        assert np.allclose(A1, A2)

    def test_Q_is_constant(self):
        """Test that Q() is independent of step size."""
        precond = PrecondIWP(q=2, d=1)
        Q1 = precond.Q()
        Q2 = precond.Q()
        assert np.allclose(Q1, Q2)

    def test_Q_is_symmetric(self):
        """Test that Q() is symmetric."""
        precond = PrecondIWP(q=2, d=1)
        Q = precond.Q()
        assert np.allclose(Q, Q.T)

    def test_Q_is_positive_definite(self):
        """Test that Q() is positive definite."""
        precond = PrecondIWP(q=2, d=1)
        Q = precond.Q()
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues > 0), "Q should be positive definite"

    def test_E0_E1_properties(self):
        """Test that E0 and E1 are accessible."""
        precond = PrecondIWP(q=2, d=2)
        assert precond.E0.shape == (2, 6)
        assert precond.E1.shape == (2, 6)


class TestPrecondIWPWithCustomXi:
    """Tests for PrecondIWP with custom Xi matrix."""

    def test_Q_incorporates_xi(self):
        """Test that Q() incorporates the Xi scaling matrix."""
        xi = np.array([[2.0, 0.0], [0.0, 3.0]])
        precond = PrecondIWP(q=1, d=2, Xi=xi)
        Q = precond.Q()

        # Q should be kron(Q_bar, xi)
        _, Q_bar, _ = _make_iwp_precond_state_matrices(1)
        expected_Q = np.kron(Q_bar, xi)

        assert np.allclose(Q, expected_Q)

    def test_A_uses_identity_regardless_of_xi(self):
        """Test that A() uses identity matrix regardless of Xi."""
        xi = np.array([[2.0, 0.0], [0.0, 3.0]])
        precond = PrecondIWP(q=1, d=2, Xi=xi)
        A = precond.A()

        # A should be kron(A_bar, I_d)
        A_bar, _, _ = _make_iwp_precond_state_matrices(1)
        expected_A = np.kron(A_bar, np.eye(2))

        assert np.allclose(A, expected_A)
