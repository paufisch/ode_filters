"""Tests for MaternPrior and _matern_companion_form."""

import jax.numpy as np
import pytest

from ode_filters.priors.gmp_priors import MaternPrior, _matern_companion_form


class TestMaternCompanionForm:
    """Tests for _matern_companion_form helper function."""

    def test_returns_correct_shapes(self):
        """Test that returned matrices have correct shapes."""
        length_scale, q = 1.0, 2
        F, L, q_coeff = _matern_companion_form(length_scale, q)
        D = q + 1

        assert F.shape == (D, D), f"F shape should be ({D}, {D})"
        assert L.shape == (D, 1), f"L shape should be ({D}, 1)"
        assert isinstance(q_coeff, float), "q coefficient should be a float"

    def test_F_has_super_diagonal_ones(self):
        """Test that F matrix has ones on super-diagonal."""
        length_scale, q = 2.0, 2
        F, _, _ = _matern_companion_form(length_scale, q)
        D = q + 1

        for i in range(D - 1):
            assert F[i, i + 1] == pytest.approx(1.0), f"F[{i}, {i + 1}] should be 1.0"

    def test_L_has_one_in_last_position(self):
        """Test that L vector has 1 in last position, zeros elsewhere."""
        length_scale, q = 1.5, 3
        _, L, _ = _matern_companion_form(length_scale, q)
        D = q + 1

        assert L[-1, 0] == pytest.approx(1.0), "Last element of L should be 1.0"
        for i in range(D - 1):
            assert L[i, 0] == pytest.approx(0.0), f"L[{i}] should be 0.0"

    def test_rejects_non_positive_length_scale(self):
        """Test that negative or zero length scale raises error."""
        with pytest.raises(ValueError, match="Length scale must be positive"):
            _matern_companion_form(0.0, 1)

        with pytest.raises(ValueError, match="Length scale must be positive"):
            _matern_companion_form(-1.0, 1)

    def test_rejects_negative_q(self):
        """Test that negative q raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            _matern_companion_form(1.0, -1)

    def test_rejects_float_q(self):
        """Test that float q raises error."""
        with pytest.raises(ValueError, match="non-negative integer"):
            _matern_companion_form(1.0, 1.5)

    @pytest.mark.parametrize("q", [0, 1, 2, 3])
    def test_q_coefficient_positive(self, q):
        """Test that diffusion coefficient is always positive."""
        _, _, q_coeff = _matern_companion_form(1.0, q)
        assert q_coeff > 0, "Diffusion coefficient should be positive"


class TestMaternPrior:
    """Tests for MaternPrior class."""

    def test_constructor_basic(self):
        """Test basic construction."""
        prior = MaternPrior(q=2, d=1, length_scale=1.0)
        assert prior.q == 2
        assert prior._dim == 1
        assert prior.n == 3  # q + 1

    def test_constructor_with_xi(self):
        """Test construction with custom Xi matrix."""
        xi = np.array([[2.0, 0.5], [0.5, 1.0]])
        prior = MaternPrior(q=1, d=2, length_scale=1.0, Xi=xi)
        assert np.allclose(prior.xi, xi)

    def test_A_returns_correct_shape(self):
        """Test that A(h) returns correct shape."""
        q, d = 2, 2
        prior = MaternPrior(q=q, d=d, length_scale=1.0)
        A = prior.A(0.5)
        expected_shape = ((q + 1) * d, (q + 1) * d)
        assert A.shape == expected_shape, f"A shape should be {expected_shape}"

    def test_Q_returns_correct_shape(self):
        """Test that Q(h) returns correct shape."""
        q, d = 2, 2
        prior = MaternPrior(q=q, d=d, length_scale=1.0)
        Q = prior.Q(0.5)
        expected_shape = ((q + 1) * d, (q + 1) * d)
        assert Q.shape == expected_shape, f"Q shape should be {expected_shape}"

    def test_b_returns_zeros(self):
        """Test that b(h) returns zero vector."""
        prior = MaternPrior(q=2, d=1, length_scale=1.0)
        b = prior.b(0.5)
        assert np.allclose(b, np.zeros(prior.n))

    def test_A_and_Q_combined(self):
        """Test that A_and_Q returns both matrices."""
        prior = MaternPrior(q=2, d=1, length_scale=1.0)
        A, Q = prior.A_and_Q(0.5)
        assert A.shape == (3, 3)
        assert Q.shape == (3, 3)

    def test_Q_is_symmetric(self):
        """Test that Q(h) is symmetric."""
        prior = MaternPrior(q=2, d=1, length_scale=1.0)
        Q = prior.Q(0.5)
        assert np.allclose(Q, Q.T), "Q should be symmetric"

    def test_Q_is_positive_semidefinite(self):
        """Test that Q(h) is positive semi-definite."""
        prior = MaternPrior(q=2, d=1, length_scale=1.0)
        Q = prior.Q(0.5)
        eigenvalues = np.linalg.eigvalsh(Q)
        assert np.all(eigenvalues >= -1e-10), "Q should be positive semi-definite"

    def test_rejects_negative_h(self):
        """Test that negative step size raises error."""
        prior = MaternPrior(q=2, d=1, length_scale=1.0)
        with pytest.raises(ValueError, match="h must be non-negative"):
            prior.A(-0.5)

    @pytest.mark.parametrize("h", [0.1, 0.5, 1.0, 2.0])
    def test_A_h_zero_is_identity(self, h):
        """Test that A(0) is identity matrix."""
        prior = MaternPrior(q=2, d=1, length_scale=1.0)
        A_0 = prior.A(0.0)
        n = prior.n
        assert np.allclose(A_0, np.eye(n)), "A(0) should be identity"

    def test_E0_E1_properties_available(self):
        """Test that E0 and E1 properties are accessible."""
        prior = MaternPrior(q=2, d=1, length_scale=1.0)
        assert prior.E0.shape == (1, 3)
        assert prior.E1.shape == (1, 3)
