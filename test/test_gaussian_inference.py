import pytest
import numpy as np
from ode_filters.gaussian_inference import marginalization


class TestMarginalization:
    """Test suite for the marginalization function."""

    def test_marginalization_basic(self):
        """Test basic functionality with simple 2D case."""
        # Setup simple case: 2D state, 1D observation
        A = np.array([[1.0, 2.0]])  # shape [1, 2]
        b = np.array([0.5])  # shape [1]
        Q = np.array([[0.1]])  # shape [1, 1]
        mu = np.array([1.0, 2.0])  # shape [2]
        Sigma = np.eye(2)  # shape [2, 2]

        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

        # Expected values
        # mu_z = A @ mu + b = [1, 2] @ [1, 2] + 0.5 = 5 + 0.5 = 5.5
        expected_mu_z = np.array([5.5])
        # Sigma_z = A @ Sigma @ A.T + Q = [1, 2] @ I @ [1, 2]^T + 0.1
        #         = [1, 2] @ [1, 2]^T + 0.1 = 5 + 0.1 = 5.1
        expected_Sigma_z = np.array([[5.1]])

        np.testing.assert_allclose(mu_z, expected_mu_z, rtol=1e-10)
        np.testing.assert_allclose(Sigma_z, expected_Sigma_z, rtol=1e-10)

    def test_marginalization_shapes(self):
        """Test that output shapes are correct for various input dimensions."""
        test_cases = [
            (1, 1),  # 1D state, 1D observation
            (2, 1),  # 2D state, 1D observation
            (3, 2),  # 3D state, 2D observation
            (5, 3),  # 5D state, 3D observation
        ]

        for n_state, n_obs in test_cases:
            A = np.random.randn(n_obs, n_state)
            b = np.random.randn(n_obs)
            Q = np.random.randn(n_obs, n_obs)
            Q = Q @ Q.T  # Make positive definite
            mu = np.random.randn(n_state)
            Sigma = np.random.randn(n_state, n_state)
            Sigma = Sigma @ Sigma.T  # Make positive definite

            mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

            assert mu_z.shape == (n_obs,), f"mu_z shape mismatch for n_state={n_state}, n_obs={n_obs}"
            assert Sigma_z.shape == (n_obs, n_obs), f"Sigma_z shape mismatch for n_state={n_state}, n_obs={n_obs}"

    def test_marginalization_zero_noise(self):
        """Test with zero observation noise (Q = 0)."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])  # Identity
        b = np.array([0.0, 0.0])  # No offset
        Q = np.zeros((2, 2))  # No noise
        mu = np.array([1.0, 2.0])
        Sigma = np.eye(2)

        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

        # With zero noise, mu_z = A @ mu + b = [1, 2]
        # and Sigma_z = A @ Sigma @ A.T + Q = I
        np.testing.assert_allclose(mu_z, mu, rtol=1e-10)
        np.testing.assert_allclose(Sigma_z, Sigma, rtol=1e-10)

    def test_marginalization_identity_transform(self):
        """Test with identity transformation (A = I, b = 0)."""
        n = 3
        A = np.eye(n)
        b = np.zeros(n)
        Q = np.eye(n) * 0.5
        mu = np.random.randn(n)
        Sigma = np.random.randn(n, n)
        Sigma = Sigma @ Sigma.T  # Make positive definite

        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

        np.testing.assert_allclose(mu_z, mu, rtol=1e-10)
        np.testing.assert_allclose(Sigma_z, Sigma + Q, rtol=1e-10)

    def test_marginalization_with_offset(self):
        """Test that offset b is correctly applied to mean."""
        A = np.eye(2)
        b = np.array([1.0, -0.5])
        Q = np.eye(2) * 0.1
        mu = np.zeros(2)
        Sigma = np.eye(2)

        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

        # With A = I and mu = 0, mu_z = A @ 0 + b = b
        np.testing.assert_allclose(mu_z, b, rtol=1e-10)

    def test_marginalization_covariance_is_positive_semidefinite(self):
        """Test that output covariance is positive semidefinite."""
        np.random.seed(42)
        for _ in range(5):
            n_state = np.random.randint(2, 6)
            n_obs = np.random.randint(1, n_state + 1)

            A = np.random.randn(n_obs, n_state)
            b = np.random.randn(n_obs)
            Q = np.random.randn(n_obs, n_obs)
            Q = Q @ Q.T  # Make positive definite
            mu = np.random.randn(n_state)
            Sigma = np.random.randn(n_state, n_state)
            Sigma = Sigma @ Sigma.T  # Make positive definite

            mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

            # Check positive semidefinite: all eigenvalues >= 0
            eigenvalues = np.linalg.eigvalsh(Sigma_z)
            assert np.all(eigenvalues >= -1e-10), f"Sigma_z is not positive semidefinite, eigenvalues: {eigenvalues}"

    def test_marginalization_dimensional_reduction(self):
        """Test case where we observe fewer dimensions than the state."""
        # 3D state, 1D observation (e.g., observing only one component)
        A = np.array([[1.0, 0.0, 0.0]])  # Observe only first component
        b = np.array([0.0])
        Q = np.array([[0.1]])
        mu = np.array([1.0, 2.0, 3.0])
        Sigma = np.eye(3)

        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

        # mu_z should be the first component of mu
        np.testing.assert_allclose(mu_z, np.array([1.0]), rtol=1e-10)
        # Sigma_z should be just the (0,0) element of Sigma plus Q
        expected_Sigma_z = np.array([[1.0 + 0.1]])
        np.testing.assert_allclose(Sigma_z, expected_Sigma_z, rtol=1e-10)

    def test_marginalization_linear_combination(self):
        """Test observation that is a linear combination of states."""
        # Observe a weighted sum of two states
        A = np.array([[2.0, 3.0]])
        b = np.array([0.0])
        Q = np.array([[0.01]])
        mu = np.array([1.0, 1.0])
        Sigma = np.eye(2)

        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

        # mu_z = 2*1 + 3*1 = 5
        np.testing.assert_allclose(mu_z, np.array([5.0]), rtol=1e-10)
        # Sigma_z = [2, 3] @ I @ [2, 3]^T + 0.01 = 4 + 9 + 0.01 = 13.01
        np.testing.assert_allclose(Sigma_z, np.array([[13.01]]), rtol=1e-10)

    def test_marginalization_repeated_calls_consistency(self):
        """Test that repeated calls with same inputs give same outputs."""
        A = np.random.randn(2, 3)
        b = np.random.randn(2)
        Q = np.random.randn(2, 2)
        Q = Q @ Q.T
        mu = np.random.randn(3)
        Sigma = np.random.randn(3, 3)
        Sigma = Sigma @ Sigma.T

        mu_z1, Sigma_z1 = marginalization(A, b, Q, mu, Sigma)
        mu_z2, Sigma_z2 = marginalization(A, b, Q, mu, Sigma)

        np.testing.assert_array_equal(mu_z1, mu_z2)
        np.testing.assert_array_equal(Sigma_z1, Sigma_z2)

    def test_marginalization_numerical_stability_small_values(self):
        """Test numerical stability with small values."""
        A = np.array([[1e-5, 1e-5]])
        b = np.array([1e-6])
        Q = np.array([[1e-10]])
        mu = np.array([1e-5, 1e-5])
        Sigma = np.eye(2) * 1e-10

        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

        # Should not produce NaN or Inf
        assert np.isfinite(mu_z).all()
        assert np.isfinite(Sigma_z).all()

    def test_marginalization_numerical_stability_large_values(self):
        """Test numerical stability with large values."""
        A = np.array([[1e5, 1e5]])
        b = np.array([1e6])
        Q = np.array([[1e10]])
        mu = np.array([1e5, 1e5])
        Sigma = np.eye(2) * 1e10

        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

        # Should not produce NaN or Inf
        assert np.isfinite(mu_z).all()
        assert np.isfinite(Sigma_z).all() 