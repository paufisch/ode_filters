import numpy as np
import pytest

from ode_filters.gaussian_inference import marginalization

# ==============================================================================
# 1. BASIC FUNCTIONALITY TESTS
# ==============================================================================


def test_marginalization_simple_2d_to_1d():
    """Test basic marginalization: 2D state to 1D observation."""
    A = np.array([[1.0, 0.0]])  # shape [1, 2]
    b = np.array([0.0])  # shape [1]
    Q = np.array([[0.1]])  # shape [1, 1]
    mu = np.array([1.0, 2.0])  # shape [2]
    Sigma = np.eye(2)  # shape [2, 2]

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Expected: mu_z = A @ mu + b = [1, 0] @ [1, 2] + 0 = [1]
    # Expected: Sigma_z = A @ Sigma @ A.T + Q = [1, 0] @ I @ [1; 0] + 0.1 = [[1.1]]
    assert mu_z == pytest.approx([1.0])
    assert Sigma_z == pytest.approx(np.array([[1.1]]))


def test_marginalization_3d_to_2d():
    """Test marginalization with larger dimensions: 3D state to 2D observation."""
    A = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.2]])  # shape [2, 3]
    b = np.array([1.0, -0.5])  # shape [2]
    Q = np.array([[0.1, 0.0], [0.0, 0.05]])  # shape [2, 2]
    mu = np.array([1.0, 2.0, 3.0])  # shape [3]
    Sigma = np.eye(3)  # shape [3, 3]

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # mu_z = [[1, 0.5, 0], [0, 1, 0.2]] @ [1, 2, 3] + [1, -0.5]
    #      = [1 + 1, 2 + 0.6] + [1, -0.5] = [3.0, 2.1]
    expected_mu_z = np.array([3.0, 2.1])
    assert mu_z == pytest.approx(expected_mu_z)

    # Covariance computation
    expected_Sigma_z = A @ Sigma @ A.T + Q
    assert Sigma_z == pytest.approx(expected_Sigma_z)


def test_marginalization_with_nonzero_offset():
    """Test with non-zero offset b."""
    A = np.array([[1.0, 0.0]])
    b = np.array([5.0])
    Q = np.array([[0.1]])
    mu = np.array([2.0, 3.0])
    Sigma = np.eye(2)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # mu_z = [1, 0] @ [2, 3] + 5 = 2 + 5 = [7]
    assert mu_z == pytest.approx([7.0])


# ==============================================================================
# 2. MATHEMATICAL PROPERTIES TESTS
# ==============================================================================


def test_marginalization_preserves_positive_definiteness():
    """Output covariance must remain positive definite."""
    A = np.array([[1.0, 0.5], [0.2, 0.8]])  # shape [2, 2]
    b = np.array([0.0, 0.0])
    Q = np.eye(2) * 0.1
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Check positive definiteness: all eigenvalues > 0
    eigenvalues = np.linalg.eigvalsh(Sigma_z)
    assert np.all(eigenvalues > -1e-10)  # Allow for numerical precision


def test_marginalization_output_symmetry():
    """Output covariance matrix must be symmetric."""
    A = np.array([[1.0, 0.5], [0.2, 0.8]])
    b = np.array([0.0, 0.0])
    Q = np.array([[0.1, 0.01], [0.01, 0.05]])
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.2], [0.2, 1.5]])

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Covariance matrices must be symmetric
    assert np.allclose(Sigma_z, Sigma_z.T)


def test_marginalization_noise_increases_uncertainty():
    """Adding observation noise should increase output covariance."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    Q_small = np.array([[0.01]])
    Q_large = np.array([[1.0]])

    _, Sigma_z_small = marginalization(A, b, Q_small, mu, Sigma)
    _, Sigma_z_large = marginalization(A, b, Q_large, mu, Sigma)

    # Larger noise should result in larger output covariance
    assert Sigma_z_large[0, 0] > Sigma_z_small[0, 0]


# ==============================================================================
# 3. EDGE CASES & BOUNDARY CONDITIONS
# ==============================================================================


def test_marginalization_identity_transformation():
    """Test with identity matrix (no state transformation)."""
    A = np.eye(2)
    b = np.zeros(2)
    Q = np.zeros((2, 2))
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Should pass through unchanged
    assert np.allclose(mu_z, mu)
    assert np.allclose(Sigma_z, Sigma)


def test_marginalization_zero_prior_mean():
    """Test with zero prior mean."""
    A = np.array([[1.0, 0.0]])
    b = np.array([3.0])
    Q = np.array([[0.1]])
    mu = np.zeros(2)
    Sigma = np.eye(2)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # mu_z should equal b when mu is zero
    assert mu_z == pytest.approx(b)


def test_marginalization_zero_observation_noise():
    """Test with zero observation noise."""
    A = np.array([[1.0, 0.5]])
    b = np.array([0.0])
    Q = np.array([[0.0]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Covariance should be just A @ Sigma @ A.T
    expected_Sigma_z = A @ Sigma @ A.T
    assert Sigma_z == pytest.approx(expected_Sigma_z)


def test_marginalization_with_negative_values():
    """Test that negative transformation coefficients work correctly."""
    A = np.array([[-1.0, 0.0]])
    b = np.array([-2.0])
    Q = np.array([[0.1]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # mu_z = [-1, 0] @ [1, 2] - 2 = -1 - 2 = [-3]
    assert mu_z == pytest.approx([-3.0])
    # Covariance should still be positive
    assert Sigma_z[0, 0] > 0


def test_marginalization_1d_to_1d():
    """Test marginalization with 1D state and 1D observation."""
    A = np.array([[2.0]])
    b = np.array([1.0])
    Q = np.array([[0.5]])
    mu = np.array([3.0])
    Sigma = np.array([[2.0]])

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # mu_z = 2 * 3 + 1 = 7
    assert mu_z == pytest.approx([7.0])
    # Sigma_z = 2 * 2 * 2 + 0.5 = 8.5
    assert Sigma_z == pytest.approx(np.array([[8.5]]))


# ==============================================================================
# 4. OUTPUT VALIDATION TESTS (Shape & Type)
# ==============================================================================


def test_marginalization_output_shape_1d_observation():
    """Output shapes must match [n_obs] and [n_obs, n_obs] for 1D observation."""
    A = np.array([[1.0, 0.0]])  # [1, 2]
    b = np.array([0.0])  # [1]
    Q = np.array([[0.1]])  # [1, 1]
    mu = np.array([1.0, 2.0])  # [2]
    Sigma = np.eye(2)  # [2, 2]

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    assert mu_z.shape == (1,)
    assert Sigma_z.shape == (1, 1)


def test_marginalization_output_shape_2d_observation():
    """Output shapes must be correct for 2D observation."""
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])  # [2, 3]
    b = np.array([0.0, 0.0])  # [2]
    Q = np.eye(2) * 0.1  # [2, 2]
    mu = np.array([1.0, 2.0, 3.0])  # [3]
    Sigma = np.eye(3)  # [3, 3]

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    assert mu_z.shape == (2,)
    assert Sigma_z.shape == (2, 2)


def test_marginalization_output_type_ndarray():
    """Output should be numpy arrays."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    Q = np.array([[0.1]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    assert isinstance(mu_z, np.ndarray)
    assert isinstance(Sigma_z, np.ndarray)


def test_marginalization_dtype_preservation():
    """Output dtype should match input dtype."""
    A = np.array([[1.0, 0.0]], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    Q = np.array([[0.1]], dtype=np.float64)
    mu = np.array([1.0, 2.0], dtype=np.float64)
    Sigma = np.eye(2, dtype=np.float64)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    assert mu_z.dtype in [np.float64, np.float32, float]
    assert Sigma_z.dtype in [np.float64, np.float32, float]


# ==============================================================================
# 5. INPUT VALIDATION TESTS
# ==============================================================================


def test_marginalization_dimension_mismatch_A_mu():
    """Function should handle dimension mismatch between A and mu."""
    A = np.array([[1.0, 0.0]])  # expects 2D input
    b = np.array([0.0])
    Q = np.array([[0.1]])
    mu = np.array([1.0])  # Wrong size (1D not 2D)
    Sigma = np.eye(1)

    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        marginalization(A, b, Q, mu, Sigma)


def test_marginalization_dimension_mismatch_Sigma():
    """Sigma dimension must match mu dimension."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    Q = np.array([[0.1]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(3)  # Wrong size

    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        marginalization(A, b, Q, mu, Sigma)


def test_marginalization_dimension_mismatch_b_A():
    """b dimension must match A's first dimension."""
    A = np.array([[1.0, 0.0]])  # [1, 2]
    b = np.array([0.0, 0.0])  # Wrong: should be [1]
    Q = np.array([[0.1]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        marginalization(A, b, Q, mu, Sigma)


def test_marginalization_non_square_Q():
    """Q must be square (observation noise covariance)."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    Q = np.array([[0.1, 0.0, 0.0]])  # Not square!
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        marginalization(A, b, Q, mu, Sigma)


# ==============================================================================
# 6. INTEGRATION TESTS
# ==============================================================================


def test_marginalization_1d_covariance_as_array():
    """Test that 1D covariance Q can be passed as 1D array (edge case)."""
    # This test documents behavior when Q is passed as 1D instead of 2D
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    Q = np.array([0.1])  # 1D instead of [[0.1]]
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    # Function should still work or raise clear error
    try:
        mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)
        # If it works, verify output is still correct
        assert mu_z == pytest.approx([1.0])
    except (ValueError, np.linalg.LinAlgError):
        # It's also acceptable to reject 1D Q
        pass
