import jax.numpy as np

# ==============================================================================
# TEST CASES: BASIC FUNCTIONALITY
# ==============================================================================


def case_simple_1d_observation():
    """Test inversion with 1D observation and 1D state."""
    A = np.array([[2.0]])  # shape [1, 1]
    mu = np.array([1.0])  # shape [1]
    Sigma = np.array([[1.0]])  # shape [1, 1]
    mu_z = np.array([2.0])  # shape [1]
    Sigma_z = np.array([[4.0]])  # shape [1, 1]

    return A, mu, Sigma, mu_z, Sigma_z


def case_2d_state_1d_observation():
    """Test inversion: 2D state with 1D observation."""
    A = np.array([[1.0, 0.0]])  # shape [1, 2]
    mu = np.array([1.0, 2.0])  # shape [2]
    Sigma = np.eye(2)  # shape [2, 2]
    mu_z = np.array([1.0])  # shape [1]
    Sigma_z = np.array([[1.0]])  # shape [1, 1]

    return A, mu, Sigma, mu_z, Sigma_z


def case_3d_state_2d_observation():
    """Test inversion: 3D state with 2D observation."""
    A = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.2]])  # shape [2, 3]
    mu = np.array([1.0, 2.0, 3.0])  # shape [3]
    Sigma = np.eye(3)  # shape [3, 3]
    mu_z = np.array([2.0, 3.0])  # shape [2]
    Sigma_z = np.array([[1.5, 0.1], [0.1, 1.2]])  # shape [2, 2]

    return A, mu, Sigma, mu_z, Sigma_z


# ==============================================================================
# TEST CASES: MATHEMATICAL PROPERTIES
# ==============================================================================


def case_identity_matrix():
    """Test inversion with identity observation matrix."""
    A = np.eye(2)  # Identity transformation
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)
    mu_z = np.array([1.0, 2.0])  # Same as mu
    Sigma_z = np.eye(2)

    return A, mu, Sigma, mu_z, Sigma_z


def case_diagonal_matrices():
    """Test inversion with diagonal covariance matrices."""
    A = np.array([[1.0, 0.5]])
    mu = np.array([1.0, 2.0])
    Sigma = np.diag(np.array([2.0, 3.0]))
    mu_z = np.array([2.0])
    Sigma_z = np.array([[2.25]])

    return A, mu, Sigma, mu_z, Sigma_z


def case_full_rank_covariance():
    """Test inversion with full-rank covariance matrices."""
    A = np.array([[1.0, 0.3], [0.2, 0.8]])  # shape [2, 2]
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.2], [0.2, 1.5]])
    mu_z = np.array([1.6, 1.8])
    Sigma_z = np.array([[1.2, 0.35], [0.35, 1.1]])

    return A, mu, Sigma, mu_z, Sigma_z


# ==============================================================================
# TEST CASES: EDGE CASES
# ==============================================================================


def case_zero_offset_mean():
    """Test inversion with zero offset in mean."""
    A = np.array([[1.0, 0.0]])
    mu = np.array([0.0, 0.0])
    Sigma = np.eye(2)
    mu_z = np.array([0.0])
    Sigma_z = np.array([[1.0]])

    return A, mu, Sigma, mu_z, Sigma_z


def case_negative_values():
    """Test inversion with negative transformation coefficients."""
    A = np.array([[-1.0, 0.5]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)
    mu_z = np.array([0.0])
    Sigma_z = np.array([[1.25]])

    return A, mu, Sigma, mu_z, Sigma_z


def case_small_observation_noise():
    """Test inversion with very small observation noise (high confidence)."""
    A = np.array([[1.0]])
    mu = np.array([0.0])
    Sigma = np.array([[1.0]])
    mu_z = np.array([1.0])
    Sigma_z = np.array([[0.01]])  # Very small noise

    return A, mu, Sigma, mu_z, Sigma_z


def case_large_observation_noise():
    """Test inversion with large observation noise (low confidence)."""
    A = np.array([[1.0]])
    mu = np.array([0.0])
    Sigma = np.array([[1.0]])
    mu_z = np.array([1.0])
    Sigma_z = np.array([[100.0]])  # Very large noise

    return A, mu, Sigma, mu_z, Sigma_z


# ==============================================================================
# TEST CASES: OUTPUT SHAPES
# ==============================================================================


def case_output_shapes_1d():
    """Test that output shapes are correct for 1D case."""
    A = np.array([[2.0]])  # [1, 1]
    mu = np.array([1.0])  # [1]
    Sigma = np.array([[1.0]])  # [1, 1]
    mu_z = np.array([2.0])  # [1]
    Sigma_z = np.array([[4.0]])  # [1, 1]

    # Expected shapes: G[n_state, n_obs], d[n_state], Lambda[n_state, n_state]
    expected_G_shape = (1, 1)
    expected_d_shape = (1,)
    expected_Lambda_shape = (1, 1)

    return (
        A,
        mu,
        Sigma,
        mu_z,
        Sigma_z,
        expected_G_shape,
        expected_d_shape,
        expected_Lambda_shape,
    )


def case_output_shapes_2d():
    """Test that output shapes are correct for 2D state, 1D observation."""
    A = np.array([[1.0, 0.0]])  # [1, 2]
    mu = np.array([1.0, 2.0])  # [2]
    Sigma = np.eye(2)  # [2, 2]
    mu_z = np.array([1.0])  # [1]
    Sigma_z = np.array([[1.0]])  # [1, 1]

    expected_G_shape = (2, 1)
    expected_d_shape = (2,)
    expected_Lambda_shape = (2, 2)

    return (
        A,
        mu,
        Sigma,
        mu_z,
        Sigma_z,
        expected_G_shape,
        expected_d_shape,
        expected_Lambda_shape,
    )


def case_output_shapes_3d():
    """Test that output shapes are correct for 3D state, 2D observation."""
    A = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.2]])  # [2, 3]
    mu = np.array([1.0, 2.0, 3.0])  # [3]
    Sigma = np.eye(3)  # [3, 3]
    mu_z = np.array([2.0, 3.0])  # [2]
    Sigma_z = np.array([[1.5, 0.1], [0.1, 1.2]])  # [2, 2]

    expected_G_shape = (3, 2)
    expected_d_shape = (3,)
    expected_Lambda_shape = (3, 3)

    return (
        A,
        mu,
        Sigma,
        mu_z,
        Sigma_z,
        expected_G_shape,
        expected_d_shape,
        expected_Lambda_shape,
    )


# ==============================================================================
# TEST CASES: INPUT VALIDATION ERRORS
# ==============================================================================


def case_dimension_mismatch_A_Sigma():
    """A's columns must match Sigma's first dimension."""
    A = np.array([[1.0, 0.0, 0.0]])  # [1, 3]
    mu = np.array([1.0, 2.0])  # [2] - mismatch!
    Sigma = np.eye(2)
    mu_z = np.array([1.0])
    Sigma_z = np.array([[1.0]])

    return A, mu, Sigma, mu_z, Sigma_z


def case_dimension_mismatch_Sigma_mu():
    """Sigma's first dimension must match mu's dimension."""
    A = np.array([[1.0, 0.0]])
    mu = np.array([1.0, 2.0, 3.0])  # [3]
    Sigma = np.eye(2)  # [2, 2] - mismatch!
    mu_z = np.array([1.0])
    Sigma_z = np.array([[1.0]])

    return A, mu, Sigma, mu_z, Sigma_z


def case_dimension_mismatch_A_mu_z():
    """A's rows must match mu_z's dimension."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])  # [2, 2]
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)
    mu_z = np.array([1.0])  # [1] - should be [2]
    Sigma_z = np.array([[1.0]])

    return A, mu, Sigma, mu_z, Sigma_z


def case_dimension_mismatch_Sigma_z():
    """Sigma_z must be square with dimension matching mu_z."""
    A = np.array([[1.0, 0.0]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)
    mu_z = np.array([1.0])
    Sigma_z = np.array([[1.0, 0.5]])  # Not square!

    return A, mu, Sigma, mu_z, Sigma_z


def case_singular_Sigma_z():
    """Sigma_z must be invertible (non-singular)."""
    A = np.array([[1.0, 0.0]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)
    mu_z = np.array([1.0])
    Sigma_z = np.array([[0.0]])  # Singular matrix!

    return A, mu, Sigma, mu_z, Sigma_z


# ==============================================================================
# TEST CASES: MATHEMATICAL PROPERTIES
# ==============================================================================


def case_posterior_covariance_positive_definite():
    """Case for testing that posterior covariance is positive definite."""
    A = np.array([[1.0, 0.5], [0.2, 0.8]])
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.2], [0.2, 1.5]])
    mu_z = np.array([1.6, 1.8])
    Sigma_z = np.array([[1.2, 0.35], [0.35, 1.1]])

    return A, mu, Sigma, mu_z, Sigma_z


def case_posterior_covariance_symmetric():
    """Case for testing that posterior covariance is symmetric."""
    A = np.array([[1.0, 0.3], [0.2, 0.8]])
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.15], [0.15, 1.5]])
    mu_z = np.array([1.6, 1.8])
    Sigma_z = np.array([[1.2, 0.35], [0.35, 1.1]])

    return A, mu, Sigma, mu_z, Sigma_z


def case_lambda_smaller_than_prior():
    """Case for testing that posterior covariance is smaller than prior."""
    A = np.array([[1.0, 0.0]])  # Perfect observation of first coordinate
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.1], [0.1, 1.0]])
    mu_z = np.array([1.0])  # Observed value
    Sigma_z = np.array([[1.1]])  # Small observation noise

    return A, mu, Sigma, mu_z, Sigma_z
