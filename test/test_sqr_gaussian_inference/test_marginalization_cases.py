import jax.numpy as np

# ==============================================================================
# TEST CASES: BASIC FUNCTIONALITY
# ==============================================================================


def case_simple_2d_to_1d():
    """Test basic marginalization: 2D state to 1D observation."""
    A = np.array([[1.0, 0.0]])  # shape [1, 2]
    b = np.array([0.0])  # shape [1]
    Q = np.array([[0.1]])  # shape [1, 1]
    mu = np.array([1.0, 2.0])  # shape [2]
    Sigma = np.eye(2)  # shape [2, 2]

    expected_mu_z = np.array([1.0])
    expected_Sigma_z = np.array([[1.1]])

    return A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z


def case_3d_to_2d():
    """Test marginalization with larger dimensions: 3D state to 2D observation."""
    A = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.2]])  # shape [2, 3]
    b = np.array([1.0, -0.5])  # shape [2]
    Q = np.array([[0.1, 0.0], [0.0, 0.05]])  # shape [2, 2]
    mu = np.array([1.0, 2.0, 3.0])  # shape [3]
    Sigma = np.eye(3)  # shape [3, 3]

    expected_mu_z = np.array([3.0, 2.1])
    expected_Sigma_z = A @ Sigma @ A.T + Q

    return A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z


def case_nonzero_offset():
    """Test with non-zero offset b."""
    A = np.array([[1.0, 0.0]])
    b = np.array([5.0])
    Q = np.array([[0.1]])
    mu = np.array([2.0, 3.0])
    Sigma = np.eye(2)

    expected_mu_z = np.array([7.0])
    expected_Sigma_z = np.array([[1.1]])

    return A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z


# ==============================================================================
# TEST CASES: EDGE CASES & BOUNDARY CONDITIONS
# ==============================================================================


def case_identity_transformation():
    """Test with identity matrix (no state transformation)."""
    A = np.eye(2)
    b = np.zeros(2)
    Q = np.zeros((2, 2))
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    expected_mu_z = mu.copy()
    expected_Sigma_z = Sigma.copy()

    return A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z


def case_zero_prior_mean():
    """Test with zero prior mean."""
    A = np.array([[1.0, 0.0]])
    b = np.array([3.0])
    Q = np.array([[0.1]])
    mu = np.zeros(2)
    Sigma = np.eye(2)

    expected_mu_z = b.copy()
    expected_Sigma_z = np.array([[1.1]])

    return A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z


def case_zero_observation_noise():
    """Test with zero observation noise."""
    A = np.array([[1.0, 0.5]])
    b = np.array([0.0])
    Q = np.array([[0.0]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    expected_mu_z = A @ mu + b
    expected_Sigma_z = A @ Sigma @ A.T

    return A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z


def case_negative_values():
    """Test that negative transformation coefficients work correctly."""
    A = np.array([[-1.0, 0.0]])
    b = np.array([-2.0])
    Q = np.array([[0.1]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    expected_mu_z = np.array([-3.0])
    expected_Sigma_z = np.array([[1.1]])

    return A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z


def case_1d_to_1d():
    """Test marginalization with 1D state and 1D observation."""
    A = np.array([[2.0]])
    b = np.array([1.0])
    Q = np.array([[0.5]])
    mu = np.array([3.0])
    Sigma = np.array([[2.0]])

    expected_mu_z = np.array([7.0])
    expected_Sigma_z = np.array([[8.5]])

    return A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z


# ==============================================================================
# TEST CASES: MATHEMATICAL PROPERTIES
# ==============================================================================


def case_positive_definite():
    """Case for testing positive definiteness of output covariance."""
    A = np.array([[1.0, 0.5], [0.2, 0.8]])  # shape [2, 2]
    b = np.array([0.0, 0.0])
    Q = np.eye(2) * 0.1
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    return A, b, Q, mu, Sigma


def case_symmetry():
    """Case for testing output covariance symmetry."""
    A = np.array([[1.0, 0.5], [0.2, 0.8]])
    b = np.array([0.0, 0.0])
    Q = np.array([[0.1, 0.01], [0.01, 0.05]])
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.2], [0.2, 1.5]])

    return A, b, Q, mu, Sigma


# ==============================================================================
# TEST CASES: OUTPUT SHAPES
# ==============================================================================


def case_output_shape_1d_observation():
    """Output shapes must match [n_obs] and [n_obs, n_obs] for 1D observation."""
    A = np.array([[1.0, 0.0]])  # [1, 2]
    b = np.array([0.0])  # [1]
    Q = np.array([[0.1]])  # [1, 1]
    mu = np.array([1.0, 2.0])  # [2]
    Sigma = np.eye(2)  # [2, 2]

    expected_mu_shape = (1,)
    expected_Sigma_shape = (1, 1)

    return A, b, Q, mu, Sigma, expected_mu_shape, expected_Sigma_shape


def case_output_shape_2d_observation():
    """Output shapes must be correct for 2D observation."""
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5]])  # [2, 3]
    b = np.array([0.0, 0.0])  # [2]
    Q = np.eye(2) * 0.1  # [2, 2]
    mu = np.array([1.0, 2.0, 3.0])  # [3]
    Sigma = np.eye(3)  # [3, 3]

    expected_mu_shape = (2,)
    expected_Sigma_shape = (2, 2)

    return A, b, Q, mu, Sigma, expected_mu_shape, expected_Sigma_shape


# ==============================================================================
# TEST CASES: INPUT VALIDATION ERRORS
# ==============================================================================


def case_dimension_mismatch_A_mu():
    """Function should handle dimension mismatch between A and mu."""
    A = np.array([[1.0, 0.0]])  # expects 2D input
    b = np.array([0.0])
    Q = np.array([[0.1]])
    mu = np.array([1.0])  # Wrong size (1D not 2D)
    Sigma = np.eye(1)

    return A, b, Q, mu, Sigma


def case_dimension_mismatch_Sigma():
    """Sigma dimension must match mu dimension."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    Q = np.array([[0.1]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(3)  # Wrong size

    return A, b, Q, mu, Sigma


def case_dimension_mismatch_b_A():
    """b dimension must match A's first dimension."""
    A = np.array([[1.0, 0.0]])  # [1, 2]
    b = np.array([0.0, 0.0])  # Wrong: should be [1]
    Q = np.array([[0.1]])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    return A, b, Q, mu, Sigma


def case_non_square_Q():
    """Q must be square (observation noise covariance)."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    Q = np.array([[0.1, 0.0, 0.0]])  # Not square!
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    return A, b, Q, mu, Sigma
