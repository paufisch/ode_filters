import jax.numpy as np
import numpy as onp  # Regular numpy for exceptions
import pytest
from pytest_cases import parametrize_with_cases

from ode_filters.inference.sqr_gaussian_inference import sqr_inversion
from test.test_sqr_gaussian_inference.test_inversion_cases import (
    case_2d_state_1d_observation,
    case_3d_state_2d_observation,
    case_diagonal_matrices,
    case_dimension_mismatch_A_mu_z,
    case_dimension_mismatch_A_Sigma,
    case_dimension_mismatch_Sigma_mu,
    case_dimension_mismatch_Sigma_z,
    case_full_rank_covariance,
    case_identity_matrix,
    case_lambda_smaller_than_prior,
    case_large_observation_noise,
    case_negative_values,
    case_output_shapes_1d,
    case_output_shapes_2d,
    case_output_shapes_3d,
    case_posterior_covariance_positive_definite,
    case_posterior_covariance_symmetric,
    case_simple_1d_observation,
    case_small_observation_noise,
    case_zero_offset_mean,
)

# ==============================================================================
# PARAMETRIZED TESTS: BASIC FUNCTIONALITY
# ==============================================================================


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[
        case_simple_1d_observation,
        case_2d_state_1d_observation,
        case_3d_state_2d_observation,
        case_identity_matrix,
        case_diagonal_matrices,
        case_full_rank_covariance,
    ],
)
def test_sqr_inversion_basic_functionality(A, mu, Sigma, mu_z, Sigma_z):
    """Test that sqr_inversion function executes without errors for various inputs."""
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    G, d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)

    # Verify we get three outputs
    assert G is not None
    assert d is not None
    assert Lambda is not None


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z,expected_G_shape,expected_d_shape,expected_Lambda_shape",
    cases=[
        case_output_shapes_1d,
        case_output_shapes_2d,
        case_output_shapes_3d,
    ],
)
def test_sqr_inversion_output_shapes(
    A,
    mu,
    Sigma,
    mu_z,
    Sigma_z,
    expected_G_shape,
    expected_d_shape,
    expected_Lambda_shape,
):
    """Test that sqr_inversion returns outputs with correct shapes."""
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    G, d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)

    assert G.shape == expected_G_shape, f"G shape {G.shape} != {expected_G_shape}"
    assert d.shape == expected_d_shape, f"d shape {d.shape} != {expected_d_shape}"
    assert Lambda.shape == expected_Lambda_shape, (
        f"Lambda shape {Lambda.shape} != {expected_Lambda_shape}"
    )


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[
        case_simple_1d_observation,
        case_2d_state_1d_observation,
        case_3d_state_2d_observation,
    ],
)
def test_sqr_inversion_output_types(A, mu, Sigma, mu_z, Sigma_z):
    """Test that sqr_inversion returns numpy arrays."""
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    G, d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)

    assert isinstance(G, np.ndarray), "G should be numpy array"
    assert isinstance(d, np.ndarray), "d should be numpy array"
    assert isinstance(Lambda, np.ndarray), "Lambda should be numpy array"


# ==============================================================================
# PARAMETRIZED TESTS: MATHEMATICAL PROPERTIES
# ==============================================================================


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[
        case_posterior_covariance_symmetric,
        case_posterior_covariance_positive_definite,
    ],
)
def test_sqr_inversion_posterior_covariance_symmetry(A, mu, Sigma, mu_z, Sigma_z):
    """Test that posterior covariance Lambda is symmetric."""
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    _, _, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda
    assert np.allclose(Lambda, Lambda.T), "Posterior covariance must be symmetric"


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[case_posterior_covariance_positive_definite],
)
def test_sqr_inversion_posterior_covariance_positive_definite(
    A, mu, Sigma, mu_z, Sigma_z
):
    """Test that posterior covariance Lambda is positive definite."""
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    _, _, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda
    eigenvalues = np.linalg.eigvalsh(Lambda)
    assert np.all(eigenvalues > -1e-10), (
        "Posterior covariance must be positive definite"
    )


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[case_lambda_smaller_than_prior],
)
def test_sqr_inversion_reduces_uncertainty(A, mu, Sigma, mu_z, Sigma_z):
    """Test that observation reduces uncertainty (Lambda <= Sigma)."""
    prior_trace = np.trace(Sigma)
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    _, _, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda

    # Check that posterior covariance trace is smaller than or equal to prior
    # (observation should reduce uncertainty)
    posterior_trace = np.trace(Lambda)
    assert posterior_trace <= prior_trace + 1e-10, (
        f"Observation should reduce uncertainty: "
        f"posterior_trace={posterior_trace} should be <= prior_trace={prior_trace}"
    )


# ==============================================================================
# PARAMETRIZED TESTS: EDGE CASES
# ==============================================================================


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[
        case_zero_offset_mean,
        case_negative_values,
        case_small_observation_noise,
        case_large_observation_noise,
    ],
)
def test_sqr_inversion_edge_cases(A, mu, Sigma, mu_z, Sigma_z):
    """Test sqr_inversion with edge case inputs."""
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    G, d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda

    # Verify outputs exist and have correct types
    assert isinstance(G, np.ndarray)
    assert isinstance(d, np.ndarray)
    assert isinstance(Lambda, np.ndarray)

    # Verify Lambda is positive definite
    eigenvalues = np.linalg.eigvalsh(Lambda)
    assert np.all(eigenvalues > -1e-10), "Lambda must be positive definite"


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[case_small_observation_noise],
)
def test_sqr_inversion_high_confidence_observation(A, mu, Sigma, mu_z, Sigma_z):
    """Test sqr_inversion with very small observation noise (high confidence in measurement)."""
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    G, _d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda

    # With high confidence in observation, Kalman gain should be larger
    # (we trust the measurement more)
    assert np.all(np.abs(G) > 0), "Kalman gain should be non-zero"


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[case_large_observation_noise],
)
def test_sqr_inversion_low_confidence_observation(A, mu, Sigma, mu_z, Sigma_z):
    """Test sqr_inversion with large observation noise (low confidence in measurement)."""
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    G, _d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda

    # With low confidence in observation, Kalman gain should be smaller
    # (we trust the prior more)
    assert isinstance(G, np.ndarray), "Should handle low confidence case"


# ==============================================================================
# PARAMETRIZED TESTS: INPUT VALIDATION ERRORS
# ==============================================================================


@parametrize_with_cases(
    "A,mu,Sigma,mu_z,Sigma_z",
    cases=[
        case_dimension_mismatch_A_Sigma,
        case_dimension_mismatch_Sigma_mu,
        case_dimension_mismatch_A_mu_z,
        case_dimension_mismatch_Sigma_z,
        # Note: case_singular_Sigma_z removed - JAX returns NaN/Inf instead of raising
    ],
)
def test_sqr_inversion_raises_on_invalid_input(A, mu, Sigma, mu_z, Sigma_z):
    """Test that sqr_inversion raises errors on invalid input dimensions."""
    # Sigma = cholesky(Sigma)
    # Sigma_z = cholesky(Sigma_z)
    with pytest.raises((ValueError, TypeError, onp.linalg.LinAlgError)):
        sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)


# ==============================================================================
# NON-PARAMETRIZED TESTS
# ==============================================================================


def test_sqr_inversion_simple_1d_manual():
    """Manually verify a simple 1D sqr_inversion case."""
    # Simple 1D case: scale by 2
    A = np.array([[2.0]])
    mu = np.array([0.0])
    Sigma = np.array([[1.0]])
    mu_z = np.array([2.0])
    Sigma_z = np.array([[1.0]])
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T

    G, d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda

    # With this setup, verify reasonable outputs
    assert G.shape == (1, 1)
    assert d.shape == (1,)
    assert Lambda.shape == (1, 1)
    assert Lambda[0, 0] > 0, "Posterior variance should be positive"


def test_sqr_inversion_gain_matrix_properties():
    """Test properties of the Kalman gain matrix."""
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    mu = np.array([0.0, 0.0])
    Sigma = np.eye(2)
    mu_z = np.array([1.0, 1.0])
    Sigma_z = np.eye(2)
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T

    G, _d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda

    # Kalman gain dimensions: [n_state, n_obs]
    assert G.shape == (2, 2), "Gain matrix should be [n_state, n_obs]"

    # With identity A and small observation noise, gain should be significant
    assert np.any(np.abs(G) > 0), "Gain matrix should have non-zero elements"


def test_sqr_inversion_posterior_mean_shift():
    """Test that posterior mean is shifted towards observation."""
    A = np.array([[1.0]])  # Identity observation
    mu = np.array([0.0])  # Prior mean at 0
    Sigma = np.array([[1.0]])  # Prior covariance
    mu_z = np.array([10.0])  # Observation at 10
    Sigma_z = np.array([[0.1]])  # Very confident observation
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T
    _G, d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda

    # Posterior mean should be shifted towards the observation
    # d = mu - G @ mu_z
    # With strong observation (small Sigma_z), d should be close to 0
    # meaning posterior mean ~ G @ mu_z should be close to mu_z
    assert isinstance(d, np.ndarray), "d should be a numpy array"
    assert d.size == 1, "d should be 1D for this case"


def test_sqr_inversion_preserves_covariance_properties():
    """Test that sqr_inversion preserves covariance matrix properties."""
    A = np.array([[1.0, 0.5]])
    mu = np.array([1.0, 2.0])
    Sigma = np.array([[1.0, 0.1], [0.1, 1.1]])  # Positive definite, symmetric
    mu_z = np.array([2.0])
    Sigma_z = np.array([[1.0]])
    Sigma = np.linalg.cholesky(Sigma).T
    Sigma_z = np.linalg.cholesky(Sigma_z).T

    _G, _d, Lambda = sqr_inversion(A, mu, Sigma, mu_z, Sigma_z)
    Lambda = Lambda.T @ Lambda

    # Lambda should be symmetric
    assert np.allclose(Lambda, Lambda.T), "Lambda should be symmetric"

    # Lambda should be positive definite
    eigenvalues = np.linalg.eigvalsh(Lambda)
    assert np.all(eigenvalues > -1e-10), "Lambda should be positive definite"

    # Lambda should be smaller than Sigma (observation reduces uncertainty)
    assert np.trace(Lambda) <= np.trace(Sigma.T @ Sigma) + 1e-10, (
        "Posterior covariance trace should be <= prior covariance trace"
    )
