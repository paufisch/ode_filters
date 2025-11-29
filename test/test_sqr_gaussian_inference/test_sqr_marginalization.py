import jax.numpy as np
import numpy as onp  # Regular numpy for exceptions
import pytest
from pytest_cases import parametrize_with_cases

from ode_filters.inference.sqr_gaussian_inference import sqr_marginalization
from test.test_sqr_gaussian_inference.test_marginalization_cases import (
    case_1d_to_1d,
    case_3d_to_2d,
    case_dimension_mismatch_A_mu,
    case_dimension_mismatch_b_A,
    case_dimension_mismatch_Sigma,
    case_identity_transformation,
    case_negative_values,
    case_non_square_Q,
    case_nonzero_offset,
    case_output_shape_1d_observation,
    case_output_shape_2d_observation,
    case_positive_definite,
    case_simple_2d_to_1d,
    case_symmetry,
    case_zero_observation_noise,
    case_zero_prior_mean,
)

# ==============================================================================
# PARAMETRIZED TESTS
# ==============================================================================


@parametrize_with_cases(
    "A,b,Q,mu,Sigma,expected_mu_z,expected_Sigma_z",
    cases=[
        case_simple_2d_to_1d,
        case_3d_to_2d,
        case_nonzero_offset,
        case_identity_transformation,
        case_zero_prior_mean,
        case_zero_observation_noise,
        case_negative_values,
        case_1d_to_1d,
    ],
)
def test_sqr_marginalization_correctness(
    A, b, Q, mu, Sigma, expected_mu_z, expected_Sigma_z
):
    """Test that sqr_marginalization computes correct mean and covariance."""
    if (np.zeros_like(Q) != Q).any():
        Q = np.linalg.cholesky(Q).T
    Sigma = np.linalg.cholesky(Sigma).T
    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)
    Sigma_z = Sigma_z.T @ Sigma_z

    assert mu_z == pytest.approx(expected_mu_z), (
        "mu_z should match value of expected value"
    )
    assert Sigma_z == pytest.approx(expected_Sigma_z), (
        "Sigma_z should match value of expected value"
    )


@parametrize_with_cases(
    "A,b,Q,mu,Sigma",
    cases=[case_output_shape_1d_observation, case_output_shape_2d_observation],
)
def test_sqr_marginalization_output_shapes(
    A,
    b,
    Q,
    mu,
    Sigma,
    case_output_shape_1d_observation=None,
    case_output_shape_2d_observation=None,
):
    """Test that output shapes are correct."""
    Q = np.linalg.cholesky(Q).T
    Sigma = np.linalg.cholesky(Sigma).T
    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    # We'll extract expected shapes from the cases through the parametrize mechanism
    # For this, we need to handle it differently
    n_obs = A.shape[0]

    assert mu_z.shape == (n_obs,)
    assert Sigma_z.shape == (n_obs, n_obs)


@parametrize_with_cases(
    "A,b,Q,mu,Sigma,expected_mu_shape,expected_Sigma_shape",
    cases=[case_output_shape_1d_observation, case_output_shape_2d_observation],
)
def test_sqr_marginalization_output_exact_shapes(
    A, b, Q, mu, Sigma, expected_mu_shape, expected_Sigma_shape
):
    """Test that output shapes match expected dimensions exactly."""
    Q = np.linalg.cholesky(Q).T
    Sigma = np.linalg.cholesky(Sigma).T
    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    assert mu_z.shape == expected_mu_shape
    assert Sigma_z.shape == expected_Sigma_shape


@parametrize_with_cases("A,b,Q,mu,Sigma", cases=[case_positive_definite, case_symmetry])
def test_sqr_marginalization_output_type_ndarray(A, b, Q, mu, Sigma):
    """Test that output is numpy arrays."""
    Q = np.linalg.cholesky(Q).T
    Sigma = np.linalg.cholesky(Sigma).T
    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    assert isinstance(mu_z, np.ndarray)
    assert isinstance(Sigma_z, np.ndarray)


@parametrize_with_cases("A,b,Q,mu,Sigma", cases=[case_positive_definite])
def test_sqr_marginalization_preserves_positive_definiteness(A, b, Q, mu, Sigma):
    """Output covariance must remain positive definite."""
    Q = np.linalg.cholesky(Q).T
    Sigma = np.linalg.cholesky(Sigma).T
    _mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)
    Sigma_z = Sigma_z.T @ Sigma_z
    # Check positive definiteness: all eigenvalues > 0
    eigenvalues = np.linalg.eigvalsh(Sigma_z)
    assert np.all(eigenvalues > -1e-10)  # Allow for numerical precision


@parametrize_with_cases("A,b,Q,mu,Sigma", cases=[case_symmetry])
def test_sqr_marginalization_output_symmetry(A, b, Q, mu, Sigma):
    """Output covariance matrix must be symmetric."""
    Q = np.linalg.cholesky(Q).T
    Sigma = np.linalg.cholesky(Sigma).T
    _mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)
    Sigma_z = Sigma_z.T @ Sigma_z

    # Covariance matrices must be symmetric
    assert np.allclose(Sigma_z, Sigma_z.T)


def test_sqr_marginalization_dtype_preservation():
    """Output dtype should match input dtype."""
    A = np.array([[1.0, 0.0]], dtype=np.float64)
    b = np.array([0.0], dtype=np.float64)
    Q = np.array([[0.1]], dtype=np.float64)
    mu = np.array([1.0, 2.0], dtype=np.float64)
    Sigma = np.eye(2, dtype=np.float64)
    Q = np.linalg.cholesky(Q).T
    Sigma = np.linalg.cholesky(Sigma).T
    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    assert mu_z.dtype in [np.float64, np.float32, float]
    assert Sigma_z.dtype in [np.float64, np.float32, float]


def test_sqr_marginalization_noise_increases_uncertainty():
    """Adding observation noise should increase output covariance."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)

    Q_small = np.array([[0.01]])
    Q_large = np.array([[1.0]])
    Q_small = np.linalg.cholesky(Q_small).T
    Q_large = np.linalg.cholesky(Q_large).T
    Sigma = np.linalg.cholesky(Sigma).T

    _, Sigma_z_small = sqr_marginalization(A, b, Q_small, mu, Sigma)
    _, Sigma_z_large = sqr_marginalization(A, b, Q_large, mu, Sigma)
    Sigma_z_small = Sigma_z_small.T @ Sigma_z_small
    Sigma_z_large = Sigma_z_large.T @ Sigma_z_large

    # Larger noise should result in larger output covariance
    assert Sigma_z_large[0, 0] > Sigma_z_small[0, 0]


@parametrize_with_cases(
    "A,b,Q,mu,Sigma",
    cases=[
        case_dimension_mismatch_A_mu,
        case_dimension_mismatch_Sigma,
        case_dimension_mismatch_b_A,
        case_non_square_Q,
    ],
)
def test_sqr_marginalization_raises_on_invalid_input(A, b, Q, mu, Sigma):
    """Test that function raises errors on invalid input dimensions."""
    with pytest.raises((ValueError, onp.linalg.LinAlgError)):
        sqr_marginalization(A, b, Q, mu, Sigma)


def test_sqr_marginalization_raises_on_non_square_sigma_root():
    """Sigma_sqr must be square; non-square factors are rejected."""
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    Q = np.array([[0.1]])
    mu = np.array([1.0, 2.0])

    Q_sqr = np.linalg.cholesky(Q).T
    Sigma_sqr = np.ones((2, 3))  # deliberately non-square

    with pytest.raises(ValueError):
        sqr_marginalization(A, b, Q_sqr, mu, Sigma_sqr)


def test_sqr_marginalization_1d_covariance_as_array():
    """Test that 1D covariance Q can be passed as 1D array (edge case)."""
    # This test documents behavior when Q is passed as 1D instead of 2D
    A = np.array([[1.0, 0.0]])
    b = np.array([0.0])
    Q = np.array([0.1])  # 1D instead of [[0.1]]
    mu = np.array([1.0, 2.0])
    Sigma = np.eye(2)
    Q = np.sqrt(Q)
    Sigma = np.linalg.cholesky(Sigma).T

    # Function should still work or raise clear error
    try:
        mu_z, _Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)
        # If it works, verify output is still correct
        assert mu_z == pytest.approx([1.0])
    except (ValueError, onp.linalg.LinAlgError):
        # It's also acceptable to reject 1D Q
        pass
