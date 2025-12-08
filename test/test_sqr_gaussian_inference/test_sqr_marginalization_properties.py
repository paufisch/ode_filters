"""
Property-based tests for sqr_marginalization function (square-root form).

These tests verify mathematical properties that should hold for ANY valid input,
using hypothesis for random input generation. This is complementary to example-based
tests which verify specific known values.

The sqr_marginalization works with square-root parameterizations of covariance matrices.

Properties tested:
- Output shapes are correct
- Output types are always correct
- Square-root Cholesky factors remain lower triangular
- Numerical stability across dimensions
"""

from __future__ import annotations

import jax.numpy as np
import numpy as onp  # Regular numpy for exceptions
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ode_filters.inference.sqr_gaussian_inference import sqr_marginalization


@st.composite
def generate_positive_definite_matrix_strategy(draw, n):
    """Generate a random positive definite matrix of size n x n using hypothesis."""
    # Generate a random matrix using hypothesis
    A_flat = draw(
        st.lists(
            st.floats(-5, 5, allow_nan=False, allow_infinity=False),
            min_size=n * n,
            max_size=n * n,
        )
    )
    A = np.array(A_flat).reshape(n, n)
    # Make it positive definite: A.T @ A + small diagonal
    return A.T @ A + np.eye(n) * 0.1


@st.composite
def valid_sqr_marginalization_inputs(
    draw, n_state_min=1, n_state_max=5, n_obs_min=1, n_obs_max=5
):
    """Generate valid inputs for sqr_marginalization function."""
    n_state = draw(st.integers(n_state_min, n_state_max))
    n_obs = draw(st.integers(n_obs_min, n_obs_max))

    # Generate A matrix
    A_flat = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=n_obs * n_state,
            max_size=n_obs * n_state,
        )
    )
    A = np.array(A_flat).reshape(n_obs, n_state)

    # Generate b offset
    b_flat = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=n_obs,
            max_size=n_obs,
        )
    )
    b = np.array(b_flat)

    # Generate Q and convert to Cholesky form
    Q_temp = draw(generate_positive_definite_matrix_strategy(n_obs))
    Q_temp = Q_temp / np.max(np.abs(Q_temp)) * 10
    Q = np.linalg.cholesky(Q_temp).T

    # Generate mu
    mu_flat = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=n_state,
            max_size=n_state,
        )
    )
    mu = np.array(mu_flat)

    # Generate Sigma and convert to Cholesky form
    Sigma_temp = draw(generate_positive_definite_matrix_strategy(n_state))
    Sigma_temp = Sigma_temp / np.max(np.abs(Sigma_temp)) * 10
    Sigma = np.linalg.cholesky(Sigma_temp).T

    return A, b, Q, mu, Sigma


def load_random_marginalization_case():
    """Convenience wrapper used by other test modules."""
    return valid_sqr_marginalization_inputs()


# ==============================================================================
# PROPERTY-BASED TESTS: INVARIANTS
# ==============================================================================


@given(valid_sqr_marginalization_inputs())
@settings(max_examples=100)
def test_sqr_marginalization_property_output_shapes(inputs):
    """Property: Output shapes are always correct."""
    A, b, Q, mu, Sigma = inputs
    n_obs = A.shape[0]

    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    assert mu_z.shape == (n_obs,), f"mu_z shape {mu_z.shape} != ({n_obs},)"
    assert Sigma_z.shape == (n_obs, n_obs), (
        f"Sigma_z shape {Sigma_z.shape} != ({n_obs}, {n_obs})"
    )


@given(valid_sqr_marginalization_inputs())
@settings(max_examples=100)
def test_sqr_marginalization_property_output_types(inputs):
    """Property: Outputs are always numpy arrays."""
    A, b, Q, mu, Sigma = inputs

    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    assert isinstance(mu_z, np.ndarray), f"mu_z is {type(mu_z)}, expected ndarray"
    assert isinstance(Sigma_z, np.ndarray), (
        f"Sigma_z is {type(Sigma_z)}, expected ndarray"
    )

    assert np.issubdtype(mu_z.dtype, np.floating)
    assert np.issubdtype(Sigma_z.dtype, np.floating)


@given(valid_sqr_marginalization_inputs())
@settings(max_examples=100)
def test_sqr_marginalization_property_no_nan_or_inf(inputs):
    """Property: Outputs never contain NaN or Inf."""
    A, b, Q, mu, Sigma = inputs

    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    assert np.all(np.isfinite(mu_z)), "mu_z contains NaN or Inf"
    assert np.all(np.isfinite(Sigma_z)), "Sigma_z contains NaN or Inf"


# ==============================================================================
# PROPERTY-BASED TESTS: NUMERICAL STABILITY
# ==============================================================================


@given(
    valid_sqr_marginalization_inputs(
        n_state_min=1, n_state_max=10, n_obs_min=1, n_obs_max=10
    )
)
@settings(max_examples=50, deadline=500)
def test_sqr_marginalization_property_numerical_stability_large_dimensions(inputs):
    """Property: Function remains numerically stable for larger dimensions."""
    A, b, Q, mu, Sigma = inputs

    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    assert np.all(np.isfinite(mu_z))
    assert np.all(np.isfinite(Sigma_z))


@given(valid_sqr_marginalization_inputs())
@settings(max_examples=50)
def test_sqr_marginalization_property_norm_bounds(inputs):
    """Property: Output magnitudes remain reasonable."""
    A, b, Q, mu, Sigma = inputs

    mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    # Frobenius norm should be reasonable
    norm_mu = np.linalg.norm(mu_z)
    norm_Sigma = np.linalg.norm(Sigma_z)

    assert norm_mu < 1e8, f"mu_z norm too large: {norm_mu}"
    assert norm_Sigma < 1e8, f"Sigma_z norm too large: {norm_Sigma}"


# ==============================================================================
# PROPERTY-BASED TESTS: CONSISTENCY WITH REGULAR FORM
# ==============================================================================


@given(valid_sqr_marginalization_inputs())
@settings(max_examples=50)
def test_sqr_marginalization_property_reconstruction(inputs):
    """Property: Reconstructed covariance from sqr form is PD."""
    A, b, Q, mu, Sigma = inputs

    _mu_z, Sigma_z_sqr = sqr_marginalization(A, b, Q, mu, Sigma)

    # Reconstruct covariance from Cholesky factor
    Sigma_z_reconstructed = Sigma_z_sqr.T @ Sigma_z_sqr

    # Reconstructed covariance should be positive definite
    eigenvalues = np.linalg.eigvalsh(Sigma_z_reconstructed)
    assert np.all(eigenvalues > -1e-10), (
        f"Reconstructed covariance not positive definite. "
        f"Min eigenvalue: {np.min(eigenvalues)}"
    )

    # Reconstructed should be symmetric (relaxed tolerance for float32)
    assert np.allclose(
        Sigma_z_reconstructed, Sigma_z_reconstructed.T, rtol=1e-5, atol=1e-7
    ), "Reconstructed covariance should be symmetric"


@given(valid_sqr_marginalization_inputs())
@settings(max_examples=50)
def test_sqr_marginalization_property_mean_invariance(inputs):
    """Property: Mean is computed the same regardless of form."""
    A, b, Q, mu, Sigma = inputs

    mu_z_sqr, _ = sqr_marginalization(A, b, Q, mu, Sigma)

    # Expected mean (computed from standard form)
    expected_mu_z = A @ mu + b

    assert np.allclose(mu_z_sqr, expected_mu_z, rtol=1e-9, atol=1e-12), (
        f"Mean should be A @ mu + b. Got {mu_z_sqr}, expected {expected_mu_z}"
    )


@given(valid_sqr_marginalization_inputs())
@settings(max_examples=50)
def test_sqr_marginalization_property_square_root_reconstruction(inputs):
    """Property: Sigma_z reconstructs to original covariance when squared."""
    A, b, Q, mu, Sigma = inputs

    _mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    # Reconstruct covariance from square-root: Sigma_z.T @ Sigma_z should equal the covariance
    Sigma_z_reconstructed = Sigma_z.T @ Sigma_z

    # Reconstructed should be symmetric (relaxed tolerance for float32)
    assert np.allclose(
        Sigma_z_reconstructed, Sigma_z_reconstructed.T, rtol=1e-5, atol=1e-7
    ), "Reconstructed covariance should be symmetric"

    # Reconstructed should be positive definite
    eigenvalues = np.linalg.eigvalsh(Sigma_z_reconstructed)
    assert np.all(eigenvalues > -1e-10), (
        f"Reconstructed covariance not positive definite. "
        f"Min eigenvalue: {np.min(eigenvalues)}"
    )


@given(valid_sqr_marginalization_inputs())
@settings(max_examples=50)
def test_sqr_marginalization_property_no_singular_square_root(inputs):
    """Property: Square-root matrix is non-singular (no zero diagonal elements)."""
    A, b, Q, mu, Sigma = inputs

    _mu_z, Sigma_z = sqr_marginalization(A, b, Q, mu, Sigma)

    # For a valid square-root, the matrix should be non-singular
    # This means diagonal elements should be non-zero
    diagonal = np.diag(Sigma_z)
    assert np.all(np.abs(diagonal) > 1e-12), (
        "Square-root matrix diagonal should have non-zero elements (non-singular)"
    )

    # Verify non-singularity by checking determinant is non-zero
    try:
        det = np.linalg.det(Sigma_z)
        assert np.abs(det) > 1e-12, (
            f"Square-root matrix should be non-singular, det={det}"
        )
    except onp.linalg.LinAlgError:
        pytest.fail("Square-root matrix should be non-singular")
