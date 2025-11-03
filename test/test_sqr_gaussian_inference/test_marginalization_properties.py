"""
Property-based tests for marginalization function.

These tests verify mathematical properties that should hold for ANY valid input,
using hypothesis for random input generation. This is complementary to example-based
tests which verify specific known values.

Properties tested:
- Output shapes depend only on input dimensions
- Output covariance is always positive definite
- Output covariance is always symmetric
- Observation noise increases output uncertainty
- Output types are always numpy arrays
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ode_filters.gaussian_inference import marginalization

# ==============================================================================
# HYPOTHESIS STRATEGIES FOR VALID INPUTS
# ==============================================================================


def generate_positive_definite_matrix(n):
    """Generate a random positive definite matrix of size n x n."""
    A = np.random.randn(n, n)
    return A.T @ A + np.eye(n) * 0.1  # Add small diagonal for stability


@st.composite
def valid_marginalization_inputs(
    draw, n_state_min=1, n_state_max=5, n_obs_min=1, n_obs_max=5
):
    """Generate valid inputs for marginalization function."""
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

    # Generate positive definite Q matrix
    Q_temp = generate_positive_definite_matrix(n_obs)
    Q = Q_temp / np.max(np.abs(Q_temp)) * 10  # Scale for numerical stability

    # Generate positive definite Sigma
    Sigma_temp = generate_positive_definite_matrix(n_state)
    Sigma = Sigma_temp / np.max(np.abs(Sigma_temp)) * 10

    # Generate mu
    mu_flat = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=n_state,
            max_size=n_state,
        )
    )
    mu = np.array(mu_flat)

    return A, b, Q, mu, Sigma


# ==============================================================================
# PROPERTY-BASED TESTS: INVARIANTS THAT SHOULD ALWAYS HOLD
# ==============================================================================


@given(valid_marginalization_inputs())
@settings(max_examples=100)
def test_marginalization_property_output_shapes(inputs):
    """Property: Output shapes depend predictably on input dimensions."""
    A, b, Q, mu, Sigma = inputs
    n_obs = A.shape[0]

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Shape of mu_z should be [n_obs]
    assert mu_z.shape == (n_obs,), f"mu_z shape {mu_z.shape} != ({n_obs},)"

    # Shape of Sigma_z should be [n_obs, n_obs]
    assert Sigma_z.shape == (n_obs, n_obs), (
        f"Sigma_z shape {Sigma_z.shape} != ({n_obs}, {n_obs})"
    )


@given(valid_marginalization_inputs())
@settings(max_examples=100)
def test_marginalization_property_output_types(inputs):
    """Property: Outputs are always numpy arrays."""
    A, b, Q, mu, Sigma = inputs

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    assert isinstance(mu_z, np.ndarray), f"mu_z is {type(mu_z)}, expected ndarray"
    assert isinstance(Sigma_z, np.ndarray), (
        f"Sigma_z is {type(Sigma_z)}, expected ndarray"
    )

    # Should be float type
    assert np.issubdtype(mu_z.dtype, np.floating)
    assert np.issubdtype(Sigma_z.dtype, np.floating)


@given(valid_marginalization_inputs())
@settings(max_examples=100)
def test_marginalization_property_covariance_symmetry(inputs):
    """Property: Output covariance Sigma_z is always symmetric."""
    A, b, Q, mu, Sigma = inputs

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Covariance should be symmetric
    assert np.allclose(Sigma_z, Sigma_z.T, rtol=1e-10, atol=1e-12), (
        "Sigma_z must be symmetric"
    )


@given(valid_marginalization_inputs())
@settings(max_examples=100)
def test_marginalization_property_covariance_positive_definite(inputs):
    """Property: Output covariance Sigma_z is always positive definite."""
    A, b, Q, mu, Sigma = inputs

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Compute eigenvalues
    eigenvalues = np.linalg.eigvalsh(Sigma_z)

    # All eigenvalues should be non-negative (positive semi-definite at minimum)
    assert np.all(eigenvalues > -1e-10), (
        f"Sigma_z not positive definite. Min eigenvalue: {np.min(eigenvalues)}"
    )


@given(valid_marginalization_inputs())
@settings(max_examples=100)
def test_marginalization_property_no_nan_or_inf(inputs):
    """Property: Outputs never contain NaN or Inf."""
    A, b, Q, mu, Sigma = inputs

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    assert np.all(np.isfinite(mu_z)), "mu_z contains NaN or Inf"
    assert np.all(np.isfinite(Sigma_z)), "Sigma_z contains NaN or Inf"


@given(valid_marginalization_inputs())
@settings(max_examples=50)
def test_marginalization_property_linear_mean_transformation(inputs):
    """Property: Mean is transformed according to the linear model z = Ax + b."""
    A, b, Q, mu, Sigma = inputs

    mu_z, _ = marginalization(A, b, Q, mu, Sigma)

    # Expected mean: E[z] = E[Ax + b] = A @ E[x] + b = A @ mu + b
    expected_mu_z = A @ mu + b

    assert np.allclose(mu_z, expected_mu_z, rtol=1e-10, atol=1e-12), (
        f"Mean not computed correctly. Got {mu_z}, expected {expected_mu_z}"
    )


@given(valid_marginalization_inputs())
@settings(max_examples=50)
def test_marginalization_property_covariance_formula(inputs):
    """Property: Covariance follows the formula Sigma_z = A @ Sigma @ A.T + Q."""
    A, b, Q, mu, Sigma = inputs

    _, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Expected covariance: Var[z] = Var[Ax + noise] = A @ Sigma @ A.T + Q
    expected_Sigma_z = A @ Sigma @ A.T + Q

    assert np.allclose(Sigma_z, expected_Sigma_z, rtol=1e-9, atol=1e-12), (
        "Covariance not computed correctly"
    )


# ==============================================================================
# PROPERTY-BASED TESTS: COMPARATIVE PROPERTIES
# ==============================================================================


@given(valid_marginalization_inputs())
@settings(max_examples=50)
def test_marginalization_property_noise_increases_variance(inputs):
    """Property: Larger observation noise increases output variance."""
    A, b, Q, mu, Sigma = inputs

    # Test with original Q
    _, Sigma_z_1 = marginalization(A, b, Q, mu, Sigma)

    # Test with scaled up Q (more noise)
    Q_scaled = Q * 2
    _, Sigma_z_2 = marginalization(A, b, Q_scaled, mu, Sigma)

    # Trace of covariance should increase with more noise
    trace_1 = np.trace(Sigma_z_1)
    trace_2 = np.trace(Sigma_z_2)

    assert trace_2 >= trace_1 - 1e-10, (
        f"Larger noise should increase trace. "
        f"trace(Q) -> trace(Q*2): {trace_1} -> {trace_2}"
    )


@given(valid_marginalization_inputs())
@settings(max_examples=50)
def test_marginalization_property_scaling_mean(inputs):
    """Property: Mean scales linearly with input scaling."""
    A, b, Q, mu, Sigma = inputs

    mu_z_1, _ = marginalization(A, b, Q, mu, Sigma)

    # Scale inputs
    scale = 2.5
    mu_z_2, _ = marginalization(A, b * scale, Q, mu * scale, Sigma)

    # Mean should scale: z = A(scale*x) + scale*b = scale*(Ax + b) = scale*z_original
    expected_mu_z_2 = scale * mu_z_1

    assert np.allclose(mu_z_2, expected_mu_z_2, rtol=1e-9, atol=1e-12), (
        f"Mean should scale linearly. Got {mu_z_2}, expected {expected_mu_z_2}"
    )


@given(valid_marginalization_inputs())
@settings(max_examples=50)
def test_marginalization_property_identity_observation(inputs):
    """Property: With identity observation (A=I), marginalization is identity."""
    _, b, Q, mu, Sigma = inputs
    n_state = Sigma.shape[0]
    n_obs = b.shape[0]

    # Only test when n_obs == n_state
    assume(n_obs == n_state)

    A = np.eye(n_state)

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # mu_z should equal mu + b
    assert np.allclose(mu_z, mu + b, rtol=1e-10, atol=1e-12)

    # Sigma_z should equal Sigma + Q
    assert np.allclose(Sigma_z, Sigma + Q, rtol=1e-10, atol=1e-12)


# ==============================================================================
# PROPERTY-BASED TESTS: NUMERICAL STABILITY
# ==============================================================================


@given(
    valid_marginalization_inputs(
        n_state_min=1, n_state_max=10, n_obs_min=1, n_obs_max=10
    )
)
@settings(max_examples=50)
def test_marginalization_property_numerical_stability_large_dimensions(inputs):
    """Property: Function remains numerically stable for larger dimensions."""
    A, b, Q, mu, Sigma = inputs

    # Should not raise or produce NaN/Inf
    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    assert np.all(np.isfinite(mu_z))
    assert np.all(np.isfinite(Sigma_z))

    # Eigenvalues should be reasonable
    eigenvalues = np.linalg.eigvalsh(Sigma_z)
    assert np.all(eigenvalues > -1e-8)


@given(valid_marginalization_inputs())
@settings(max_examples=50)
def test_marginalization_property_determinant_condition_number(inputs):
    """Property: Output covariance has reasonable conditioning."""
    A, b, Q, mu, Sigma = inputs

    _, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Compute condition number
    try:
        cond = np.linalg.cond(Sigma_z)
        # Condition number should be finite and not too large (< 1e10)
        assert np.isfinite(cond) and cond < 1e10, f"Condition number too large: {cond}"
    except np.linalg.LinAlgError:
        # If singular, that's acceptable
        pass
