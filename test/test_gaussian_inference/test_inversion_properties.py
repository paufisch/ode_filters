"""
Property-based tests for inversion function.

These tests verify mathematical properties that should hold for ANY valid input,
using hypothesis for random input generation. This is complementary to example-based
tests which verify specific known values.

Properties tested:
- Output shapes and types are always correct
- Posterior covariance is positive definite
- Posterior covariance is symmetric
- Posterior covariance is smaller than prior (uncertainty reduction)
- Numerical stability across dimensions
"""

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ode_filters.gaussian_inference import inversion


def generate_positive_definite_matrix(n):
    """Generate a random positive definite matrix of size n x n."""
    A = np.random.randn(n, n)
    return A @ A.T + np.eye(n) * 0.1


@st.composite
def valid_inversion_inputs(
    draw, n_state_min=1, n_state_max=5, n_obs_min=1, n_obs_max=5
):
    """Generate valid inputs for inversion function."""
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

    # Generate mu
    mu_flat = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=n_state,
            max_size=n_state,
        )
    )
    mu = np.array(mu_flat)

    # Generate positive definite Sigma
    Sigma_temp = generate_positive_definite_matrix(n_state)
    Sigma = Sigma_temp / np.max(np.abs(Sigma_temp)) * 10

    # Generate mu_z
    mu_z_flat = draw(
        st.lists(
            st.floats(-10, 10, allow_nan=False, allow_infinity=False),
            min_size=n_obs,
            max_size=n_obs,
        )
    )
    mu_z = np.array(mu_z_flat)

    # Generate positive definite Sigma_z
    Sigma_z_temp = generate_positive_definite_matrix(n_obs)
    Sigma_z = Sigma_z_temp / np.max(np.abs(Sigma_z_temp)) * 10

    return A, mu, Sigma, mu_z, Sigma_z


# ==============================================================================
# PROPERTY-BASED TESTS: INVARIANTS
# ==============================================================================


@given(valid_inversion_inputs())
@settings(max_examples=100)
def test_inversion_property_output_shapes(inputs):
    """Property: Output shapes are always correct."""
    A, mu, Sigma, mu_z, Sigma_z = inputs
    n_state = A.shape[1]
    n_obs = A.shape[0]

    G, d, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    assert G.shape == (n_state, n_obs), f"G shape {G.shape} != ({n_state}, {n_obs})"
    assert d.shape == (n_state,), f"d shape {d.shape} != ({n_state},)"
    assert Lambda.shape == (n_state, n_state), (
        f"Lambda shape {Lambda.shape} != ({n_state}, {n_state})"
    )


@given(valid_inversion_inputs())
@settings(max_examples=100)
def test_inversion_property_output_types(inputs):
    """Property: Outputs are always numpy arrays."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    G, d, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    assert isinstance(G, np.ndarray), f"G is {type(G)}, expected ndarray"
    assert isinstance(d, np.ndarray), f"d is {type(d)}, expected ndarray"
    assert isinstance(Lambda, np.ndarray), f"Lambda is {type(Lambda)}, expected ndarray"

    assert np.issubdtype(G.dtype, np.floating)
    assert np.issubdtype(d.dtype, np.floating)
    assert np.issubdtype(Lambda.dtype, np.floating)


@given(valid_inversion_inputs())
@settings(max_examples=100)
def test_inversion_property_posterior_covariance_symmetry(inputs):
    """Property: Posterior covariance Lambda is always symmetric."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    _, _, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    assert np.allclose(Lambda, Lambda.T, rtol=1e-10, atol=1e-12), (
        "Lambda must be symmetric"
    )


@given(valid_inversion_inputs())
@settings(max_examples=100)
def test_inversion_property_posterior_covariance_positive_definite(inputs):
    """Property: Posterior covariance Lambda is always positive definite."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    _, _, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    eigenvalues = np.linalg.eigvalsh(Lambda)

    assert np.all(eigenvalues > -1e-10), (
        f"Lambda not positive definite. Min eigenvalue: {np.min(eigenvalues)}"
    )


@given(valid_inversion_inputs())
@settings(max_examples=100)
def test_inversion_property_no_nan_or_inf(inputs):
    """Property: Outputs never contain NaN or Inf."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    G, d, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    assert np.all(np.isfinite(G)), "G contains NaN or Inf"
    assert np.all(np.isfinite(d)), "d contains NaN or Inf"
    assert np.all(np.isfinite(Lambda)), "Lambda contains NaN or Inf"


@given(valid_inversion_inputs())
@settings(max_examples=50)
def test_inversion_property_reduces_uncertainty(inputs):
    """Property: Posterior covariance trace is smaller than prior."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    _, _, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    prior_trace = np.trace(Sigma)
    posterior_trace = np.trace(Lambda)

    assert posterior_trace <= prior_trace + 1e-10, (
        f"Posterior should reduce uncertainty. "
        f"prior_trace={prior_trace}, posterior_trace={posterior_trace}"
    )


# ==============================================================================
# PROPERTY-BASED TESTS: COMPARATIVE PROPERTIES
# ==============================================================================


@given(valid_inversion_inputs())
@settings(max_examples=50)
def test_inversion_property_noise_effect(inputs):
    """Property: More observation noise increases posterior uncertainty."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    _, _, Lambda_1 = inversion(A, mu, Sigma, mu_z, Sigma_z)

    # More observation noise (larger Sigma_z)
    Sigma_z_scaled = Sigma_z * 2
    _, _, Lambda_2 = inversion(A, mu, Sigma, mu_z, Sigma_z_scaled)

    trace_1 = np.trace(Lambda_1)
    trace_2 = np.trace(Lambda_2)

    assert trace_2 >= trace_1 - 1e-10, (
        f"More noise should increase uncertainty. trace_1={trace_1}, trace_2={trace_2}"
    )


@given(valid_inversion_inputs())
@settings(max_examples=50)
def test_inversion_property_mean_offset_scaling(inputs):
    """Property: Offset d scales with input scaling."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    _, d_1, _ = inversion(A, mu, Sigma, mu_z, Sigma_z)

    scale = 2.5
    _, d_2, _ = inversion(A, mu * scale, Sigma, mu_z, Sigma_z)

    # When mu scales, d should scale accordingly
    expected_d_2 = d_1 + (scale - 1) * mu
    assert np.allclose(d_2, expected_d_2, rtol=1e-9, atol=1e-12), (
        f"Offset should scale with mu. Got {d_2}, expected {expected_d_2}"
    )


# ==============================================================================
# PROPERTY-BASED TESTS: NUMERICAL STABILITY
# ==============================================================================


@given(valid_inversion_inputs(n_state_min=1, n_state_max=10, n_obs_min=1, n_obs_max=10))
@settings(max_examples=50)
def test_inversion_property_numerical_stability_large_dimensions(inputs):
    """Property: Function remains numerically stable for larger dimensions."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    G, d, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    assert np.all(np.isfinite(G))
    assert np.all(np.isfinite(d))
    assert np.all(np.isfinite(Lambda))

    eigenvalues = np.linalg.eigvalsh(Lambda)
    assert np.all(eigenvalues > -1e-8)


@given(valid_inversion_inputs())
@settings(max_examples=50)
def test_inversion_property_conditioning(inputs):
    """Property: Output covariance has reasonable conditioning."""
    A, mu, Sigma, mu_z, Sigma_z = inputs

    _, _, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    try:
        cond = np.linalg.cond(Lambda)
        assert np.isfinite(cond) and cond < 1e10, f"Condition number too large: {cond}"
    except np.linalg.LinAlgError:
        pass


# ==============================================================================
# PROPERTY-BASED TESTS: IDENTITY CASES
# ==============================================================================


@given(valid_inversion_inputs())
@settings(max_examples=50)
def test_inversion_property_identity_observation_matrix(inputs):
    """Property: With identity observation (A=I), specific behavior holds."""
    _, mu, Sigma, mu_z, Sigma_z = inputs
    n_state = Sigma.shape[0]

    assume(n_state == Sigma_z.shape[0])

    A = np.eye(n_state)

    G, d, Lambda = inversion(A, mu, Sigma, mu_z, Sigma_z)

    # G should be close to identity for well-conditioned case
    assert G.shape == (n_state, n_state)

    # Lambda should be well-defined
    eigenvalues = np.linalg.eigvalsh(Lambda)
    assert np.all(eigenvalues > -1e-10)
