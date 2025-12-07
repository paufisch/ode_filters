"""Tests for ODEconservation and ODEconservationmeasurement factory functions."""

import jax.numpy as np
import pytest

from ode_filters.measurement.measurement_models import (
    Conservation,
    ODEconservation,
    ODEconservationmeasurement,
)


def make_projection_matrices(d: int, q: int):
    """Create E0 and E1 projection matrices for given d and q."""
    eye_d = np.eye(d)
    basis = np.eye(q + 1)
    E0 = np.kron(basis[0:1], eye_d)
    E1 = np.kron(basis[1:2], eye_d)
    return E0, E1


class TestODEconservationConstruction:
    """Tests for ODEconservation constructor."""

    def test_basic_construction(self):
        """Test basic construction of ODEconservation."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 1.0]])  # Conservation: x1 + x2 = const
        p = np.array([1.0])

        model = ODEconservation(vf=vf, E0=E0, E1=E1, A=A, p=p)
        # Model has one Conservation constraint
        assert len(model._constraints) == 1
        assert isinstance(model._constraints[0], Conservation)

    def test_rejects_mismatched_A_p_shapes(self):
        """Test that mismatched A and p shapes raise error."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 1.0], [1.0, -1.0]])  # 2 constraints
        p = np.array([1.0])  # Only 1 value

        with pytest.raises(ValueError, match="must match"):
            ODEconservation(vf=vf, E0=E0, E1=E1, A=A, p=p)

    def test_rejects_mismatched_A_E0_shapes(self):
        """Test that mismatched A and E0 shapes raise error."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 1.0, 1.0]])  # 3 columns but d=2
        p = np.array([1.0])

        with pytest.raises(ValueError):
            ODEconservation(vf=vf, E0=E0, E1=E1, A=A, p=p)


class TestODEconservationMethods:
    """Tests for ODEconservation methods."""

    @pytest.fixture
    def conservation_model(self):
        """Create a standard ODEconservation for testing."""

        def vf(x, *, t):
            return -x  # Simple decay

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 1.0]])  # Conservation: x1 + x2 = const
        p = np.array([2.0])  # x1 + x2 = 2

        return ODEconservation(vf=vf, E0=E0, E1=E1, A=A, p=p)

    def test_g_returns_combined_residual(self, conservation_model):
        """Test that g returns ODE residual plus conservation residual."""
        # State: [x1, x1', x2, x2'] = [1.0, 0.5, 1.0, -0.5]
        state = np.array([1.0, 0.5, 1.0, -0.5])
        result = conservation_model.g(state, t=0.0)

        # Should have d + k = 2 + 1 = 3 components
        assert result.shape == (3,)

    def test_g_conservation_constraint_satisfied(self, conservation_model):
        """Test g when conservation constraint is satisfied."""
        # State layout for d=2, q=1: [x1, x2, x1', x2']
        # x1 + x2 = 2 (satisfied), with derivatives x' = -x (matching vf)
        state = np.array([1.0, 1.0, -1.0, -1.0])  # x=[1,1], x'=[-1,-1]
        result = conservation_model.g(state, t=0.0)

        # ODE residual: x' - vf(x) = x' - (-x) = x' + x
        # For x1: -1 + 1 = 0, x2: -1 + 1 = 0
        # Conservation: 1 + 1 - 2 = 0
        assert np.allclose(result, np.zeros(3), atol=1e-6)

    def test_jacobian_g_returns_correct_shape(self, conservation_model):
        """Test that jacobian_g returns correct shape."""
        state = np.array([1.0, 0.5, 1.0, -0.5])
        jacobian = conservation_model.jacobian_g(state, t=0.0)

        # Shape should be (d + k, (q+1)*d) = (3, 4)
        assert jacobian.shape == (3, 4)

    def test_get_noise_returns_correct_shape(self, conservation_model):
        """Test that get_noise returns correct shape."""
        R = conservation_model.get_noise(t=0.0)

        # Shape should be (d + k, d + k) = (3, 3)
        assert R.shape == (3, 3)


class TestODEconservationmeasurement:
    """Tests for ODEconservationmeasurement class."""

    @pytest.fixture
    def measurement_model(self):
        """Create ODEconservationmeasurement for testing."""

        def vf(x, *, t):
            return -x

        E0, E1 = make_projection_matrices(d=1, q=1)
        C = np.array([[1.0]])  # Conservation constraint
        p = np.array([1.0])
        A = np.array([[1.0]])  # Observation matrix
        z = np.array([[0.5], [0.8]])  # Measurements
        z_t = np.array([0.5, 1.0])  # Measurement times

        return ODEconservationmeasurement(
            vf=vf, E0=E0, E1=E1, C=C, p=p, A=A, z=z, z_t=z_t
        )

    def test_g_without_measurement(self, measurement_model):
        """Test g at time without measurement."""
        state = np.array([1.0, 0.5])
        result = measurement_model.g(state, t=0.0)  # No measurement at t=0

        # Should only have ODE + conservation info (d + k = 1 + 1 = 2)
        assert result.shape == (2,)

    def test_g_with_measurement(self, measurement_model):
        """Test g at time with measurement."""
        state = np.array([1.0, 0.5])
        result = measurement_model.g(state, t=0.5)  # Measurement at t=0.5

        # Should have ODE + conservation + measurement info (d + k + k_meas = 1 + 1 + 1 = 3)
        assert result.shape == (3,)

    def test_jacobian_g_without_measurement(self, measurement_model):
        """Test jacobian_g at time without measurement."""
        state = np.array([1.0, 0.5])
        jacobian = measurement_model.jacobian_g(state, t=0.0)

        # Shape: (d + k, (q+1)*d) = (2, 2)
        assert jacobian.shape == (2, 2)

    def test_jacobian_g_with_measurement(self, measurement_model):
        """Test jacobian_g at time with measurement."""
        state = np.array([1.0, 0.5])
        jacobian = measurement_model.jacobian_g(state, t=0.5)

        # Shape: (d + k + k_meas, (q+1)*d) = (3, 2)
        assert jacobian.shape == (3, 2)

    def test_get_noise_without_measurement(self, measurement_model):
        """Test get_noise at time without measurement."""
        R = measurement_model.get_noise(t=0.0)

        # Shape: (d + k, d + k) = (2, 2)
        assert R.shape == (2, 2)

    def test_get_noise_with_measurement(self, measurement_model):
        """Test get_noise at time with measurement."""
        R = measurement_model.get_noise(t=0.5)

        # Shape: (d + k + k_meas, d + k + k_meas) = (3, 3)
        assert R.shape == (3, 3)
