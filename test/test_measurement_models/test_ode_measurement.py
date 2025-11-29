"""Tests for ODEmeasurement class and LinearMeasurementBase mixin."""

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.measurement.measurement_models import (
    ODEInformation,
    ODEmeasurement,
)


def make_projection_matrices(d: int, q: int):
    """Create E0 and E1 projection matrices for given d and q."""
    eye_d = np.eye(d)
    basis = np.eye(q + 1)
    E0 = np.kron(basis[0:1], eye_d)
    E1 = np.kron(basis[1:2], eye_d)
    return E0, E1


class TestODEmeasurementConstruction:
    """Tests for ODEmeasurement constructor."""

    def test_basic_construction(self):
        """Test basic construction of ODEmeasurement."""
        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5], [0.8]])
        z_t = np.array([0.5, 1.0])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)
        assert model._A_meas.shape == (1, 1)
        assert model._z_meas.shape == (2, 1)
        assert model._z_t_meas.shape == (2,)

    def test_rejects_invalid_A_shape(self):
        """Test that invalid A shape raises error."""
        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([1.0])  # 1D instead of 2D
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        with pytest.raises(ValueError, match="must be 2D"):
            ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

    def test_rejects_mismatched_z_shape(self):
        """Test that mismatched z shape raises error."""
        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])  # k=1
        z = np.array([[0.5, 0.3]])  # Shape (1, 2) but should be (n, k)=(n, 1)
        z_t = np.array([0.5])

        with pytest.raises(ValueError, match="must be 2D"):
            ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

    def test_rejects_mismatched_z_t_length(self):
        """Test that mismatched z_t length raises error."""
        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5], [0.8]])  # 2 measurements
        z_t = np.array([0.5])  # Only 1 time

        with pytest.raises(ValueError, match="must match"):
            ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

    def test_accepts_2d_z_t(self):
        """Test that 2D z_t with shape (n, 1) is accepted."""
        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([[0.5]])  # Shape (1, 1)

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)
        assert model._z_t_meas.shape == (1,)


class TestODEmeasurementMethods:
    """Tests for ODEmeasurement methods."""

    @pytest.fixture
    def measurement_model(self):
        """Create a standard ODEmeasurement for testing."""
        def vf(x, *, t):
            return -x  # Simple decay

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])  # Direct observation
        z = np.array([[0.5], [0.8], [0.3]])  # 3 measurements
        z_t = np.array([0.5, 1.0, 1.5])  # Measurement times

        return ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

    def test_g_without_measurement(self, measurement_model):
        """Test g at time without measurement."""
        state = np.array([1.0, 0.5])
        result = measurement_model.g(state, t=0.0)  # No measurement at t=0

        # Should only have ODE info (dimension d=1)
        assert result.shape == (1,)

    def test_g_with_measurement(self, measurement_model):
        """Test g at time with measurement."""
        state = np.array([1.0, 0.5])
        result = measurement_model.g(state, t=0.5)  # Measurement at t=0.5

        # Should have ODE + measurement info (d + k = 1 + 1 = 2)
        assert result.shape == (2,)

    def test_g_measurement_residual_correct(self, measurement_model):
        """Test that measurement residual is computed correctly."""
        state = np.array([0.5, -0.5])  # x=0.5, x'=-0.5 (consistent with vf)
        result = measurement_model.g(state, t=0.5)

        # ODE residual: x' - (-x) = -0.5 + 0.5 = 0
        # Measurement residual: A @ E0 @ state - z = 0.5 - 0.5 = 0
        assert np.allclose(result, np.zeros(2), atol=1e-6)

    def test_jacobian_g_without_measurement(self, measurement_model):
        """Test jacobian_g at time without measurement."""
        state = np.array([1.0, 0.5])
        jacobian = measurement_model.jacobian_g(state, t=0.0)

        # Shape: (d, (q+1)*d) = (1, 2)
        assert jacobian.shape == (1, 2)

    def test_jacobian_g_with_measurement(self, measurement_model):
        """Test jacobian_g at time with measurement."""
        state = np.array([1.0, 0.5])
        jacobian = measurement_model.jacobian_g(state, t=0.5)

        # Shape: (d + k, (q+1)*d) = (2, 2)
        assert jacobian.shape == (2, 2)

    def test_get_noise_without_measurement(self, measurement_model):
        """Test get_noise at time without measurement."""
        R = measurement_model.get_noise(t=0.0)

        # Shape: (d, d) = (1, 1)
        assert R.shape == (1, 1)

    def test_get_noise_with_measurement(self, measurement_model):
        """Test get_noise at time with measurement."""
        R = measurement_model.get_noise(t=0.5)

        # Shape: (d + k, d + k) = (2, 2)
        assert R.shape == (2, 2)


class TestODEInformationLinearize:
    """Tests for the linearize method."""

    def test_linearize_returns_H_and_c(self):
        """Test that linearize returns H and c matrices."""
        def vf(x, *, t):
            return x ** 2

        E0, E1 = make_projection_matrices(d=1, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)
        state = np.array([1.0, 0.5])

        H, c = model.linearize(state, t=0.0)

        assert H.shape == (1, 2)  # (d, (q+1)*d)
        assert c.shape == (1,)  # (d,)

    def test_linearize_satisfies_affine_approximation(self):
        """Test that H @ state + c ≈ g(state)."""
        def vf(x, *, t):
            return x ** 2

        E0, E1 = make_projection_matrices(d=1, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)
        state = np.array([1.0, 0.5])

        H, c = model.linearize(state, t=0.0)
        g_val = model.g(state, t=0.0)

        # At the linearization point: H @ state + c = g(state)
        reconstructed = H @ state + c
        assert np.allclose(reconstructed, g_val)

    def test_linearize_H_matches_jacobian(self):
        """Test that H from linearize matches jacobian_g."""
        def vf(x, *, t):
            return x ** 2

        E0, E1 = make_projection_matrices(d=1, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)
        state = np.array([1.0, 0.5])

        H, _ = model.linearize(state, t=0.0)
        jacobian = model.jacobian_g(state, t=0.0)

        assert np.allclose(H, jacobian)


class TestODEInformationGetNoise:
    """Tests for get_noise method."""

    def test_get_noise_returns_R(self):
        """Test that get_noise returns _R matrix."""
        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        R = model.get_noise(t=0.0)
        assert R.shape == (2, 2)
        assert np.allclose(R, model._R)

    def test_noise_can_be_modified(self):
        """Test that noise matrix can be modified."""
        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        # Modify noise
        model._R = model._R.at[0, 0].set(0.1)

        R = model.get_noise(t=0.0)
        assert R[0, 0] == pytest.approx(0.1)


class TestMultiDimensionalMeasurement:
    """Tests for multi-dimensional measurement scenarios."""

    def test_2d_state_with_measurements(self):
        """Test ODEmeasurement with 2D state."""
        def vf(x, *, t):
            return -x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 0.0], [0.0, 1.0]])  # Full state observation
        z = np.array([[0.5, 0.3]])
        z_t = np.array([1.0])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

        state = np.array([1.0, 0.5, 0.8, 0.2])
        result = model.g(state, t=1.0)

        # d + k = 2 + 2 = 4
        assert result.shape == (4,)

    def test_partial_observation(self):
        """Test ODEmeasurement with partial observation."""
        def vf(x, *, t):
            return -x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 0.0]])  # Observe only first component
        z = np.array([[0.5]])
        z_t = np.array([1.0])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

        state = np.array([1.0, 0.5, 0.8, 0.2])
        result = model.g(state, t=1.0)

        # d + k = 2 + 1 = 3
        assert result.shape == (3,)

