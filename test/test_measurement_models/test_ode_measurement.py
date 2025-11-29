"""Tests for ODEmeasurement class and LinearMeasurementBase mixin."""

import jax.numpy as np
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

    def test_rejects_invalid_2d_z_t(self):
        """Test that 2D z_t with shape (n, k) where k > 1 raises error."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([[0.5, 0.6]])  # Shape (1, 2) - invalid

        with pytest.raises(ValueError, match="must be 1D shape"):
            ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)


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
            return x**2

        E0, E1 = make_projection_matrices(d=1, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)
        state = np.array([1.0, 0.5])

        H, c = model.linearize(state, t=0.0)

        assert H.shape == (1, 2)  # (d, (q+1)*d)
        assert c.shape == (1,)  # (d,)

    def test_linearize_satisfies_affine_approximation(self):
        """Test that H @ state + c ≈ g(state)."""

        def vf(x, *, t):
            return x**2

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
            return x**2

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


class TestNoisePropertyAndSetters:
    """Tests for R property and setter methods."""

    def test_R_property_returns_noise_matrix(self):
        """Test that R property returns the noise matrix."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        assert model.R.shape == (2, 2)
        assert np.allclose(model.R, np.zeros((2, 2)))

    def test_R_setter_with_matrix(self):
        """Test that R setter accepts a full matrix."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        new_R = np.array([[0.1, 0.0], [0.0, 0.2]])
        model.R = new_R

        assert np.allclose(model.R, new_R)

    def test_R_setter_with_vector(self):
        """Test that R setter accepts a vector for diagonal."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        model.R = np.array([0.1, 0.2])

        assert model.R[0, 0] == pytest.approx(0.1)
        assert model.R[1, 1] == pytest.approx(0.2)
        assert model.R[0, 1] == pytest.approx(0.0)

    def test_R_setter_with_scalar(self):
        """Test that R setter accepts a scalar for uniform diagonal."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        model.R = 0.5

        assert model.R[0, 0] == pytest.approx(0.5)
        assert model.R[1, 1] == pytest.approx(0.5)
        assert model.R[0, 1] == pytest.approx(0.0)

    def test_R_setter_rejects_wrong_matrix_shape(self):
        """Test that R setter rejects incorrect matrix shapes."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        with pytest.raises(ValueError, match="must have shape"):
            model.R = np.array([[0.1]])

    def test_R_setter_rejects_wrong_vector_length(self):
        """Test that R setter rejects incorrect vector length."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        with pytest.raises(ValueError, match="must have length"):
            model.R = np.array([0.1])

    def test_R_setter_rejects_3d_array(self):
        """Test that R setter rejects 3D+ arrays."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        with pytest.raises(ValueError, match="must be scalar, 1D, or 2D"):
            model.R = np.array([[[0.1]]])


class TestMeasurementNoiseDefaults:
    """Tests for default measurement noise in ODEmeasurement."""

    def test_default_measurement_noise_is_nonzero(self):
        """Test that default measurement noise is non-zero."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

        R_meas = model.R_measure
        # ODE part (1x1) should be zero, measurement part should be non-zero
        assert R_meas[0, 0] == pytest.approx(0.0)  # ODE noise
        assert R_meas[1, 1] == pytest.approx(1e-6)  # Default measurement noise

    def test_custom_measurement_noise_at_construction(self):
        """Test custom measurement noise at construction."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = ODEmeasurement(
            vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t, measurement_noise=0.01
        )

        R_meas = model.R_measure
        assert R_meas[1, 1] == pytest.approx(0.01)

    def test_set_measurement_noise_scalar(self):
        """Test setting measurement noise with a scalar."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)
        model.set_measurement_noise(0.05)

        assert model.R_measure[1, 1] == pytest.approx(0.05)

    def test_set_measurement_noise_vector(self):
        """Test setting measurement noise with a vector."""

        def vf(x, *, t):
            return -x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2D measurement
        z = np.array([[0.5, 0.3]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)
        model.set_measurement_noise(np.array([0.01, 0.02]))

        # ODE part is 2x2, measurement part is 2x2
        assert model.R_measure[2, 2] == pytest.approx(0.01)
        assert model.R_measure[3, 3] == pytest.approx(0.02)

    def test_set_measurement_noise_matrix(self):
        """Test setting measurement noise with a full matrix."""

        def vf(x, *, t):
            return -x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        z = np.array([[0.5, 0.3]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)
        noise_matrix = np.array([[0.1, 0.01], [0.01, 0.2]])
        model.set_measurement_noise(noise_matrix)

        assert model.R_measure[2, 2] == pytest.approx(0.1)
        assert model.R_measure[3, 3] == pytest.approx(0.2)
        assert model.R_measure[2, 3] == pytest.approx(0.01)

    def test_R_measure_setter_full_matrix(self):
        """Test setting the full R_measure matrix via setter."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)
        new_R = np.array([[0.01, 0.0], [0.0, 0.1]])
        model.R_measure = new_R

        assert np.allclose(model.R_measure, new_R)

    def test_R_measure_setter_rejects_wrong_shape(self):
        """Test that R_measure setter rejects incorrect shapes."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

        with pytest.raises(ValueError, match="must have shape"):
            model.R_measure = np.array([[0.1]])

    def test_set_measurement_noise_rejects_wrong_vector_length(self):
        """Test that set_measurement_noise rejects wrong vector length."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

        with pytest.raises(ValueError, match="must have length"):
            model.set_measurement_noise(
                np.array([0.1, 0.2])
            )  # 2 values but only 1 measurement

    def test_set_measurement_noise_rejects_wrong_matrix_shape(self):
        """Test that set_measurement_noise rejects wrong matrix shape."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

        with pytest.raises(ValueError, match="must have shape"):
            model.set_measurement_noise(
                np.array([[0.1, 0.0], [0.0, 0.2]])
            )  # 2x2 but only 1 measurement

    def test_set_measurement_noise_rejects_invalid_ndim(self):
        """Test that set_measurement_noise rejects 3D+ arrays."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = ODEmeasurement(vf=vf, E0=E0, E1=E1, A=A, z=z, z_t=z_t)

        with pytest.raises(ValueError, match="must be scalar, 1D, or 2D"):
            model.set_measurement_noise(np.array([[[0.1]]]))  # 3D array


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
