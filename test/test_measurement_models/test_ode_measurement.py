"""Tests for ODEmeasurement and ODE information classes."""

import jax.numpy as np
import pytest

from ode_filters.measurement.measurement_models import (
    Conservation,
    Measurement,
    ODEInformation,
    ODEInformationWithHidden,
    ODEmeasurement,
    SecondOrderODEInformationWithHidden,
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
        # Model now uses constraints internally
        assert len(model._constraints) == 1
        assert isinstance(model._constraints[0], Measurement)

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
        # Should work without raising an error
        assert len(model._constraints) == 1

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
        """Test that get_noise returns R matrix."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        R = model.get_noise(t=0.0)
        assert R.shape == (2, 2)
        assert np.allclose(R, model.R)

    def test_noise_can_be_modified(self):
        """Test that noise matrix can be modified."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=1, q=1)
        model = ODEInformation(vf=vf, E0=E0, E1=E1)

        # Modify noise via property
        model.R = 0.1

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

        R_meas = model.get_noise(t=0.5)
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

        R_meas = model.get_noise(t=0.5)
        assert R_meas[1, 1] == pytest.approx(0.01)


class TestMeasurementDataclass:
    """Tests for the Measurement dataclass."""

    def test_measurement_creation(self):
        """Test creating a Measurement constraint."""
        A = np.array([[1.0]])
        z = np.array([[0.5], [0.8]])
        z_t = np.array([0.5, 1.0])

        m = Measurement(A=A, z=z, z_t=z_t)
        assert m.dim == 1
        assert m.find_index(0.5) == 0
        assert m.find_index(1.0) == 1
        assert m.find_index(0.0) is None

    def test_measurement_residual(self):
        """Test measurement residual computation."""
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        m = Measurement(A=A, z=z, z_t=z_t)
        x = np.array([0.8])
        residual = m.residual(x, t=0.5)
        assert residual is not None
        assert np.allclose(residual, np.array([0.3]))

    def test_measurement_noise_scalar(self):
        """Test measurement noise from scalar."""
        A = np.array([[1.0], [0.0]])
        z = np.array([[0.5, 0.3]])
        z_t = np.array([0.5])

        m = Measurement(A=A, z=z, z_t=z_t, noise=0.1)
        R = m.get_noise_matrix()
        assert R.shape == (2, 2)
        assert np.allclose(R, 0.1 * np.eye(2))


class TestConservationDataclass:
    """Tests for the Conservation dataclass."""

    def test_conservation_creation(self):
        """Test creating a Conservation constraint."""
        A = np.array([[1.0, 1.0]])
        p = np.array([2.0])

        c = Conservation(A=A, p=p)
        assert c.dim == 1

    def test_conservation_residual(self):
        """Test conservation residual computation."""
        A = np.array([[1.0, 1.0]])
        p = np.array([2.0])

        c = Conservation(A=A, p=p)
        x = np.array([1.0, 1.0])
        residual = c.residual(x)
        assert np.allclose(residual, np.array([0.0]))

    def test_conservation_jacobian(self):
        """Test conservation Jacobian."""
        A = np.array([[1.0, 1.0]])
        p = np.array([2.0])

        c = Conservation(A=A, p=p)
        jac = c.jacobian()
        assert np.allclose(jac, A)


class TestComposableConstraints:
    """Tests for composing constraints directly."""

    def test_ode_with_multiple_constraints(self):
        """Test ODEInformation with both conservation and measurement."""

        def vf(x, *, t):
            return -x

        E0, E1 = make_projection_matrices(d=2, q=1)
        conservation = Conservation(
            A=np.array([[1.0, 1.0]]),  # x1 + x2 = const
            p=np.array([2.0]),
        )
        measurement = Measurement(
            A=np.array([[1.0, 0.0]]),  # observe x1
            z=np.array([[0.5]]),
            z_t=np.array([0.5]),
            noise=0.01,
        )

        model = ODEInformation(vf, E0, E1, constraints=[conservation, measurement])

        # At t=0 (no measurement): ODE (2) + conservation (1) = 3
        state = np.array([1.0, 1.0, -1.0, -1.0])
        g_t0 = model.g(state, t=0.0)
        assert g_t0.shape == (3,)

        # At t=0.5 (with measurement): ODE (2) + conservation (1) + measurement (1) = 4
        g_t05 = model.g(state, t=0.5)
        assert g_t05.shape == (4,)


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


class TestHiddenStates:
    """Tests for ODE models with hidden states using separate classes."""

    def test_first_order_with_hidden_state(self):
        """Test first-order ODE with hidden parameter: dx/dt = -u * x."""

        # Vector field with hidden parameter
        def vf(x, u, *, t):
            return -u * x

        # State is [x, x', u, u'] for q=1, d_x=1, d_u=1
        # Joint state dimension: (q+1)*d_x + (q+1)*d_u = 2 + 2 = 4
        d_x, d_u, q = 1, 1, 1
        D_x = (q + 1) * d_x  # 2
        D_u = (q + 1) * d_u  # 2
        D = D_x + D_u  # 4

        # Build projection matrices for joint state [x_block, u_block]
        # E0 extracts x (first d_x elements of x_block)
        E0 = np.zeros((d_x, D))
        E0 = E0.at[0, 0].set(1.0)

        # E1 extracts x' (derivative part of x_block)
        E1 = np.zeros((d_x, D))
        E1 = E1.at[0, 1].set(1.0)

        # E0_hidden extracts u (first d_u elements of u_block)
        E0_hidden = np.zeros((d_u, D))
        E0_hidden = E0_hidden.at[0, 2].set(1.0)

        model = ODEInformationWithHidden(vf, E0, E1, E0_hidden)

        # State: [x=1, x'=-0.5, u=0.5, u'=0]
        # At equilibrium: x' = -u*x = -0.5*1 = -0.5 ✓
        state = np.array([1.0, -0.5, 0.5, 0.0])

        # Residual should be zero at equilibrium
        g_val = model.g(state, t=0.0)
        assert g_val.shape == (1,)
        assert np.allclose(g_val, np.zeros(1), atol=1e-6)

    def test_first_order_hidden_jacobian(self):
        """Test Jacobian computation with hidden state."""

        def vf(x, u, *, t):
            return -u * x

        d_x, d_u, q = 1, 1, 1
        D = (q + 1) * (d_x + d_u)

        E0 = np.zeros((d_x, D))
        E0 = E0.at[0, 0].set(1.0)
        E1 = np.zeros((d_x, D))
        E1 = E1.at[0, 1].set(1.0)
        E0_hidden = np.zeros((d_u, D))
        E0_hidden = E0_hidden.at[0, 2].set(1.0)

        model = ODEInformationWithHidden(vf, E0, E1, E0_hidden)

        state = np.array([1.0, -0.5, 0.5, 0.0])
        jacobian = model.jacobian_g(state, t=0.0)

        # Jacobian shape: (d_x, D) = (1, 4)
        assert jacobian.shape == (1, 4)

        # Manual computation:
        # g = x' - vf(x, u) = x' + u*x
        # dg/d[x, x', u, u'] = [u, 1, x, 0] = [0.5, 1, 1, 0]
        expected = np.array([[0.5, 1.0, 1.0, 0.0]])
        assert np.allclose(jacobian, expected, atol=1e-5)

    def test_second_order_with_hidden_state(self):
        """Test second-order ODE with hidden parameter: d²x/dt² = -omega²*x - u*v."""

        # Damped oscillator with unknown damping coefficient
        def vf(x, v, u, *, t):
            omega = 1.0
            return -(omega**2) * x - u * v

        # State: [x, x', x'', u, u', u''] for q=2, d_x=1, d_u=1
        d_x, d_u, q = 1, 1, 2
        D_x = (q + 1) * d_x  # 3
        D_u = (q + 1) * d_u  # 3
        D = D_x + D_u  # 6

        # E0 extracts x
        E0 = np.zeros((d_x, D))
        E0 = E0.at[0, 0].set(1.0)

        # E1 extracts x'
        E1 = np.zeros((d_x, D))
        E1 = E1.at[0, 1].set(1.0)

        # E2 extracts x''
        E2 = np.zeros((d_x, D))
        E2 = E2.at[0, 2].set(1.0)

        # E0_hidden extracts u
        E0_hidden = np.zeros((d_u, D))
        E0_hidden = E0_hidden.at[0, 3].set(1.0)

        model = SecondOrderODEInformationWithHidden(vf, E0, E1, E2, E0_hidden)

        # State: x=1, x'=0, x''=-1, u=0, u'=0, u''=0
        # vf = -1*1 - 0*0 = -1, so residual = x'' - vf = -1 - (-1) = 0
        state = np.array([1.0, 0.0, -1.0, 0.0, 0.0, 0.0])

        g_val = model.g(state, t=0.0)
        assert g_val.shape == (1,)
        assert np.allclose(g_val, np.zeros(1), atol=1e-6)

    def test_hidden_with_measurement_constraint(self):
        """Test hidden states combined with measurement constraints."""

        def vf(x, u, *, t):
            return -u * x

        d_x, d_u, q = 1, 1, 1
        D = (q + 1) * (d_x + d_u)

        E0 = np.zeros((d_x, D))
        E0 = E0.at[0, 0].set(1.0)
        E1 = np.zeros((d_x, D))
        E1 = E1.at[0, 1].set(1.0)
        E0_hidden = np.zeros((d_u, D))
        E0_hidden = E0_hidden.at[0, 2].set(1.0)

        # Measurement constraint on x
        measurement = Measurement(
            A=np.array([[1.0]]),
            z=np.array([[0.5]]),
            z_t=np.array([0.5]),
            noise=0.01,
        )

        model = ODEInformationWithHidden(
            vf, E0, E1, E0_hidden, constraints=[measurement]
        )

        state = np.array([1.0, -0.5, 0.5, 0.0])

        # At t=0: only ODE constraint (d_x=1)
        g_t0 = model.g(state, t=0.0)
        assert g_t0.shape == (1,)

        # At t=0.5: ODE (1) + measurement (1) = 2
        g_t05 = model.g(state, t=0.5)
        assert g_t05.shape == (2,)

    def test_second_order_hidden_jacobian(self):
        """Test Jacobian computation for second-order with hidden state."""

        def vf(x, v, u, *, t):
            return -x - u * v

        d_x, d_u, q = 1, 1, 2
        D = (q + 1) * (d_x + d_u)

        E0 = np.zeros((d_x, D))
        E0 = E0.at[0, 0].set(1.0)
        E1 = np.zeros((d_x, D))
        E1 = E1.at[0, 1].set(1.0)
        E2 = np.zeros((d_x, D))
        E2 = E2.at[0, 2].set(1.0)
        E0_hidden = np.zeros((d_u, D))
        E0_hidden = E0_hidden.at[0, 3].set(1.0)

        model = SecondOrderODEInformationWithHidden(vf, E0, E1, E2, E0_hidden)

        # State: x=1, x'=0.5, x''=-1, u=2, u'=0, u''=0
        state = np.array([1.0, 0.5, -1.0, 2.0, 0.0, 0.0])
        jacobian = model.jacobian_g(state, t=0.0)

        # Jacobian shape: (d_x, D) = (1, 6)
        assert jacobian.shape == (1, 6)

        # g = x'' - vf(x, v, u) = x'' + x + u*v
        # dg/dx = 1, dg/dv = u = 2, dg/du = v = 0.5
        # dg/d[x, x', x'', u, u', u''] = [1, 2, 1, 0.5, 0, 0]
        expected = np.array([[1.0, 2.0, 1.0, 0.5, 0.0, 0.0]])
        assert np.allclose(jacobian, expected, atol=1e-5)
