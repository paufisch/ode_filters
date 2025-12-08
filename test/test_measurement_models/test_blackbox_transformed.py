"""Tests for BlackBoxMeasurement, TransformedMeasurement, and SecondOrderODE classes."""

import jax
import jax.numpy as np
import pytest

from ode_filters.measurement.measurement_models import (
    BlackBoxMeasurement,
    Conservation,
    Measurement,
    ODEInformation,
    SecondOrderODEconservation,
    SecondOrderODEconservationmeasurement,
    SecondOrderODEInformation,
    SecondOrderODEmeasurement,
    TransformedMeasurement,
)


def make_projection_matrices(d: int, q: int):
    """Create E0, E1, E2 projection matrices for given d and q."""
    eye_d = np.eye(d)
    basis = np.eye(q + 1)
    E0 = np.kron(basis[0:1], eye_d)
    E1 = np.kron(basis[1:2], eye_d)
    E2 = np.kron(basis[2:3], eye_d) if q >= 2 else None
    return E0, E1, E2


# =============================================================================
# BlackBoxMeasurement Tests
# =============================================================================


class TestBlackBoxMeasurementConstruction:
    """Tests for BlackBoxMeasurement constructor."""

    def test_basic_construction(self):
        """Test basic construction of BlackBoxMeasurement."""

        def g_func(state, *, t):
            return state[:2]

        model = BlackBoxMeasurement(g_func, state_dim=4, obs_dim=2)
        assert model._state_dim == 4
        assert model._obs_dim == 2

    def test_construction_with_scalar_noise(self):
        """Test construction with scalar noise."""

        def g_func(state, *, t):
            return state[:2]

        model = BlackBoxMeasurement(g_func, state_dim=4, obs_dim=2, noise=0.1)
        assert model.R.shape == (2, 2)
        assert np.allclose(model.R, 0.1 * np.eye(2))

    def test_construction_with_vector_noise(self):
        """Test construction with diagonal noise vector."""

        def g_func(state, *, t):
            return state[:2]

        model = BlackBoxMeasurement(
            g_func, state_dim=4, obs_dim=2, noise=np.array([0.1, 0.2])
        )
        assert model.R[0, 0] == pytest.approx(0.1)
        assert model.R[1, 1] == pytest.approx(0.2)

    def test_construction_with_matrix_noise(self):
        """Test construction with full noise matrix."""

        def g_func(state, *, t):
            return state[:2]

        noise_matrix = np.array([[0.1, 0.05], [0.05, 0.2]])
        model = BlackBoxMeasurement(g_func, state_dim=4, obs_dim=2, noise=noise_matrix)
        assert np.allclose(model.R, noise_matrix)

    def test_rejects_non_callable_g_func(self):
        """Test that non-callable g_func raises error."""
        with pytest.raises(TypeError, match="must be callable"):
            BlackBoxMeasurement("not_a_function", state_dim=4, obs_dim=2)

    def test_rejects_invalid_state_dim(self):
        """Test that invalid state_dim raises error."""

        def g_func(state, *, t):
            return state[:2]

        with pytest.raises(ValueError, match="must be a positive integer"):
            BlackBoxMeasurement(g_func, state_dim=0, obs_dim=2)

        with pytest.raises(ValueError, match="must be a positive integer"):
            BlackBoxMeasurement(g_func, state_dim=-1, obs_dim=2)

    def test_rejects_invalid_obs_dim(self):
        """Test that invalid obs_dim raises error."""

        def g_func(state, *, t):
            return state[:2]

        with pytest.raises(ValueError, match="must be a positive integer"):
            BlackBoxMeasurement(g_func, state_dim=4, obs_dim=0)

    def test_rejects_mismatched_vector_noise(self):
        """Test that mismatched vector noise raises error."""

        def g_func(state, *, t):
            return state[:2]

        with pytest.raises(ValueError, match="must have length"):
            BlackBoxMeasurement(g_func, state_dim=4, obs_dim=2, noise=np.array([0.1]))

    def test_rejects_mismatched_matrix_noise(self):
        """Test that mismatched matrix noise raises error."""

        def g_func(state, *, t):
            return state[:2]

        with pytest.raises(ValueError, match="must have shape"):
            BlackBoxMeasurement(g_func, state_dim=4, obs_dim=2, noise=np.array([[0.1]]))

    def test_rejects_3d_noise(self):
        """Test that 3D noise raises error."""

        def g_func(state, *, t):
            return state[:2]

        with pytest.raises(ValueError, match="must be scalar, 1D, or 2D"):
            BlackBoxMeasurement(
                g_func, state_dim=4, obs_dim=2, noise=np.array([[[0.1]]])
            )


class TestBlackBoxMeasurementMethods:
    """Tests for BlackBoxMeasurement methods."""

    @pytest.fixture
    def bb_model(self):
        """Create a BlackBoxMeasurement for testing."""

        def g_func(state, *, t):
            # Nonlinear observation: squared first component + second component
            return np.array([state[0] ** 2, state[1]])

        return BlackBoxMeasurement(g_func, state_dim=4, obs_dim=2, noise=0.01)

    def test_g_evaluation(self, bb_model):
        """Test g function evaluation."""
        state = np.array([2.0, 3.0, 1.0, 1.0])
        result = bb_model.g(state, t=0.0)

        assert result.shape == (2,)
        assert result[0] == pytest.approx(4.0)  # 2^2
        assert result[1] == pytest.approx(3.0)

    def test_jacobian_g_computation(self, bb_model):
        """Test Jacobian computation via autodiff."""
        state = np.array([2.0, 3.0, 1.0, 1.0])
        jacobian = bb_model.jacobian_g(state, t=0.0)

        assert jacobian.shape == (2, 4)
        # d(x^2)/dx = 2x = 4, d(x^2)/dy = 0
        assert jacobian[0, 0] == pytest.approx(4.0)
        assert jacobian[0, 1] == pytest.approx(0.0)
        # d(y)/dx = 0, d(y)/dy = 1
        assert jacobian[1, 0] == pytest.approx(0.0)
        assert jacobian[1, 1] == pytest.approx(1.0)

    def test_get_noise(self, bb_model):
        """Test get_noise returns R matrix."""
        R = bb_model.get_noise(t=0.0)
        assert R.shape == (2, 2)
        assert np.allclose(R, 0.01 * np.eye(2))

    def test_linearize(self, bb_model):
        """Test linearize method."""
        state = np.array([2.0, 3.0, 1.0, 1.0])
        H, c = bb_model.linearize(state, t=0.0)

        assert H.shape == (2, 4)
        assert c.shape == (2,)

        # Verify H @ state + c = g(state)
        g_val = bb_model.g(state, t=0.0)
        reconstructed = H @ state + c
        assert np.allclose(reconstructed, g_val)

    def test_R_setter_scalar(self, bb_model):
        """Test R setter with scalar."""
        bb_model.R = 0.5
        assert np.allclose(bb_model.R, 0.5 * np.eye(2))

    def test_R_setter_vector(self, bb_model):
        """Test R setter with vector."""
        bb_model.R = np.array([0.1, 0.2])
        assert bb_model.R[0, 0] == pytest.approx(0.1)
        assert bb_model.R[1, 1] == pytest.approx(0.2)

    def test_R_setter_matrix(self, bb_model):
        """Test R setter with matrix."""
        new_R = np.array([[0.1, 0.05], [0.05, 0.2]])
        bb_model.R = new_R
        assert np.allclose(bb_model.R, new_R)

    def test_R_setter_rejects_wrong_vector_length(self, bb_model):
        """Test R setter rejects wrong vector length."""
        with pytest.raises(ValueError, match="must have length"):
            bb_model.R = np.array([0.1])

    def test_R_setter_rejects_wrong_matrix_shape(self, bb_model):
        """Test R setter rejects wrong matrix shape."""
        with pytest.raises(ValueError, match="must have shape"):
            bb_model.R = np.array([[0.1]])

    def test_R_setter_rejects_3d_array(self, bb_model):
        """Test R setter rejects 3D array."""
        with pytest.raises(ValueError, match="must be scalar, 1D, or 2D"):
            bb_model.R = np.array([[[0.1]]])

    def test_validate_state_wrong_dimension(self, bb_model):
        """Test that wrong state dimension raises error."""
        state = np.array([1.0, 2.0])  # Wrong length
        with pytest.raises(ValueError, match="must have length"):
            bb_model.g(state, t=0.0)

    def test_validate_state_wrong_rank(self, bb_model):
        """Test that non-1D state raises error."""
        state = np.array([[1.0, 2.0, 3.0, 4.0]])  # 2D instead of 1D
        with pytest.raises(ValueError, match="must be a one-dimensional"):
            bb_model.g(state, t=0.0)


# =============================================================================
# TransformedMeasurement Tests
# =============================================================================


class TestTransformedMeasurementConstruction:
    """Tests for TransformedMeasurement constructor."""

    def test_basic_construction_with_autodiff(self):
        """Test basic construction with autodiff Jacobian."""

        def vf(x, *, t):
            return -x

        E0, E1, _ = make_projection_matrices(d=2, q=1)
        base = ODEInformation(vf, E0, E1)

        def sigma(state):
            return state * 2  # Simple scaling

        model = TransformedMeasurement(base, sigma)
        assert model._use_autodiff is True

    def test_construction_with_explicit_jacobian(self):
        """Test construction with explicit Jacobian."""

        def vf(x, *, t):
            return -x

        E0, E1, _ = make_projection_matrices(d=2, q=1)
        base = ODEInformation(vf, E0, E1)

        def sigma(state):
            return state * 2

        def sigma_jacobian(state):
            return 2 * np.eye(state.shape[0])

        model = TransformedMeasurement(
            base, sigma, use_autodiff_jacobian=False, sigma_jacobian=sigma_jacobian
        )
        assert model._use_autodiff is False

    def test_rejects_non_callable_sigma(self):
        """Test that non-callable sigma raises error."""

        def vf(x, *, t):
            return -x

        E0, E1, _ = make_projection_matrices(d=2, q=1)
        base = ODEInformation(vf, E0, E1)

        with pytest.raises(TypeError, match="must be callable"):
            TransformedMeasurement(base, "not_a_function")

    def test_rejects_missing_explicit_jacobian(self):
        """Test that missing explicit Jacobian raises error when autodiff disabled."""

        def vf(x, *, t):
            return -x

        E0, E1, _ = make_projection_matrices(d=2, q=1)
        base = ODEInformation(vf, E0, E1)

        def sigma(state):
            return state * 2

        with pytest.raises(ValueError, match="Must provide 'sigma_jacobian'"):
            TransformedMeasurement(base, sigma, use_autodiff_jacobian=False)


class TestTransformedMeasurementMethods:
    """Tests for TransformedMeasurement methods."""

    @pytest.fixture
    def transformed_model(self):
        """Create a TransformedMeasurement for testing."""

        def vf(x, *, t):
            return -x

        E0, E1, _ = make_projection_matrices(d=2, q=1)
        base = ODEInformation(vf, E0, E1)

        def sigma(state):
            # Apply softplus to first 2 components
            return state.at[:2].set(jax.nn.softplus(state[:2]))

        return TransformedMeasurement(base, sigma)

    def test_g_evaluation(self, transformed_model):
        """Test g function evaluation on transformed state."""
        state = np.array([1.0, 1.0, 0.5, 0.5])
        result = transformed_model.g(state, t=0.0)

        # Should return ODE residual for transformed state
        assert result.shape == (2,)

    def test_jacobian_g_with_chain_rule(self, transformed_model):
        """Test Jacobian computation includes chain rule."""
        state = np.array([1.0, 1.0, 0.5, 0.5])
        jacobian = transformed_model.jacobian_g(state, t=0.0)

        assert jacobian.shape == (2, 4)

    def test_get_noise(self, transformed_model):
        """Test get_noise delegates to base model."""
        R = transformed_model.get_noise(t=0.0)
        assert R.shape == (2, 2)

    def test_linearize(self, transformed_model):
        """Test linearize method."""
        state = np.array([1.0, 1.0, 0.5, 0.5])
        H, c = transformed_model.linearize(state, t=0.0)

        assert H.shape == (2, 4)
        assert c.shape == (2,)

        # Verify H @ state + c = g(state)
        g_val = transformed_model.g(state, t=0.0)
        reconstructed = H @ state + c
        assert np.allclose(reconstructed, g_val)

    def test_R_property(self, transformed_model):
        """Test R property delegates to base model."""
        R = transformed_model.R
        assert R.shape == (2, 2)

    def test_R_setter(self, transformed_model):
        """Test R setter modifies base model."""
        transformed_model.R = 0.1
        assert np.allclose(transformed_model.R, 0.1 * np.eye(2))

    def test_with_blackbox_base(self):
        """Test TransformedMeasurement with BlackBoxMeasurement base."""

        def g_func(state, *, t):
            return state[:2]

        base = BlackBoxMeasurement(g_func, state_dim=4, obs_dim=2, noise=0.01)

        def sigma(state):
            return state**2

        model = TransformedMeasurement(base, sigma)

        state = np.array([2.0, 3.0, 1.0, 1.0])
        result = model.g(state, t=0.0)

        # g(sigma(state)) = sigma(state)[:2] = [4, 9]
        assert result.shape == (2,)
        assert result[0] == pytest.approx(4.0)
        assert result[1] == pytest.approx(9.0)

    def test_with_explicit_jacobian(self):
        """Test using explicit Jacobian instead of autodiff."""

        def vf(x, *, t):
            return -x

        E0, E1, _ = make_projection_matrices(d=2, q=1)
        base = ODEInformation(vf, E0, E1)

        def sigma(state):
            return state * 2

        def sigma_jacobian(state):
            return 2 * np.eye(state.shape[0])

        model = TransformedMeasurement(
            base, sigma, use_autodiff_jacobian=False, sigma_jacobian=sigma_jacobian
        )

        state = np.array([1.0, 1.0, 0.5, 0.5])
        jacobian = model.jacobian_g(state, t=0.0)

        assert jacobian.shape == (2, 4)


# =============================================================================
# SecondOrderODEInformation Tests
# =============================================================================


class TestSecondOrderODEInformation:
    """Tests for SecondOrderODEInformation class."""

    def test_basic_construction(self):
        """Test basic construction of SecondOrderODEInformation."""

        def vf(x, v, *, t):
            return -x - 0.1 * v  # Damped oscillator

        E0, E1, E2 = make_projection_matrices(d=1, q=2)
        model = SecondOrderODEInformation(vf, E0, E1, E2)

        assert model._d == 1
        assert model._state_dim == 3

    def test_g_evaluation(self):
        """Test g function evaluation."""

        def vf(x, v, *, t):
            return -x  # Simple harmonic oscillator

        E0, E1, E2 = make_projection_matrices(d=1, q=2)
        model = SecondOrderODEInformation(vf, E0, E1, E2)

        # State: [x, x', x''] = [1, 0, -1]
        # vf(1, 0) = -1, so residual = x'' - vf = -1 - (-1) = 0
        state = np.array([1.0, 0.0, -1.0])
        result = model.g(state, t=0.0)

        assert result.shape == (1,)
        assert np.allclose(result, np.zeros(1), atol=1e-6)

    def test_jacobian_g(self):
        """Test Jacobian computation."""

        def vf(x, v, *, t):
            return -x - 0.5 * v

        E0, E1, E2 = make_projection_matrices(d=1, q=2)
        model = SecondOrderODEInformation(vf, E0, E1, E2)

        state = np.array([1.0, 2.0, -1.5])
        jacobian = model.jacobian_g(state, t=0.0)

        assert jacobian.shape == (1, 3)
        # g = x'' - vf(x, v) = x'' + x + 0.5*v
        # dg/d[x, x', x''] = [1, 0.5, 1]
        expected = np.array([[1.0, 0.5, 1.0]])
        assert np.allclose(jacobian, expected, atol=1e-5)

    def test_linearize(self):
        """Test linearize method."""

        def vf(x, v, *, t):
            return -x

        E0, E1, E2 = make_projection_matrices(d=1, q=2)
        model = SecondOrderODEInformation(vf, E0, E1, E2)

        state = np.array([1.0, 0.0, -1.0])
        H, c = model.linearize(state, t=0.0)

        assert H.shape == (1, 3)
        assert c.shape == (1,)

        # Verify H @ state + c = g(state)
        g_val = model.g(state, t=0.0)
        reconstructed = H @ state + c
        assert np.allclose(reconstructed, g_val)

    def test_with_conservation_constraint(self):
        """Test with conservation constraint."""

        def vf(x, v, *, t):
            return -x

        E0, E1, E2 = make_projection_matrices(d=2, q=2)
        conservation = Conservation(
            A=np.array([[1.0, 1.0]]),
            p=np.array([1.0]),
        )
        model = SecondOrderODEInformation(vf, E0, E1, E2, constraints=[conservation])

        state = np.array([0.5, 0.5, 0.0, 0.0, -0.5, -0.5])
        result = model.g(state, t=0.0)

        # ODE (2) + conservation (1) = 3
        assert result.shape == (3,)

    def test_multidimensional(self):
        """Test with multi-dimensional state."""

        def vf(x, v, *, t):
            return -x - 0.1 * v

        E0, E1, E2 = make_projection_matrices(d=2, q=2)
        model = SecondOrderODEInformation(vf, E0, E1, E2)

        state = np.ones(6)
        result = model.g(state, t=0.0)

        assert result.shape == (2,)


# =============================================================================
# SecondOrder Factory Function Tests
# =============================================================================


class TestSecondOrderFactoryFunctions:
    """Tests for second-order ODE factory functions."""

    def test_SecondOrderODEconservation(self):
        """Test SecondOrderODEconservation factory."""

        def vf(x, v, *, t):
            return -x

        E0, E1, E2 = make_projection_matrices(d=2, q=2)
        A = np.array([[1.0, 1.0]])
        p = np.array([1.0])

        model = SecondOrderODEconservation(vf, E0, E1, E2, A, p)

        assert len(model._constraints) == 1
        assert isinstance(model._constraints[0], Conservation)

        state = np.array([0.5, 0.5, 0.0, 0.0, -0.5, -0.5])
        result = model.g(state, t=0.0)

        # ODE (2) + conservation (1) = 3
        assert result.shape == (3,)

    def test_SecondOrderODEmeasurement(self):
        """Test SecondOrderODEmeasurement factory."""

        def vf(x, v, *, t):
            return -x

        E0, E1, E2 = make_projection_matrices(d=1, q=2)
        A = np.array([[1.0]])
        z = np.array([[0.5], [0.8]])
        z_t = np.array([0.5, 1.0])

        model = SecondOrderODEmeasurement(vf, E0, E1, E2, A, z, z_t)

        assert len(model._constraints) == 1
        assert isinstance(model._constraints[0], Measurement)

        state = np.array([1.0, 0.0, -1.0])

        # At t=0 (no measurement): only ODE (1)
        result_t0 = model.g(state, t=0.0)
        assert result_t0.shape == (1,)

        # At t=0.5 (with measurement): ODE (1) + measurement (1) = 2
        result_t05 = model.g(state, t=0.5)
        assert result_t05.shape == (2,)

    def test_SecondOrderODEconservationmeasurement(self):
        """Test SecondOrderODEconservationmeasurement factory."""

        def vf(x, v, *, t):
            return -x

        E0, E1, E2 = make_projection_matrices(d=2, q=2)
        C = np.array([[1.0, 1.0]])
        p = np.array([1.0])
        A = np.array([[1.0, 0.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        model = SecondOrderODEconservationmeasurement(
            vf, E0, E1, E2, C, p, A, z, z_t, measurement_noise=0.01
        )

        assert len(model._constraints) == 2
        assert isinstance(model._constraints[0], Conservation)
        assert isinstance(model._constraints[1], Measurement)

        state = np.array([0.5, 0.5, 0.0, 0.0, -0.5, -0.5])

        # At t=0: ODE (2) + conservation (1) = 3
        result_t0 = model.g(state, t=0.0)
        assert result_t0.shape == (3,)

        # At t=0.5: ODE (2) + conservation (1) + measurement (1) = 4
        result_t05 = model.g(state, t=0.5)
        assert result_t05.shape == (4,)


# =============================================================================
# Measurement Dataclass Edge Cases
# =============================================================================


class TestMeasurementEdgeCases:
    """Tests for edge cases in Measurement dataclass."""

    def test_measurement_noise_vector(self):
        """Test Measurement with vector noise (diagonal)."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        z = np.array([[0.5, 0.3]])
        z_t = np.array([0.5])

        m = Measurement(A=A, z=z, z_t=z_t, noise=np.array([0.1, 0.2]))
        R = m.get_noise_matrix()

        assert R.shape == (2, 2)
        assert R[0, 0] == pytest.approx(0.1)
        assert R[1, 1] == pytest.approx(0.2)
        assert R[0, 1] == pytest.approx(0.0)

    def test_measurement_noise_matrix(self):
        """Test Measurement with full noise matrix."""
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        z = np.array([[0.5, 0.3]])
        z_t = np.array([0.5])
        noise_matrix = np.array([[0.1, 0.05], [0.05, 0.2]])

        m = Measurement(A=A, z=z, z_t=z_t, noise=noise_matrix)
        R = m.get_noise_matrix()

        assert np.allclose(R, noise_matrix)

    def test_measurement_residual_no_measurement_at_time(self):
        """Test that residual returns None when no measurement at time."""
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        m = Measurement(A=A, z=z, z_t=z_t)

        # No measurement at t=0
        residual = m.residual(np.array([1.0]), t=0.0)
        assert residual is None

    def test_measurement_jacobian_no_measurement_at_time(self):
        """Test that jacobian returns None when no measurement at time."""
        A = np.array([[1.0]])
        z = np.array([[0.5]])
        z_t = np.array([0.5])

        m = Measurement(A=A, z=z, z_t=z_t)

        # No measurement at t=0
        jacobian = m.jacobian(t=0.0)
        assert jacobian is None

    def test_measurement_constraint_dimension_mismatch(self):
        """Test that mismatched constraint dimension raises error."""

        def vf(x, *, t):
            return -x

        E0, E1, _ = make_projection_matrices(d=2, q=1)

        # Conservation A has wrong dimension (3 instead of 2)
        conservation = Conservation(
            A=np.array([[1.0, 1.0, 1.0]]),
            p=np.array([1.0]),
        )

        with pytest.raises(ValueError, match="must match state dimension"):
            ODEInformation(vf, E0, E1, constraints=[conservation])

    def test_measurement_mismatched_A_dimension(self):
        """Test that mismatched Measurement A dimension raises error."""

        def vf(x, *, t):
            return -x

        E0, E1, _ = make_projection_matrices(d=2, q=1)

        # Measurement A has wrong dimension (3 instead of 2)
        measurement = Measurement(
            A=np.array([[1.0, 1.0, 1.0]]),
            z=np.array([[0.5]]),
            z_t=np.array([0.5]),
        )

        with pytest.raises(ValueError, match="must match state dimension"):
            ODEInformation(vf, E0, E1, constraints=[measurement])
