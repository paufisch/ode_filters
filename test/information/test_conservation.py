"""Tests for conservation laws on ODE information."""

import jax.numpy as np
import pytest

from ode_filters.information import Conservation, ODEInformation


def make_projection_matrices(d: int, q: int):
    """Create E0 and E1 projection matrices for given d and q."""
    eye_d = np.eye(d)
    basis = np.eye(q + 1)
    E0 = np.kron(basis[0:1], eye_d)
    E1 = np.kron(basis[1:2], eye_d)
    return E0, E1


class TestConservationConstruction:
    """Tests for ODE information with a conservation law."""

    def test_basic_construction(self):
        """Test basic construction with a conservation law."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 1.0]])  # Conservation: x1 + x2 = const
        p = np.array([1.0])

        model = ODEInformation(vf, E0, E1, constraints=[Conservation(A, p)])
        # Model has one Conservation constraint
        assert len(model.constraints) == 1
        assert isinstance(model.constraints[0], Conservation)

    def test_rejects_mismatched_A_p_shapes(self):
        """Test that mismatched A and p shapes raise error."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 1.0], [1.0, -1.0]])  # 2 constraints
        p = np.array([1.0])  # Only 1 value

        with pytest.raises(ValueError, match="must match"):
            ODEInformation(vf, E0, E1, constraints=[Conservation(A, p)])

    def test_rejects_mismatched_A_E0_shapes(self):
        """Test that mismatched A and E0 shapes raise error."""

        def vf(x, *, t):
            return x

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 1.0, 1.0]])  # 3 columns but d=2
        p = np.array([1.0])

        with pytest.raises(ValueError):
            ODEInformation(vf, E0, E1, constraints=[Conservation(A, p)])


class TestConservationMethods:
    """Tests for the conservation rows."""

    @pytest.fixture
    def conservation_model(self):
        """A first-order ODE information with one conservation law."""

        def vf(x, *, t):
            return -x  # Simple decay

        E0, E1 = make_projection_matrices(d=2, q=1)
        A = np.array([[1.0, 1.0]])  # Conservation: x1 + x2 = const
        p = np.array([2.0])  # x1 + x2 = 2

        return ODEInformation(vf, E0, E1, constraints=[Conservation(A, p)])

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
        R = conservation_model.noise_sqr(t=0.0)

        # Shape should be (d + k, d + k) = (3, 3)
        assert R.shape == (3, 3)
