"""Tests for ODE information classes and constraint dataclasses."""

import jax
import jax.numpy as np
import pytest

from ode_filters.information import (
    BlackBoxMeasurement,
    Conservation,
    Measurement,
    ODEInformation,
    ODEInformationWithHidden,
    SecondOrderODEInformation,
    SecondOrderODEInformationWithHidden,
)
from ode_filters.priors import IWP


def make_projection_matrices(d: int, q: int):
    """Create E0 and E1 projection matrices for given d and q."""
    eye_d = np.eye(d)
    basis = np.eye(q + 1)
    E0 = np.kron(basis[0:1], eye_d)
    E1 = np.kron(basis[1:2], eye_d)
    return E0, E1


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

        R_sqr = model.get_noise(t=0.0)
        R_reconstructed = R_sqr.T @ R_sqr
        assert R_reconstructed[0, 0] == pytest.approx(0.1)


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

    def test_ode_with_conservation_constraint(self):
        """Test ODEInformation with a conservation constraint."""

        def vf(x, *, t):
            return -x

        E0, E1 = make_projection_matrices(d=2, q=1)
        conservation = Conservation(
            A=np.array([[1.0, 1.0]]),  # x1 + x2 = const
            p=np.array([2.0]),
        )

        model = ODEInformation(vf, E0, E1, constraints=[conservation])

        # ODE (2) + conservation (1) = 3, independent of time
        state = np.array([1.0, 1.0, -1.0, -1.0])
        g_t0 = model.g(state, t=0.0)
        assert g_t0.shape == (3,)

        g_t05 = model.g(state, t=0.5)
        assert g_t05.shape == (3,)


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


class TestSelectorValidation:
    """Mismatched selection matrices used to broadcast silently."""

    @staticmethod
    def vf(x, *, t):
        return -x

    def test_rejects_E1_with_fewer_rows_than_E0(self):
        prior = IWP(q=2, d=2)
        with pytest.raises(ValueError, match="E1 must have as many rows as E0"):
            ODEInformation(self.vf, prior.E0, prior.E1[:1])

    def test_rejects_mismatched_columns(self):
        with pytest.raises(ValueError, match="one column per state entry"):
            ODEInformation(self.vf, IWP(q=2, d=1).E0, IWP(q=1, d=1).E1)

    def test_rejects_non_matrix_selector(self):
        prior = IWP(q=2, d=1)
        with pytest.raises(ValueError, match="E0 must be a 2-D selection matrix"):
            ODEInformation(self.vf, prior.E0[0], prior.E1)

    def test_missing_E2_names_the_cause(self):
        prior = IWP(q=1, d=1)  # E2 needs q >= 2
        with pytest.raises(ValueError, match=r"E2 is None .*q >= 2"):
            SecondOrderODEInformation(
                lambda x, v, *, t: -x, prior.E0, prior.E1, prior.E2
            )

    def test_missing_other_selector(self):
        prior = IWP(q=2, d=1)
        with pytest.raises(ValueError, match="E1 is None"):
            ODEInformation(self.vf, prior.E0, None)


class TestConstraintInputs:
    def test_conservation_accepts_lists(self):
        cons = Conservation(A=[[1.0, 1.0]], p=[1.0])
        assert cons.A.shape == (1, 2) and cons.p.shape == (1,)

    def test_conservation_rejects_wrong_ranks(self):
        with pytest.raises(ValueError, match="2-D 'A' and a 1-D 'p'"):
            Conservation(A=np.ones(2), p=np.ones(1))

    def test_unknown_constraint_type_is_rejected(self):
        prior = IWP(q=1, d=2)
        with pytest.raises(TypeError, match="must be Conservation objects"):
            ODEInformation(
                lambda x, *, t: -x, prior.E0, prior.E1, constraints=[(1.0, 2.0)]
            )

    def test_constraints_list_is_copied(self):
        prior = IWP(q=1, d=2)
        constraints = [Conservation(np.array([[1.0, 1.0]]), np.array([1.0]))]
        model = ODEInformation(
            lambda x, *, t: -x, prior.E0, prior.E1, constraints=constraints
        )
        constraints.append(Conservation(np.array([[1.0, -1.0]]), np.array([0.0])))
        assert model.g(np.ones(4), t=0.0).shape == (3,)

    def test_measurement_accepts_lists(self):
        meas = Measurement(A=[[1.0]], z=[[0.5], [0.6]], z_t=[0.1, 0.2])
        assert meas.z.shape == (2, 1) and meas.z_t.shape == (2,)


class TestNoiseCovarianceValidation:
    """A singular noise covariance that mixes channels used to give NaN deep
    inside a solve; noise-free channels (zero rows/columns) stay allowed."""

    singular = np.array([[1.0, 1.0], [1.0, 1.0]])

    def test_measurement_rejects_singular_correlated_noise(self):
        with pytest.raises(ValueError, match="positive-definite covariance"):
            Measurement(
                A=np.eye(2), z=np.zeros((1, 2)), z_t=np.zeros(1), noise=self.singular
            )

    def test_measurement_rejects_wrongly_shaped_noise(self):
        with pytest.raises(ValueError, match="'noise' must be a scalar"):
            Measurement(
                A=np.eye(2), z=np.zeros((1, 2)), z_t=np.zeros(1), noise=np.ones(3)
            )

    def test_noise_free_channels_are_allowed(self):
        Measurement(
            A=np.eye(2),
            z=np.zeros((1, 2)),
            z_t=np.zeros(1),
            noise=np.diag(np.array([0.0, 1.0])),
        )
        prior = IWP(q=1, d=2)
        model = ODEInformation(lambda x, *, t: -x, prior.E0, prior.E1)
        model.R = 0.0
        assert np.all(np.isfinite(model.get_noise(t=0.0)))

    def test_R_setter_rejects_singular_correlated_noise(self):
        prior = IWP(q=1, d=2)
        model = ODEInformation(lambda x, *, t: -x, prior.E0, prior.E1)
        with pytest.raises(ValueError, match="positive-definite covariance"):
            model.R = self.singular

    def test_black_box_rejects_singular_correlated_noise(self):
        with pytest.raises(ValueError, match="positive-definite covariance"):
            BlackBoxMeasurement(
                lambda s, *, t: s[:2], state_dim=4, obs_dim=2, noise=self.singular
            )
        model = BlackBoxMeasurement(lambda s, *, t: s[:2], state_dim=4, obs_dim=2)
        with pytest.raises(ValueError, match="positive-definite covariance"):
            model.R = self.singular

    def test_check_is_skipped_for_traced_noise(self):
        """Noise as a differentiated parameter must not trip the eager check."""

        def noise_matrix(n):
            return Measurement(
                A=np.eye(1), z=np.zeros((1, 1)), z_t=np.zeros(1), noise=n
            ).get_noise_matrix()

        assert np.allclose(jax.jit(noise_matrix)(0.5), 0.5)
