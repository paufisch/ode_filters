import jax
import jax.numpy as np
import numpy as onp  # Regular numpy for testing utilities
import pytest

from ode_filters.measurement.measurement_models import ODEInformation


def make_projection_matrices(d: int, q: int):
    """Create E0 and E1 projection matrices for given d and q."""
    eye_d = np.eye(d)
    basis = np.eye(q + 1)
    E0 = np.kron(basis[0:1], eye_d)
    E1 = np.kron(basis[1:2], eye_d)
    return E0, E1


def test_observation_matches_manual_computation():
    def vf(state, *, t):
        return state**2

    E0, E1 = make_projection_matrices(d=1, q=1)
    model = ODEInformation(vf=vf, E0=E0, E1=E1)
    state = np.array([2.0, 3.0])
    time = 0.0

    manual = model._E1 @ state - vf(model._E0 @ state, t=time)
    computed = model.g(state, t=time)

    onp.testing.assert_allclose(np.asarray(computed), np.asarray(manual))


def test_jacobian_matches_expected_linearization():
    def vf(state, *, t):
        return state**2

    E0, E1 = make_projection_matrices(d=1, q=1)
    model = ODEInformation(vf=vf, E0=E0, E1=E1)
    state = np.array([1.5, -0.5])
    time = 1.0

    jacobian = model.jacobian_g(state, t=time)
    jacobian_vf = jax.jacfwd(vf)(model._E0 @ state, t=time)
    expected = model._E1 - jacobian_vf @ model._E0

    onp.testing.assert_allclose(np.asarray(jacobian), np.asarray(expected))


@pytest.mark.parametrize("d, q", [(1, 1), (2, 2), (3, 1)])
def test_projection_matrices_have_expected_shapes(d, q):
    def vf(state, *, t):
        return state

    E0, E1 = make_projection_matrices(d=d, q=q)
    model = ODEInformation(vf=vf, E0=E0, E1=E1)
    state_dim = (q + 1) * d

    assert model._E0.shape == (d, state_dim)
    assert model._E1.shape == (d, state_dim)


def test_invalid_state_dimension_raises_error():
    def vf(state, *, t):
        return state

    E0, E1 = make_projection_matrices(d=1, q=1)
    model = ODEInformation(vf=vf, E0=E0, E1=E1)

    with pytest.raises(ValueError):
        model.g(np.array([1.0]), t=0.0)


def test_state_with_wrong_rank_raises_error():
    def vf(state, *, t):
        return state

    E0, E1 = make_projection_matrices(d=1, q=1)
    model = ODEInformation(vf=vf, E0=E0, E1=E1)
    bad_state = np.array([[1.0, 2.0]])  # shape (1, 2) → ndim = 2

    with pytest.raises(ValueError, match="must be a one-dimensional"):
        model.g(bad_state, t=0.0)
