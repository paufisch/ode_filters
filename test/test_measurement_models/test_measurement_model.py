import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ode_filters.measurement_models import ODEInformation


def test_observation_matches_manual_computation():
    def vf(state):
        return state**2

    model = ODEInformation(vf=vf, d=1, q=1)
    state = jnp.array([2.0, 3.0])

    manual = model._E1 @ state - vf(model._E0 @ state)
    computed = model.g(state)

    np.testing.assert_allclose(np.asarray(computed), np.asarray(manual))


def test_jacobian_matches_expected_linearization():
    def vf(state):
        return state**2

    model = ODEInformation(vf=vf, d=1, q=1)
    state = jnp.array([1.5, -0.5])

    jacobian = model.jacobian_g(state)
    jacobian_vf = jax.jacfwd(vf)(model._E0 @ state)
    expected = model._E1 - jacobian_vf @ model._E0

    np.testing.assert_allclose(np.asarray(jacobian), np.asarray(expected))


@pytest.mark.parametrize("d, q", [(1, 1), (2, 2), (3, 1)])
def test_projection_matrices_have_expected_shapes(d, q):
    def vf(state):
        return state

    model = ODEInformation(vf=vf, d=d, q=q)
    state_dim = (q + 1) * d

    assert model._E0.shape == (d, state_dim)
    assert model._E1.shape == (d, state_dim)


def test_invalid_state_dimension_raises_error():
    def vf(state):
        return state

    model = ODEInformation(vf=vf, d=1, q=1)

    with pytest.raises(ValueError):
        model.g(jnp.array([1.0]))


def test_invalid_constructor_parameters_raise():
    def vf(state):
        return state

    with pytest.raises(ValueError):
        ODEInformation(vf=vf, d=0, q=1)

    with pytest.raises(ValueError):
        ODEInformation(vf=vf, d=1, q=0)
