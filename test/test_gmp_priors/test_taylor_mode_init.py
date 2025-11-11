import jax.numpy as jnp
import numpy as np
import pytest

from ode_filters.priors.GMP_priors import taylor_mode_initialization


def test_taylor_mode_initialization_q0_returns_state_flattened():
    def vf(y, *, t):
        return y

    x0 = jnp.array([1.0, 2.0])
    result, _ = taylor_mode_initialization(vf, x0, q=0)

    assert result.ndim == 1
    assert np.array_equal(np.asarray(result), np.asarray(x0).ravel())


def test_taylor_mode_initialization_scalar_linear_field_matches_closed_form():
    def vf(y, *, t):
        return -y  # u(t) = e^{-t} → u^{(k)}(0) = (-1)^k

    x0 = jnp.array([1.0])
    expected = jnp.array([(-1.0) ** k for k in range(4)])
    result, _ = taylor_mode_initialization(vf, x0, q=3)

    assert result.shape == expected.shape
    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_taylor_mode_initialization_vector_field_runs_and_flattens():
    def vf(y, *, t):
        a, b = 0.5, -0.3
        return jnp.array([a * y[0] - y[0] * y[1], b * y[1] + y[0] * y[1]])

    x0 = jnp.array([1.0, 2.0])
    result, _ = taylor_mode_initialization(vf, x0, q=3)

    assert result.ndim == 1
    assert result.shape[0] == len(x0) * (3 + 1)


def test_taylor_mode_initialization_rejects_invalid_inputs():
    def vf(y, *, t):
        return y

    with pytest.raises(TypeError):
        taylor_mode_initialization(123, jnp.array([0.0]), q=1)

    with pytest.raises(ValueError):
        taylor_mode_initialization(vf, jnp.array([0.0]), q=-1)


def test_taylor_mode_initialization_q1_linear_vector_field_matches_matrix_product():
    A = jnp.array([[0.0, 1.0], [-2.0, -3.0]])

    def vf(y, *, t):
        return A @ y

    x0 = jnp.array([1.0, 2.0])

    expected = jnp.concatenate((x0, A @ x0))
    result, _ = taylor_mode_initialization(vf, x0, q=1)

    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_taylor_mode_initialization_linear_vector_field_matches_matrix_powers():
    A = jnp.array([[0.0, 1.0], [-2.0, -3.0]])

    def vf(y, *, t):
        return A @ y

    x0 = jnp.array([1.0, 2.0])

    Ax = A @ x0
    A2x = A @ (A @ x0)
    expected = jnp.concatenate((x0, Ax, A2x))
    result, _ = taylor_mode_initialization(vf, x0, q=2)

    assert np.allclose(np.asarray(result), np.asarray(expected))
