import jax.numpy as jnp
import numpy as np
import pytest

from ode_filters.GMP_priors import (
    taylor_mode_initialization,
)


def test_taylor_mode_initialization_q0_returns_state_flattened():
    x0 = jnp.array([1.0, 2.0])
    result = taylor_mode_initialization(lambda y: y, x0, q=0)
    assert result.ndim == 1
    np.testing.assert_array_equal(result, x0.ravel())


def test_taylor_mode_initialization_scalar_linear_field_matches_closed_form():
    def vf(y):
        return -y  # u(t) = e^{-t} → u^{(k)}(0) = (-1)^k

    x0 = jnp.array([1.0])
    expected = jnp.array([(-1.0) ** k for k in range(4)])
    result = taylor_mode_initialization(vf, x0, q=3)

    assert result.shape == expected.shape
    np.testing.assert_allclose(result, expected)


def test_taylor_mode_initialization_vector_field_runs_and_flattens():
    def lf(y):
        a, b = 0.5, -0.3
        return jnp.array([a * y[0] - y[0] * y[1], b * y[1] + y[0] * y[1]])

    x0 = jnp.array([1.0, 2.0])
    result = taylor_mode_initialization(lf, x0, q=3)

    assert result.ndim == 1
    assert result.shape[0] == (len(x0) * (3 + 1))


def test_taylor_mode_initialization_rejects_invalid_inputs():
    with pytest.raises(TypeError):
        taylor_mode_initialization(123, jnp.array([0.0]), q=1)

    with pytest.raises(ValueError):
        taylor_mode_initialization(lambda y: y, jnp.array([0.0]), q=-1)
