import jax
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


def test_taylor_mode_initialization_q1_linear_vector_field_matches_matrix_product():
    A = jnp.array([[0.0, 1.0], [-2.0, -3.0]])

    def vf(y):
        return A @ y

    x0 = jnp.array([1.0, 2.0])

    expected = jnp.concatenate((x0, A @ x0))
    result = taylor_mode_initialization(vf, x0, q=1)

    np.testing.assert_allclose(result, expected)


def test_taylor_mode_initialization_linear_vector_field_matches_matrix_powers():
    A = jnp.array([[0.0, 1.0], [-2.0, -3.0]])

    def vf(y):
        return A @ y

    x0 = jnp.array([1.0, 2.0])

    Ax = A @ x0
    A2x = A @ (A @ x0)
    expected = jnp.concatenate((x0, Ax, A2x))
    result = taylor_mode_initialization(vf, x0, q=2)

    np.testing.assert_allclose(result, expected)


def test_taylor_mode_initialization_passes_expected_series_to_jet(monkeypatch):
    x0 = jnp.array([1.0, 2.0])

    primals_outputs = [
        jnp.array([10.0, 20.0]),
        jnp.array([30.0, 40.0]),
        jnp.array([50.0, 60.0]),
    ]
    series_outputs = [
        (jnp.array([100.0, 200.0]),),
        (jnp.array([300.0, 400.0]),),
        (jnp.array([500.0, 600.0]),),
    ]

    calls = {"idx": 0}
    expected_series_terms: list[jnp.ndarray] = []

    def fake_jet(fun, primals, series, factorial_scaled=True):
        assert fun is vf
        np.testing.assert_array_equal(np.asarray(primals[0]), np.asarray(x0))

        (incoming_series,) = series
        assert len(incoming_series) == len(expected_series_terms)
        for got, want in zip(incoming_series, expected_series_terms):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))

        idx = calls["idx"]
        primals_out = primals_outputs[idx]
        terms_out = series_outputs[idx]
        calls["idx"] += 1

        expected_series_terms.clear()
        expected_series_terms.extend([primals_out, *terms_out])

        return primals_out, terms_out

    def vf(y):
        return y  # will never run because fake_jet intercepts

    monkeypatch.setattr(jax.experimental.jet, "jet", fake_jet)

    result = taylor_mode_initialization(vf, x0, q=len(primals_outputs))

    expected_flat = jnp.concatenate(
        [jnp.ravel(arr) for arr in [x0, *[terms[-1] for terms in series_outputs]]]
    )
    np.testing.assert_allclose(result, expected_flat)
    assert calls["idx"] == len(primals_outputs)
