"""Validation / dataclass coverage for Measurement, prepare_observations, and the
Conservation-only constraint guard on the ODE-information models."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters.measurement.measurement_models import (
    Measurement,
    ODEInformation,
    prepare_observations,
)


def _E0_E1(d=1, q=2):
    eye = np.eye(d)
    basis = np.eye(q + 1)
    return np.kron(basis[0:1], eye), np.kron(basis[1:2], eye)


def test_prepare_observations_empty_returns_none():
    E0, _ = _E0_E1()
    assert prepare_observations([], E0, np.linspace(0.0, 1.0, 5)) is None


def test_measurement_A_must_be_2d():
    with pytest.raises(ValueError, match="'A' must be 2D"):
        Measurement(A=np.array([1.0]), z=np.zeros((2, 1)), z_t=np.zeros(2))


def test_measurement_z_must_be_2d():
    with pytest.raises(ValueError, match="'z' must be 2D"):
        Measurement(A=np.eye(1), z=np.zeros(2), z_t=np.zeros(2))


def test_measurement_z_t_column_vector_is_normalized():
    m = Measurement(A=np.eye(1), z=np.zeros((2, 1)), z_t=np.zeros((2, 1)))
    assert m.z_t.ndim == 1


def test_measurement_z_t_bad_ndim_raises():
    with pytest.raises(ValueError, match="'z_t' must be 1D"):
        Measurement(A=np.eye(1), z=np.zeros((2, 1)), z_t=np.zeros((2, 2)))


def test_measurement_z_t_length_mismatch_raises():
    with pytest.raises(ValueError, match="length must match"):
        Measurement(A=np.eye(1), z=np.zeros((2, 1)), z_t=np.zeros(3))


def test_measurement_jacobian_at_matching_time():
    m = Measurement(A=np.eye(1), z=np.zeros((2, 1)), z_t=np.array([0.0, 1.0]))
    assert np.allclose(m.jacobian(0.0), np.eye(1))


def test_measurement_rejected_as_measure_constraint():
    E0, E1 = _E0_E1()
    m = Measurement(A=np.eye(1), z=np.zeros((1, 1)), z_t=np.array([0.0]))
    with pytest.raises(TypeError, match="no longer accepted"):
        ODEInformation(lambda x, *, t: -x, E0, E1, constraints=[m])
