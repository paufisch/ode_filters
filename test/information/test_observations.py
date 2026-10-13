"""Data observations: ``Measurement`` and ``prepare_observations``."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters.information import Measurement, prepare_observations
from ode_filters.priors import IWP


class TestMeasurementDataclass:
    """Tests for the Measurement dataclass."""

    def test_measurement_creation(self):
        """Test creating a Measurement constraint."""
        A = np.array([[1.0]])
        z = np.array([[0.5], [0.8]])
        z_t = np.array([0.5, 1.0])

        m = Measurement(A=A, z=z, z_t=z_t)
        assert m.dim == 1

    def test_measurement_noise_scalar(self):
        """Test measurement noise from scalar."""
        A = np.array([[1.0], [0.0]])
        z = np.array([[0.5, 0.3]])
        z_t = np.array([0.5])

        m = Measurement(A=A, z=z, z_t=z_t, noise=0.1)
        R = m.R
        assert R.shape == (2, 2)
        assert np.allclose(R, 0.1 * np.eye(2))


def test_prepare_observations_empty_returns_none():
    assert prepare_observations([], IWP(q=2, d=1).E0, np.linspace(0.0, 1.0, 5)) is None


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
