"""Data observations: ``Measurement`` and ``prepare_observations``."""

from __future__ import annotations

import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.information import Measurement, prepare_observations
from ode_filters.priors import IWP
from test.helpers import assert_close, cov


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


def test_each_sensor_keeps_its_own_rows_noise_data_and_times():
    """Three sensors with different noises, dimensions and times, entry by entry.

    Sensor ``a`` observes ``x_0`` (variance 0.1), sensor ``b`` two combinations of
    ``x`` with a correlated noise, and sensor ``c`` (``full_state``) the
    derivative ``x_1'`` (variance 0.5); their times overlap only in part.
    """
    prior = IWP(q=1, d=2)
    ts = np.linspace(0.0, 2.0, 11)
    R_b = np.array([[0.04, 0.01], [0.01, 0.09]])
    steps_a, steps_b, steps_c = (
        onp.array([1, 4, 7]),
        onp.array([4, 9]),
        onp.array([2, 7]),
    )
    z_a = np.array([[1.0], [2.0], [3.0]])
    z_b = np.array([[4.0, 5.0], [6.0, 7.0]])
    z_c = np.array([[8.0], [9.0]])
    sensors = [
        Measurement(np.array([[1.0, 0.0]]), z_a, ts[1:][steps_a], noise=0.1),
        Measurement(np.array([[0.0, 1.0], [1.0, 1.0]]), z_b, ts[1:][steps_b], R_b),
        Measurement(prior.E1[1:], z_c, ts[1:][steps_c], noise=0.5, full_state=True),
    ]
    obs = prepare_observations(sensors, prior.E0, ts)
    assert obs is not None

    H = np.concatenate(
        [np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]) @ prior.E0, prior.E1[1:]]
    )
    assert_close(obs.H, H, rtol=0.0)
    R = onp.zeros((4, 4))
    R[0, 0], R[1:3, 1:3], R[3, 3] = 0.1, R_b, 0.5
    assert_close(cov(obs.R_sqr), R, rtol=1e-14)

    c_seq = onp.zeros((10, 4))
    mask = onp.zeros((10, 4), dtype=bool)
    for steps, rows, z in [
        (steps_a, slice(0, 1), z_a),
        (steps_b, slice(1, 3), z_b),
        (steps_c, slice(3, 4), z_c),
    ]:
        c_seq[steps, rows] = -z
        mask[steps, rows] = True
    assert_close(obs.c_seq, c_seq, rtol=0.0)
    assert (onp.asarray(obs.mask) == mask).all()
