"""Data observations: ``Measurement``, ``prepare_observations`` and
``merge_measurement_times``."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import gaussian_filter, taylor_mode_initialization
from ode_filters.information import (
    Measurement,
    ODEInformation,
    merge_measurement_times,
    prepare_observations,
)
from ode_filters.information.observations import MEASUREMENT_TIME_ATOL
from ode_filters.priors import IWP
from test.helpers import assert_close, assert_grad_matches_fd, cov


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


# --------------------------------------------------------------------------
# Measurements built inside a trace
# --------------------------------------------------------------------------

PRIOR = IWP(q=2, d=1)
TS = np.linspace(0.0, 1.0, 11)
T_OBS = onp.array([0.33, 0.71])  # off the grid: merged into it in the trace
Z = np.array([[0.7], [0.5]])


def _decay(x, *, t):
    return -x


def _data_evidence(z, noise=1e-3):
    """The data evidence, with the measurement, grid and ObsModel built here."""
    meas = Measurement(np.eye(1), z, T_OBS, noise=noise)
    ts = merge_measurement_times(TS, [meas])
    obs = prepare_observations([meas], PRIOR.E0, ts)
    m_0, P_0_sqr = taylor_mode_initialization(_decay, np.array([1.0]), q=2)
    measure = ODEInformation(_decay, PRIOR.E0, PRIOR.E1)
    result = gaussian_filter(m_0, P_0_sqr, PRIOR, measure, ts, obs_model=obs)
    return result.log_likelihood_obs


def test_concrete_times_are_a_new_numpy_array():
    times = onp.array([0.5, 1.0])
    meas = Measurement(np.eye(1), np.zeros((2, 1)), times)
    assert isinstance(meas.z_t, onp.ndarray)
    assert meas.z_t is not times


def test_a_measurement_built_under_jit_gives_the_eager_evidence():
    expected = _data_evidence(Z)
    assert np.isfinite(expected)
    assert_close(jax.jit(_data_evidence)(Z), expected, rtol=1e-12)
    assert_close(eqx.filter_jit(_data_evidence)(Z), expected, rtol=1e-12)


def test_data_and_noise_of_a_measurement_built_in_the_trace_are_differentiable():
    assert_grad_matches_fd(lambda s: _data_evidence(s * Z), np.array(1.0))
    assert_grad_matches_fd(
        lambda log_r: _data_evidence(Z, noise=np.exp(log_r)), np.array(-7.0)
    )


def test_vmap_over_the_data_of_a_measurement_built_in_the_trace():
    zs = np.stack([Z, Z + 0.1, Z - 0.2])
    expected = np.stack([_data_evidence(z) for z in zs])
    assert_close(jax.vmap(_data_evidence)(zs), expected, rtol=1e-12)


def test_traced_times_raise_naming_the_cause():
    def from_times(times):
        return prepare_observations([Measurement(np.eye(1), Z, times)], PRIOR.E0, TS)

    with pytest.raises(TypeError, match="Measurement times are traced"):
        jax.jit(from_times)(TS[1:3])
    # The times of a measurement passed in as an argument are traced as well.
    meas = Measurement(np.eye(1), Z, TS[1:3])
    with pytest.raises(TypeError, match="Measurement times are traced"):
        eqx.filter_jit(lambda m: prepare_observations([m], PRIOR.E0, TS))(meas)
    with pytest.raises(TypeError, match="nodes of the grid ts are traced"):
        jax.jit(lambda ts: prepare_observations([meas], PRIOR.E0, ts))(TS)
    with pytest.raises(TypeError, match="nodes of the grid ts are traced"):
        jax.jit(lambda ts: merge_measurement_times(ts, [meas]))(TS)


# --------------------------------------------------------------------------
# merge_measurement_times
# --------------------------------------------------------------------------

GRID = onp.linspace(0.0, 1.0, 5)  # nodes 0, 0.25, 0.5, 0.75, 1


def _sensor(times):
    times = onp.asarray(times, dtype=float)
    return Measurement(np.eye(1), np.ones((times.shape[0], 1)), times)


def test_merge_adds_the_off_grid_times_as_nodes():
    a, b = _sensor([0.25, 0.6]), _sensor([0.6, 0.9, 1.0])
    with pytest.raises(ValueError, match="merge_measurement_times"):
        prepare_observations([a, b], PRIOR.E0, GRID)
    merged = merge_measurement_times(GRID, [a, b])
    assert isinstance(merged, onp.ndarray)
    assert_close(merged, onp.array([0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]), rtol=0.0)
    obs = prepare_observations([a, b], PRIOR.E0, merged)
    assert obs is not None
    assert int(onp.asarray(obs.mask).sum()) == 5


def test_merge_adds_no_node_for_near_duplicate_times():
    # Within the matching tolerance of a node, and of each other.
    near = 0.5 * MEASUREMENT_TIME_ATOL
    a, b = _sensor([0.25 + near, 0.6]), _sensor([0.6 + near])
    merged = merge_measurement_times(GRID, [a, b])
    assert_close(merged, onp.array([0.0, 0.25, 0.5, 0.6, 0.75, 1.0]), rtol=0.0)
    obs = prepare_observations([a, b], PRIOR.E0, merged)
    assert obs is not None
    # Both sensors' data land on the node at 0.6 (step 2).
    assert onp.asarray(obs.mask)[2].all()


def test_merge_without_new_times_returns_a_copy_of_the_grid():
    for measurements in ([], [_sensor(GRID[1:3])]):
        merged = merge_measurement_times(GRID, measurements)
        assert merged is not GRID
        assert_close(merged, GRID, rtol=0.0)


def test_merge_rejects_two_times_of_one_measurement_at_one_node():
    near = 0.5 * MEASUREMENT_TIME_ATOL
    for times in ([0.6, 0.6 + near], [0.5, 0.5 + near], [0.6, 0.6]):
        with pytest.raises(ValueError, match="several times at one node"):
            merge_measurement_times(GRID, [_sensor(times)])


def test_merge_rejects_data_at_the_initial_time():
    with pytest.raises(ValueError, match="initial time"):
        merge_measurement_times(GRID, [_sensor([0.0, 0.6])])


def test_merge_rejects_times_outside_the_grid():
    for times in ([1.5], [-0.5, 0.6]):
        with pytest.raises(ValueError, match="outside"):
            merge_measurement_times(GRID, [_sensor(times)])


def test_merge_validates_the_grid():
    with pytest.raises(ValueError, match="1-D array"):
        merge_measurement_times(onp.zeros((2, 2)), [])
    with pytest.raises(ValueError, match="at least two"):
        merge_measurement_times(onp.zeros(1), [])
    with pytest.raises(ValueError, match="strictly increasing"):
        merge_measurement_times(onp.array([0.0, 1.0, 0.5]), [])
