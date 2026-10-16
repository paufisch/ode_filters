"""Data observations: ``Measurement``, ``ObsModel`` and ``prepare_observations``."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import gaussian_filter, taylor_mode_initialization
from ode_filters.information import (
    Measurement,
    ObsModel,
    ODEInformation,
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
T_OBS = onp.array([0.33, 0.71])  # off the grid: data-only nodes
Z = np.array([[0.7], [0.5]])


def _decay(x, *, t):
    return -x


def _data_evidence(z, noise=1e-3):
    """The data evidence, with the measurement and ObsModel built here."""
    meas = Measurement(np.eye(1), z, T_OBS, noise=noise)
    obs = prepare_observations([meas], PRIOR.E0, TS)
    m_0, P_0_sqr = taylor_mode_initialization(_decay, np.array([1.0]), q=2)
    measure = ODEInformation(_decay, PRIOR.E0, PRIOR.E1)
    result = gaussian_filter(m_0, P_0_sqr, PRIOR, measure, TS, obs_model=obs)
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


# --------------------------------------------------------------------------
# Data-only nodes, shared nodes and repeated times
# --------------------------------------------------------------------------

GRID = onp.linspace(0.0, 1.0, 5)  # nodes 0, 0.25, 0.5, 0.75, 1


def _sensor(times, values=None):
    times = onp.asarray(times, dtype=float)
    z = onp.arange(1.0, times.shape[0] + 1.0) if values is None else values
    return Measurement(np.eye(1), np.asarray(z, dtype=float)[:, None], times)


def _obs(*measurements, grid=GRID) -> ObsModel:
    obs = prepare_observations(list(measurements), PRIOR.E0, grid)
    assert obs is not None
    return obs


def test_on_grid_data_have_no_timeline():
    obs = _obs(_sensor([0.25, 1.0]))
    assert obs.t is None and obs.grid_node is None
    assert obs.c_seq.shape == (4, 1)


def test_off_grid_times_become_data_only_nodes():
    a, b = _sensor([0.25, 0.6]), _sensor([0.6, 0.9, 1.0])
    obs = _obs(a, b)
    assert isinstance(obs.t, onp.ndarray)
    assert_close(obs.t, onp.array([0.0, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0]), rtol=0.0)
    assert (onp.asarray(obs.grid_node) == [True, True, False, True, False, True]).all()
    mask = onp.zeros((6, 2), dtype=bool)
    c_seq = onp.zeros((6, 2))
    for step, row, value in [(0, 0, 1.0), (2, 0, 2.0), (2, 1, 1.0), (4, 1, 2.0)]:
        mask[step, row], c_seq[step, row] = True, -value
    mask[5, 1], c_seq[5, 1] = True, -3.0
    assert (onp.asarray(obs.mask) == mask).all()
    assert_close(obs.c_seq, c_seq, rtol=0.0)


def test_near_duplicate_times_share_a_node():
    # Within the matching tolerance of a grid node, and of each other.
    near = 0.5 * MEASUREMENT_TIME_ATOL
    a, b = _sensor([0.25 + near, 0.6]), _sensor([0.6 + near])
    obs = _obs(a, b)
    assert obs.t is not None
    assert_close(obs.t, onp.array([0.0, 0.25, 0.5, 0.6, 0.75, 1.0]), rtol=0.0)
    # Sensor a's datum at 0.25 + near is at the grid node 0.25 (step 0); both
    # sensors' data at 0.6 share the data-only node 0.6 (step 2).
    assert onp.asarray(obs.mask)[0, 0] and onp.asarray(obs.mask)[2].all()


def test_repeated_times_of_one_measurement_take_copies_of_its_rows():
    near = 0.5 * MEASUREMENT_TIME_ATOL
    meas = Measurement(
        np.array([[1.0], [2.0]]) @ np.eye(1),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]),
        onp.array([0.5, 0.6, 0.5 + near, 0.5]),
        noise=np.array([[0.04, 0.01], [0.01, 0.09]]),
    )
    obs = _obs(meas)
    # Three data at the grid node 0.5 (step 1): three copies of the two rows.
    assert obs.H.shape == (6, 3)
    assert_close(obs.H, np.tile(np.array([[1.0], [2.0]]) @ PRIOR.E0, (3, 1)), rtol=0.0)
    R = jax.scipy.linalg.block_diag(*[np.array([[0.04, 0.01], [0.01, 0.09]])] * 3)
    assert_close(cov(obs.R_sqr), R, rtol=1e-14)
    assert_close(obs.c_seq[1], -np.array([1.0, 2.0, 5.0, 6.0, 7.0, 8.0]), rtol=0.0)
    assert onp.asarray(obs.mask)[1].all()
    # One datum at the data-only node 0.6 (step 2) takes the first copy only.
    assert (onp.asarray(obs.mask)[2] == [True, True, False, False, False, False]).all()
    assert_close(obs.c_seq[2, :2], -np.array([3.0, 4.0]), rtol=0.0)


def test_without_data_keeps_the_timeline_and_drops_every_row():
    obs = _obs(_sensor([0.25, 0.6]))
    free = obs.without_data()
    assert free.H.shape == (0, 3) and free.R_sqr.shape == (0, 0)
    assert free.c_seq.shape == (5, 0) and free.mask.shape == (5, 0)
    assert free.t is obs.t and free.grid_node is obs.grid_node


def test_times_at_or_before_the_initial_time_raise():
    near = 0.5 * MEASUREMENT_TIME_ATOL
    for times in ([0.0, 0.6], [near], [-0.5, 0.6]):
        with pytest.raises(ValueError, match="at or before the initial time"):
            _obs(_sensor(times))


def test_times_after_the_final_time_raise():
    with pytest.raises(ValueError, match="after the final time"):
        _obs(_sensor([0.6, 1.5]))
    # Within the tolerance of the last node, a time is that node.
    assert _obs(_sensor([1.0 + 0.5 * MEASUREMENT_TIME_ATOL])).t is None


def test_the_grid_is_validated():
    with pytest.raises(ValueError, match="1-D array"):
        _obs(_sensor([0.6]), grid=onp.zeros((2, 2)))
    with pytest.raises(ValueError, match="at least two"):
        _obs(_sensor([0.6]), grid=onp.zeros(1))
    with pytest.raises(ValueError, match="strictly increasing"):
        _obs(_sensor([0.6]), grid=onp.array([0.0, 1.0, 0.5]))
