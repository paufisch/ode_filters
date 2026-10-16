"""Data between the grid nodes: the fixed-grid filters on the timeline of the data.

A datum between two grid nodes becomes a data-only node, where the filter
conditions on the data without an update on the ODE information. Two
independent constructions pin the result:

- the merged grid (the timeline as an ordinary grid) with the ODE information
  switched off at the data-only nodes (a zero row with unit noise, which
  carries no information), for a frozen linearization and for a dense Kalman
  oracle of a linear ODE;
- the grid alone, where a solve without data must reproduce the timeline solve
  at the grid nodes, whatever the calibration.

Reference problem: the damped rotation ``x' = L x`` with two sensors, one on
each component, at times on and between the nodes of a 13-node grid.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    Measurement,
    ODEInformation,
    PrecondIWP,
    affine_filter,
    gaussian_filter,
    interpolate,
    prepare_observations,
    rts_smoother,
    sample_posterior,
    taylor_mode_initialization,
    taylor_surrogate,
)
from ode_filters.calibration import as_calibration
from ode_filters.filters.fixed_grid import grid_transitions
from ode_filters.filters.kernel import OnlineInformation, filter_step
from ode_filters.gaussian import AffineModel, Gaussian
from ode_filters.information import ObsModel
from ode_filters.linearization import TaylorCorrection
from test.helpers import (
    DenseStep,
    assert_close,
    cov,
    dense_kalman,
    surrogate_on_merged_grid,
)

L = np.array([[-0.5, 1.0], [-1.0, -0.5]])
TS = onp.linspace(0.0, 3.0, 13)  # h = 0.25
T_A = onp.array([0.3, 0.75, 1.1, 1.15, 2.6])  # 0.75 is a grid node
T_B = onp.array([0.3, 1.5, 2.95, 3.0])  # 1.5 and 3.0 are grid nodes
NOISE = 0.05**2
R_ODE = 1e-4
LOG_2PI = float(onp.log(2.0 * onp.pi))
PRIORS = {"iwp": lambda: IWP(q=2, d=2), "precond_iwp": lambda: PrecondIWP(q=3, d=2)}


def _linear(x, *, t):
    return L @ x


def _lotka_volterra(x, *, t):
    return np.array([1.5 * x[0] - x[0] * x[1], -3.0 * x[1] + x[0] * x[1]])


def _sensors() -> list[Measurement]:
    z_a = 0.5 + 0.1 * onp.sin(7.0 * T_A)
    z_b = -0.2 + 0.1 * onp.cos(5.0 * T_B)
    return [
        Measurement(np.array([[1.0, 0.0]]), z_a[:, None], T_A, noise=NOISE),
        Measurement(np.array([[0.0, 1.0]]), z_b[:, None], T_B, noise=NOISE),
    ]


def _data(prior) -> ObsModel:
    obs = prepare_observations(_sensors(), prior.E0, TS)
    assert obs is not None and obs.t is not None and obs.grid_node is not None
    return obs


def _merged(prior) -> tuple[onp.ndarray, onp.ndarray, ObsModel]:
    """The timeline as a grid, its grid-node flags, and the data on it."""
    timeline = _data(prior)
    assert timeline.t is not None and timeline.grid_node is not None
    obs = prepare_observations(_sensors(), prior.E0, timeline.t)
    assert obs is not None and obs.t is None
    return onp.asarray(timeline.t), onp.asarray(timeline.grid_node), obs


def _grid_nodes(grid_node) -> onp.ndarray:
    """The indices of the grid nodes ``0..N`` among the nodes of the timeline."""
    return onp.concatenate([[0], onp.flatnonzero(grid_node) + 1])


def _intervals(grid_node) -> onp.ndarray:
    steps = onp.asarray(grid_node).astype(int)
    return onp.cumsum(steps) - steps


def _setup(prior_name, vf=_linear, x0=(1.0, 0.0)):
    prior = PRIORS[prior_name]()
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array(x0), q=prior.q)
    return prior, ODEInformation(vf, prior.E0, prior.E1), m_0, P_0_sqr


# --------------------------------------------------------------------------
# Exact conditioning
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prior_name", sorted(PRIORS))
def test_a_frozen_linearization_equals_the_merged_grid_without_ode_rows(prior_name):
    prior, measure, m_0, P_0_sqr = _setup(prior_name, _lotka_volterra, (1.0, 1.0))
    plain = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="none")
    surrogate = taylor_surrogate(measure, plain.m_pred, ts=TS[1:])
    obs = _data(prior)
    result = affine_filter(m_0, P_0_sqr, prior, surrogate, TS, obs_model=obs)

    nodes, grid_node, merged_obs = _merged(prior)
    k = surrogate.H.shape[1]
    merged_surrogate = surrogate_on_merged_grid(surrogate, grid_node)
    expected = affine_filter(
        m_0, P_0_sqr, prior, merged_surrogate, nodes, obs_model=merged_obs
    )

    assert_close(result.t, nodes, rtol=0.0)
    assert_close(result.m, expected.m, rtol=1e-12)
    assert_close(cov(result.P_sqr), cov(expected.P_sqr), rtol=1e-12)
    assert_close(result.log_likelihood_obs, expected.log_likelihood_obs, rtol=1e-12)
    # Each switched-off step adds log N(0; 0, I_k) to the merged grid's evidence.
    n_off = int((~grid_node).sum())
    assert_close(
        result.log_likelihood,
        expected.log_likelihood + 0.5 * k * n_off * LOG_2PI,
        rtol=1e-12,
    )
    smoothed, smoothed_merged = rts_smoother(result), rts_smoother(expected)
    assert_close(smoothed.m, smoothed_merged.m, rtol=1e-12)
    assert_close(cov(smoothed.P_sqr), cov(smoothed_merged.P_sqr), rtol=1e-12)


def _dense_steps(prior, nodes, grid_node, diffusion, obs):
    """Dense steps on the merged grid: ODE rows at the grid nodes only."""
    H_ode = prior.E1 - L @ prior.E0
    R_obs = cov(obs.R_sqr)
    steps = []
    for n, h in enumerate(onp.diff(nodes)):
        Q_sqr = prior.Q_sqr(h)
        present = onp.asarray(obs.mask[n])
        on = bool(grid_node[n])
        steps.append(
            DenseStep(
                A=prior.A(h),
                Q=diffusion[n] * (Q_sqr.T @ Q_sqr),
                H=H_ode if on else np.zeros_like(H_ode),
                c=np.zeros(2),
                R=R_ODE * np.eye(2) if on else np.eye(2),
                H_obs=obs.H[present],
                c_obs=obs.c_seq[n][present],
                R_obs=R_obs[onp.ix_(present, present)],
            )
        )
    return steps


@pytest.mark.parametrize("prior_name", sorted(PRIORS))
@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_a_linear_ode_matches_the_dense_oracle_on_the_merged_grid(
    prior_name, calibration
):
    """The oracle propagates each step with the reported diffusion; that this
    is the quasi-MLE of the step's grid interval, estimated at the interval's
    start, is checked against the oracle's filtered means."""
    prior, measure, m_0, P_0_sqr = _setup(prior_name)
    measure = measure.with_noise(R_ODE)
    obs = _data(prior)
    result = gaussian_filter(
        m_0, P_0_sqr, prior, measure, TS, calibration=calibration, obs_model=obs
    )
    nodes, grid_node, merged_obs = _merged(prior)
    diffusion = onp.asarray(result.diffusion)
    ref = dense_kalman(
        m_0,
        cov(P_0_sqr),
        _dense_steps(prior, nodes, grid_node, diffusion, merged_obs),
    )
    assert_close(result.m, ref.m, rtol=1e-12)
    assert_close(cov(result.P_sqr), ref.P, rtol=1e-12)
    assert_close(result.log_likelihood_obs, ref.log_likelihood_obs, rtol=1e-12)
    n_off = int((~grid_node).sum())
    assert_close(
        result.log_likelihood, ref.log_likelihood + n_off * LOG_2PI, rtol=1e-12
    )
    smoothed = rts_smoother(result)
    assert_close(smoothed.m, ref.m_smooth, rtol=1e-12)
    assert_close(cov(smoothed.P_sqr), ref.P_smooth, rtol=1e-12)

    # The diffusion of interval k: r.T (H Q H.T)^-1 r / d at the prediction of
    # the whole interval from the oracle's state at its first node.
    H = prior.E1 - L @ prior.E0
    h = TS[1] - TS[0]
    Q_sqr = prior.Q_sqr(h)
    HQH = H @ Q_sqr.T @ Q_sqr @ H.T
    intervals = _intervals(grid_node)
    starts = _grid_nodes(grid_node)[:-1]
    for k, start in enumerate(starts):
        r = H @ prior.A(h) @ ref.m[start]
        sigma_sqr = r @ np.linalg.solve(HQH, r) / 2.0
        steps = intervals == k
        assert_close(
            result.sigma_sqr[steps], np.full(steps.sum(), sigma_sqr), rtol=1e-9
        )
        expected = sigma_sqr if calibration == "dynamic" else 1.0
        assert_close(diffusion[steps], np.full(steps.sum(), expected), rtol=1e-9)


# --------------------------------------------------------------------------
# Without data, the timeline is the grid
# --------------------------------------------------------------------------


@pytest.mark.parametrize("prior_name", sorted(PRIORS))
@pytest.mark.parametrize("calibration", ["none", "dynamic", "diagonal"])
@pytest.mark.parametrize("empty", ["mask", "without_data"])
def test_without_data_the_grid_nodes_are_those_of_the_grid(
    prior_name, calibration, empty
):
    prior, measure, m_0, P_0_sqr = _setup(prior_name, _lotka_volterra, (1.0, 1.0))
    obs = _data(prior)
    if empty == "mask":
        obs = obs._replace(mask=np.zeros_like(obs.mask))
    else:
        obs = obs.without_data()
    result = gaussian_filter(
        m_0, P_0_sqr, prior, measure, TS, calibration=calibration, obs_model=obs
    )
    plain = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration=calibration)
    assert obs.grid_node is not None
    grid_node = onp.asarray(obs.grid_node)
    at_grid = _grid_nodes(grid_node)

    assert_close(result.m[at_grid], plain.m, rtol=1e-12)
    assert_close(cov(result.P_sqr[at_grid]), cov(plain.P_sqr), rtol=1e-12)
    assert_close(result.log_likelihood, plain.log_likelihood, rtol=1e-12)
    smoothed, smoothed_plain = rts_smoother(result), rts_smoother(plain)
    assert_close(smoothed.m[at_grid], smoothed_plain.m, rtol=1e-12)
    assert_close(cov(smoothed.P_sqr[at_grid]), cov(smoothed_plain.P_sqr), rtol=1e-12)
    # Every step of an interval reports the interval's diffusion.
    intervals = _intervals(grid_node)
    assert_close(result.sigma_sqr, plain.sigma_sqr[intervals], rtol=1e-12)
    assert_close(result.diffusion, plain.diffusion[intervals], rtol=1e-12)
    # The ODE innovation: the grid's at the grid nodes, none at data-only nodes.
    assert_close(result.mz[grid_node], plain.mz, rtol=1e-12, atol=1e-12)
    k = plain.mz.shape[1]
    assert_close(
        result.mz[~grid_node], np.zeros((int((~grid_node).sum()), k)), rtol=0.0
    )
    assert_close(
        result.Pz_sqr[~grid_node],
        np.broadcast_to(np.eye(k), (int((~grid_node).sum()), k, k)),
        rtol=0.0,
    )
    if empty == "without_data":
        assert result.log_likelihood_obs is None and result.mz_obs is None
    else:
        assert float(result.log_likelihood_obs) == 0.0


def test_data_inside_an_interval_leave_its_diffusion_unchanged():
    """The diffusion of an interval is estimated before its data-only nodes."""
    prior, measure, m_0, P_0_sqr = _setup("iwp", _lotka_volterra, (1.0, 1.0))
    obs = _data(prior)
    result = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, obs_model=obs)
    plain = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    # Interval 1, from 0.25 to 0.5, has the first data-only node (0.3): its
    # two steps propagate with the diffusion estimated at 0.25, before the data.
    assert_close(result.sigma_sqr[:3], plain.sigma_sqr[np.array([0, 1, 1])], rtol=1e-12)
    # The data at 0.3 move the state at 0.5 (node 3 of the timeline).
    assert float(np.max(np.abs(result.m[3] - plain.m[2]))) > 1e-3


# --------------------------------------------------------------------------
# On the grid, nothing changes
# --------------------------------------------------------------------------


def _scan_of_filter_steps(m_0, P_0_sqr, prior, measure, ts, obs, calibration):
    """The fixed-grid filter as it is without data-only nodes: one step per node."""
    constant, _ = grid_transitions(prior, ts)
    assert constant is not None
    information = OnlineInformation(measure, TaylorCorrection())

    def body(x, inputs):
        t, c, mask = inputs
        out = filter_step(
            x,
            constant,
            information,
            t=t,
            calibration=as_calibration(calibration),
            scale_noise=prior.apply_state_sigma_sqr,
            observation=AffineModel(obs.H, c, obs.R_sqr),
            mask=mask,
        )
        return out.filtered, out

    _, out = jax.lax.scan(body, Gaussian(m_0, P_0_sqr), (ts[1:], obs.c_seq, obs.mask))
    return out


@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_data_on_grid_nodes_run_the_steps_of_the_grid(calibration):
    prior, measure, m_0, P_0_sqr = _setup("iwp", _lotka_volterra, (1.0, 1.0))
    on_grid = [
        Measurement(np.array([[1.0, 0.0]]), np.ones((3, 1)), TS[[2, 5, 9]], NOISE),
        Measurement(np.array([[0.0, 1.0]]), np.ones((2, 1)), TS[[5, 12]], NOISE),
    ]
    obs = prepare_observations(on_grid, prior.E0, TS)
    assert obs is not None and obs.t is None
    result = gaussian_filter(
        m_0, P_0_sqr, prior, measure, TS, calibration=calibration, obs_model=obs
    )
    out = _scan_of_filter_steps(m_0, P_0_sqr, prior, measure, TS, obs, calibration)
    assert_close(result.t, TS, rtol=0.0)
    assert_close(result.m[1:], out.filtered.m, rtol=0.0)
    assert_close(result.P_sqr[1:], out.filtered.P_sqr, rtol=0.0)
    assert_close(result.log_likelihood_obs, np.sum(out.log_likelihood_obs), rtol=0.0)


# --------------------------------------------------------------------------
# Repeated times
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t_obs", [0.75, 1.1], ids=["grid_node", "data_only"])
def test_two_data_at_one_node_are_both_conditioned_on(t_obs):
    prior, measure, m_0, P_0_sqr = _setup("iwp")
    A = np.array([[1.0, 0.0]])
    times = onp.array([t_obs, t_obs])
    z = np.array([[0.4], [0.6]])
    repeated = [Measurement(A, z, times, noise=NOISE)]
    separate = [Measurement(A, z[i : i + 1], times[i : i + 1], NOISE) for i in (0, 1)]
    one = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, measurements=repeated)
    two = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, measurements=separate)
    assert_close(one.m, two.m, rtol=1e-12)
    assert_close(cov(one.P_sqr), cov(two.P_sqr), rtol=1e-12)
    assert_close(one.log_likelihood_obs, two.log_likelihood_obs, rtol=1e-12)
    # Conditioning on the mean of the two data with half the noise variance
    # gives the same posterior.
    mean = [Measurement(A, np.array([[0.5]]), times[:1], noise=NOISE / 2)]
    averaged = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, measurements=mean)
    assert_close(one.m, averaged.m, rtol=1e-12)
    assert_close(cov(one.P_sqr), cov(averaged.P_sqr), rtol=1e-12)


# --------------------------------------------------------------------------
# The result on the timeline
# --------------------------------------------------------------------------


def test_interpolation_and_samples_work_on_the_timeline():
    prior, measure, m_0, P_0_sqr = _setup("iwp")
    result = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, measurements=_sensors())
    smoothed = rts_smoother(result)
    at_nodes = interpolate(result, prior, result.t)
    assert_close(at_nodes.m, smoothed.m, rtol=1e-12)
    # Between a grid node and a data-only node.
    inside = interpolate(result, prior, np.array([1.05, 1.12]))
    assert bool(np.all(np.isfinite(inside.m)))
    filtered = interpolate(result, prior, result.t, smoothed=False)
    assert_close(filtered.m, result.m, rtol=1e-12)
    path = sample_posterior(jax.random.PRNGKey(0), result)
    assert path.shape == result.m.shape


def test_measurements_are_the_obs_model_of_the_state_selector():
    prior, measure, m_0, P_0_sqr = _setup("precond_iwp")
    surrogate = taylor_surrogate(
        measure, np.zeros((TS.shape[0] - 1, prior.state_dim)), ts=TS[1:]
    )
    for solve in (
        lambda **kw: gaussian_filter(m_0, P_0_sqr, prior, measure, TS, **kw),
        lambda **kw: affine_filter(m_0, P_0_sqr, prior, surrogate, TS, **kw),
    ):
        expected = solve(obs_model=_data(prior))
        result = solve(measurements=_sensors())
        assert_close(result.m, expected.m, rtol=0.0)
        assert_close(result.log_likelihood_obs, expected.log_likelihood_obs, rtol=0.0)
        with pytest.raises(ValueError, match="not both"):
            solve(obs_model=_data(prior), measurements=_sensors())
    no_data = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, measurements=[])
    assert no_data.log_likelihood_obs is None


def test_jit_with_the_obs_model_closed_over_or_traced():
    prior, measure, m_0, P_0_sqr = _setup("iwp", _lotka_volterra, (1.0, 1.0))
    obs = _data(prior)

    def solve(obs_model, ts=TS):
        return gaussian_filter(m_0, P_0_sqr, prior, measure, ts, obs_model=obs_model)

    expected = solve(obs)
    for result in (
        jax.jit(lambda: solve(obs))(),
        jax.jit(solve)(obs),  # the timeline is traced
        jax.jit(lambda ts: solve(obs, ts))(TS),  # the grid is traced
        jax.jit(lambda o, ts: solve(o, ts))(obs, TS),
    ):
        assert_close(result.t, expected.t, rtol=0.0)
        assert_close(result.m, expected.m, rtol=1e-12)
        assert_close(result.log_likelihood_obs, expected.log_likelihood_obs, rtol=1e-12)


def test_checkpointing_gives_the_same_result_and_gradient():
    prior, measure, m_0, P_0_sqr = _setup("iwp", _lotka_volterra, (1.0, 1.0))
    obs = _data(prior)

    def evidence(scale, checkpoint):
        result = gaussian_filter(
            scale * m_0,
            P_0_sqr,
            prior,
            measure,
            TS,
            obs_model=obs,
            checkpoint=checkpoint,
        )
        return result.log_likelihood_obs

    for f in (evidence, jax.grad(evidence)):
        assert_close(f(1.0, True), f(1.0, False), rtol=1e-12)


def test_a_grid_from_jax_linspace_takes_the_data_of_a_numpy_grid():
    """The grid nodes of the timeline take the solver's values, which differ
    from NumPy's linspace in the last digits."""
    prior, measure, m_0, P_0_sqr = _setup("iwp")
    obs = _data(prior)
    ts = np.linspace(0.0, 3.0, 13)
    result = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, obs_model=obs)
    expected = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, obs_model=obs)
    assert_close(result.t[_grid_nodes(onp.asarray(obs.grid_node))], ts, rtol=0.0)
    assert_close(result.m, expected.m, rtol=1e-10)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_a_timeline_of_another_grid_raises():
    prior, measure, m_0, P_0_sqr = _setup("iwp")
    obs = _data(prior)

    def solve(ts, obs_model=obs):
        return gaussian_filter(m_0, P_0_sqr, prior, measure, ts, obs_model=obs_model)

    for ts in (TS[:-1], onp.linspace(0.0, 3.3, 13)):
        with pytest.raises(ValueError, match="belongs to another grid"):
            solve(ts)
    # A grid node shifted onto the data-only node at 1.1: not increasing.
    with pytest.raises(ValueError, match="belongs to another grid"):
        solve(TS, obs._replace(t=onp.where(obs.t == 1.0, 1.1 - 1e-8, obs.t)))
    with pytest.raises(ValueError, match="both be given"):
        solve(TS, obs._replace(grid_node=None))
    with pytest.raises(ValueError, match="one entry per step"):
        solve(TS, obs._replace(grid_node=obs.grid_node[1:]))
    with pytest.raises(ValueError, match="steps but its timeline has"):
        solve(TS, obs._replace(c_seq=obs.c_seq[1:]))
