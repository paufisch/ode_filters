"""Reference scenarios: solver outputs recorded before the restructure.

Each scenario runs one public solver configuration and returns its outputs as
*coordinate-invariant* arrays -- means and covariances (never square-root
factors, which are unique only up to an orthogonal factor) in the original
state coordinates, log-likelihoods and diffusion estimates. These survive any
change of internal representation, so ``test_reference_outputs`` can compare a
rewritten solver against the recorded values.

To record: ``uv run python -m test.integration.reference.generate``. Every
regeneration that changes a value must be explained in ``README.md`` next to
this file.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as np
from jax import Array

from ode_filters import (
    IWP,
    AffineSurrogate,
    BlackBoxMeasurement,
    InferenceProblem,
    IOUPPrior,
    IteratedTaylorCorrection,
    JointPrior,
    MaternPrior,
    ODEInformation,
    PController,
    PrecondIWP,
    PrecondMaternPrior,
    QuadratureCorrection,
    TaylorCorrection,
    affine_filter,
    aggregate_sigma_sqr,
    gaussian_filter,
    gaussian_filter_adaptive,
    iterated_smoother,
    marginal_loglik,
    posthoc_mle_sigma_sqr,
    prepare_observations,
    rescale_sqr_seq,
    rts_smoother,
    taylor_mode_initialization,
)
from ode_filters.information import (
    Conservation,
    Measurement,
    ODEInformationWithHidden,
    SecondOrderODEInformation,
)
from test.helpers import LINEAR_2D, LOGISTIC, SIR, cov

N = 20

Outputs = dict[str, Array]


def _logistic_setup(prior):
    m_0, P_0_sqr = taylor_mode_initialization(LOGISTIC.vf, LOGISTIC.x0, q=prior.q)
    measure = ODEInformation(LOGISTIC.vf, prior.E0, prior.E1)
    return m_0, P_0_sqr, measure


def _logistic_observations(prior):
    ts = np.linspace(*LOGISTIC.tspan, N + 1)
    t_obs = ts[5::5]
    z = (1.0 / (1.0 + np.exp(-t_obs)) + 0.02 * np.cos(7.0 * t_obs))[:, None]
    return prepare_observations(
        [Measurement(np.eye(1), z, t_obs, noise=1e-3)], prior.E0, ts
    )


def _grid(tspan) -> Array:
    return np.linspace(*tspan, N + 1)


def _filter_outputs(res, *, smooth: bool = True) -> Outputs:
    """Coordinate-invariant outputs of a ``gaussian_filter`` result."""
    m_pred, P_pred = res.m_pred, cov(res.P_pred_sqr)
    G, d, P_back = res.G_back, res.d_back, cov(res.P_back_sqr)
    out = {
        "m": res.m,
        "P": cov(res.P_sqr),
        "log_likelihood": res.log_likelihood,
        "m_pred": m_pred,
        "P_pred": P_pred,
        "G_back": G,
        "d_back": d,
        "P_back": P_back,
        "mz": res.mz,
        "Pz": cov(res.Pz_sqr),
        "sigma_sqr": res.sigma_sqr,
    }
    if res.log_likelihood_obs is not None:
        out["log_likelihood_obs"] = res.log_likelihood_obs
        out["mz_obs"] = res.mz_obs
        out["Pz_obs"] = cov(res.Pz_obs_sqr)
    if smooth:
        m_s, P_s_sqr = rts_smoother(res)
        out["m_smooth"] = m_s
        out["P_smooth"] = cov(P_s_sqr)
    return out


CORRECTIONS = {
    "ek1": None,
    "ek0": TaylorCorrection(order=0),
    "iekf": IteratedTaylorCorrection(max_iters=3),
    "slr": QuadratureCorrection(n_nodes=3),
}


def _logistic(calibration: str, correction: str) -> Callable[[], Outputs]:
    def run() -> Outputs:
        prior = IWP(q=2, d=1)
        m_0, P_0_sqr, measure = _logistic_setup(prior)
        res = gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            _grid(LOGISTIC.tspan),
            calibration=calibration,
            correction=CORRECTIONS[correction],
        )
        return _filter_outputs(res)

    return run


def _logistic_obs(calibration: str) -> Callable[[], Outputs]:
    def run() -> Outputs:
        prior = IWP(q=2, d=1)
        m_0, P_0_sqr, measure = _logistic_setup(prior)
        res = gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            _grid(LOGISTIC.tspan),
            calibration=calibration,
            obs_model=_logistic_observations(prior),
        )
        return _filter_outputs(res)

    return run


def _linear_2d_precond(calibration: str) -> Callable[[], Outputs]:
    def run() -> Outputs:
        prior = PrecondIWP(q=3, d=2)
        m_0, P_0_sqr = taylor_mode_initialization(LINEAR_2D.vf, LINEAR_2D.x0, q=3)
        measure = ODEInformation(LINEAR_2D.vf, prior.E0, prior.E1)
        res = gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            _grid(LINEAR_2D.tspan),
            calibration=calibration,
        )
        return _filter_outputs(res)

    return run


def _matern_dynamic() -> Outputs:
    prior = MaternPrior(q=2, d=1, length_scale=1.5)
    m_0, P_0_sqr, measure = _logistic_setup(prior)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, _grid(LOGISTIC.tspan))
    return _filter_outputs(res)


def _ioup_dynamic() -> Outputs:
    prior = IOUPPrior(q=2, d=1, rate=-0.5)
    m_0, P_0_sqr, measure = _logistic_setup(prior)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, _grid(LOGISTIC.tspan))
    return _filter_outputs(res)


def _hidden_setup(prior):
    def vf(x, u, *, t):
        return -0.8 * x + u

    measure = ODEInformationWithHidden(vf, prior.E0_x, prior.E1, prior.E0_hidden)
    D = prior.E0.shape[1]
    m_0 = np.zeros(D).at[0].set(1.0)
    P_0_sqr = np.diag(np.full(D, 0.1))
    return m_0, P_0_sqr, measure


def _joint_hidden_obs(calibration: str) -> Callable[[], Outputs]:
    def run() -> Outputs:
        prior = JointPrior(IWP(q=2, d=1), MaternPrior(q=1, d=1, length_scale=1.0))
        m_0, P_0_sqr, measure = _hidden_setup(prior)
        ts = np.linspace(0.0, 2.0, N + 1)
        t_obs = ts[4::4]
        z = np.exp(-t_obs)[:, None] + 0.1 * np.sin(3.0 * t_obs)[:, None]
        obs = prepare_observations(
            [Measurement(np.array([[1.0, 0.0]]), z, t_obs, noise=1e-3)],
            prior.E0,
            ts,
        )
        res = gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            _grid((0.0, 2.0)),
            calibration=calibration,
            obs_model=obs,
        )
        return _filter_outputs(res)

    return run


def _precond_joint_dynamic() -> Outputs:
    prior = JointPrior(PrecondIWP(q=2, d=1), PrecondIWP(q=1, d=1))
    m_0, P_0_sqr, measure = _hidden_setup(prior)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, _grid((0.0, 2.0)))
    return _filter_outputs(res)


def _second_order(calibration: str) -> Callable[[], Outputs]:
    def run() -> Outputs:
        def vf(x, v, *, t):
            return -4.0 * x - 0.1 * v

        prior = IWP(q=3, d=1)
        m_0, P_0_sqr = taylor_mode_initialization(
            vf, (np.array([1.0]), np.array([0.0])), q=3, order=2
        )
        measure = SecondOrderODEInformation(vf, prior.E0, prior.E1, prior.E2)
        res = gaussian_filter(
            m_0, P_0_sqr, prior, measure, _grid((0.0, 3.0)), calibration=calibration
        )
        return _filter_outputs(res)

    return run


def _sir_conservation() -> Outputs:
    prior = IWP(q=2, d=3)
    m_0, P_0_sqr = taylor_mode_initialization(SIR.vf, SIR.x0, q=2)
    measure = ODEInformation(
        SIR.vf,
        prior.E0,
        prior.E1,
        constraints=[Conservation(np.ones((1, 3)), np.array([1.0]))],
    )
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, _grid(SIR.tspan))
    return _filter_outputs(res)


def _adaptive(with_obs: bool) -> Callable[[], Outputs]:
    def run() -> Outputs:
        prior = IWP(q=2, d=1)
        m_0, P_0_sqr, measure = _logistic_setup(prior)
        save_at = np.linspace(*LOGISTIC.tspan, 6)
        obs = None
        if with_obs:
            z = (1.0 / (1.0 + np.exp(-save_at[2:4])))[:, None] + 0.01
            obs = prepare_observations(
                [Measurement(np.eye(1), z, save_at[2:4], noise=1e-3)],
                prior.E0,
                save_at,
            )
        res = gaussian_filter_adaptive(
            m_0,
            P_0_sqr,
            prior,
            measure,
            save_at,
            obs_model=obs,
            atol=1e-6,
            rtol=1e-4,
            smoother=not with_obs,
        )
        # Recorded when the adaptive path still summed the data evidence into
        # log_likelihood; it now reports it separately.
        log_likelihood = res.log_likelihood
        if with_obs:
            log_likelihood = log_likelihood + res.log_likelihood_obs
        out = {
            "m": res.m,
            "P": cov(res.P_sqr),
            "log_likelihood": log_likelihood,
            "success": res.success,
            "mz": res.mz,
            "Pz": cov(res.Pz_sqr),
        }
        if with_obs:
            out["mz_obs"] = res.mz_obs
            out["Pz_obs"] = cov(res.Pz_obs_sqr)
        else:
            m_s, P_s_sqr = rts_smoother(res)
            out["m_smooth"] = m_s
            out["P_smooth"] = cov(P_s_sqr)
        return out

    return run


def _ipls() -> Outputs:
    prior = IWP(q=2, d=1)
    m_0, P_0_sqr, measure = _logistic_setup(prior)
    res = iterated_smoother(
        m_0,
        P_0_sqr,
        prior,
        measure,
        _grid(LOGISTIC.tspan),
        linearization="slr",
        n_iters=2,
        n_nodes=3,
        obs_model=_logistic_observations(prior),
    )
    return {
        "m": res.m,
        "P": cov(res.P_sqr),
        "m_filt": res.filter.m,
        "P_filt": cov(res.filter.P_sqr),
        "log_likelihood": res.filter.log_likelihood,
        "log_likelihood_obs": res.filter.log_likelihood_obs,
    }


def _affine_filter() -> Outputs:
    """Frozen-linearization filter: EK1 linearized once at the prior means."""
    prior = IWP(q=2, d=1)
    m_0, P_0_sqr, measure = _logistic_setup(prior)
    ts, h = np.linspace(*LOGISTIC.tspan, N + 1, retstep=True)
    A = prior.A(h)
    m_lin = np.stack([np.linalg.matrix_power(A, n) @ m_0 for n in range(1, N + 1)])
    H_seq, c_seq = jax.vmap(lambda m, t: measure.linearize(m, t=t))(m_lin, ts[1:])
    surrogate = AffineSurrogate(H_seq, c_seq, np.zeros((N, 1, 1)))
    res = affine_filter(
        m_0, P_0_sqr, prior, surrogate, ts, obs_model=_logistic_observations(prior)
    )
    return {
        "m": res.m,
        "P": cov(res.P_sqr),
        "G_back": res.G_back,
        "d_back": res.d_back,
        "P_back": cov(res.P_back_sqr),
        "log_likelihood": res.log_likelihood,
        "log_likelihood_obs": res.log_likelihood_obs,
    }


def _marginal_loglik() -> Outputs:
    prior = IWP(q=2, d=1)
    ts = np.linspace(0.0, 2.0, N + 1)
    t_obs = ts[2::2]
    obs = prepare_observations(
        [Measurement(np.eye(1), np.exp(-1.3 * t_obs)[:, None], t_obs, noise=1e-4)],
        prior.E0,
        ts,
    )

    def build(theta):
        def vf(x, *, t):
            return -theta * x

        m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
        return m_0, P_0_sqr, ODEInformation(vf, prior.E0, prior.E1)

    problem = InferenceProblem(build=build, prior=prior, ts=ts)
    theta = np.array(1.1)
    ll_ode, ll_obs = marginal_loglik(theta, obs, model=problem, channel="both")
    return {
        "log_likelihood": ll_ode,
        "log_likelihood_obs": ll_obs,
        "grad_obs": jax.grad(lambda th: marginal_loglik(th, obs, model=problem))(theta),
        "grad_ode": jax.grad(
            lambda th: marginal_loglik(th, obs, model=problem, channel="ode")
        )(theta),
    }


# ---------------------------------------------------------------------------
# Added 2026-10-14 (see README.md). Matern scenarios use length_scale=1.0, where
# the library's length-scale convention and the textbook one coincide. Their
# outputs are kept compact: no per-step predictions or backward conditionals
# unless the scenario is about them.
# ---------------------------------------------------------------------------


def _compact_outputs(res) -> Outputs:
    """Filtered and smoothed marginals, evidence and diffusion of a fixed-grid run."""
    m_s, P_s_sqr = rts_smoother(res)
    return {
        "m": res.m,
        "P": cov(res.P_sqr),
        "log_likelihood": res.log_likelihood,
        "sigma_sqr": res.sigma_sqr,
        "m_smooth": m_s,
        "P_smooth": cov(P_s_sqr),
    }


def _adaptive_outputs(res) -> Outputs:
    """Save-time marginals, evidence, step counts and the accepted steps."""
    steps = res.steps
    n = int(steps.count)
    return {
        "m": res.m,
        "P": cov(res.P_sqr),
        "log_likelihood": res.log_likelihood,
        "n_accepted": res.n_accepted,
        "n_rejected": res.n_rejected,
        "h": steps.h[:n],
        "sigma_sqr": steps.sigma_sqr[:n],
    }


def _adaptive_variant(**options) -> Callable[[], Outputs]:
    """The logistic problem of ``adaptive_smoother`` with another error control."""

    def run() -> Outputs:
        prior = IWP(q=2, d=1)
        m_0, P_0_sqr, measure = _logistic_setup(prior)
        res = gaussian_filter_adaptive(
            m_0,
            P_0_sqr,
            prior,
            measure,
            np.linspace(*LOGISTIC.tspan, 6),
            atol=1e-6,
            rtol=1e-4,
            record_steps=256,
            **options,
        )
        return _adaptive_outputs(res)

    return run


def _adaptive_diagonal() -> Outputs:
    """Per-component calibration of a two-dimensional linear ODE."""
    prior = IWP(q=2, d=2)
    m_0, P_0_sqr = taylor_mode_initialization(LINEAR_2D.vf, LINEAR_2D.x0, q=2)
    measure = ODEInformation(LINEAR_2D.vf, prior.E0, prior.E1)
    res = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*LINEAR_2D.tspan, 6),
        calibration="diagonal",
        atol=1e-6,
        rtol=1e-4,
        record_steps=256,
    )
    return _adaptive_outputs(res)


def _ieks(n_iters: int, with_obs: bool) -> Callable[[], Outputs]:
    """IEKS on a coarse grid through the logistic's steep rise, where passes matter.

    Each pass moves the smoothed means by up to ``2e-4`` (the first) and
    ``3e-6`` (the next two). The convergence diagnostic ``result.step`` is not
    recorded: after the first pass it is a difference of means at the rounding
    level.
    """

    def run() -> Outputs:
        x0 = 0.05
        prior = IWP(q=2, d=1)
        m_0, P_0_sqr = taylor_mode_initialization(LOGISTIC.vf, np.array([x0]), q=2)
        measure = ODEInformation(LOGISTIC.vf, prior.E0, prior.E1)
        ts = _grid((0.0, 10.0))
        obs = None
        if with_obs:
            t_obs = ts[5::5]
            z = x0 * np.exp(t_obs) / (1.0 - x0 + x0 * np.exp(t_obs))
            z = z + 0.02 * np.cos(7.0 * t_obs)
            obs = prepare_observations(
                [Measurement(np.eye(1), z[:, None], t_obs, noise=1e-3)], prior.E0, ts
            )
        res = iterated_smoother(
            m_0,
            P_0_sqr,
            prior,
            measure,
            ts,
            linearization="taylor",
            n_iters=n_iters,
            obs_model=obs,
        )
        out = {
            "m": res.m,
            "P": cov(res.P_sqr),
            "m_filt": res.filter.m,
            "log_likelihood": res.filter.log_likelihood,
        }
        if with_obs:
            out["log_likelihood_obs"] = res.filter.log_likelihood_obs
        return out

    return run


def _precond_matern_dynamic() -> Outputs:
    prior = PrecondMaternPrior(q=2, d=1, length_scale=1.0)
    m_0, P_0_sqr, measure = _logistic_setup(prior)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, _grid(LOGISTIC.tspan))
    return _filter_outputs(res)


def _blackbox_dynamic() -> Outputs:
    """The logistic ODE residual as a black box (Jacobian by autodiff)."""
    prior = IWP(q=2, d=1)
    m_0, P_0_sqr, _ = _logistic_setup(prior)

    def residual(state, *, t):
        return state[1:2] - LOGISTIC.vf(state[:1], t=t)

    measure = BlackBoxMeasurement(residual, state_dim=3, obs_dim=1)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, _grid(LOGISTIC.tspan))
    return _compact_outputs(res)


def _posthoc_rescale() -> Outputs:
    """Runs without calibration, rescaled by the global and the running MLE.

    The joint run rescales its state block only (``prior=``).
    """
    prior = IWP(q=2, d=1)
    m_0, P_0_sqr, measure = _logistic_setup(prior)
    res = gaussian_filter(
        m_0, P_0_sqr, prior, measure, _grid(LOGISTIC.tspan), calibration="none"
    )
    sigma_sqr = posthoc_mle_sigma_sqr(res.mz, res.Pz_sqr)
    running = aggregate_sigma_sqr(res.sigma_sqr, kind="running")
    _, P_s_sqr = rts_smoother(res)

    joint = JointPrior(IWP(q=2, d=1), MaternPrior(q=1, d=1, length_scale=1.0))
    m_0, P_0_sqr, measure = _hidden_setup(joint)
    joint_res = gaussian_filter(
        m_0, P_0_sqr, joint, measure, _grid((0.0, 2.0)), calibration="none"
    )
    joint_sigma_sqr = posthoc_mle_sigma_sqr(joint_res.mz, joint_res.Pz_sqr)
    return {
        "sigma_sqr": sigma_sqr,
        "P": cov(rescale_sqr_seq(res.P_sqr, sigma_sqr)),
        "P_smooth": cov(rescale_sqr_seq(P_s_sqr, sigma_sqr)),
        "sigma_sqr_running": running,
        "P_running": cov(rescale_sqr_seq(res.P_sqr[1:], running)),
        "joint_sigma_sqr": joint_sigma_sqr,
        "joint_P": cov(rescale_sqr_seq(joint_res.P_sqr, joint_sigma_sqr, prior=joint)),
    }


def _non_autonomous() -> Outputs:
    """``x' = cos(t) x`` from the exact initial derivatives of ``exp(sin t)``."""

    def vf(x, *, t):
        return np.cos(t) * x

    def solution(t):
        return np.exp(np.sin(t))

    derivatives, f = [], solution
    for _ in range(4):
        derivatives.append(f(0.0))
        f = jax.grad(f)
    prior = IWP(q=3, d=1)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    res = gaussian_filter(
        np.array(derivatives), np.zeros((4, 4)), prior, measure, _grid((0.0, 5.0))
    )
    return _compact_outputs(res)


SCENARIOS: dict[str, Callable[[], Outputs]] = {
    **{
        f"logistic_{cal}_ek1": _logistic(cal, "ek1")
        for cal in ("none", "dynamic", "diagonal", "diagonal_ekf0")
    },
    **{f"logistic_dynamic_{c}": _logistic("dynamic", c) for c in ("ek0", "iekf")},
    "logistic_none_ek0": _logistic("none", "ek0"),
    "logistic_none_slr": _logistic("none", "slr"),
    "logistic_obs_none": _logistic_obs("none"),
    "logistic_obs_dynamic": _logistic_obs("dynamic"),
    **{
        f"linear_2d_precond_{cal}": _linear_2d_precond(cal)
        for cal in ("none", "dynamic", "diagonal")
    },
    "matern_dynamic": _matern_dynamic,
    "ioup_dynamic": _ioup_dynamic,
    "joint_hidden_obs_none": _joint_hidden_obs("none"),
    "joint_hidden_obs_dynamic": _joint_hidden_obs("dynamic"),
    "precond_joint_dynamic": _precond_joint_dynamic,
    "second_order_dynamic": _second_order("dynamic"),
    "second_order_diagonal_ekf0": _second_order("diagonal_ekf0"),
    "sir_conservation_dynamic": _sir_conservation,
    "adaptive_smoother": _adaptive(with_obs=False),
    "adaptive_obs": _adaptive(with_obs=True),
    "ipls": _ipls,
    "affine_filter": _affine_filter,
    "marginal_loglik": _marginal_loglik,
    "adaptive_running_mean": _adaptive_variant(sigma_in_error="running_mean"),
    "adaptive_p_controller": _adaptive_variant(controller=PController(order=2)),
    "adaptive_diagonal": _adaptive_diagonal,
    **{
        f"ieks_{n}{suffix}": _ieks(n, with_obs)
        for n in (0, 1, 3)
        for suffix, with_obs in (("", False), ("_obs", True))
    },
    "precond_matern_dynamic": _precond_matern_dynamic,
    "blackbox_dynamic": _blackbox_dynamic,
    "posthoc_rescale": _posthoc_rescale,
    "non_autonomous": _non_autonomous,
}


def run_all() -> dict[str, Array]:
    """All scenario outputs, flattened to ``"<scenario>/<field>"`` keys."""
    flat = {}
    for name, scenario in SCENARIOS.items():
        for field, value in scenario().items():
            flat[f"{name}/{field}"] = np.asarray(value)
    return flat
