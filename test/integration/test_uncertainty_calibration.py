"""Is the posterior uncertainty the right size? Two-sided calibration checks.

Two kinds of check:

**Calibration bands on ODEs with known solutions.** For each calibration
policy -- ``"dynamic"``, post-hoc rescaling of a ``"none"`` run by
``posthoc_mle_sigma_sqr``, and ``"diagonal"`` -- and four problems, the
standardized error ``z = (mean - truth) / std`` of the solution components over
the trajectory (initial node excluded: it is exact, with zero variance) has an
RMS that must lie in ``[10^-1.5, 10^1.5]``: a posterior that is over- or
under-confident by more than one and a half orders of magnitude fails. The
numerical error of an ODE solver is not a draw from its posterior, so an RMS of
exactly one is not expected; well-calibrated solvers sit within an order of
magnitude of it. Setup: IWP(q=3), EK1, ``h = 0.1`` on ``[0, 10]``. Measured
RMS z (filter / smoother):

==============  ============  ============  ============  ============
policy          logistic      cubic decay   linear 2D     Hopf
==============  ============  ============  ============  ============
dynamic         0.33 / 0.27   0.21 / 0.20   0.56 / 0.58   0.36 / 0.31
diagonal        0.33 / 0.27   0.21 / 0.20   0.33 / 0.29   0.38 / 0.32
post hoc        0.35 / 5e-4   0.39 / 4e-3   0.53 / 1e-3   0.26 / 1e-2
==============  ============  ============  ============  ============

The post-hoc smoother is the exception, and a property of the method rather
than a defect: under a constant diffusion the smoothed mean superconverges at
the grid nodes (on the logistic problem its interior error is 1e-11 -- 1e-13
against the filter's 1e-7) while its standard deviation is only 1.1 -- 1.6
times smaller than the filter's, so the smoother is conservative by two to
three and a half orders of magnitude. A per-step (dynamic) diffusion destroys
the superconvergence, and the smoother is calibrated like the filter. The
post-hoc smoother is therefore held to its own band, which still rejects an
over-confident posterior and a further loss of more than one and a half orders
of magnitude below the lowest measured value.

The RMS z depends on the order, and ``q = 3`` is where EK1 is best
calibrated. Over the same problems and policies (post-hoc smoother aside) it
measures 0.11 -- 4.5 at ``q = 1``, 0.058 -- 0.27 (filter) and 0.023 -- 0.096
(smoother) at ``q = 2``, and 0.061 -- 0.78 (filter) and 0.022 -- 0.10
(smoother) at ``q = 4, 5``: there the smoother is conservative by up to 1.7
orders of magnitude and would fail the lower edge of the band.

**Exact calibration on a model the filter gets exactly right.** The truth is
drawn from the prior itself (``BasePrior.sample``); the ODE information
``x' = L x + g(t)`` is observed with noise, by making the forcing ``g`` at the
grid nodes equal the sample's residual plus noise, and the first solution
component is observed with noise at every other node. The filter then computes
the exact posterior of a linear-Gaussian model (EK1 is exact for a linear
field, the diffusion is the sampling one), so across independent draws

- each filtered and each smoothed marginal gives a chi-square statistic
  ``(x - m).T P^-1 (x - m)`` with ``D`` degrees of freedom, and their sum over
  the draws is chi-square with ``n_draws * D``;
- whitening the whole path through the smoother's backward conditionals gives
  independent standard normals, ``(K * D)`` per draw;
- the whitened innovations of both channels are independent standard normals.

Each test is two-sided at ``ALPHA`` (Bonferroni-corrected over the nodes), so
an exactly calibrated filter fails it with probability below ``ALPHA`` for any
seed; the seed is fixed. Measured over the four priors: chi2/dof 0.997 --
1.009 (filter marginals), 0.988 -- 1.012 (smoother marginals), 1.002 -- 1.012
(whole paths, p = 0.04 -- 0.74), 1.000 -- 1.012 (innovations). The check has
power: a solver diffusion 10% too large gives chi2/dof 0.92 on the paths
(p = 7e-35), 1.5 times too large 0.71
(``test_the_exact_check_rejects_a_wrong_diffusion``). The coverage of the 95%
intervals is a much weaker diagnostic -- it stays at 0.951 even at 1.5 times
the diffusion, where the data pin the observed component -- and is checked
only against its sampling error.
"""

from __future__ import annotations

from functools import cache

import equinox as eqx
import jax
import jax.numpy as np
import pytest
from scipy import stats

from ode_filters import (
    IWP,
    IOUPPrior,
    MaternPrior,
    Measurement,
    ODEInformation,
    PrecondIWP,
    gaussian_filter,
    posthoc_mle_sigma_sqr,
    prepare_observations,
    rescale_sqr_seq,
    rts_smoother,
    taylor_mode_initialization,
)
from test.helpers import Problem

# ---------------------------------------------------------------------------
# Calibration bands on problems with known solutions
# ---------------------------------------------------------------------------

L = np.array([[-0.5, 1.0], [-1.0, -0.5]])


def _logistic(x, *, t):
    return x * (1.0 - x)


def _cubic_decay(x, *, t):
    return -(x**3)


def _linear_2d(x, *, t):
    return L @ x


def _hopf(x, *, t):
    """Hopf normal form: radius ``r' = r (1 - r^2)``, angle ``theta' = 1``."""
    r_sqr = x[0] ** 2 + x[1] ** 2
    return np.array([-x[1] + x[0] * (1 - r_sqr), x[0] + x[1] * (1 - r_sqr)])


def _hopf_solution(ts, r_0=0.1):
    r = 1.0 / np.sqrt(1.0 + (1.0 / r_0**2 - 1.0) * np.exp(-2.0 * ts))
    return np.stack([r * np.cos(ts), r * np.sin(ts)], axis=-1)


TSPAN = (0.0, 10.0)
PROBLEMS = {
    "logistic": Problem(
        _logistic,
        np.array([0.1]),
        TSPAN,
        solution=lambda ts: (1.0 / (1.0 + 9.0 * np.exp(-ts)))[:, None],
    ),
    "cubic_decay": Problem(
        _cubic_decay,
        np.array([1.0]),
        TSPAN,
        solution=lambda ts: (1.0 / np.sqrt(1.0 + 2.0 * ts))[:, None],
    ),
    "linear_2d": Problem(
        _linear_2d,
        np.array([1.0, 0.0]),
        TSPAN,
        solution=lambda ts: jax.vmap(
            lambda t: jax.scipy.linalg.expm(L * t) @ np.array([1.0, 0.0])
        )(ts),
    ),
    "hopf": Problem(_hopf, np.array([0.1, 0.0]), TSPAN, solution=_hopf_solution),
}
POLICIES = ("dynamic", "posthoc", "diagonal")
Q = 3
N_STEPS = 100

BAND = (10**-1.5, 10**1.5)
# See the module docstring: conservative by up to 3.5 orders, measured.
POSTHOC_SMOOTHER_BAND = (1e-5, 10**1.5)


def _rms_z(m, P_sqr, E0, truth):
    """RMS of ``(E0 m - truth) / std`` over the nodes after the initial one."""
    std = np.sqrt(np.sum((P_sqr @ E0.T) ** 2, axis=-2))
    z = (m @ E0.T - truth) / std
    return np.sqrt(np.mean(z[1:] ** 2))


@cache
def _rms_z_of(problem_name: str, policy: str) -> tuple[float, float]:
    """RMS z of the filter and the smoother of one solve."""
    problem = PROBLEMS[problem_name]
    assert problem.solution is not None
    prior = IWP(Q, problem.d)
    m_0, P_0_sqr = taylor_mode_initialization(problem.vf, problem.x0, Q)
    measure = ODEInformation(problem.vf, prior.E0, prior.E1)
    ts = np.linspace(*problem.tspan, N_STEPS + 1)

    @jax.jit
    def run():
        calibration = "none" if policy == "posthoc" else policy
        res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration=calibration)
        m_s, P_s_sqr = rts_smoother(res)
        P_f_sqr = res.P_sqr
        if policy == "posthoc":
            # A constant diffusion scales every covariance of the run; the means
            # do not change (zero ODE noise, exact initial state).
            sigma_sqr = posthoc_mle_sigma_sqr(res.mz, res.Pz_sqr)
            P_f_sqr = rescale_sqr_seq(P_f_sqr, sigma_sqr)
            P_s_sqr = rescale_sqr_seq(P_s_sqr, sigma_sqr)
        truth = problem.solution(ts)
        return (
            _rms_z(res.m, P_f_sqr, prior.E0, truth),
            _rms_z(m_s, P_s_sqr, prior.E0, truth),
        )

    z_filter, z_smoother = run()
    return float(z_filter), float(z_smoother)


@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("problem_name", sorted(PROBLEMS))
def test_rms_z_rejects_over_and_under_confidence(problem_name, policy):
    z_filter, z_smoother = _rms_z_of(problem_name, policy)
    low, high = BAND
    assert low <= z_filter <= high, f"filter RMS z {z_filter:.3g}"
    if policy == "posthoc":
        low, high = POSTHOC_SMOOTHER_BAND
        # The superconvergent smoother is the more conservative one.
        assert z_smoother < z_filter
    assert low <= z_smoother <= high, f"smoother RMS z {z_smoother:.3g}"


# ---------------------------------------------------------------------------
# Exact calibration: truth drawn from the prior
# ---------------------------------------------------------------------------

EXACT_TS = np.linspace(0.0, 4.0, 41)
EXACT_H = 0.1
R_ODE = 0.05**2
R_OBS = 0.1**2
OBS_EVERY = 2
N_DRAWS = 200
ALPHA = 1e-6
Z_95 = 1.959963984540054

EXACT_PRIORS = {
    "iwp": lambda: IWP(q=2, d=2, Xi=np.array([[1.0, 0.3], [0.3, 0.5]])),
    "precond_iwp": lambda: PrecondIWP(q=3, d=2),
    # length_scale=1: the same process under either length-scale convention.
    "matern": lambda: MaternPrior(q=2, d=2, length_scale=1.0),
    # The rate is the ODE's linear part: a probabilistic exponential integrator.
    "ioup": lambda: IOUPPrior(q=2, d=2, rate=L),
}


def _whitened_sqr_norm(x, P_sqr):
    """``x.T P^-1 x`` for ``P = P_sqr.T @ P_sqr``."""
    v = jax.scipy.linalg.solve_triangular(P_sqr.T, x, lower=True)
    return np.sum(v**2)


def _two_sided_p(statistic, dof: int) -> float:
    cdf = stats.chi2.cdf(float(statistic), dof)
    return 2.0 * min(cdf, 1.0 - cdf)


@cache
def _exact_statistics(prior_name: str, solver_xi_scale: float = 1.0):
    """Chi-square statistics of ``N_DRAWS`` solves on data drawn from the prior.

    ``solver_xi_scale`` multiplies the diffusion the *solver* assumes (the data
    always come from the prior itself); one is the exact model.
    """
    prior = EXACT_PRIORS[prior_name]()
    solver_prior = eqx.tree_at(lambda p: p.xi, prior, solver_xi_scale * prior.xi)
    D = prior.state_dim
    K = EXACT_TS.shape[0]
    key_P0, key_draws = jax.random.split(jax.random.PRNGKey(0))
    M = jax.random.normal(key_P0, (D, D))
    P_0_sqr = np.linalg.cholesky(0.5 * (M @ M.T / D + np.eye(D))).T
    m_0 = np.zeros(D)
    H_ode = prior.E1 - L @ prior.E0
    obs_nodes = np.arange(OBS_EVERY, K, OBS_EVERY)
    # Concrete (built outside the traced draw): prepare_observations matches
    # the times on the host.
    obs_times = EXACT_TS[obs_nodes]
    E0 = prior.E0

    def one_draw(key):
        key_x, key_ode, key_obs = jax.random.split(key, 3)
        x = prior.sample(key_x, EXACT_TS, m_0=m_0, P_0_sqr=P_0_sqr)
        # The forcing at the nodes is the sample's ODE residual plus noise, so
        # the ODE information E1 x - L E0 x - g(t_n) = 0 is a noisy observation.
        g = x @ H_ode.T + np.sqrt(R_ODE) * jax.random.normal(key_ode, (K, 2))
        y = x[obs_nodes] @ E0[:1].T + np.sqrt(R_OBS) * jax.random.normal(
            key_obs, (obs_nodes.shape[0], 1)
        )

        def vf(state, *, t):
            node = np.round((t - EXACT_TS[0]) / EXACT_H).astype(int)
            return L @ state + g[node]

        measure = ODEInformation(vf, prior.E0, prior.E1).with_noise(R_ODE)
        data = Measurement(np.array([[1.0, 0.0]]), y, obs_times, noise=R_OBS)
        obs_model = prepare_observations([data], E0, EXACT_TS)
        res = gaussian_filter(
            m_0,
            P_0_sqr,
            solver_prior,
            measure,
            EXACT_TS,
            calibration="none",
            obs_model=obs_model,
        )
        smoothed = rts_smoother(res)
        chi_filter = jax.vmap(lambda a, m, P: _whitened_sqr_norm(a - m, P))(
            x, res.m, res.P_sqr
        )
        chi_smoother = jax.vmap(lambda a, m, P: _whitened_sqr_norm(a - m, P))(
            x, smoothed.m, smoothed.P_sqr
        )
        back = res.backward
        chi_path = _whitened_sqr_norm(x[-1] - res.m[-1], res.P_sqr[-1]) + np.sum(
            jax.vmap(lambda a, b, G, d, P: _whitened_sqr_norm(a - G @ b - d, P))(
                x[:-1], x[1:], back.G, back.d, back.P_sqr
            )
        )
        assert res.mz_obs is not None and res.Pz_obs_sqr is not None
        chi_innovations = np.sum(
            jax.vmap(_whitened_sqr_norm)(res.mz, res.Pz_sqr)
        ) + np.sum(jax.vmap(_whitened_sqr_norm)(res.mz_obs, res.Pz_obs_sqr))
        std = np.sqrt(np.sum((smoothed.P_sqr @ E0.T) ** 2, axis=-2))
        covered = np.abs(x @ E0.T - smoothed.m @ E0.T)[1:] < Z_95 * std[1:]
        return chi_filter, chi_smoother, chi_path, chi_innovations, covered

    keys = jax.random.split(key_draws, N_DRAWS)
    out = jax.jit(jax.vmap(one_draw))(keys)
    innovation_rows = (K - 1) * 2 + obs_nodes.shape[0]
    return out, D, K, innovation_rows


@pytest.mark.parametrize("prior_name", sorted(EXACT_PRIORS))
def test_filter_and_smoother_marginals_are_exactly_calibrated(prior_name):
    (chi_filter, chi_smoother, *_), D, K, _ = _exact_statistics(prior_name)
    alpha_node = ALPHA / (2 * K)
    for label, chi in (("filter", chi_filter), ("smoother", chi_smoother)):
        p_values = [_two_sided_p(np.sum(chi[:, k]), N_DRAWS * D) for k in range(K)]
        worst = min(range(K), key=lambda k: p_values[k])
        assert p_values[worst] > alpha_node, (
            f"{label} marginal at node {worst}: chi2/dof "
            f"{float(np.mean(chi[:, worst])) / D:.3f}, p = {p_values[worst]:.2e}"
        )


@pytest.mark.parametrize("prior_name", sorted(EXACT_PRIORS))
def test_smoothed_paths_and_innovations_are_white(prior_name):
    (_, _, chi_path, chi_innovations, covered), D, K, rows = _exact_statistics(
        prior_name
    )
    for label, chi, dof in (
        ("smoothed path", np.sum(chi_path), N_DRAWS * K * D),
        ("innovations", np.sum(chi_innovations), N_DRAWS * rows),
    ):
        p = _two_sided_p(chi, dof)
        assert p > ALPHA / 2, f"{label}: chi2/dof {float(chi) / dof:.4f}, p = {p:.2e}"
    # The 95% intervals cover 95% of the truth. The nodes of one draw are
    # correlated, so the standard error of the coverage is at most the one of
    # N_DRAWS independent nodes; five of those leave a tail below 1e-6.
    coverage = float(np.mean(covered))
    assert abs(coverage - 0.95) < 5.0 * (0.95 * 0.05 / N_DRAWS) ** 0.5, coverage


def test_the_exact_check_rejects_a_wrong_diffusion():
    """A solver whose diffusion is 1.5 times too large fails the same check."""
    (_, _, chi_path, *_), D, K, _ = _exact_statistics("iwp", solver_xi_scale=1.5)
    assert _two_sided_p(np.sum(chi_path), N_DRAWS * K * D) < ALPHA
