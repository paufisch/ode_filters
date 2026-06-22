"""Work-precision benchmark: accuracy vs runtime, with calibration check.

Unlike ``benchmark_ode_solvers.py`` (wallclock only), this measures *accuracy*
against a high-accuracy reference and plots work-precision diagrams (relative
**trajectory RMSE** vs runtime -- not a single endpoint, which is noisy and phase-
sensitive on oscillatory problems). To keep the comparison apples-to-apples it
produces **two** diagrams, each holding the stepping strategy fixed across *all*
solvers:

- ``work_precision.png`` -- **all adaptive** (the headline). Every solver sweeps
  its tolerance and runs its own step-size controller:
  ``ode_filters`` EK1 / EK0 (``gaussian_filter_adaptive``, PI controller),
  ``probdiffeq`` ts1 / ts0 (``solve_adaptive_save_at``), and **Diffrax**
  Tsit5 (PID controller). This is what a user actually runs, and it lets the
  probabilistic solvers use the adaptive stepping they ship with. The solution is
  scored on a common dense save grid (via each solver's dense output, so the save
  points do not force extra steps).
- ``work_precision_fixed.png`` -- **all fixed grid**. Every solver runs on the
  same uniform grid (``N`` sweep), including Diffrax Tsit5 forced onto a constant
  step (``ConstantStepSize``); the RMSE is over that native grid.

Mixing the two (fixed-grid probabilistic vs adaptive classical) conflates the
probabilistic-vs-classical axis with the fixed-vs-adaptive axis, so we keep them
in separate figures.

NOTE: this script enables ``jax_enable_x64`` itself. Standalone scripts do not
see ``conftest.py`` (which is what turns on float64 for the test suite); in float32
the probabilistic solvers hit a ~1e-6 round-off floor that *grows* with step count,
which would show up as spurious "error increases with runtime" curves.

All solvers (probabilistic and classical) use ``Q=3`` smoothness / matched order.
The reference is Diffrax ``Dopri8`` at ``rtol=1e-12`` (a tolerance achievable in
float64 for these problems, and 2-3 orders tighter than the best method measured).
A third figure is a **chi-squared calibration sweep**: for a well-calibrated
probabilistic solver the standardized residual
``(x_true - mean)^T P^{-1} (x_true - mean) / d`` should hover near 1 (much larger =
over-confident, much smaller = under-confident). Coarse grids where a solver
diverges are dropped (the convergent branch is what a work-precision diagram shows).

Run with: ``uv run --extra benchmarks python benchmarks/work_precision.py``
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from probdiffeq import ivpsolve, ivpsolvers, taylor

from ode_filters.filters import (
    TaylorCorrection,
    gaussian_filter,
    gaussian_filter_adaptive,
)
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization

# Standalone benchmark scripts do not see `conftest.py`, which is what enables
# float64 for the test suite. Without this, JAX defaults to float32 and the
# probabilistic solvers hit a ~1e-6 round-off floor that *grows* with step count
# -- producing spurious "error increases with runtime" work-precision curves.
jax.config.update("jax_enable_x64", True)

Q = 3  # smoothness order for the probabilistic solvers (matched across them)
N_EVAL = 100  # save points for the adaptive trajectory-RMSE evaluation grid


@dataclass
class ODEProblem:
    name: str
    vf: Callable  # vf(x, *, t)
    x0: jnp.ndarray
    tspan: tuple[float, float]
    dim: int


def _problems() -> list[ODEProblem]:
    def logistic(x, *, t):
        del t
        return x * (1 - x)

    def lotka_volterra(z, *, t):
        del t
        a, b, c, d = 1.5, 1.0, 3.0, 1.0
        return jnp.array([a * z[0] - b * z[0] * z[1], -c * z[1] + d * z[0] * z[1]])

    def van_der_pol(z, *, t):
        del t
        mu = 1.0
        return jnp.array([z[1], mu * (1 - z[0] ** 2) * z[1] - z[0]])

    return [
        ODEProblem("Logistic (1D)", logistic, jnp.array([0.01]), (0.0, 10.0), 1),
        ODEProblem(
            "Lotka-Volterra (2D)",
            lotka_volterra,
            jnp.array([10.0, 5.0]),
            (0.0, 10.0),
            2,
        ),
        ODEProblem(
            "Van der Pol (2D)", van_der_pol, jnp.array([2.0, 0.0]), (0.0, 20.0), 2
        ),
    ]


# --------------------------------------------------------------------------- #
# Reference (Diffrax Dopri8 @ 1e-12)                                          #
# --------------------------------------------------------------------------- #


def _diffrax_term(problem: ODEProblem):
    return diffrax.ODETerm(lambda t, y, args: problem.vf(y, t=t))


def reference_trajectory(problem: ODEProblem, ts: jnp.ndarray) -> jnp.ndarray:
    # rtol=1e-12 (achievable in float64) keeps the reference 2-3 orders tighter
    # than the best method measured, so even the tightest sweep point is not
    # reference-limited.
    sol = diffrax.diffeqsolve(
        _diffrax_term(problem),
        diffrax.Dopri8(),
        t0=problem.tspan[0],
        t1=problem.tspan[1],
        dt0=None,
        y0=problem.x0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-12, atol=1e-14),
        max_steps=2_000_000,
    )
    return sol.ys


def _dense_grid(problem: ODEProblem) -> jnp.ndarray:
    """Common save grid for the adaptive trajectory-RMSE comparison."""
    return jnp.linspace(problem.tspan[0], problem.tspan[1], N_EVAL + 1)


# --------------------------------------------------------------------------- #
# Solvers -- each `make_*` returns a no-arg callable returning the *trajectory* #
# (the solution sampled on its evaluation grid), so the caller can score it     #
# with a trajectory RMSE rather than a single endpoint.                         #
#                                                                              #
# `*_fixed`    take a grid size N; the trajectory is the N+1 native grid points.#
# `*_adaptive` take a tolerance `tol` (atol = rtol = tol) and run a controller, #
#              saving on the common dense `_dense_grid(problem)`.               #
# --------------------------------------------------------------------------- #


def make_ode_filters_fixed(problem: ODEProblem, N: int, order: int) -> Callable:
    prior = IWP(Q, problem.dim, Xi=0.5 * jnp.eye(problem.dim))
    mu_0, S0 = taylor_mode_initialization(problem.vf, problem.x0, Q)
    measure = ODEInformation(problem.vf, prior.E0, prior.E1)
    corr = TaylorCorrection(order=order)
    e0 = prior.E0

    @jax.jit
    def run():
        res = gaussian_filter(
            mu_0,
            S0,
            prior,
            measure,
            problem.tspan,
            N,
            correction=corr,
            calibration="dynamic",
        )
        return res.m @ e0.T  # [N+1, d]

    return run


def make_ode_filters_adaptive(problem: ODEProblem, tol: float, order: int) -> Callable:
    prior = IWP(Q, problem.dim, Xi=0.5 * jnp.eye(problem.dim))
    mu_0, S0 = taylor_mode_initialization(problem.vf, problem.x0, Q)
    measure = ODEInformation(problem.vf, prior.E0, prior.E1)
    corr = TaylorCorrection(order=order)
    e0 = prior.E0
    save_at = _dense_grid(problem)

    @jax.jit
    def run():
        res = gaussian_filter_adaptive(
            mu_0,
            S0,
            prior,
            measure,
            save_at,
            correction=corr,
            atol=tol,
            rtol=tol,
            calibration="dynamic",
            max_steps=8192,
        )
        traj = res.m @ e0.T  # [N_EVAL+1, d]
        # A solve that fails to reach every save time is a divergence -- emit NaN
        # so it drops out of the plotted (convergent) branch.
        return jnp.where(res.success, traj, jnp.nan)

    return run


def _probdiffeq_setup(problem: ODEProblem, kind: str):
    def vf(y, *, t):
        return problem.vf(y, t=t)

    tcoeffs = taylor.odejet_padded_scan(
        lambda y: vf(y, t=problem.tspan[0]), (problem.x0,), num=Q
    )
    fact = "dense" if kind == "ts1" else "isotropic"
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact=fact)
    correction = (
        ivpsolvers.correction_ts1(vf, ode_order=1, ssm=ssm)
        if kind == "ts1"
        else ivpsolvers.correction_ts0(vf, ode_order=1, ssm=ssm)
    )
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=correction, ssm=ssm)
    return init, solver, ssm


def make_probdiffeq_fixed(problem: ODEProblem, N: int, kind: str) -> Callable:
    grid = jnp.linspace(problem.tspan[0], problem.tspan[1], N + 1)
    init, solver, ssm = _probdiffeq_setup(problem, kind)

    @jax.jit
    def solve():
        return ivpsolve.solve_fixed_grid(init, grid=grid, solver=solver, ssm=ssm)

    def run():
        # `u` is the list of Taylor coefficients per grid point; `u[0]` is the
        # state trajectory, shape [N+1, d].
        return solve().u[0]

    return run


def make_probdiffeq_adaptive(problem: ODEProblem, tol: float, kind: str) -> Callable:
    init, solver, ssm = _probdiffeq_setup(problem, kind)
    adaptive = ivpsolvers.adaptive(solver, ssm=ssm, atol=tol, rtol=tol)
    save_at = _dense_grid(problem)
    dt0 = (problem.tspan[1] - problem.tspan[0]) / 100.0

    @jax.jit
    def solve():
        # `save_at` uses the solver's dense output (interpolation), so it does not
        # force extra steps -- the runtime still reflects the natural adaptive cost.
        return ivpsolve.solve_adaptive_save_at(
            init, save_at=save_at, adaptive_solver=adaptive, dt0=dt0, ssm=ssm
        )

    def run():
        return solve().u[0]  # [N_EVAL+1, d]

    return run


def make_diffrax_adaptive(problem: ODEProblem, tol: float) -> Callable:
    term = _diffrax_term(problem)
    save_at = _dense_grid(problem)

    @jax.jit
    def run():
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=problem.tspan[0],
            t1=problem.tspan[1],
            dt0=None,
            y0=problem.x0,
            saveat=diffrax.SaveAt(ts=save_at),  # dense output, no forced steps
            stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol),
            max_steps=1_000_000,
        )
        return sol.ys  # [N_EVAL+1, d]

    return run


def make_diffrax_fixed(problem: ODEProblem, N: int) -> Callable:
    term = _diffrax_term(problem)
    t0, t1 = problem.tspan
    grid = jnp.linspace(t0, t1, N + 1)
    dt0 = (t1 - t0) / N

    @jax.jit
    def run():
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=problem.x0,
            saveat=diffrax.SaveAt(ts=grid),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=N + 16,  # constant step lands ~N steps; small buffer for rounding
        )
        return sol.ys  # [N+1, d]

    return run


# --------------------------------------------------------------------------- #
# Timing + sweep                                                              #
# --------------------------------------------------------------------------- #


def median_time_ms(run: Callable, n_warmup: int = 2, n_runs: int = 7) -> float:
    for _ in range(n_warmup):
        jax.block_until_ready(run())
    samples = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(run())
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples)) * 1000.0


def rel_trmse(traj: jnp.ndarray, ref: jnp.ndarray) -> float:
    """Relative trajectory RMSE between a solution and the reference.

    Both are ``[K, d]`` (solution sampled on the evaluation grid). Averaging the
    error over the whole trajectory -- rather than reading a single endpoint --
    is robust to the phase sensitivity of oscillatory problems and is the metric
    used across the probabilistic-ODE literature.
    """
    num = jnp.sqrt(jnp.mean(jnp.sum((traj - ref) ** 2, axis=-1)))
    den = jnp.sqrt(jnp.mean(jnp.sum(ref**2, axis=-1)))
    return float(num / den)


# (label, color, marker, make(problem, knob) -> run). `make` takes a grid size N
# for the fixed methods and a tolerance for the adaptive methods.
FIXED_METHODS = [
    ("ode_filters EK1", "C0", "o", lambda p, n: make_ode_filters_fixed(p, n, 1)),
    ("ode_filters EK0", "C4", "v", lambda p, n: make_ode_filters_fixed(p, n, 0)),
    ("probdiffeq ts1", "C1", "s", lambda p, n: make_probdiffeq_fixed(p, n, "ts1")),
    ("probdiffeq ts0", "C2", "D", lambda p, n: make_probdiffeq_fixed(p, n, "ts0")),
    ("Diffrax Tsit5", "C3", "^", make_diffrax_fixed),
]
ADAPTIVE_METHODS = [
    ("ode_filters EK1", "C0", "o", lambda p, tol: make_ode_filters_adaptive(p, tol, 1)),
    ("ode_filters EK0", "C4", "v", lambda p, tol: make_ode_filters_adaptive(p, tol, 0)),
    (
        "probdiffeq ts1",
        "C1",
        "s",
        lambda p, tol: make_probdiffeq_adaptive(p, tol, "ts1"),
    ),
    (
        "probdiffeq ts0",
        "C2",
        "D",
        lambda p, tol: make_probdiffeq_adaptive(p, tol, "ts0"),
    ),
    ("Diffrax Tsit5", "C3", "^", make_diffrax_adaptive),
]
N_VALUES = [80, 160, 320, 640, 1280]
TOLS = [1e-3, 1e-5, 1e-7, 1e-9]
# A probabilistic solver at too-coarse a grid / loose a tolerance diverges; we
# plot only the convergent branch (finite error below this threshold).
_CONVERGED = 1.0


def work_precision(
    problems: list[ODEProblem],
    methods: list,
    sweep: list,
    eval_grid: Callable[[ODEProblem, object], jnp.ndarray],
    save_path: str,
    title: str,
) -> None:
    """Plot trajectory-RMSE-vs-runtime for `methods`, swept over `sweep`.

    `eval_grid(problem, knob)` returns the save grid that each method's `run()`
    samples (and on which the reference is evaluated): the native N+1 grid for the
    fixed methods, a common dense grid for the adaptive ones.
    """
    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4.2))
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems, strict=True):
        print(f"\n{problem.name}")
        # The reference depends only on the eval grid, which (per knob) is shared
        # across methods -- compute it once per knob.
        refs = {
            knob: reference_trajectory(problem, eval_grid(problem, knob))
            for knob in sweep
        }

        for name, color, marker, make in methods:
            times, errs = [], []
            for knob in sweep:
                run = make(problem, knob)
                err = rel_trmse(run(), refs[knob])
                t = median_time_ms(run)
                print(f"  {name:18s} {knob!s:>8s}  tRMSE={err:.2e}  t={t:8.3f} ms")
                if np.isfinite(err) and err < _CONVERGED:  # drop diverged points
                    times.append(t)
                    errs.append(err)
            if times:
                ax.plot(times, errs, color=color, marker=marker, label=name)

        ax.set(
            xlabel="runtime (ms)",
            ylabel="relative trajectory RMSE",
            title=problem.name,
            xscale="log",
            yscale="log",
        )
        ax.legend(fontsize="small")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {save_path}")


# --------------------------------------------------------------------------- #
# Chi-squared calibration sweep                                               #
# --------------------------------------------------------------------------- #


def _ode_filters_mean_cov(problem: ODEProblem, N: int):
    prior = IWP(Q, problem.dim, Xi=0.5 * jnp.eye(problem.dim))
    mu_0, S0 = taylor_mode_initialization(problem.vf, problem.x0, Q)
    measure = ODEInformation(problem.vf, prior.E0, prior.E1)
    res = gaussian_filter(
        mu_0,
        S0,
        prior,
        measure,
        problem.tspan,
        N,
        correction=TaylorCorrection(order=1),
        calibration="dynamic",
    )
    e0 = prior.E0
    m = res.m @ e0.T  # [N+1, d]
    p_full = jnp.einsum("nij,nik->njk", res.P_sqr, res.P_sqr)  # P = P_sqr.T @ P_sqr
    p_x = jnp.einsum("ai,nij,bj->nab", e0, p_full, e0)  # [N+1, d, d]
    return m, p_x


def chi2_statistic(problem: ODEProblem, N: int) -> float:
    """Mean standardized squared residual over the grid (~1 when calibrated)."""
    ts = jnp.linspace(problem.tspan[0], problem.tspan[1], N + 1)
    ref = reference_trajectory(problem, ts)  # [N+1, d]
    m, p_x = _ode_filters_mean_cov(problem, N)
    err = ref - m  # [N+1, d]
    jitter = 1e-12 * jnp.eye(problem.dim)
    # quadratic form err_i^T P_i^{-1} err_i, skipping the (exact) initial point
    quad = jax.vmap(lambda e, P: e @ jnp.linalg.solve(P + jitter, e))(err[1:], p_x[1:])
    return float(jnp.mean(quad) / problem.dim)


def calibration_sweep(problems: list[ODEProblem], save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.2))
    # Use the convergent regime; coarse grids on the fast 2D problems are
    # numerically unstable (non-PD covariance) and are skipped.
    n_values = [160, 320, 640, 1280]
    print("\nChi-squared calibration (ode_filters EK1, dynamic calibration):")
    for problem, color in zip(problems, ["C0", "C1", "C2"], strict=False):
        ns, chi2 = [], []
        for n in n_values:
            c = chi2_statistic(problem, n)
            print(f"  {problem.name:20s} N={n:4d}  chi2/d={c:.3f}")
            if np.isfinite(c) and c > 0:
                ns.append(n)
                chi2.append(c)
        if ns:
            ax.plot(ns, chi2, marker="o", color=color, label=problem.name)
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, label="ideal (= 1)")
    ax.set(
        xlabel="grid size N",
        ylabel=r"$\chi^2 / d$  (standardized residual)",
        title="Calibration: standardized residual vs resolution",
        xscale="log",
        yscale="log",
    )
    ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nsaved {save_path}")


def main() -> None:
    problems = _problems()
    # Adaptive methods all save on the common dense grid; fixed methods save on
    # their own native N+1 grid (the resolution *is* the sweep knob).
    work_precision(
        problems,
        ADAPTIVE_METHODS,
        TOLS,
        lambda p, _tol: _dense_grid(p),
        "benchmarks/work_precision.png",
        "Work-precision (all adaptive): trajectory RMSE vs runtime (lower-left is better)",
    )
    work_precision(
        problems,
        FIXED_METHODS,
        N_VALUES,
        lambda p, n: jnp.linspace(p.tspan[0], p.tspan[1], n + 1),
        "benchmarks/work_precision_fixed.png",
        "Work-precision (all fixed-grid): trajectory RMSE vs runtime (lower-left is better)",
    )
    calibration_sweep(problems, "benchmarks/calibration_chi2.png")


if __name__ == "__main__":
    main()
