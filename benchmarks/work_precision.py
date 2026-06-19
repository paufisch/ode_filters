"""Work-precision benchmark: accuracy vs runtime, with calibration check.

Unlike ``benchmark_ode_solvers.py`` (wallclock only), this measures *accuracy*
against a high-accuracy reference and plots work-precision diagrams (error vs
runtime) comparing:

- **ode_filters** EK1 / EK0 (via the pluggable ``TaylorCorrection``),
- **probdiffeq** ts1 / ts0 (the JAX probabilistic-solver oracle), and
- **Diffrax** Tsit5 (a classical adaptive JAX solver, the speed baseline).

The reference is Diffrax ``Dopri8`` at ``rtol=1e-9`` (a tolerance achievable in
float64 for these problems, and 2+ orders tighter than the best method measured).
A second figure is a **chi-squared calibration sweep**: for a well-calibrated
probabilistic solver the standardized residual
``(x_true - mean)^T P^{-1} (x_true - mean) / d`` should hover near 1 (much larger =
over-confident, much smaller = under-confident). Coarse grids where a solver
diverges are dropped (the convergent branch is what a work-precision diagram shows).

Run with: ``uv run python benchmarks/work_precision.py``
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

from ode_filters.filters import TaylorCorrection
from ode_filters.filters.ode_filter_loop import ekf1_sqr_loop_dynamic_scan
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization

Q = 3  # smoothness order for the probabilistic solvers (matched across them)


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
# Reference (Diffrax Dopri8 @ 1e-13)                                          #
# --------------------------------------------------------------------------- #


def _diffrax_term(problem: ODEProblem):
    return diffrax.ODETerm(lambda t, y, args: problem.vf(y, t=t))


def reference_trajectory(problem: ODEProblem, ts: jnp.ndarray) -> jnp.ndarray:
    sol = diffrax.diffeqsolve(
        _diffrax_term(problem),
        diffrax.Dopri8(),
        t0=problem.tspan[0],
        t1=problem.tspan[1],
        dt0=None,
        y0=problem.x0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=1e-9, atol=1e-11),
        max_steps=2_000_000,
    )
    return sol.ys


# --------------------------------------------------------------------------- #
# Solvers -- each `make_*` returns a no-arg callable returning the final state #
# --------------------------------------------------------------------------- #


def make_ode_filters(problem: ODEProblem, N: int, order: int) -> Callable:
    prior = IWP(Q, problem.dim, Xi=0.5 * jnp.eye(problem.dim))
    mu_0, S0 = taylor_mode_initialization(problem.vf, problem.x0, Q)
    measure = ODEInformation(problem.vf, prior.E0, prior.E1)
    corr = TaylorCorrection(order=order)
    e0 = prior.E0

    @jax.jit
    def run():
        res = ekf1_sqr_loop_dynamic_scan(
            mu_0,
            S0,
            prior,
            measure,
            problem.tspan,
            N,
            correction=corr,
            calibration="dynamic",
        )
        return (res[0] @ e0.T)[-1]

    return run


def make_probdiffeq(problem: ODEProblem, N: int, kind: str) -> Callable:
    grid = jnp.linspace(problem.tspan[0], problem.tspan[1], N + 1)

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

    @jax.jit
    def solve():
        return ivpsolve.solve_fixed_grid(init, grid=grid, solver=solver, ssm=ssm)

    def run():
        return solve().u[0][-1]

    return run


def make_diffrax(problem: ODEProblem, tol: float) -> Callable:
    term = _diffrax_term(problem)

    @jax.jit
    def run():
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=problem.tspan[0],
            t1=problem.tspan[1],
            dt0=None,
            y0=problem.x0,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=diffrax.PIDController(rtol=tol, atol=tol),
            max_steps=1_000_000,
        )
        return sol.ys[-1]

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


def rel_error(x: jnp.ndarray, ref: jnp.ndarray) -> float:
    return float(jnp.linalg.norm(x - ref) / jnp.linalg.norm(ref))


METHODS = [
    ("ode_filters EK1", "C0", "o", lambda p, n: make_ode_filters(p, n, 1)),
    ("ode_filters EK0", "C4", "v", lambda p, n: make_ode_filters(p, n, 0)),
    ("probdiffeq ts1", "C1", "s", lambda p, n: make_probdiffeq(p, n, "ts1")),
    ("probdiffeq ts0", "C2", "D", lambda p, n: make_probdiffeq(p, n, "ts0")),
]
N_VALUES = [80, 160, 320, 640, 1280]
DIFFRAX_TOLS = [1e-3, 1e-5, 1e-7, 1e-9]
# A probabilistic solver at too-coarse a grid diverges; we plot only the
# convergent branch (finite error below this threshold).
_CONVERGED = 1.0


def work_precision(problems: list[ODEProblem], save_path: str) -> None:
    fig, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4.2))
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems, strict=True):
        ref_final = reference_trajectory(problem, jnp.array(list(problem.tspan)))[-1]
        print(f"\n{problem.name}")

        for name, color, marker, make in METHODS:
            times, errs = [], []
            for n in N_VALUES:
                run = make(problem, n)
                err = rel_error(run(), ref_final)
                t = median_time_ms(run)
                print(f"  {name:18s} N={n:4d}  err={err:.2e}  t={t:7.3f} ms")
                if np.isfinite(err) and err < _CONVERGED:  # drop diverged points
                    times.append(t)
                    errs.append(err)
            if times:
                ax.plot(times, errs, color=color, marker=marker, label=name)

        # Diffrax (classical, adaptive): sweep tolerance.
        d_times, d_errs = [], []
        for tol in DIFFRAX_TOLS:
            run = make_diffrax(problem, tol)
            err = rel_error(run(), ref_final)
            t = median_time_ms(run)
            print(f"  {'Diffrax Tsit5':18s} tol={tol:.0e} err={err:.2e}  t={t:7.3f} ms")
            if np.isfinite(err) and err < _CONVERGED:
                d_times.append(t)
                d_errs.append(err)
        if d_times:
            ax.plot(d_times, d_errs, color="C3", marker="^", label="Diffrax Tsit5")

        ax.set(
            xlabel="runtime (ms)",
            ylabel="relative error (final state)",
            title=problem.name,
            xscale="log",
            yscale="log",
        )
        ax.legend(fontsize="small")

    fig.suptitle("Work-precision: accuracy vs runtime (lower-left is better)")
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
    res = ekf1_sqr_loop_dynamic_scan(
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
    m = res[0] @ e0.T  # [N+1, d]
    p_full = jnp.einsum("nij,nik->njk", res[1], res[1])  # P = P_sqr.T @ P_sqr
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
    work_precision(problems, "benchmarks/work_precision.png")
    calibration_sweep(problems, "benchmarks/calibration_chi2.png")


if __name__ == "__main__":
    main()
