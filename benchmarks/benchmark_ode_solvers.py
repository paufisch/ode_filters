"""Benchmark ODE solvers: ode_filters vs probdiffeq vs scipy.

This benchmark compares fixed-grid ODE solving performance across:
1. ode_filters (this library) - EKF1-based probabilistic solver (first-order linearization)
2. probdiffeq - Highly optimized JAX probabilistic ODE solver
3. scipy.integrate.solve_ivp - Classical RK45 solver

Fair comparison notes:
- ode_filters uses EKF1 (first-order Taylor series linearization of the ODE constraint)
- probdiffeq's equivalent is `correction_ts1` with `dense` factorization
- For ts0 (zeroth-order), probdiffeq recommends `isotropic` factorization
- Both probdiffeq variants use `strategy_filter` (forward pass only) to match
  ode_filters' ekf1_sqr_loop. Using `strategy_fixedpoint` (a smoother) would
  do ~2x the work and is not a fair comparison.
- scipy RK45 is an adaptive solver; we force evaluation at fixed grid points via t_eval
  but it still uses adaptive internal stepping. This is a fundamentally different approach.

Run with: python benchmarks/benchmark_ode_solvers.py
"""

from __future__ import annotations  # noqa: I001

import time
from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from probdiffeq import ivpsolve, ivpsolvers, taylor
from scipy.integrate import solve_ivp

from ode_filters.filters import ekf1_sqr_loop
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization


# =============================================================================
# Test Problems
# =============================================================================


@dataclass
class ODEProblem:
    """Container for an ODE test problem."""

    name: str
    vf: Callable  # Vector field f(x, *, t)
    vf_scipy: Callable  # Vector field for scipy f(t, x)
    x0: jnp.ndarray
    tspan: tuple[float, float]
    dim: int


def logistic_problem() -> ODEProblem:
    """Logistic ODE: dx/dt = x(1-x), x(0) = 0.01."""

    def vf(x, *, t):
        del t
        return x * (1 - x)

    def vf_scipy(t, x):
        del t
        return x * (1 - x)

    return ODEProblem(
        name="Logistic (1D)",
        vf=vf,
        vf_scipy=vf_scipy,
        x0=jnp.array([0.01]),
        tspan=(0.0, 10.0),
        dim=1,
    )


def lotka_volterra_problem() -> ODEProblem:
    """Lotka-Volterra predator-prey: dx/dt = ax - bxy, dy/dt = -cy + dxy."""
    a, b, c, d = 1.5, 1.0, 3.0, 1.0

    def vf(z, *, t):
        del t
        x, y = z[0], z[1]
        return jnp.array([a * x - b * x * y, -c * y + d * x * y])

    def vf_scipy(t, z):
        del t
        x, y = z
        return [a * x - b * x * y, -c * y + d * x * y]

    return ODEProblem(
        name="Lotka-Volterra (2D)",
        vf=vf,
        vf_scipy=vf_scipy,
        x0=jnp.array([10.0, 5.0]),
        tspan=(0.0, 15.0),
        dim=2,
    )


def van_der_pol_problem(mu: float = 1.0) -> ODEProblem:
    """Van der Pol oscillator: x'' - mu(1-x^2)x' + x = 0."""

    def vf(z, *, t):
        del t
        x, y = z[0], z[1]
        return jnp.array([y, mu * (1 - x**2) * y - x])

    def vf_scipy(t, z):
        del t
        x, y = z
        return [y, mu * (1 - x**2) * y - x]

    return ODEProblem(
        name="Van der Pol (2D)",
        vf=vf,
        vf_scipy=vf_scipy,
        x0=jnp.array([2.0, 0.0]),
        tspan=(0.0, 20.0),
        dim=2,
    )


# =============================================================================
# Solver Wrappers
# =============================================================================


def setup_ode_filters(problem: ODEProblem, N: int, q: int = 2) -> Callable:
    """Set up ode_filters solver for a problem."""
    xi = 0.5 * jnp.eye(problem.dim)
    prior = IWP(q, problem.dim, Xi=xi)
    mu_0, Sigma_0_sqr = taylor_mode_initialization(problem.vf, problem.x0, q)
    measure = ODEInformation(problem.vf, prior.E0, prior.E1)

    def solve():
        result = ekf1_sqr_loop(mu_0, Sigma_0_sqr, prior, measure, problem.tspan, N)
        return result[0][-1]  # Return final mean

    return solve


def setup_ode_filters_jit(problem: ODEProblem, N: int, q: int = 2) -> Callable:
    """Set up JIT-compiled ode_filters solver."""
    xi = 0.5 * jnp.eye(problem.dim)
    prior = IWP(q, problem.dim, Xi=xi)
    mu_0, Sigma_0_sqr = taylor_mode_initialization(problem.vf, problem.x0, q)
    measure = ODEInformation(problem.vf, prior.E0, prior.E1)

    @jax.jit
    def solve(mu_0, Sigma_0_sqr, tspan):
        result = ekf1_sqr_loop(mu_0, Sigma_0_sqr, prior, measure, tspan, N)
        return result[0][-1]

    def run():
        return solve(mu_0, Sigma_0_sqr, problem.tspan).block_until_ready()

    return run


def setup_probdiffeq_ts1(problem: ODEProblem, N: int, num: int = 1) -> Callable:
    """Set up probdiffeq fixed-grid solver with TS1 (first-order linearization).

    This matches ode_filters' EKF1 which uses first-order Taylor series linearization.
    Uses 'dense' factorization as recommended for ts1 in probdiffeq docs.
    Uses strategy_filter (forward pass only) to match ode_filters' ekf1_sqr_loop.
    """
    grid = jnp.linspace(problem.tspan[0], problem.tspan[1], N + 1)

    def vf_probdiffeq(y, *, t):
        return problem.vf(y, t=t)

    tcoeffs = taylor.odejet_padded_scan(
        lambda y: vf_probdiffeq(y, t=problem.tspan[0]), (problem.x0,), num=num
    )
    # Use 'dense' factorization with ts1 as recommended by probdiffeq docs
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact="dense")
    ts1 = ivpsolvers.correction_ts1(vf_probdiffeq, ode_order=1, ssm=ssm)
    # strategy_filter = forward pass only, matching ode_filters' ekf1_sqr_loop
    # (strategy_fixedpoint is a smoother: forward + backward pass = ~2x work)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts1, ssm=ssm)

    @jax.jit
    def solve(ssm_init, grid):
        return ivpsolve.solve_fixed_grid(ssm_init, grid=grid, solver=solver, ssm=ssm)

    def run():
        result = solve(init, grid)
        # u is a list of arrays for different derivative orders; take the 0th order
        return result.u[0][-1].block_until_ready()

    return run


def setup_probdiffeq_ts0(problem: ODEProblem, N: int, num: int = 1) -> Callable:
    """Set up probdiffeq fixed-grid solver with TS0 (zeroth-order linearization).

    This is faster but less accurate than TS1. Uses 'isotropic' factorization
    as recommended for ts0 in probdiffeq docs.
    Uses strategy_filter (forward pass only) to match ode_filters' ekf1_sqr_loop.
    """
    grid = jnp.linspace(problem.tspan[0], problem.tspan[1], N + 1)

    def vf_probdiffeq(y, *, t):
        return problem.vf(y, t=t)

    tcoeffs = taylor.odejet_padded_scan(
        lambda y: vf_probdiffeq(y, t=problem.tspan[0]), (problem.x0,), num=num
    )
    # Use 'isotropic' factorization with ts0 as recommended by probdiffeq docs
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact="isotropic")
    ts0 = ivpsolvers.correction_ts0(vf_probdiffeq, ode_order=1, ssm=ssm)
    # strategy_filter = forward pass only, matching ode_filters' ekf1_sqr_loop
    # (strategy_fixedpoint is a smoother: forward + backward pass = ~2x work)
    strategy = ivpsolvers.strategy_filter(ssm=ssm)
    solver = ivpsolvers.solver(strategy, prior=ibm, correction=ts0, ssm=ssm)

    @jax.jit
    def solve(ssm_init, grid):
        return ivpsolve.solve_fixed_grid(ssm_init, grid=grid, solver=solver, ssm=ssm)

    def run():
        result = solve(init, grid)
        # u is a list of arrays for different derivative orders; take the 0th order
        return result.u[0][-1].block_until_ready()

    return run


def setup_scipy(problem: ODEProblem, N: int) -> Callable:
    """Set up scipy solve_ivp with fixed evaluation grid."""
    t_eval = np.linspace(problem.tspan[0], problem.tspan[1], N + 1)
    x0_np = np.array(problem.x0)

    def solve():
        result = solve_ivp(
            problem.vf_scipy,
            problem.tspan,
            x0_np,
            method="RK45",
            t_eval=t_eval,
        )
        return result.y[:, -1]

    return solve


# =============================================================================
# Benchmarking Infrastructure
# =============================================================================


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    solver_name: str
    problem_name: str
    N: int
    mean_time_ms: float
    std_time_ms: float
    compile_time_ms: float | None = None


def benchmark_solver(
    solver_fn: Callable,
    n_warmup: int = 3,
    n_runs: int = 10,
) -> tuple[float, float]:
    """Benchmark a solver function.

    Args:
        solver_fn: Callable that runs the solver
        n_warmup: Number of warmup runs (for JIT compilation)
        n_runs: Number of timed runs

    Returns:
        Tuple of (mean_time_ms, std_time_ms)
    """
    for _ in range(n_warmup):
        solver_fn()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        solver_fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def measure_compile_time(setup_fn: Callable, *args) -> float:
    """Measure JIT compilation time."""
    start = time.perf_counter()
    solver_fn = setup_fn(*args)
    solver_fn()  # First call triggers compilation
    end = time.perf_counter()
    return (end - start) * 1000


# =============================================================================
# Main Benchmark Runner
# =============================================================================


def run_benchmarks(
    problems: list[ODEProblem],
    N_values: list[int],
    n_warmup: int = 3,
    n_runs: int = 10,
) -> list[BenchmarkResult]:
    """Run all benchmarks."""
    results = []

    for problem in problems:
        print(f"\n{'=' * 60}")
        print(f"Problem: {problem.name}")
        print(f"{'=' * 60}")

        for N in N_values:
            print(f"\n  N = {N}")
            print(f"  {'-' * 50}")

            # ode_filters (JIT) - uses EKF1 (first-order linearization)
            try:
                solver_fn = setup_ode_filters_jit(problem, N)
                mean_t, std_t = benchmark_solver(solver_fn, n_warmup, n_runs)
                results.append(
                    BenchmarkResult(
                        solver_name="ode_filters (EKF1)",
                        problem_name=problem.name,
                        N=N,
                        mean_time_ms=mean_t,
                        std_time_ms=std_t,
                    )
                )
                print(f"  ode_filters (EKF1):    {mean_t:8.3f} ± {std_t:6.3f} ms")
            except Exception as e:
                print(f"  ode_filters (EKF1):    FAILED - {e}")

            # probdiffeq TS1 (first-order) - matches ode_filters' EKF1
            try:
                solver_fn = setup_probdiffeq_ts1(problem, N)
                mean_t, std_t = benchmark_solver(solver_fn, n_warmup, n_runs)
                results.append(
                    BenchmarkResult(
                        solver_name="probdiffeq (TS1)",
                        problem_name=problem.name,
                        N=N,
                        mean_time_ms=mean_t,
                        std_time_ms=std_t,
                    )
                )
                print(f"  probdiffeq (TS1):      {mean_t:8.3f} ± {std_t:6.3f} ms")
            except Exception as e:
                print(f"  probdiffeq (TS1):      FAILED - {e}")

            # probdiffeq TS0 (zeroth-order) - faster but different model
            try:
                solver_fn = setup_probdiffeq_ts0(problem, N)
                mean_t, std_t = benchmark_solver(solver_fn, n_warmup, n_runs)
                results.append(
                    BenchmarkResult(
                        solver_name="probdiffeq (TS0)",
                        problem_name=problem.name,
                        N=N,
                        mean_time_ms=mean_t,
                        std_time_ms=std_t,
                    )
                )
                print(f"  probdiffeq (TS0):      {mean_t:8.3f} ± {std_t:6.3f} ms")
            except Exception as e:
                print(f"  probdiffeq (TS0):      FAILED - {e}")

            # scipy RK45 (adaptive, evaluated at fixed grid)
            try:
                solver_fn = setup_scipy(problem, N)
                mean_t, std_t = benchmark_solver(solver_fn, n_warmup, n_runs)
                results.append(
                    BenchmarkResult(
                        solver_name="scipy (RK45)",
                        problem_name=problem.name,
                        N=N,
                        mean_time_ms=mean_t,
                        std_time_ms=std_t,
                    )
                )
                print(f"  scipy (RK45):          {mean_t:8.3f} ± {std_t:6.3f} ms")
            except Exception as e:
                print(f"  scipy (RK45):          FAILED - {e}")

    return results


def plot_results(results: list[BenchmarkResult], save_path: str | None = None):
    """Create comparison plots."""
    problems = list({r.problem_name for r in results})
    solvers = list({r.solver_name for r in results})

    # Color and marker schemes for different solvers
    colors = {
        "ode_filters (EKF1)": "C0",
        "probdiffeq (TS1)": "C1",
        "probdiffeq (TS0)": "C2",
        "scipy (RK45)": "C3",
    }
    markers = {
        "ode_filters (EKF1)": "o",
        "probdiffeq (TS1)": "s",
        "probdiffeq (TS0)": "D",
        "scipy (RK45)": "^",
    }

    _, axes = plt.subplots(1, len(problems), figsize=(5 * len(problems), 4))
    if len(problems) == 1:
        axes = [axes]

    for ax, problem in zip(axes, problems, strict=True):
        for solver in solvers:
            solver_results = [
                r
                for r in results
                if r.problem_name == problem and r.solver_name == solver
            ]
            if not solver_results:
                continue

            N_vals = [r.N for r in solver_results]
            times = [r.mean_time_ms for r in solver_results]
            stds = [r.std_time_ms for r in solver_results]

            ax.errorbar(
                N_vals,
                times,
                yerr=stds,
                label=solver,
                color=colors.get(solver, "gray"),
                marker=markers.get(solver, "x"),
                capsize=3,
            )

        ax.set_xlabel("Grid size (N)")
        ax.set_ylabel("Time (ms)")
        ax.set_title(problem)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")

    plt.show()


def print_summary_table(results: list[BenchmarkResult]):
    """Print a summary table of results."""
    problems = sorted({r.problem_name for r in results})
    solvers = [
        "ode_filters (EKF1)",
        "probdiffeq (TS1)",
        "probdiffeq (TS0)",
        "scipy (RK45)",
    ]
    N_values = sorted({r.N for r in results})

    print("\n" + "=" * 100)
    print("SUMMARY TABLE (times in ms)")
    print("=" * 100)

    for problem in problems:
        print(f"\n{problem}")
        print("-" * 90)

        header = f"{'N':>6}"
        for solver in solvers:
            header += f" | {solver:>18}"
        print(header)
        print("-" * 90)

        for N in N_values:
            row = f"{N:>6}"
            for solver in solvers:
                matching = [
                    r
                    for r in results
                    if r.problem_name == problem
                    and r.solver_name == solver
                    and r.N == N
                ]
                if matching:
                    r = matching[0]
                    row += f" | {r.mean_time_ms:>18.3f}"
                else:
                    row += f" | {'N/A':>18}"
            print(row)


def main():
    """Run the benchmark suite."""
    print("ODE Solver Benchmark")
    print("=" * 70)
    print("Comparing fixed-grid ODE solving performance")
    print("=" * 70)
    print("\nSolvers:")
    print("  - ode_filters (EKF1): This library, first-order linearization")
    print("  - probdiffeq (TS1):   First-order Taylor series (matches EKF1)")
    print("  - probdiffeq (TS0):   Zeroth-order Taylor series (faster, less accurate)")
    print("  - scipy (RK45):       Adaptive RK45, evaluated at fixed grid points")
    print()
    print("Notes:")
    print(
        "  - ode_filters uses Python for-loops unrolled by JIT (slow compile for large N)"
    )
    print("  - probdiffeq uses jax.lax.scan (O(1) compilation time, faster at large N)")
    print(
        "  - Both JAX solvers use strategy_filter (forward pass only) for a fair comparison"
    )
    print("  - scipy is adaptive internally; t_eval only controls output points")
    print("=" * 70)

    # Define test problems
    problems = [
        logistic_problem(),
        lotka_volterra_problem(),
        van_der_pol_problem(),
    ]

    # Define grid sizes - kept moderate due to ode_filters compilation time
    # For large N (>500), ode_filters JIT compilation becomes very slow
    N_values = [50, 100, 200, 500]

    # Run benchmarks
    results = run_benchmarks(
        problems=problems,
        N_values=N_values,
        n_warmup=3,
        n_runs=10,
    )

    # Print summary
    print_summary_table(results)

    # Create plots
    plot_results(results, save_path="benchmarks/benchmark_results.png")


if __name__ == "__main__":
    main()
