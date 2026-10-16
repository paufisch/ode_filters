"""Work-precision benchmark: accuracy against runtime, across libraries.

Measures the accuracy of ode_filters, probdiffeq and Diffrax against a reference
solution and plots it against the runtime. Accuracy is the relative trajectory
RMSE, averaged over the solution path rather than taken at the endpoint, which
is noisy and sensitive to the phase on oscillatory problems. Within each figure
all solvers use the same stepping strategy:

- ``work_precision.{pdf,svg}``: adaptive steps. Every solver sweeps its
  tolerance and runs its own step-size controller: ``ode_filters`` EK1 / EK0
  (``gaussian_filter_adaptive``, PI controller), ``probdiffeq`` ts1 / ts0
  (``solve_adaptive_save_at``) and Diffrax's Tsit5 (PID controller). The
  solution is scored on a common save grid, through each solver's dense output,
  so the save points do not force extra steps.
- ``work_precision_fixed.{pdf,svg}``: fixed grid. Every solver runs on the same
  uniform grid (a sweep over ``N``), Diffrax's Tsit5 with ``ConstantStepSize``;
  the RMSE is over that grid.

Mixing fixed-grid probabilistic solvers with an adaptive classical one would
confound the two comparisons, so they are separate figures. Next to the
figures, ``work_precision.json`` holds every number (error, median runtime and
interquartile range, the runtime of every round, step counts) and the metadata
of the run (package versions, git commit, CPU, load average, date).

Timing. Every configuration is compiled and run once first; that call also
gives the error and the step counts. The timing then runs in ``--repeats``
rounds. Each round visits all configurations (every problem, solver and
setting) in a new random order and times one warm-up call plus
``CALLS_PER_REPEAT`` calls of each (``block_until_ready``), keeping the median
of those calls. A slow drift in the machine load therefore affects all
configurations equally instead of biasing the ones that ran last. The figures
show the median over the rounds, with the interquartile range as a horizontal
bar.

The script enables ``jax_enable_x64`` itself. Standalone scripts do not read
``conftest.py``, which enables float64 for the test suite. In float32 the
probabilistic solvers reach a round-off floor of about 1e-6 that grows with
the number of steps, which would show as an error that increases with runtime.

All probabilistic solvers use priors of order ``Q = 3``. Both probabilistic
libraries calibrate the diffusion dynamically, at every step (ode_filters
``calibration="dynamic"``, probdiffeq ``solver_dynamic``): with the same
calibration they compute the same means on a fixed grid, so the figures compare
runtime rather than calibration choices (``probdiffeq.ivpsolvers.solver`` does
not calibrate, which is far less accurate on Lotka-Volterra). The reference is
Diffrax's ``Dopri8`` at ``rtol=1e-12``, two to three orders of magnitude below
the smallest error measured. Settings where a solver diverges (relative error
above 1) are left out of the figures. Calibration quality, i.e. whether the
posterior uncertainty matches the error, is a property of ode_filters alone and
is studied in the docs (``docs/examples/calibration-modes.ipynb`` and
``docs/calibration.md``), not here.

Run with ``uv run --extra benchmarks python benchmarks/work_precision.py``
(``--help`` lists the options; ``--quick`` is a smoke run of under a minute).
"""

# pyright runs in the dev environment (pre-commit, CI), which lacks the
# `benchmarks` extra: there Diffrax, matplotlib and tueplots are unresolved and
# type as unknown, while the ode_filters, JAX and probdiffeq calls stay checked.
# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import datetime
import gc
import importlib.metadata
import json
import logging
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D
from probdiffeq import ivpsolve, ivpsolvers, taylor
from tueplots import bundles, cycler
from tueplots.constants.color import palettes

import ode_filters
from ode_filters.filters import gaussian_filter, gaussian_filter_adaptive
from ode_filters.information import ODEInformation
from ode_filters.linearization import TaylorCorrection
from ode_filters.priors import IWP, taylor_mode_initialization

# Standalone scripts do not read `conftest.py`, which enables float64 for the
# test suite. In float32 the probabilistic solvers reach a round-off floor of
# about 1e-6 that grows with the number of steps, so the error would increase
# with the runtime.
jax.config.update("jax_enable_x64", True)

DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "results"
RESULTS_NAME = "work_precision.json"
FIGURE_STEMS = {"adaptive": "work_precision", "fixed": "work_precision_fixed"}

Q = 3  # smoothness order for the probabilistic solvers (matched across them)
N_EVAL = 100  # save points for the adaptive trajectory-RMSE evaluation grid
REFERENCE_RTOL, REFERENCE_ATOL = 1e-12, 1e-14

TOLS = (1e-3, 1e-5, 1e-7, 1e-9)
N_VALUES = (80, 160, 320, 640, 1280)
REPEATS = 20
# --quick: a smoke run (CI) that exercises every solver, problem and output.
QUICK_TOLS = (1e-5, 1e-8)
QUICK_N_VALUES = (160, 640)
QUICK_REPEATS = 3

CALLS_PER_REPEAT = 5  # timed calls per configuration and round (median kept)
SEED = 0  # of the per-round random order of the configurations
# A probabilistic solver at too-coarse a grid / loose a tolerance diverges; we
# plot only the convergent branch (finite error below this threshold).
CONVERGED_BELOW = 1.0
NOT_REPORTED = -1  # step count a solver does not expose

# A run returns (trajectory [K, d], accepted steps, rejected steps).
Out = tuple[jax.Array, jax.Array, jax.Array]
Run = Callable[[], Out]
AdaptiveRun = Callable[[jax.Array], Out]  # takes the tolerance (atol = rtol)


@dataclass(frozen=True)
class ODEProblem:
    key: str
    title: str
    vf: Callable  # vf(x, *, t)
    x0: jax.Array
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
        ODEProblem("logistic", "Logistic", logistic, jnp.array([0.01]), (0.0, 10.0), 1),
        ODEProblem(
            "lotka_volterra",
            "Lotka-Volterra",
            lotka_volterra,
            jnp.array([10.0, 5.0]),
            (0.0, 10.0),
            2,
        ),
        ODEProblem(
            "van_der_pol",
            "Van der Pol",
            van_der_pol,
            jnp.array([2.0, 0.0]),
            (0.0, 20.0),
            2,
        ),
    ]


def fixed_grid(problem: ODEProblem, n: int) -> np.ndarray:
    """The uniform ``n``-step grid every fixed-grid solver runs on."""
    return np.linspace(problem.tspan[0], problem.tspan[1], n + 1)


def dense_grid(problem: ODEProblem) -> np.ndarray:
    """Common save grid for the adaptive trajectory-RMSE comparison."""
    return np.linspace(problem.tspan[0], problem.tspan[1], N_EVAL + 1)


# --------------------------------------------------------------------------- #
# Reference (Diffrax Dopri8 @ 1e-12)                                          #
# --------------------------------------------------------------------------- #


def _diffrax_term(problem: ODEProblem) -> diffrax.ODETerm:
    return diffrax.ODETerm(lambda t, y, args: problem.vf(y, t=t))


def reference_on(problem: ODEProblem, grids: Sequence[np.ndarray]) -> list[np.ndarray]:
    """The reference solution on each grid, from one solve on their union."""
    union = np.unique(np.concatenate(grids))
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
        saveat=diffrax.SaveAt(ts=jnp.asarray(union)),
        stepsize_controller=diffrax.PIDController(
            rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
        ),
        max_steps=2_000_000,
    )
    ys = np.asarray(sol.ys)
    return [ys[np.searchsorted(union, grid)] for grid in grids]


# --------------------------------------------------------------------------- #
# Solvers. Each returns the trajectory (the solution sampled on its            #
# evaluation grid, so the caller can score it with a trajectory RMSE rather    #
# than a single endpoint) and its accepted / rejected step counts.             #
#                                                                              #
# `*_fixed`    take a grid size N; the trajectory is the N+1 native grid points.#
# `*_adaptive` return a jitted function of the tolerance (atol = rtol = tol,   #
#              traced, so one compilation serves the whole sweep) that runs a  #
#              controller and saves on the common dense `dense_grid(problem)`. #
# --------------------------------------------------------------------------- #


def _ode_filters_setup(problem: ODEProblem, order: int):
    prior = IWP(Q, problem.dim, Xi=0.5 * jnp.eye(problem.dim))
    m_0, P_0_sqr = taylor_mode_initialization(problem.vf, problem.x0, Q)
    measure = ODEInformation(problem.vf, prior.E0, prior.E1)
    return prior, m_0, P_0_sqr, measure, TaylorCorrection(order=order)


def make_ode_filters_fixed(problem: ODEProblem, n: int, order: int) -> Run:
    prior, m_0, P_0_sqr, measure, correction = _ode_filters_setup(problem, order)
    grid = fixed_grid(problem, n)  # concrete: the transition is computed once

    @jax.jit
    def run() -> Out:
        res = gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            grid,
            correction=correction,
            calibration="dynamic",
        )
        return res.m @ prior.E0.T, jnp.asarray(n), jnp.asarray(0)

    return run


def make_ode_filters_adaptive(problem: ODEProblem, order: int) -> AdaptiveRun:
    prior, m_0, P_0_sqr, measure, correction = _ode_filters_setup(problem, order)
    save_at = dense_grid(problem)

    @jax.jit
    def run(tol: float) -> Out:
        res = gaussian_filter_adaptive(
            m_0,
            P_0_sqr,
            prior,
            measure,
            save_at,
            correction=correction,
            atol=tol,
            rtol=tol,
            calibration="dynamic",
            max_steps=8192,
        )
        assert res.success is not None
        assert res.n_accepted is not None and res.n_rejected is not None
        # A solve that fails to reach every save time has diverged; NaN drops it
        # from the plotted (convergent) branch.
        traj = jnp.where(res.success, res.m @ prior.E0.T, jnp.nan)
        return traj, jnp.sum(res.n_accepted), jnp.sum(res.n_rejected)

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
    # Dynamic calibration, as ode_filters' calibration="dynamic" (see the
    # module docstring and test/integration/test_probdiffeq_agreement.py).
    solver = ivpsolvers.solver_dynamic(
        strategy, prior=ibm, correction=correction, ssm=ssm
    )
    return init, solver, ssm


def make_probdiffeq_fixed(problem: ODEProblem, n: int, kind: str) -> Run:
    grid = jnp.asarray(fixed_grid(problem, n))
    init, solver, ssm = _probdiffeq_setup(problem, kind)

    @jax.jit
    def run() -> Out:
        sol = ivpsolve.solve_fixed_grid(init, grid=grid, solver=solver, ssm=ssm)
        # `u` is the list of Taylor coefficients per grid point; `u[0]` is the
        # state trajectory, shape [N+1, d].
        return sol.u[0], jnp.asarray(n), jnp.asarray(0)

    return run


def make_probdiffeq_adaptive(problem: ODEProblem, kind: str) -> AdaptiveRun:
    init, solver, ssm = _probdiffeq_setup(problem, kind)
    save_at = jnp.asarray(dense_grid(problem))
    dt0 = (problem.tspan[1] - problem.tspan[0]) / 100.0

    @jax.jit
    def run(tol: float) -> Out:
        adaptive = ivpsolvers.adaptive(solver, ssm=ssm, atol=tol, rtol=tol)
        # `save_at` uses the solver's dense output (interpolation), so it does not
        # force extra steps, and the runtime is that of the adaptive steps alone.
        sol = ivpsolve.solve_adaptive_save_at(
            init, save_at=save_at, adaptive_solver=adaptive, dt0=dt0, ssm=ssm
        )
        # `num_steps` counts the accepted steps cumulatively; rejections are not
        # recorded.
        n_accepted = jnp.max(sol.num_steps).astype(int)
        return sol.u[0], n_accepted, jnp.asarray(NOT_REPORTED)

    return run


def make_diffrax_fixed(problem: ODEProblem, n: int) -> Run:
    term = _diffrax_term(problem)
    t0, t1 = problem.tspan
    grid = jnp.asarray(fixed_grid(problem, n))

    @jax.jit
    def run() -> Out:
        sol = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            t0=t0,
            t1=t1,
            dt0=(t1 - t0) / n,
            y0=problem.x0,
            saveat=diffrax.SaveAt(ts=grid),
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=n + 16,  # constant step lands ~N steps; small buffer for rounding
        )
        assert sol.ys is not None and sol.stats is not None
        return sol.ys, sol.stats["num_accepted_steps"], sol.stats["num_rejected_steps"]

    return run


def make_diffrax_adaptive(problem: ODEProblem) -> AdaptiveRun:
    term = _diffrax_term(problem)
    save_at = jnp.asarray(dense_grid(problem))

    @jax.jit
    def run(tol: float) -> Out:
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
        assert sol.ys is not None and sol.stats is not None
        return sol.ys, sol.stats["num_accepted_steps"], sol.stats["num_rejected_steps"]

    return run


@dataclass(frozen=True)
class Solver:
    library: str  # colour in the figures
    method: str  # line style and marker in the figures
    make_fixed: Callable[[ODEProblem, int], Run]
    make_adaptive: Callable[[ODEProblem], AdaptiveRun]

    @property
    def name(self) -> str:
        return f"{self.library} {self.method}"


SOLVERS = (
    Solver(
        "ode_filters",
        "EK1",
        partial(make_ode_filters_fixed, order=1),
        partial(make_ode_filters_adaptive, order=1),
    ),
    Solver(
        "ode_filters",
        "EK0",
        partial(make_ode_filters_fixed, order=0),
        partial(make_ode_filters_adaptive, order=0),
    ),
    Solver(
        "probdiffeq",
        "ts1",
        partial(make_probdiffeq_fixed, kind="ts1"),
        partial(make_probdiffeq_adaptive, kind="ts1"),
    ),
    Solver(
        "probdiffeq",
        "ts0",
        partial(make_probdiffeq_fixed, kind="ts0"),
        partial(make_probdiffeq_adaptive, kind="ts0"),
    ),
    Solver("Diffrax", "Tsit5", make_diffrax_fixed, make_diffrax_adaptive),
)


# --------------------------------------------------------------------------- #
# Configurations, accuracy, timing                                            #
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    """One point of a work-precision curve: a solver on a problem at a setting."""

    study: str  # "adaptive" (knob: tolerance) or "fixed" (knob: number of steps)
    problem: ODEProblem
    solver: Solver
    knob: float
    run: Run
    reference: np.ndarray
    error: float = math.nan
    n_steps: int | None = None
    n_rejected: int | None = None
    first_call_s: float = math.nan
    samples_ms: list[float] = field(default_factory=list)


def rel_trmse(traj: np.ndarray, ref: np.ndarray) -> float:
    """Relative trajectory RMSE between a solution and the reference.

    Both are ``[K, d]`` (solution sampled on the evaluation grid). The error is
    averaged over the whole trajectory instead of read at the endpoint, which is
    sensitive to the phase on oscillatory problems.
    """
    with np.errstate(over="ignore", invalid="ignore"):  # a diverged run: inf/NaN
        num = np.sqrt(np.mean(np.sum((traj - ref) ** 2, axis=-1)))
    den = np.sqrt(np.mean(np.sum(ref**2, axis=-1)))
    return float(num / den)


def build_configs(
    problems: Sequence[ODEProblem], tols: Sequence[float], n_values: Sequence[int]
) -> list[Config]:
    configs = []
    for problem in problems:
        grids = [dense_grid(problem)] + [fixed_grid(problem, n) for n in n_values]
        ref_dense, *ref_fixed = reference_on(problem, grids)
        for solver in SOLVERS:
            adaptive = solver.make_adaptive(problem)
            for tol in tols:
                run = partial(adaptive, jnp.asarray(tol))
                configs.append(Config("adaptive", problem, solver, tol, run, ref_dense))
            for n, ref in zip(n_values, ref_fixed, strict=True):
                run = solver.make_fixed(problem, n)
                configs.append(Config("fixed", problem, solver, n, run, ref))
    return configs


def prepare(config: Config) -> None:
    """Compile and run once: the error and the step counts, plus a warm-up."""
    t0 = time.perf_counter()
    traj, n_accepted, n_rejected = jax.block_until_ready(config.run())
    config.first_call_s = time.perf_counter() - t0
    config.error = rel_trmse(np.asarray(traj), config.reference)
    config.n_steps = None if int(n_accepted) < 0 else int(n_accepted)
    config.n_rejected = None if int(n_rejected) < 0 else int(n_rejected)
    jax.block_until_ready(config.run())


def _load_average() -> list[float] | None:
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except (AttributeError, OSError):  # not available on Windows
        return None


def time_interleaved(
    configs: list[Config], *, repeats: int
) -> list[list[float] | None]:
    """Time every configuration in ``repeats`` interleaved rounds.

    Each round visits all configurations in a fresh random order; per visit, one
    warm-up call (the caches are cold after switching configurations) and
    ``CALLS_PER_REPEAT`` timed calls, whose median is the round's sample.
    Returns the load average after each round.
    """
    rng = random.Random(SEED)
    order = list(range(len(configs)))
    loads = []
    start = time.perf_counter()
    for r in range(repeats):
        rng.shuffle(order)
        gc.collect()
        gc.disable()  # as timeit does: no collector pauses inside the timed calls
        try:
            for i in order:
                run = configs[i].run
                jax.block_until_ready(run())
                calls = []
                for _ in range(CALLS_PER_REPEAT):
                    t0 = time.perf_counter()
                    jax.block_until_ready(run())
                    calls.append(time.perf_counter() - t0)
                configs[i].samples_ms.append(1e3 * float(np.median(calls)))
        finally:
            gc.enable()
        loads.append(_load_average())
        load = "n/a" if loads[-1] is None else f"{loads[-1][0]:.1f}"
        elapsed = time.perf_counter() - start
        print(f"  round {r + 1}/{repeats} ({elapsed:.0f} s, load {load})", flush=True)
    return loads


# --------------------------------------------------------------------------- #
# Raw data and metadata                                                       #
# --------------------------------------------------------------------------- #


def _finite_or_none(x: float) -> float | None:
    return x if math.isfinite(x) else None


def result_record(config: Config) -> dict[str, Any]:
    samples = np.asarray(config.samples_ms)
    q25, median, q75 = np.percentile(samples, [25, 50, 75])
    converged = math.isfinite(config.error) and config.error < CONVERGED_BELOW
    return {
        "study": config.study,
        "problem": config.problem.key,
        "problem_title": config.problem.title,
        "solver": config.solver.name,
        "library": config.solver.library,
        "method": config.solver.method,
        "knob": "tol" if config.study == "adaptive" else "N",
        "value": config.knob,
        "error": _finite_or_none(config.error),
        "converged": converged,
        "time_ms_median": float(median),
        "time_ms_q25": float(q25),
        "time_ms_q75": float(q75),
        "time_ms_samples": [round(s, 6) for s in config.samples_ms],
        "n_steps": config.n_steps,
        "n_rejected": config.n_rejected,
        "first_call_s": round(config.first_call_s, 4),
    }


def _version(dist: str) -> str | None:
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_state(path: Path) -> dict[str, Any]:
    """Commit (and uncommitted changes) of the checkout ode_filters runs from."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):  # installed from a wheel
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty}


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        if sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine()


def metadata(args: argparse.Namespace, settings: dict[str, Any]) -> dict[str, Any]:
    dists = (
        "jax",
        "jaxlib",
        "ode-filters",
        "probdiffeq",
        "diffrax",
        "equinox",
        "numpy",
        "matplotlib",
        "tueplots",
    )
    return {
        "date": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "quick": args.quick,
        "versions": {"python": platform.python_version()}
        | {d: _version(d) for d in dists},
        "ode_filters_git": _git_state(Path(ode_filters.__file__).resolve().parent),
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "x64": bool(jax.config.read("jax_enable_x64")),
        "settings": settings,
    }


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #

LIBRARIES = ("ode_filters", "probdiffeq", "Diffrax")  # colour = library
# Line style and marker = linearization order (Diffrax is classical: its own).
METHOD_STYLE = {
    "EK1": ("-", "o"),
    "ts1": ("-", "o"),
    "EK0": ("--", "v"),
    "ts0": ("--", "v"),
    "Tsit5": ("-", "s"),
}
ORDER_LEGEND = (("EK1 / ts1", "EK1"), ("EK0 / ts0", "EK0"))
MARKER_SIZE = 3.5


def _label_runtimes(ax: Any) -> None:
    """Runtime ticks as plain numbers, with 1-3 or 1-2-5 steps on a narrow axis.

    A log axis spanning about two decades or less would otherwise carry one or
    two labelled ticks. The errors keep matplotlib's powers of ten.
    """
    lo, hi = ax.get_xlim()
    decades = math.log10(hi / lo)
    subs = (1.0, 2.0, 5.0) if decades < 1.2 else (1.0, 3.0) if decades < 2.3 else (1.0,)
    ax.xaxis.set_major_locator(ticker.LogLocator(subs=subs))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def set_style() -> None:
    plt.rcParams.update(bundles.tue_ai_beamer())
    plt.rcParams.update(cycler.cycler(color=palettes.paultol_bright))
    # TrueType instead of Type 3 fonts in the PDF (publishers reject Type 3).
    plt.rcParams["pdf.fonttype"] = 42
    # Fall back to the default font quietly where Roboto Condensed is not installed.
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def plot_study(results: list[dict[str, Any]], study: str, stem: Path) -> list[Path]:
    """One work-precision panel per problem; error vs median runtime (IQR bars)."""
    rows = [r for r in results if r["study"] == study]
    problems = list(dict.fromkeys(r["problem_title"] for r in rows))
    solvers = list(dict.fromkeys(r["solver"] for r in rows))
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color = {library: palette[i] for i, library in enumerate(LIBRARIES)}
    neutral = palette[6]  # grey: the legend entries for style, not library

    fig, axes = plt.subplots(1, len(problems), squeeze=False)
    for ax, problem in zip(axes.flat, problems, strict=True):
        for solver in solvers:
            points = [
                r
                for r in rows
                if r["problem_title"] == problem
                and r["solver"] == solver
                and r["converged"]
            ]
            if not points:
                continue
            # In sweep order: loosest tolerance / coarsest grid first.
            points.sort(
                key=lambda r: -r["value"] if study == "adaptive" else r["value"]
            )
            median = np.array([r["time_ms_median"] for r in points])
            q25 = np.array([r["time_ms_q25"] for r in points])
            q75 = np.array([r["time_ms_q75"] for r in points])
            linestyle, marker = METHOD_STYLE[points[0]["method"]]
            ax.errorbar(
                median,
                [r["error"] for r in points],
                xerr=np.stack([median - q25, q75 - median]),
                color=color[points[0]["library"]],
                linestyle=linestyle,
                marker=marker,
                markersize=MARKER_SIZE,
                elinewidth=0.6,
                capsize=0,
            )
        ax.set(xscale="log", yscale="log", title=problem, xlabel="runtime (ms)")
        _label_runtimes(ax)
        ax.grid(which="major")
    axes.flat[0].set_ylabel("relative trajectory RMSE")

    # Factorized legend: colour = library, line style / marker = linearization
    # order (Diffrax Tsit5 has a style of its own).
    def handle(c: str, method: str | None = None) -> Line2D:
        linestyle, marker = ("-", "") if method is None else METHOD_STYLE[method]
        return Line2D(
            [], [], color=c, linestyle=linestyle, marker=marker, markersize=MARKER_SIZE
        )

    handles = [handle(color["ode_filters"]), handle(color["probdiffeq"])]
    handles += [handle(color["Diffrax"], "Tsit5")]
    handles += [handle(neutral, method) for _, method in ORDER_LEGEND]
    labels = ["ode_filters", "probdiffeq", "Diffrax Tsit5"]
    labels += [label for label, _ in ORDER_LEGEND]
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncols=len(handles),
        frameon=False,
    )

    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path)
    plt.close(fig)
    return paths


def plot_all(results: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    set_style()
    paths = []
    for study, stem in FIGURE_STEMS.items():
        if any(r["study"] == study for r in results):
            paths += plot_study(results, study, out_dir / stem)
    return paths


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def print_table(results: list[dict[str, Any]]) -> None:
    print(
        f"\n{'study':8s} {'problem':14s} {'solver':16s} {'setting':>9s} "
        f"{'rel. tRMSE':>10s} {'median ms':>10s} {'IQR ms':>17s} {'steps':>11s}"
    )
    for r in sorted(results, key=lambda r: list(FIGURE_STEMS).index(r["study"])):
        setting = f"{r['value']:.0e}" if r["knob"] == "tol" else str(r["value"])
        error = "nan" if r["error"] is None else f"{r['error']:.2e}"
        if not r["converged"]:
            error += "*"
        iqr = f"{r['time_ms_q25']:.3f}-{r['time_ms_q75']:.3f}"
        steps = "" if r["n_steps"] is None else str(r["n_steps"])
        if r["n_rejected"]:
            steps += f"+{r['n_rejected']}"
        print(
            f"{r['study']:8s} {r['problem_title']:14s} {r['solver']:16s} "
            f"{setting:>9s} {error:>10s} {r['time_ms_median']:10.3f} {iqr:>17s} "
            f"{steps:>11s}"
        )
    print(
        f"* diverged (error >= {CONVERGED_BELOW:g} or NaN): not plotted. "
        "steps: accepted+rejected (probdiffeq reports accepted only)."
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Work-precision benchmark of ode_filters vs probdiffeq and Diffrax."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"where the figures and {RESULTS_NAME} go (default: %(default)s)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "smoke run: a reduced sweep with few repeats, well under a minute; "
            "its timings are not meant for publication"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=None,
        help=(
            f"interleaved timing rounds per configuration (default: {REPEATS}, "
            f"{QUICK_REPEATS} with --quick)"
        ),
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help=f"redraw the figures from {RESULTS_NAME} in --out-dir; no benchmark run",
    )
    args = parser.parse_args(argv)
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir: Path = args.out_dir.resolve()
    results_path = out_dir / RESULTS_NAME
    if args.replot:
        if not results_path.is_file():
            sys.exit(f"--replot: no {results_path}; run the benchmark first.")
        results = json.loads(results_path.read_text())["results"]
        for path in plot_all(results, out_dir):
            print(f"saved {path}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    tols = QUICK_TOLS if args.quick else TOLS
    n_values = QUICK_N_VALUES if args.quick else N_VALUES
    repeats = args.repeats or (QUICK_REPEATS if args.quick else REPEATS)
    load_start = _load_average()
    cpus = os.cpu_count() or 1
    if load_start is not None and load_start[0] > 0.25 * cpus:
        print(
            f"warning: load average {load_start[0]:.1f} on {cpus} CPUs; the "
            "timings will be noisy. Produce final figures on an idle machine."
        )
    wall_start = time.perf_counter()

    configs = build_configs(_problems(), tols, n_values)
    print(f"compiling and checking {len(configs)} configurations ...", flush=True)
    for config in configs:
        prepare(config)
    print(f"timing: {repeats} interleaved rounds of {len(configs)} configurations")
    loads = time_interleaved(configs, repeats=repeats)

    settings = {
        "Q": Q,
        "n_eval": N_EVAL,
        "tols": list(tols),
        "n_values": list(n_values),
        "repeats": repeats,
        "calls_per_repeat": CALLS_PER_REPEAT,
        "seed": SEED,
        "reference": (
            f"diffrax Dopri8, rtol={REFERENCE_RTOL:g}, atol={REFERENCE_ATOL:g}"
        ),
        "converged_below": CONVERGED_BELOW,
    }
    meta = metadata(args, settings)
    meta["load_average"] = {
        "start": load_start,
        "end": _load_average(),
        "per_round": loads,
    }
    meta["wall_time_s"] = round(time.perf_counter() - wall_start, 1)
    results = [result_record(c) for c in configs]
    results_path.write_text(
        json.dumps({"metadata": meta, "results": results}, indent=1) + "\n"
    )

    print_table(results)
    print(f"\nsaved {results_path}")
    for path in plot_all(results, out_dir):
        print(f"saved {path}")


if __name__ == "__main__":
    main()
