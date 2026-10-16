# Benchmarks

Performance studies of `ode_filters` against probdiffeq and Diffrax. Each
script runs on its own; the `benchmarks` extra installs Diffrax, probdiffeq,
matplotlib and tueplots:

```bash
uv run --extra benchmarks python benchmarks/<name>.py
```

> Studies of the calibration and the uncertainty compare `ode_filters` with
> itself, not with other libraries, so they are part of the documentation:
> [`docs/examples/calibration-modes.ipynb`](../docs/examples/calibration-modes.ipynb)
> and the [calibration guide](../docs/calibration.md).

## Inventory

- **`work_precision.py`**: the work-precision study, accuracy against runtime.
  Accuracy is the relative trajectory RMSE, averaged over the solution path
  rather than taken at the endpoint, which is noisy and sensitive to the phase
  on oscillatory problems. It compares `ode_filters` (EK1 and EK0 through
  `TaylorCorrection`), probdiffeq (ts1 and ts0) and Diffrax's Tsit5, the
  classical JAX solver, against a Diffrax `Dopri8` reference (rtol=1e-12), on
  the logistic, Lotka–Volterra and van der Pol problems. The script enables
  float64 itself, since standalone scripts do not read `conftest.py`. In
  float32 the probabilistic solvers reach a round-off floor of about 1e-6 that
  grows with the number of steps, which would show as an error that increases
  with runtime. Within each figure all solvers use the same stepping strategy,
  and both probabilistic libraries calibrate the diffusion at every step
  (`calibration="dynamic"` in `ode_filters`, `solver_dynamic` in probdiffeq),
  so on a fixed grid they compute the same solution. The script produces two
  figures:
  - `work_precision.{pdf,svg}`, adaptive steps: every solver sweeps its
    tolerance and runs its own step-size controller (`gaussian_filter_adaptive`,
    probdiffeq's `solve_adaptive_save_at`, Diffrax's PID controller). probdiffeq
    is faster than `ode_filters`, and Diffrax is faster than both.
  - `work_precision_fixed.{pdf,svg}`, fixed grid: every solver runs the same
    uniform grid of `N` steps, Diffrax with `ConstantStepSize`. This isolates
    the accuracy and cost per step from the step-size controller.

  Points where a solver diverges (relative trajectory RMSE above 1, e.g. a
  probabilistic solver at the loosest tolerance or the coarsest grid) are left
  out of the figures, so a curve can have fewer markers than the sweep has
  settings. The table printed at the end marks them with `*`.

## `work_precision.py`

### Outputs

The outputs go to `benchmarks/results/` next to the script, independent of the
working directory (`--out-dir PATH` changes it). The folder is git-ignored,
since the timings depend on the machine and its load; they are regenerated, not
versioned.

| File | Content |
| --- | --- |
| `work_precision.pdf`, `.svg` | all-adaptive figure (PDF for papers and slides, SVG for the web) |
| `work_precision_fixed.pdf`, `.svg` | all-fixed-grid figure |
| `work_precision.json` | every number, see below |

`work_precision.json` holds one record per solver, problem and setting: the
relative trajectory RMSE, the median runtime and its interquartile range (ms), the
runtime of every timing round, the accepted and rejected steps where the solver
reports them (probdiffeq reports accepted steps only), and the first-call time
(compilation). Its `metadata` records the date, the command, the versions of
Python, JAX, ode_filters, probdiffeq, Diffrax and the plotting packages, the git
commit of the ode_filters checkout (and whether it had uncommitted changes), the
platform, CPU and JAX backend, the load average at the start, after every round
and at the end, and the sweep settings.

### Options

| Flag | Effect |
| --- | --- |
| `--out-dir PATH` | write the outputs to `PATH` (default `benchmarks/results/`) |
| `--repeats R` | interleaved timing rounds (default 20; 3 with `--quick`) |
| `--quick` | smoke run with two tolerances, two grids and 3 rounds, under a minute; it runs every solver, problem and output, but its timings are not for publication |
| `--replot` | redraw the figures from the JSON in `--out-dir` without running the benchmark (e.g. to restyle) |

### How the timing works

Each configuration (problem, solver, and tolerance or grid) is compiled and
run once first. That call gives the error and the step counts. The adaptive
solvers take the tolerance as a traced argument, so one compilation serves the
whole tolerance sweep. The timing then runs in rounds. Each round visits every
configuration in a new random order (seeded) and times one warm-up call and
five timed calls (`jax.block_until_ready`, with Python's garbage collector
off), of which it keeps the median. A slow drift in the machine load therefore
affects all solvers and settings equally, not only the ones that ran last. Each
point in a figure is the median over the rounds, and its horizontal bar is the
interquartile range.

### Producing the final figures

Timings are comparable only on an idle machine. The script prints a warning
when the load average exceeds a quarter of the CPU count, and the JSON records
the load after every round.

1. Stop other CPU-heavy work (builds, test suites, other benchmarks). On a
   laptop, connect the power supply and keep it awake.
2. Check that the machine is idle: the load average from `uptime` should stay
   below about a quarter of the number of cores.
3. Run the full sweep (about two minutes on an Apple M4 Pro, half of it
   compilation):

   ```bash
   uv run --extra benchmarks python benchmarks/work_precision.py
   ```

4. Check the printed table and the interquartile ranges in the JSON. An
   interquartile range above about 10 % of the median means the machine was
   disturbed; rerun, or raise `--repeats`.
5. The figures are in `benchmarks/results/`. To restyle them without timing
   again, edit the plotting code and run with `--replot`.

### Suggested captions

The figures have no title. Their description belongs in the caption, e.g.:

> **Work-precision, all adaptive.** Relative trajectory RMSE against the median
> runtime of one solve, sweeping the tolerance (atol = rtol) on three problems.
> Color: library (ode_filters, probdiffeq, Diffrax). Solid circles: first-order
> linearization (EK1, ts1). Dashed triangles: zeroth order (EK0, ts0). Squares:
> the classical Tsit5. Horizontal bars: interquartile range of the runtime over
> the timing rounds. Lower left is better.

> **Work-precision, all fixed grid.** As above, but every solver runs the same
> uniform grid of N steps (Diffrax with a constant step size).
