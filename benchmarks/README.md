# Benchmarks

Cross-library **performance studies** of `ode_filters` (vs probdiffeq and Diffrax).
Each script is runnable standalone (the `benchmarks` extra pulls in Diffrax,
probdiffeq, matplotlib and tueplots):

```bash
uv run --extra benchmarks python benchmarks/<name>.py
```

> Calibration / UQ studies (which are *internal* ode_filters comparisons, not
> cross-library) live with the documentation, not here — see
> [`docs/examples/calibration-modes.ipynb`](../docs/examples/calibration-modes.ipynb)
> and the [calibration guide](../docs/calibration.md).

## Inventory

- **`work_precision.py`** — The headline **work-precision** study: *accuracy vs
  runtime*, not just wallclock. Accuracy is the relative **trajectory RMSE**
  (averaged over the solution path, not a single endpoint — the endpoint is noisy
  and phase-sensitive on oscillatory problems). Compares `ode_filters` (EK1 / EK0
  via the pluggable `TaylorCorrection`), `probdiffeq` (ts1 / ts0), and **Diffrax**
  Tsit5 (the classical JAX solver, the speed baseline) against a high-accuracy
  Diffrax `Dopri8` reference (rtol=1e-12), on logistic / Lotka-Volterra / van der
  Pol. The script enables float64 itself (standalone scripts do not see
  `conftest.py`; in float32 the probabilistic solvers hit a ~1e-6 round-off floor
  that *grows* with step count, producing spurious "error increases with runtime"
  curves). To keep the comparison apples-to-apples the stepping strategy is held
  fixed across *all* solvers within each figure (rather than mixing fixed-grid
  probabilistic solvers with an adaptive classical one). Produces two figures:
  - `work_precision.{pdf,svg}` — **all adaptive** (headline): every solver sweeps
    its tolerance and runs its own step-size controller (ode_filters via
    `gaussian_filter_adaptive`, probdiffeq via `solve_adaptive_save_at`,
    Diffrax via its PID controller). This is what a user actually runs. The
    probabilistic solvers track each other; Diffrax sets the speed/accuracy frontier.
  - `work_precision_fixed.{pdf,svg}` — **all fixed grid**: every solver runs the
    same uniform `N`-step grid, including Diffrax forced onto a constant step
    (`ConstantStepSize`). Isolates per-step accuracy/cost with no controller in the loop.

  Points where a solver diverges (relative trajectory RMSE > 1, e.g. a probabilistic
  solver at the loosest tolerance / coarsest grid) are dropped — a work-precision
  diagram shows the convergent branch — so a curve may have fewer markers than the
  sweep length. The table printed at the end marks them with `*`.

## `work_precision.py`

### Outputs

Everything goes to `benchmarks/results/` next to the script, whatever the working
directory (`--out-dir PATH` to change it). The folder is git-ignored: the timings
are specific to the machine and the moment, so they are regenerated rather than
versioned. Only the copies shown in the docs (`docs/assets/work-precision*.svg`)
are committed, see step 6 below.

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
| `--quick` | smoke run: two tolerances, two grids, 3 rounds; well under a minute. Exercises every solver, problem and output, but its timings are not for publication |
| `--replot` | redraw the figures from the JSON in `--out-dir` without running the benchmark (e.g. to restyle) |

### How the timing works

Each configuration (problem x solver x tolerance or grid) is compiled and run
once first; that call gives the error and the step counts. The adaptive solvers
take the tolerance as a traced argument, so one compilation serves the whole
tolerance sweep. Then the timing runs in rounds: each round visits *every*
configuration in a fresh random order (seeded) and times one warm-up call and
five timed calls (`jax.block_until_ready`, Python's garbage collector off),
keeping their median. A slow drift in the machine load therefore hits all
solvers and settings alike instead of whichever ran last. Each point in a figure
is the median over the rounds; its horizontal bar is the interquartile range.

### Producing the final figures

Timings are only meaningful on a quiet machine. The script prints a warning when
the load average is above a quarter of the CPU count; check the load trace in the
JSON afterwards.

1. Close other heavy work (builds, browsers, other benchmarks); on a laptop, plug
   in the power and keep it awake.
2. Check that the machine is idle: `uptime` should show a load average well
   below the number of cores.
3. Run the full sweep (a few minutes; the compilation takes about half):

   ```bash
   uv run --extra benchmarks python benchmarks/work_precision.py
   ```

4. Check the printed table and the interquartile ranges in the JSON: wide bars
   mean the machine was disturbed; rerun, or raise `--repeats`.
5. The figures are in `benchmarks/results/`. To restyle them later without
   timing again, edit the plotting code and run with `--replot`.
6. For a release, copy the two SVGs to the docs, where the comparison page
   shows them, and update the measurement note below them in
   `docs/comparison.md` (date, machine, versions, commit; all in the JSON):

   ```bash
   cp benchmarks/results/work_precision.svg docs/assets/work-precision.svg
   cp benchmarks/results/work_precision_fixed.svg docs/assets/work-precision-fixed.svg
   ```

### Suggested captions

The figures carry no title; their description belongs in the caption, e.g.:

> **Work-precision, all adaptive.** Relative trajectory RMSE against the median
> runtime of one solve, sweeping the tolerance (atol = rtol) on three problems.
> Colour: library (ode_filters, probdiffeq, Diffrax); solid circles: first-order
> linearization (EK1, ts1), dashed triangles: zeroth order (EK0, ts0); squares:
> the classical Tsit5. Horizontal bars: interquartile range of the runtime over
> the timing rounds. Lower left is better.

> **Work-precision, all fixed grid.** As above, but every solver runs the same
> uniform grid of N steps (Diffrax with a constant step size).
