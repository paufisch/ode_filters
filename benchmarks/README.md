# Benchmarks

Cross-library **performance studies** of `ode_filters` (vs probdiffeq and Diffrax).
Each script is runnable standalone (the `benchmarks` extra pulls in Diffrax /
matplotlib; probdiffeq comes from the `dev` group):

```bash
uv run --extra benchmarks python benchmarks/<name>.py
```

> Calibration / UQ studies (which are *internal* ode_filters comparisons, not
> cross-library) live with the documentation, not here — see
> [`docs/examples/adaptive-and-calibration.ipynb`](../docs/examples/adaptive-and-calibration.ipynb)
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
  - `work_precision.png` — **all adaptive** (headline): every solver sweeps its
    tolerance and runs its own step-size controller (ode_filters via
    `gaussian_filter_adaptive`, probdiffeq via `solve_adaptive_save_at`,
    Diffrax via its PID controller). This is what a user actually runs. The
    probabilistic solvers track each other; Diffrax sets the speed/accuracy frontier.
  - `work_precision_fixed.png` — **all fixed grid**: every solver runs the same
    uniform `N`-step grid, including Diffrax forced onto a constant step
    (`ConstantStepSize`). Isolates per-step accuracy/cost with no controller in the loop.

  Points where a solver diverges (relative trajectory RMSE > 1, e.g. a probabilistic
  solver at the loosest tolerance / coarsest grid) are dropped — a work-precision
  diagram shows the convergent branch — so a curve may have fewer markers than the
  sweep length. The dropped values are printed to stdout.
