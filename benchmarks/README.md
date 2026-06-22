# Benchmarks

Performance studies of `ode_filters`. Each script is runnable standalone (the
`benchmarks` extra pulls in Diffrax / SciPy / matplotlib; probdiffeq comes from
the `dev` group):

```bash
uv run --extra benchmarks python benchmarks/<name>.py
```

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
  probabilistic solvers with an adaptive classical one). Produces three figures:
  - `work_precision.png` — **all adaptive** (headline): every solver sweeps its
    tolerance and runs its own step-size controller (ode_filters via
    `gaussian_filter_adaptive`, probdiffeq via `solve_adaptive_save_at`,
    Diffrax via its PID controller). This is what a user actually runs. The
    probabilistic solvers track each other; Diffrax sets the speed/accuracy frontier.
  - `work_precision_fixed.png` — **all fixed grid**: every solver runs the same
    uniform `N`-step grid, including Diffrax forced onto a constant step
    (`ConstantStepSize`). Isolates per-step accuracy/cost with no controller in the loop.
  - `calibration_chi2.png` — a **chi-squared calibration sweep**: the standardized
    residual `(x_true - mean)^T P^{-1} (x_true - mean) / d` vs grid size `N`. A
    well-calibrated solver hovers near 1. In float64, `ode_filters`' per-step
    dynamic calibration is *consistently conservative* here (chi2/d < 1, i.e.
    posterior intervals too wide): it approaches well-calibrated (~1) only on
    Lotka-Volterra at coarse `N`, and grows more under-confident as `N` increases.
    (An earlier float32 run reported over-confidence on van der Pol at fine `N`;
    that was a round-off artifact, not a calibration property.) A useful reminder
    that per-step dynamic calibration is not a universal guarantee (see the
    [calibration guide](../docs/calibration.md)).

- **`benchmark_ode_solvers.py`** — Fixed-grid wallclock comparison of
  `ode_filters` (EKF1) vs `probdiffeq` (TS1, TS0) vs `scipy.solve_ivp`
  (RK45). Filter-only (no smoother) configurations matched between
  libraries. (Superseded by `work_precision.py` for accuracy claims.)

- **`benchmark_calibration_modes.py`** — Five-mode comparison on
  moderately stiff van der Pol (mu=10): fixed/global post-hoc MLE,
  fixed/dynamic per-step, fixed/diagonal per-component, adaptive/dynamic,
  and adaptive/diagonal. Demonstrates (a) the per-step sigma^2 variation
  (~25 orders of magnitude on the relaxation cycle), (b) that diagonal
  outperforms dynamic at fixed step by ~4x in accuracy, and (c) that
  diagonal does *not* help when component scales are similar -- van der
  Pol is multi-d but single-scale, so adaptive+dynamic beats
  adaptive+diagonal here.
