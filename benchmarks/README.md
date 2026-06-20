# Benchmarks

Performance studies of `ode_filters`. Each script is runnable standalone:

```bash
uv run python benchmarks/<name>.py
```

## Inventory

- **`work_precision.py`** — The headline **work-precision** study: *accuracy vs
  runtime*, not just wallclock. Compares `ode_filters` (EK1 / EK0 via the
  pluggable `TaylorCorrection`), `probdiffeq` (ts1 / ts0), and **Diffrax** Tsit5
  (a classical adaptive JAX solver, the speed baseline) against a high-accuracy
  Diffrax `Dopri8` reference, on logistic / Lotka-Volterra / van der Pol. Produces
  two figures:
  - `work_precision.png` — error vs runtime (lower-left is better); the
    probabilistic solvers track each other, and Diffrax sets the classical
    speed/accuracy frontier.
  - `calibration_chi2.png` — a **chi-squared calibration sweep**: the standardized
    residual `(x_true - mean)^T P^{-1} (x_true - mean) / d` vs grid size `N`. A
    well-calibrated solver hovers near 1. `ode_filters`' per-step dynamic
    calibration lands within an order or two on most points (order-1 on
    Lotka-Volterra), but it is *conservative* on the smooth logistic (intervals
    too wide) and becomes *over-confident* on van der Pol at fine `N` — a useful
    reminder that per-step dynamic calibration is not a universal guarantee
    (see the [calibration guide](../docs/calibration.md)).

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
