# Benchmarks

Performance studies of `ode_filters`. Each script is runnable standalone:

```bash
uv run python benchmarks/<name>.py
```

## Inventory

- **`benchmark_ode_solvers.py`** — Fixed-grid wallclock comparison of
  `ode_filters` (EKF1) vs `probdiffeq` (TS1, TS0) vs `scipy.solve_ivp`
  (RK45). Filter-only (no smoother) configurations matched between
  libraries.

- **`benchmark_calibration_modes.py`** — Five-mode comparison on
  moderately stiff van der Pol (mu=10): fixed/global post-hoc MLE,
  fixed/dynamic per-step, fixed/diagonal per-component, adaptive/dynamic,
  and adaptive/diagonal. Demonstrates (a) the per-step sigma^2 variation
  (~25 orders of magnitude on the relaxation cycle), (b) that diagonal
  outperforms dynamic at fixed step by ~4x in accuracy, and (c) that
  diagonal does *not* help when component scales are similar -- van der
  Pol is multi-d but single-scale, so adaptive+dynamic beats
  adaptive+diagonal here.
