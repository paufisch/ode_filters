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

- **`benchmark_calibration_modes.py`** — Diffusion-calibration comparison
  on moderately stiff van der Pol (mu=10): fixed-step + global post-hoc
  MLE vs fixed-step + online dynamic per-step vs adaptive + online
  dynamic. Demonstrates the per-step sigma^2 variation (~25 orders of
  magnitude on the relaxation cycle) and the work / accuracy trade-off
  across the three modes.
