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
