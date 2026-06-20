# Changelog

All notable changes to **ode-filters** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
[Semantic Versioning](https://semver.org/) (pre-1.0: breaking changes ship in a
minor bump).

## [0.7.0] - Unreleased

A major refactor that re-scopes the library from "a square-root EK1 + RTS solver"
into a **pluggable Gaussian-filtering inference substrate**: the measurement
model and the linearization scheme are now independent, pluggable axes, with a
single consolidated solver API and a differentiable parameter-inference layer.

> **Upgrading:** this release intentionally makes a clean break (no deprecation
> aliases). See **Migration** below.

### Breaking

- **Consolidated solver API.** The `ekf1_sqr_loop*` family
  (`{plain/preconditioned} x {joint/sequential} x {for-loop/scan}`) is replaced
  by three auto-dispatching entry points returning a single
  [`FilterResult`][res] NamedTuple:
  - `ekf1_sqr_loop*` → **`gaussian_filter`** (preconditioning is selected from
    the prior type; observations via `obs_model=`; linearization via
    `correction=`).
  - `ekf1_sqr_adaptive_loop` → **`gaussian_filter_adaptive`** (jit/vmap/grad-safe
    save-at-grid solver).
  - `rts_sqr_smoother_loop*` → **`rts_smoother(prior, result)`**.
- **Return types changed** from wide positional tuples to the named
  `FilterResult`. Index results by name (`result.m`, `result.P_sqr`,
  `result.log_likelihood`, ...) instead of by position. `AdaptiveLoopResult` is
  no longer a public export.
- **Low-level loop variants removed entirely** (not just de-exported): the
  non-scan and sequential `ekf1_sqr_loop*` variants no longer exist, even via
  submodule import. The surviving internal scan loops live in
  `ode_filters.filters.{ode_filter_loop,ode_filter_step,ode_filter_adaptive}`
  and are implementation detail, not public API.
- **Measurement factory family removed:** `ODEmeasurement`,
  `ODEconservationmeasurement`, `SecondOrderODEmeasurement`,
  `SecondOrderODEconservationmeasurement`, and `build_obs_at_time`. Use
  `ODEInformation` / `ODEconservation` for the model, and `prepare_observations`
  → `ObsModel` for data.
- **`calibration="cumulative"` removed** (non-Markovian global-sigma
  post-multiply; superseded by `diagonal_ekf0`). Valid modes are now
  `"dynamic"` / `"diagonal"` / `"diagonal_ekf0"` / `"none"`.
- **Python requirement** remains `>=3.13`.

### Added

- **Pluggable `Correction` abstraction** (`TaylorCorrection(order=0|1)` = EK0/EK1,
  `IteratedTaylorCorrection` = IEKF, grad-safe fixed-iteration). Selectable per
  solve via `correction=` on `gaussian_filter`, `gaussian_filter_adaptive`, and
  the preconditioned path.
- **Differentiable inference layer:** a pure `jit`/`grad`/`vmap`-able
  `marginal_loglik(theta, data, *, model)` with `InferenceProblem`, an
  `equinox`-based `ODEFilter` + `fit`, and an unconstrained-parameter layer
  (`PositiveReal`, `Real`, `unwrap`). NumPyro and BlackJAX inference examples.
- **Adaptive fixed-point smoother:** `gaussian_filter_adaptive(..., smoother=True)`
  returns a backward pass that `rts_smoother` consumes, using `O(#save points)`
  memory regardless of the adaptive sub-step count (Krämer 2025). jit/grad-safe.
  (Not supported together with `obs_model` yet.)
- **`FilterResult.success`** — whether an adaptive solve reached every save time.
- **Square-root process noise `prior.Q_sqr(h)`** (closed form for `IWP` /
  `PrecondIWP`); the filter no longer Choleskys a dense, ill-conditioned `Q(h)`.
- **Block-aware post-hoc calibration:** `rescale_sqr` / `rescale_sqr_seq` accept
  `prior=` to scale only the ODE-state block for `JointPrior` / `PrecondJointPrior`.
- Matern (`MaternPrior`, `PrecondMaternPrior`) and `JointPrior` latent-force
  priors (+ preconditioned variants); conservation-law and black-box / transformed
  measurement models; work-precision benchmark vs probdiffeq / Diffrax.
- A `benchmarks` optional-dependency extra (`pip install ode-filters[benchmarks]`).

### Fixed

- `MaternPrior.b` returned the wrong shape `(q+1,)` instead of `(q+1)*d` —
  a silent dimension bug for `d > 1` multi-output Matern models.
- The adaptive `success` flag now requires **every** save interval to be reached,
  not just the final one (a stalled middle interval previously reported success).
- Diagonal-mode calibration guards a zero denominator (was `0/0 → NaN`,
  grad-safe).
- The jittable adaptive solver guards non-finite local errors (no silent NaN
  output; reports `success=False`).
- Asynchronous / partial-dimension observations now fail loudly
  (`NotImplementedError` in `prepare_observations`) instead of silently
  corrupting inactive channels. (The full per-dimension fix is on the backlog.)
- Declared `scipy` (a test dependency) explicitly; dropped a defunct `polyfill.io`
  reference from the docs build.

### Notes

- Reverse-mode autodiff is now regression-tested across the full
  `{plain, preconditioned} x {EK0, EK1, IEKF}` matrix, the RTS smoother, the
  adaptive solver, and `marginal_loglik` (`test/test_filters/test_grad_safety.py`).
- Square-root predict/update are pinned against a dense moment-form Kalman oracle.

### Migration

| Old | New |
| --- | --- |
| `ekf1_sqr_loop(...)` / `_preconditioned` / `_sequential` / `_dynamic_scan` | `gaussian_filter(..., correction=, obs_model=, calibration=)` |
| `ekf1_sqr_adaptive_loop(...)` (public) | `gaussian_filter_adaptive(..., save_at=)` (`smoother=True` for smoothing) |
| `rts_sqr_smoother_loop(...)` / `_preconditioned` | `rts_smoother(prior, result)` |
| positional tuple unpack `m, P, *_ = ...` | named fields `result.m`, `result.P_sqr`, ... |
| `ODEmeasurement(...)` and the factory family | `ODEInformation` / `ODEconservation` + `prepare_observations` |
| `calibration="cumulative"` | `"dynamic"` (online) or `"diagonal_ekf0"`, or post-hoc `rescale_sqr` |

`ekf1_sqr_adaptive_loop` still exists as an *internal* driver (importable from
`ode_filters.filters.ode_filter_adaptive`) for the dense per-step diffusion trace
(`sigma_sqr_seq`) and the `sigma_in_error="running_mean"` controller; it is not
part of the public API and is not jit/grad-able. Prefer `gaussian_filter_adaptive`.

[res]: https://paufisch.github.io/ode_filters/api/filters/
[0.7.0]: https://github.com/paufisch/ode_filters/compare/v0.6.6...HEAD
