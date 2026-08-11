# Changelog

All notable changes to **ode-filters** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/) (pre-1.0: breaking changes ship in a
minor bump, non-breaking changes in a patch bump).

## [Unreleased]

### Added

- `marginal_loglik(..., channel=...)` selects which evidence channel to return:
  `"obs"` (default, unchanged -- the data evidence / Fenrir objective), `"ode"`
  (the ODE-defect residual evidence, which needs no observations), or `"both"`
  (the `(ll_ode, ll_obs)` pair). Methods that weight the two channels separately
  -- split / hybrid / tiered hyperparameter selection -- need the pair, and
  previously had to call `gaussian_filter` directly to get it. Note the two are
  sums over *different* numbers of terms (`N` filter steps vs `K` observations),
  so the raw sum makes the residual channel's influence scale with grid density.
- `InferenceProblem.prior_fn`, an optional `theta -> prior` callable for fitting
  the *prior's* hyperparameters (diffusion scale, length scale) rather than --
  or in addition to -- vector-field parameters. When set it takes precedence
  over the static `prior` and is evaluated inside the traced region, so the
  hyperparameters are differentiated. Defaults to `None` (static prior,
  behaviour unchanged).

- `IOUPPrior` -- an integrated Ornstein-Uhlenbeck process prior (Bosch, Hennig &
  Tronarp, *Probabilistic Exponential Integrators*, NeurIPS 2023), exported at the
  top level. Its highest derivative follows linear dynamics `dY^(q) = R Y^(q) dt + dW`
  with a constant `rate` `R` (scalar, per-dimension vector, or full `d x d` matrix);
  baking the ODE's linear part into the prior turns the solver into a probabilistic
  exponential integrator (markedly better on stiff / semi-linear problems). It is a
  drop-in prior: the measurement model and the filter/smoother recursion are
  unchanged, and `Q_sqr(h)` reuses the same square-root matrix-fraction decomposition
  as `MaternPrior`. `IWP` is the `rate = 0` special case. (The re-linearized
  exponential-Rosenbrock variant, which needs per-step re-discretisation, is not
  included.)
- `MaternPrior` / `PrecondMaternPrior` accept an `n_quad` keyword argument
  controlling the Gauss-Legendre node count of the new square-root process-noise
  decomposition (default 64, robust to the float64 order ceiling).

### Changed

- `MaternPrior.Q_sqr` / `PrecondMaternPrior.Q_sqr` now compute the square-root
  process noise by a square-root matrix-fraction decomposition (QR of the
  Gauss-Legendre-propagated rank-1 diffusion) in length-scale-normalized
  coordinates, instead of a Cholesky of the dense `expm`-derived `Q(h)`. The factor
  the square-root filter consumes stays finite and accurate up to the float64 order
  ceiling (~`q = 18`) rather than NaN-ing around `q >= 6`; results match the old
  path (to ~1e-13) wherever it was valid. The dense `Q(h)` itself is unchanged
  (still ill-conditioned at high `q` -- prefer `Q_sqr` downstream).

### Fixed

- **Every prior is constructible from traced hyperparameters again.** Two
  construction-time defects introduced with the square-root process-noise work
  above made `jax.jit` / `jax.grad` over a prior's own hyperparameters raise, which
  breaks gradient-based hyperparameter inference — the library's primary use case —
  for `IWP`, `PrecondIWP`, `MaternPrior`, `PrecondMaternPrior`, and any `JointPrior`
  built from them. Eager construction was unaffected, which is why the suite did not
  catch it.
  - The `Q_bar` finiteness backstop tested a JAX array with a Python `bool()`,
    raising `TracerBoolConversionError` for any traced `Xi`. It is now skipped under
    tracing (where it cannot be evaluated) and keeps its eager behaviour, which is
    where a pathological `q` would be introduced.
  - `_make_matern_sqr_noise` cast `lambda = sqrt((2q+1)/length_scale)` with
    `float()`, raising `ConcretizationTypeError` for a traced `length_scale`. The
    cast was unnecessary — nothing downstream needs a concrete value — and is
    removed, so `length_scale` is differentiable again.
  - Regression coverage: `test/test_gmp_priors/test_traced_construction.py`
    constructs every prior under `jit`/`grad` and pins `grad` against central finite
    differences, so a future cast that silently *freezes* a hyperparameter (rather
    than raising) fails too. This is the prior-construction analogue of
    `test/test_filters/test_grad_safety.py`, which guards the solver loops.
- `IWP` / `PrecondIWP` now factor the constant Hilbert `Q_bar` with a closed-form
  square-root factor (exact integer-factorial formula, re-triangularized via QR)
  instead of a numerical Cholesky that lost positive-definiteness and returned `NaN`
  around `q >= 13`. The factor now stays finite and reconstructs `Q_bar` to machine
  precision well past `q = 20` (matching `probdiffeq` and `ProbNumDiffEq.jl`, which
  both use closed-form IWP factors); results are unchanged for `q <= 12`. A cheap
  construction-time finiteness guard remains as a defensive backstop.

## [0.7.1] - 2026-06-23

Documentation, packaging, and API-surface polish on top of the 0.7.0 refactor.
No breaking changes.

### Added

- `PrecondJointPrior` is now exported from the top-level `ode_filters` namespace
  (previously importable only from `ode_filters.priors`, even though the solver
  already dispatched on it).
- `ode_filters.__version__`.
- An `inference` optional-dependency extra (`pip install ode-filters[inference]` --
  NumPyro / BlackJAX / Optax) for the `examples/` parameter-inference scripts, which
  now also enable `jax_enable_x64`.

### Changed

- Narrowed the ODE-information `constraints=` parameter type to
  `list[Conservation] | None` (a `Measurement` was already rejected at runtime).

### Fixed

- Documentation accuracy: `FilterResult.log_likelihood` is documented as the
  post-calibration ODE-residual marginal likelihood (not comparable across
  calibration modes); `success` also requires a finite log-likelihood; the adaptive
  solver's `log_likelihood_obs` is always `None` (its observation likelihood is
  folded into `log_likelihood`). Documented the Matern high-smoothness (`q >= 6`)
  dense-`Q` limitation. Removed overclaiming / "calibrated-guarantee" framing across
  the guides and example notebooks.

## [0.7.0] - 2026-06-20

A major refactor that re-scopes the library from "a square-root EK1 + RTS solver"
into a **pluggable Gaussian-filtering inference substrate**: the measurement
model and the linearization scheme are now independent, pluggable axes, with a
single consolidated solver API and a differentiable parameter-inference layer.

> **Upgrading:** this release intentionally makes a clean break (no deprecation
> aliases) -- every entry under **Changed** and **Removed** below is breaking. The
> Python requirement is unchanged (`>=3.13`). See **Migration** at the end.

### Changed

- **Consolidated solver API.** The `ekf1_sqr_loop*` family
  (`{plain/preconditioned} x {joint/sequential} x {for-loop/scan}`) is replaced
  by three auto-dispatching entry points returning a single
  [`FilterResult`][res] NamedTuple:
  - `ekf1_sqr_loop*` -> **`gaussian_filter`** (preconditioning is selected from
    the prior type; observations via `obs_model=`; linearization via
    `correction=`).
  - `ekf1_sqr_adaptive_loop` -> **`gaussian_filter_adaptive`** (jit/vmap/grad-safe
    save-at-grid solver).
  - `rts_sqr_smoother_loop*` -> **`rts_smoother(prior, result)`**.
- **Return types** changed from wide positional tuples to the named
  `FilterResult`. Index results by name (`result.m`, `result.P_sqr`,
  `result.log_likelihood`, ...) instead of by position. `AdaptiveLoopResult` is
  no longer a public export.

### Removed

- **Low-level loop variants removed entirely** (not just de-exported): the
  non-scan and sequential `ekf1_sqr_loop*` variants no longer exist, even via
  submodule import. The surviving internal scan loops live in
  `ode_filters.filters.{ode_filter_loop,ode_filter_step,ode_filter_adaptive}`
  and are implementation detail, not public API.
- **Measurement factory family:** `ODEmeasurement`, `ODEconservationmeasurement`,
  `SecondOrderODEmeasurement`, `SecondOrderODEconservationmeasurement`, and
  `build_obs_at_time`. Use `ODEInformation` / `ODEconservation` for the model,
  and `prepare_observations` -> `ObsModel` for data.
- **`calibration="cumulative"`** (non-Markovian global-sigma post-multiply;
  superseded by `diagonal_ekf0`). Valid modes are now `"dynamic"` / `"diagonal"`
  / `"diagonal_ekf0"` / `"none"`.

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
  `PrecondIWP`); the IWP filter path no longer Choleskys a dense, ill-conditioned
  `Q(h)`.
- **Block-aware post-hoc calibration:** `rescale_sqr` / `rescale_sqr_seq` accept
  `prior=` to scale only the ODE-state block for `JointPrior` / `PrecondJointPrior`.
- Matern (`MaternPrior`, `PrecondMaternPrior`) and `JointPrior` latent-force
  priors (+ preconditioned variants); conservation-law and black-box / transformed
  measurement models; work-precision benchmark vs probdiffeq / Diffrax.
- A `benchmarks` optional-dependency extra (`pip install ode-filters[benchmarks]`).
- **Reverse-mode autodiff is regression-tested** across the full
  `{plain, preconditioned} x {EK0, EK1, IEKF}` matrix, the RTS smoother, the
  adaptive solver, and `marginal_loglik` (`test/test_filters/test_grad_safety.py`);
  square-root predict/update are pinned against a dense moment-form Kalman oracle.

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
[Unreleased]: https://github.com/paufisch/ode_filters/compare/v0.7.1...HEAD
[0.7.1]: https://github.com/paufisch/ode_filters/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/paufisch/ode_filters/compare/v0.6.6...v0.7.0
