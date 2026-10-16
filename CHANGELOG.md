# Changelog

All notable changes to **ode-filters** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres
to [Semantic Versioning](https://semver.org/) (pre-1.0: breaking changes ship in a
minor bump, non-breaking changes in a patch bump).

## [Unreleased]

### Added

- `merge_measurement_times(ts, measurements)` adds off-grid measurement times to
  a grid, with the matching tolerance of `prepare_observations`.
- A `Measurement` and its `ObsModel` can be built inside `jax.jit`, `grad` or
  `vmap` when the times are concrete and the data traced.
- `InferenceProblem(..., adaptive=AdaptiveSettings(...))` runs
  `marginal_loglik` with adaptive steps (channels `"obs"`, `"ode"`, `"both"`
  and `"fenrir"`).
- `checkpoint=True` for `gaussian_filter`, `affine_filter` and
  `iterated_smoother` wraps each step in `jax.checkpoint`, which reduces the
  memory of reverse-mode differentiation.
- `gaussian_filter_adaptive(..., h_init="auto")` chooses the first step from
  the initial state and the tolerances (Hairer, Nørsett & Wanner, Sec. II.4).
- `ode_filters.information` protocols for the optional members of a
  measurement model: `HasLHSSelector`, `EK0Linearizable`, `HasNoiseCovariance`
  and `NoiseSettable`. A model that lacks one raises an error naming it.

### Changed

- Docs: improve clarity of writing.

### Fixed

- The work-precision benchmark ran probdiffeq without calibration. Both
  libraries now calibrate at every step, and the figures on the comparison page
  are rerun.
- Docs: the comparison table uses one name per method across the libraries and
  was checked again against their source.

## [0.8.0] - 2026-10-15

A breaking release that rebuilds the package around one filter step and turns it
towards inference from data. Every solver -- fixed grid, adaptive, frozen
linearization, iterated smoother -- now runs the same step; preconditioned priors
work everywhere; priors and information models are `equinox` modules; and the
release adds the conditional evidence, information criteria, posterior samples,
dense output and consistency diagnostics. There is no deprecation period: the
[migration guide](https://paufisch.github.io/ode_filters/migration/) lists every
change a 0.7 user can hit, old next to new. The input checks and correctness
fixes below come on top.

### Added

- **Time grids.** `gaussian_filter` takes the time grid `ts` (any strictly
  increasing array) instead of `tspan` and `N`; on a concrete uniform grid the
  prior's transition is computed once, also when a jitted function closes over
  the grid.
- **Calibration policies** `NoCalibration`, `DynamicCalibration` and
  `DiagonalCalibration(jacobian="ek1" | "ek0")`, which own the floor
  `min_sigma_sqr`. The strings `"none"`, `"dynamic"`, `"diagonal"` and
  `"diagonal_ekf0"` remain shorthands.
- **Preconditioning on every path.** Preconditioning is a change of variables
  inside each step, so `PrecondIWP` and `PrecondMaternPrior` work with adaptive
  steps, data and the iterated smoother, and `JointPrior` combines plain and
  preconditioned blocks. `prior.transition(h)` returns the matrices a step
  computes with (a `Transition`).
- **Data.** Each step conditions on the rows that have data, so measurements
  can have different times. The adaptive solver reports the data evidence in
  `log_likelihood_obs`, smooths with data (`smoother=True`), records its steps
  (`record_steps=`, `FilterResult.steps`), counts accepted and rejected steps
  (`n_accepted`, `n_rejected`) and offers `sigma_in_error="running_mean"`.
- **Frozen linearizations.** `AffineSurrogate`, `taylor_surrogate`,
  `slr_surrogate` and `affine_filter`: a square-root Kalman filter of a fixed
  affine model per step.
- **`iterated_smoother`**: the IEKS (`linearization="taylor"`) and the IPLS
  (`"slr"`), with warm starts and a per-pass `diagnostics` callback.
- **Evidence and information criteria.** `fenrir_loglik` and
  `conditional_loglik` estimate the conditional evidence `log p(Y | Z)` of the
  data given the ODE information (`marginal_loglik` channels `"fenrir"` and
  `"dalton"`); `information_criteria` measures what the ODE information and
  the data say about a hidden input.
- **Hidden inputs.** `joint_taylor_initialization` gives a consistent initial
  law for an ODE driven by a hidden input; `MaternPrior.stationary_cov_sqr()`
  the stationary law, finite at short length scales; `BasePrior.sample` exact
  prior sample paths on any grid.
- **Outputs.** `sample_posterior` draws joint sample paths from the smoothing
  posterior; `interpolate` evaluates filtered or smoothed marginals between the
  grid nodes, using the diffusion each step propagated with
  (`FilterResult.diffusion`).
- **Diagnostics.** `whitened_residuals`, `nis` and `residual_autocorrelation`
  check a fitted model against its data; `observation_predictive` gives the
  predictive distribution of linear observations.
- **Fitting.** `fit_scipy` minimizes with SciPy's quasi-Newton methods
  (L-BFGS-B by default) on JAX gradients; SciPy joins the `inference` extra.
  `NonNegativeReal` is a softplus parameter that may start at zero.
- **Types.** `Gaussian`, `Conditional` and `AffineModel` with the algebra
  `marginalize`, `invert`, `update`, `compose` and `log_likelihood`
  (`ode_filters.gaussian`); `Gaussian.cov` and `Gaussian.std` (e.g.
  `rts_smoother(result).std`); the `MeasurementModel` protocol; `BasePrior`;
  `filters.kernel.filter_step`.
- `BlackBoxMeasurement(..., ode_dim=...)` and `BlackBoxMeasurement.ode_dim`: the
  number of leading rows of `g` that drive diffusion calibration (default: all
  rows, i.e. `g` is the ODE residual).
- Docs: an *Inference from data* tutorial track (parameter estimation, a hidden
  input, Bayesian sampling), a *How it compares* page, the migration guide, and
  an API page for `ode_filters.calibration`. CI checks every internal link of
  the built site (`.github/scripts/check_site_links.py`). A project logo,
  animated in the docs header, tabs and navigation.
- `probdiffeq` joins the `benchmarks` extra.
- Tooling: ruff's docstring rules (Google convention) and an ASCII-only check
  for the package source; ruff also lints the notebooks; a `slow` marker for
  the MCMC runs of the example scripts; a float32 smoke test.
- `gaussian_filter_adaptive(..., max_step=)` caps every step, the first one
  included, so a loose tolerance cannot step over a short feature such as a
  brief pulse; `None` (default) bounds the steps by the save times only. It
  takes over the `h_max` of the removed Python loop.
- Tooling: the tests run in parallel (pytest-xdist) and treat warnings as
  errors; CI also runs them under strict JAX promotion and runtime type checks,
  against the newest allowed dependencies (weekly and on pull requests), and
  runs the benchmark in a quick mode.
- Tests: a consistency battery that runs every prior, correction, calibration
  policy, driver and output under `jit`, `vmap` and `grad` (checked against
  finite differences), with a float32 tier compared against float64; a
  linear test problem driven by a hidden input, with its closed-form solution;
  value checks in place of many shape- or finiteness-only assertions.
- Tests of the solvers' statistical claims and convergence, modelled on
  probdiffeq, ProbNumDiffEq.jl and Diffrax: two-sided calibration bands for the
  dynamic, diagonal and post-hoc calibrations; an exact chi-square calibration
  check on data drawn from the prior (`BasePrior.sample`); a convergence-order
  matrix (q = 1..6, four priors, EK0 and EK1, filter and smoother); stiff van
  der Pol against SciPy's Radau, a DETEST subset, a 20-dimensional linear
  system against `expm`; and agreement with probdiffeq 0.8 on a fixed grid.

### Changed

- **Layout.** `ode_filters.measurement` is now `ode_filters.information`,
  `priors.gmp_priors` is split into `priors.base`, `iwp`, `matern`, `ioup`,
  `joint` and `initialization`, the corrections and statistical linearization
  move to `ode_filters.linearization`, the square-root algebra to
  `ode_filters.gaussian`, and `inference.parameter_inference` becomes
  `inference.likelihood`.
- **Models are `equinox` modules.** Hyperparameters, selection matrices and
  noise covariances are pytree leaves; orders, dimensions and vector fields are
  static. Compile with `equinox.filter_jit` (`jax.jit` with static model
  arguments fails). `measure.R = ...` becomes `measure.with_noise(...)`,
  `E_constraint` becomes `E_lhs`, `get_noise(t=...)` becomes `noise_sqr(t=...)`;
  the sigma estimators take `mz`, `Pz_sqr`; `slr_linearize` and
  `check_arg_projection` take `t` without a default.
- **Results.** `FilterResult` is in the original state coordinates for every
  prior (`m_bar`, `P_bar_sqr` and `T` are gone), gains `diffusion`,
  `n_accepted`, `n_rejected` and `steps`, and its field order changed.
  `rts_smoother(result)` takes no prior and returns a `Gaussian`. On the
  adaptive path `log_likelihood` is the ODE-information evidence only.
- **Corrections.** `linearize(measure, m, t=...)` returns an `AffineModel` and
  `correct(measure, x, model, t=...)` returns `(posterior, innovation)`. Each
  step linearizes once, so EK0 (`TaylorCorrection(order=0)`) calibrates with
  its own linearization and evaluates no vector-field Jacobian; its
  `sigma_sqr` changes slightly.
- **Priors.** `PrecondIWP` and `PrecondMaternPrior` report `A(h)`, `Q(h)` and
  `Q_sqr(h)` in the original coordinates. A prior implements `A(h)` and
  `Q_sqr(h)`; `Q(h)` is derived.
- **Data.** `prepare_observations` raises when a measurement time is not a grid
  node (it dropped the data silently). Data innovations at rows without data
  are zero with unit variance.
- **Inference.** `InferenceProblem` and `ODEFilter` take `ts` instead of
  `tspan` and `N`, and a calibration policy instead of `min_sigma_sqr`; `fit`
  trains `ode_params` only.
- `PController.propose` / `PIController.propose` are written with `jax.numpy`
  so the jittable solver can call them; they return a 0-d JAX array instead of
  a Python `float`, and a non-positive `err_prev` now means "no memory", like
  `None`. The `StepSizeController` protocol requires only `propose`. A custom
  controller passed to `gaussian_filter_adaptive` must be traceable.
- `TransformedMeasurement` uses autodiff exactly when no `sigma_jacobian` is
  given.
- **Changes results:** `MaternPrior` and `PrecondMaternPrior` use the textbook
  length scale `l` (Rasmussen & Williams 2006, eq. 4.14): the rate of the SDE is
  `sqrt(2q+1) / l`, where it was `sqrt((2q+1) / l)`, so the old `length_scale`
  acted as the square of the usual one. Pass `length_scale=sqrt(l_old)` to
  reproduce an earlier result; `length_scale=1` is unchanged. The priors are
  now tested against independent oracles (the textbook kernels, and Van Loan's
  block exponential of each prior's SDE).
- Docs: the home page and README lead with inference; wider layout, home-page
  cards, API pages grouped by topic; tutorials compile with
  `equinox.filter_jit`; the "Calibration modes" notebook explains what
  calibration does; the adaptive-steps guide is a short executed page; figures
  use `tueplots` with Paul Tol's bright palette as SVG. The docs build fails
  when a notebook cell raises.
- Benchmarks: `benchmarks/work_precision.py` writes next to itself, to
  `benchmarks/results/` (`--out-dir` to change), instead of into the working
  directory: PDF and SVG figures in the docs' `tueplots` style with one
  factorized legend, and a JSON with every number and the run's metadata
  (versions, git commit, CPU, load average). The timing compiles first, then
  interleaves all configurations over repeated rounds and reports the median
  and interquartile range. `--quick` is a smoke run of under a minute,
  `--repeats` sets the number of rounds, `--replot` redraws from the JSON.
  `tueplots` joins the `benchmarks` extra, and pyright now checks
  `benchmarks/`.

### Removed

- `ipls_smoother` and `IPLSResult`: use `iterated_smoother(..., linearization="slr")`.
- `affine_filter_scan`: use `affine_filter`.
- `PrecondJointPrior`: use `JointPrior` with preconditioned blocks.
- `ODEconservation` and `SecondOrderODEconservation`: pass
  `constraints=[Conservation(A, p)]` to the information classes.
- `CorrectionResult`; `sqr_marginalization`, `sqr_inversion` and
  `compose_backward_conditionals` (use `marginalize`, `invert`, `compose`).
- The `min_sigma_sqr=` arguments of the solvers (use a calibration policy).
- `TransformedMeasurement(use_autodiff_jacobian=...)`;
  `Measurement.find_index`, `.residual`, `.jacobian` and `.get_noise_matrix()`
  (use `Measurement.R`); `Conservation.jacobian()` (use `.A`).
- The internal drivers `filters.ode_filter_loop`, `ode_filter_step`,
  `ode_filter_adaptive` (with the adaptive loop `sqr_adaptive_loop`, named
  `ekf1_sqr_adaptive_loop` before 0.7.2) and `ipls`. Its per-step trace is
  `record_steps=` and its `sigma_in_error="running_mean"` an option of
  `gaussian_filter_adaptive`; its `h_min` and `h_max` have no counterpart.
- The top-level exports `SLRModel` and `slr_linearize` (import them from
  `ode_filters.linearization`), and the exports of the quadrature rules
  `gauss_hermite_rule`, `cubature_rule` and `quadrature_nodes` (choose a rule
  with `rule=` and `n_nodes=`).
- `MaternPrior` internals `_F`, `_L`, `_q`, `S`, `n` and `A_and_Q`.
- `docs/examples/ode_filters.mplstyle`.

### Fixed

- Switching `jax_enable_x64` within one process no longer corrupts later
  solves. From JAX 0.9.2 on, JAX keeps the device copy of a NumPy constant
  captured under `jit` across the switch, keyed by the array object, and the
  priors reused cached constant arrays: a float64 solve after a float32 one ran
  on float32 constants (eight digits short), and a float32 solve after a
  float64 one warned. The cached constants are now handed out as fresh copies.
- `rts_smoother`, `sample_posterior` and smoothed `interpolate` returned NaN
  after a calibrated diffusion of exactly zero. The default dynamic calibration
  gives one for a solve started exactly at an equilibrium (or a solution the
  prior reproduces, like `x' = 1`), diagonal calibration for a component at
  rest. `invert` now gives a component with exactly zero variance a zero gain
  (the pseudo-inverse), the limit of a vanishing `min_sigma_sqr`; all other
  results are unchanged bit for bit. Gradients at such a point still need a
  floor (see the sharp bits).
- `taylor_mode_initialization` and `joint_taylor_initialization` held the time
  at `t0`, so the higher initial derivatives of a time-dependent vector field
  were wrong (`x''' = 1` instead of `0` for `x' = cos(t) x` at `t0 = 0`). Time
  is now expanded with the state. **This changes the initial state, and so the
  results, of time-dependent problems.**
- A prior with a singular `Xi` (e.g. `Xi=0`) produced a NaN process noise
  without an error; a concrete `Xi` must now be symmetric positive definite.
- `PrecondMaternPrior` returned NaN at short length scales (e.g. `q = 2`,
  `length_scale = 2.6e-3`, `h = 1e-2`): the 0.7.3 fix reached only
  `MaternPrior`. Both now share the length-scale-normalized `A(h)` and the
  square-root `Q_sqr(h)`.
- `gaussian_filter_adaptive` used a preconditioned prior's preconditioned
  matrices as if they were in the original coordinates, and `JointPrior`
  ignored a preconditioned block's coordinates; both returned wrong solutions
  without an error. Both now work (see Added).
- `calibration="diagonal_ekf0"` took the EK0 Jacobian from the prior's `E1`
  instead of the model's left-hand-side selector. For second-order ODEs that
  selects the wrong derivative and inflated `sigma_sqr` by orders of magnitude
  (median 190 instead of 0.026 on `x'' = -x`); the means were right. **This
  changes the calibrated output of second-order models run with
  `diagonal_ekf0`.**
- `gaussian_filter_adaptive` did not validate `calibration`: a misspelled mode
  ran as `"none"`, and a diagonal mode with a non-diagonal `Xi` was accepted.
- A diagonal calibration policy accepted a non-diagonal `Xi` under `jit`, where
  `Xi` is traced. Priors now record at construction whether `Xi` is diagonal
  (`xi_diagonal`), and the check uses that record under `jit`.
- The `log_likelihood_obs` docstring called it the objective of Fenrir. It is
  the sequential data evidence; Fenrir's objective is the new `fenrir_loglik`.
- Reverse-mode gradients were NaN wherever a diffusion estimate or the adaptive
  local error is exactly zero -- e.g. any system started at an equilibrium or at
  rest: `sqrt'(0) = inf` turned the backward pass into `0 * inf`. The scaling
  and the adaptive error norm use a zero-safe square root; forward values are
  unchanged.
- `gaussian_filter_adaptive` ignored a custom controller's `propose` and ran
  its own copy of the PI law. The built-in controllers produce the same step
  sizes as before.
- `BlackBoxMeasurement` could not be used in any solver (`AttributeError: ...
  'ode_dim'`); `TransformedMeasurement` around it now forwards `ode_dim` too.
- `IOUPPrior` accepted `n_quad <= q` and returned a wrongly shaped `Q_sqr`.
- The opt-in runtime type checking (`ODE_FILTERS_TYPECHECK=1`) failed at import.
- `JointPrior.xi` raised `AttributeError`; it returns `blkdiag(Xi_x, Xi_u)`.
- `taylor_mode_initialization` failed with JAX 0.11 for a vector field that
  unpacks its state (`s, i, r = x`): `jax.experimental.jet` cannot push an
  empty series through the unpacking. The first coefficient, the vector field
  itself, is now evaluated directly.
- `IWP`, `PrecondIWP`, `PrecondMaternPrior`, `gaussian_filter_adaptive`,
  `aggregate_sigma_sqr(kind="running")` and `residual_autocorrelation` raised
  under JAX's strict promotion (`jax_numpy_dtype_promotion="strict"`,
  `jax_numpy_rank_promotion="raise"`). Values and gradients are unchanged.
- Input checks for configurations that failed silently or deep inside a solve:
  - the information classes validate their selection matrices (a sliced `E1`
    used to broadcast into a wrong residual);
  - noise covariances are checked up front: noise-free channels are fine, but a
    singular covariance that mixes channels, or an indefinite one, raises
    instead of producing NaN;
  - `constraints=` rejects entries that are not `Conservation` and copies the
    list; `Conservation` and `Measurement` accept array-likes;
  - `TransformedMeasurement` used a given `sigma_jacobian` only with
    `use_autodiff_jacobian=False`.
- Docs: `gaussian_filter` was missing from the API reference, cross-links inside
  the notebooks were broken on the site, the joint state-parameter example
  crashed, `sharp-bits` claimed the solvers were already jitted, and the README
  install command installed no dev tools.
- Tests: the uncertainty-coverage test checked a band ~1e4 times too wide, the
  logistic observation test compared against the wrong initial value, about 20
  parity tests compared a function with itself, and the accuracy tolerances
  were 5-9 orders of magnitude looser than the measured errors. The solvers are
  now also checked against recorded reference outputs and a dense Kalman
  oracle.
- Tests: gaps a mutation run exposed are closed -- non-autonomous ODEs on every
  solver path, `interpolate` under dynamic calibration (covariances, off-midpoint
  times, against a dense bridge), several sensors with their own noises and
  times, demeaning in `residual_autocorrelation`, gradients of `affine_filter`
  in the surrogate, the evidence at extreme diffusions, an independent oracle
  for `joint_taylor_initialization`, and whole runs of `TransformedMeasurement`
  and `SecondOrderODEInformationWithHidden`; thirteen more reference scenarios.
- Tests: the dense Kalman oracle updated covariances in the short form
  `P - K S K.T` and smoothed with `pinv`, which cost it seven digits under a
  dynamic diffusion at q = 3 (the solvers were right throughout); with the
  Joseph form it agrees with them to 1e-12 instead of 1e-7. The remaining
  coverage test passed for any band wider than the error (the posterior was
  ~6e6 times wider), and the post-hoc "unit whitened residuals" test held by
  construction; both are replaced by the calibration checks above.

## [0.7.3] - 2026-10-04

A patch release: `MaternPrior` no longer returns NaN transition/process-noise
matrices at short length scales, and `TransformedMeasurement` now forwards
`ode_dim` and `E_args` so transformed measures run through the filter loops.
Non-breaking.

### Fixed

- `MaternPrior.A_and_Q` / `A` / `Q` returned NaN at short length scales: the
  scaling-and-squaring `expm` of the `[[F, S], [0, -F.T]]` Hamiltonian block
  carries `S ~ lam^(2q+1)` and overflowed already at `q = 2`,
  `length_scale = 2.6e-3` (`lam ~ 44`), `h = 1e-2` -- i.e. `lam h = 0.44`, which
  is what any problem re-expressed on a unit time interval looks like. `A(h)` is
  now `D_lam expm(lam M h) D_lam^-1` in the length-scale-normalized coordinates
  `_make_matern_sqr_noise` already builds, and `Q(h)` is `R.T @ R` from the
  square-root matrix-fraction decomposition `Q_sqr` already uses. Agrees with
  the block form to ~1e-13 wherever that was finite; regression test
  `test/test_gmp_priors/test_matern_short_length_scale.py`.

### Added

- `TransformedMeasurement.ode_dim` and `TransformedMeasurement.E_args`, both
  delegated to the wrapped base model. A state transformation reparameterises
  *where* the residual is evaluated; it changes neither how many rows are ODE
  defect nor -- for the usual case of a link function on a coordinate the base
  already reads -- which coordinates the residual is nonlinear in. Previously
  only `R` was forwarded, so a transformed measure could not be run through the
  filter loops at all: they slice `H_t[:ode_dim]` for calibration. `E_args`
  additionally keeps statistical linearization cheap -- `slr_linearize` falls
  back to the identity when it is absent, which is correct but quadratures over
  the whole state (`n_nodes ** state_dim` nodes instead of
  `n_nodes ** len(args)`). A `sigma` that mixes in coordinates the base does not
  read must override `E_args` with the union of the two sets. Over a
  `BlackBoxMeasurement` base, which declares neither, both raise
  `AttributeError` rather than guessing -- that is the signal `slr_linearize`
  probes for, so raising is what keeps its identity fallback correct.

## [0.7.2] - 2026-08-11

A feature release on top of 0.7.1: a new prior family (`IOUPPrior`), a new
correction family (statistical linearization / `QuadratureCorrection`) and the
trajectory-level `ipls_smoother`, numerically stable high-order square-root
process noise for the Matern and IWP priors, and two automatic-differentiation
fixes that unblock gradient-based hyperparameter inference. Non-breaking.

### Added

- `marginal_loglik(..., channel=...)` selects which evidence channel to return:
  `"obs"` (default, unchanged -- the data evidence / Fenrir objective), `"ode"`
  (the ODE-defect residual evidence, which needs no observations, and so is the
  one channel whose overload accepts `data=None`), or `"both"`
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
- **Statistical linearization (`QuadratureCorrection`) as a `Correction` strategy.**
  Fits the affine surrogate over the predictive *spread* rather than tangent to the
  predicted mean, evaluating the Gaussian expectations by quadrature: `H` becomes
  the expected Jacobian and the linearization-residual covariance `Omega` enters the
  innovation covariance as inflated measurement noise (`H P H^T + Omega + R`). Both
  differences vanish identically for an affine vector field, so the correction is a
  no-op on linear problems and on well-resolved pure solves. Rules: `"gauss_hermite"`
  (default, exactness `2 * n_nodes - 1`) and `"cubature"` (third-degree spherical,
  `2p` nodes). `max_iters > 1` gives the iterated posterior-linearization filter
  (IPLF). Works on the plain, sequential-observation, and preconditioned paths.
- **`ipls_smoother`** — the iterated posterior-linearization smoother. Refits the
  surrogate at every step over the *smoothed* marginal and re-runs filter and
  smoother for a fixed number of passes; `n_iters=0` reproduces the one-shot SLR
  filter plus RTS exactly. Fixed grid, plain priors, fixed process noise, with
  optional `obs_model`. Also exports `affine_filter_scan`, the linear-Gaussian
  filter over a precomputed per-step surrogate that drives it.
- **`ode_filters.filters.statistical_linearization`** — the SLR primitives:
  `slr_linearize`, `SLRModel`, `gauss_hermite_rule`, `cubature_rule`,
  `quadrature_nodes`, and `check_arg_projection`. The quadrature runs over the
  *projected* marginal `N(E_args m, E_args P E_args^T)`, so its dimension is the ODE
  dimension rather than the `d (q+1)` state dimension — which is both what makes the
  spectrally accurate rule affordable and why no square root of the ill-conditioned
  full `P` is ever taken.
- **`BaseODEInformation.E_args`** — selection matrix for the coordinates the vector
  field reads (`E0` for first-order, `[E0; E1]` for second-order, plus `E0_hidden`
  for the hidden-state variants). Custom subclasses whose residual reads more than
  `E0 @ state` must override it; `check_arg_projection` validates a declaration by
  perturbing along the null space of `E_args`.
- `MaternPrior` / `PrecondMaternPrior` accept an `n_quad` keyword argument
  controlling the Gauss-Legendre node count of the new square-root process-noise
  decomposition (default 64, robust to the float64 order ceiling). `n_quad <= q`
  is rejected at construction: the rule contributes one row per node to the QR, so
  too few nodes yield a non-square `[n_quad, q+1]` factor -- a wrong shape rather
  than a coarser approximation, which otherwise surfaced only later as an opaque
  shape error from inside a solve.

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

- **Reverse-mode gradients no longer come back `NaN` from a deterministic initial
  condition.** `np.linalg.qr` has an *undefined* derivative at a rank-deficient
  input, and an exactly-zero stacked factor is not a corner case here: it is what
  the first backward conditional *is* when the filter starts from a Dirac initial
  state, which is exactly what `taylor_mode_initialization` returns. Because a zero
  cotangent times a `NaN` derivative is still `NaN`, that single degenerate factor
  poisoned the gradient of *everything* downstream -- including quantities that did
  not depend on it at all, such as the smoothed *mean*. `sqr_marginalization` and
  `sqr_inversion` now route their QR through `_safe_qr`, which substitutes a
  well-conditioned stand-in on the exactly-zero branch (the standard double-`where`
  idiom) and defines the derivative there to be zero. That is the correct value,
  not a fudge: `P = P_sqr.T @ P_sqr` is smooth even where `P_sqr` is not, so
  `dP = 2 P_sqr.T d(P_sqr)` vanishes at `P_sqr = 0` for any finite `d(P_sqr)`.
  Values are bit-unchanged; `jax.grad` through `rts_smoother` (and `ipls_smoother`)
  from a singular `P_0_sqr` now matches central finite differences instead of
  returning `NaN`, so the jitter workaround previously documented for those
  functions is no longer needed. Two degeneracies remain deliberately unhandled: a
  *merely* rank-deficient factor (which needs a rank-revealing factorization), and a
  singular *innovation* covariance, which stays `NaN` -- correctly, since
  conditioning a zero-variance state on a zero-variance observation is ill-posed.
  Regression coverage:
  `test/test_sqr_gaussian_inference/test_degenerate_factor_gradients.py`.
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
[Unreleased]: https://github.com/paufisch/ode_filters/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/paufisch/ode_filters/compare/v0.7.3...v0.8.0
[0.7.3]: https://github.com/paufisch/ode_filters/compare/v0.7.2...v0.7.3
[0.7.2]: https://github.com/paufisch/ode_filters/compare/v0.7.1...v0.7.2
[0.7.1]: https://github.com/paufisch/ode_filters/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/paufisch/ode_filters/compare/v0.6.6...v0.7.0
