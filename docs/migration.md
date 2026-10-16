# Migrating from 0.7 to 0.8

Release 0.8.0 reorganizes the package around one filter step and makes the
models JAX pytrees. It breaks the 0.7 API in a single release, with no
deprecation period. This page lists every change that affects code written for
0.7, with the old form on the left and the new one on the right. The
[changelog](https://github.com/paufisch/ode_filters/blob/main/CHANGELOG.md)
lists the additions.

Most code needs three edits: pass a time grid instead of `tspan, N`, call
`rts_smoother(result)` without the prior, and compile with `equinox.filter_jit`
instead of `jax.jit` with static arguments.

```python
# 0.7
result = gaussian_filter(mu_0, P_0_sqr, prior, measure, (0.0, 5.0), 50)
m, P_sqr = rts_smoother(prior, result)
solve = jax.jit(gaussian_filter, static_argnums=(2, 3, 4, 5))

# now
ts = np.linspace(0.0, 5.0, 51)  # N + 1 nodes; non-uniform grids work too
result = gaussian_filter(m_0, P_0_sqr, prior, measure, ts)
m, P_sqr = rts_smoother(result)  # a Gaussian; unpacks as before
solve = eqx.filter_jit(gaussian_filter)
```

## Module layout

The top-level `ode_filters` namespace exports the public API, so most imports
from it keep working. Imports from subpackages move:

| 0.7 | now |
| --- | --- |
| `ode_filters.measurement` | `ode_filters.information` (same class names) |
| `ode_filters.priors.gmp_priors` | `ode_filters.priors` (modules `base`, `iwp`, `matern`, `ioup`, `joint`, `initialization`) |
| `ode_filters.inference.sqr_gaussian_inference` | `ode_filters.gaussian` (see [Gaussian algebra](#gaussian-algebra)) |
| `ode_filters.filters.correction` | `ode_filters.linearization`; the corrections are no longer re-exported from `ode_filters.filters` |
| `ode_filters.filters.statistical_linearization` | `ode_filters.linearization.statistical` |
| top-level `SLRModel`, `slr_linearize` | `ode_filters.linearization` (next to `taylor_linearize`) |
| `gauss_hermite_rule`, `cubature_rule`, `quadrature_nodes` | no longer exported: choose the rule by name (`rule=`, `n_nodes=`); the functions stay in `ode_filters.linearization.statistical` |
| `CalibrationMode` (in `ode_filters.filters.ode_filter_adaptive`) | `ode_filters.calibration.CalibrationMode` |
| `ode_filters.inference.parameter_inference` | `ode_filters.inference.likelihood` |
| `ode_filters.filters.gaussian_filter`, `ode_filter_loop`, `ode_filter_step`, `ode_filter_adaptive`, `ipls` | `ode_filters.filters.fixed_grid`, `adaptive`, `smoothing`, `kernel`, `iterated`, `result` |

## Solvers

| 0.7 | now |
| --- | --- |
| `gaussian_filter(mu_0, P_0_sqr, prior, measure, tspan, N, ...)` | `gaussian_filter(m_0, P_0_sqr, prior, measure, ts, ...)` with `ts = np.linspace(t0, t1, N + 1)`; any strictly increasing grid works |
| `gaussian_filter(..., min_sigma_sqr=s)` | `gaussian_filter(..., calibration=DynamicCalibration(min_sigma_sqr=s))` (or `DiagonalCalibration(min_sigma_sqr=s)`) |
| `gaussian_filter_adaptive(mu_0, ...)` | `gaussian_filter_adaptive(m_0, ...)`; `min_sigma_sqr=` moves into the calibration policy as above |
| `rts_smoother(prior, result)` returning `(m, P_sqr)` | `rts_smoother(result)` returning a `Gaussian` (unpacks as `m, P_sqr`) |
| `ipls_smoother(mu_0, P_0_sqr, prior, measure, tspan, N, ...)`, `IPLSResult` | `iterated_smoother(m_0, P_0_sqr, prior, measure, ts, linearization="slr", ...)`, `IteratedSmootherResult`: `result.m` is smoothed, `result.filter` the last filter pass |
| `affine_filter_scan(...)` | `affine_filter(m_0, P_0_sqr, prior, AffineSurrogate(H, c, R_sqr), ts, obs_model=...)`, returning a `FilterResult`. The prior replaces the raw transition matrices `A_h, b_h, Q_sqr`. Without data, `log_likelihood_obs` is `None` (it was `0.0`). |

### The adaptive loop

The internal Python-loop driver `sqr_adaptive_loop` is removed;
`gaussian_filter_adaptive` covers it:

- `gaussian_filter_adaptive(..., record_steps=n)` records the accepted steps
  (`result.steps`: `t`, `h`, `m`, `sigma_sqr`, `count`). The loop also returned
  covariances and backward conditionals per step; to get them, run
  `gaussian_filter` on the grid `[t_0, *result.steps.t[:count]]`, which
  repeats the same steps.
- `sigma_in_error="running_mean"` is an argument of `gaussian_filter_adaptive`.
- The loop's `h_max` is `gaussian_filter_adaptive(..., max_step=)`. Its
  `h_min` has no counterpart: steps never cross a save time, and a solve that
  would need more than `max_steps` steps in one save interval stops with
  `result.success = False` (it never raises under `jit`).

The calibration strings `"none"`, `"dynamic"`, `"diagonal"` and
`"diagonal_ekf0"` are still accepted; they stand for `NoCalibration()`,
`DynamicCalibration()`, `DiagonalCalibration()` and
`DiagonalCalibration(jacobian="ek0")`.

### `FilterResult`

- Every field is in the original state coordinates, also for preconditioned
  priors. The preconditioned-coordinate fields `m_bar`, `P_bar_sqr` and `T` are
  removed; use `m` and `P_sqr`.
- The field order changed and fields were added (`diffusion`, `n_accepted`,
  `n_rejected`, `steps`): access fields by name, not by position.
- `gaussian_filter_adaptive`: `log_likelihood` is the ODE-information evidence
  only; the data evidence is in `log_likelihood_obs`, as on the fixed grid.
  Their sum is the old `log_likelihood`.

### Preconditioned priors

`PrecondIWP` and `PrecondMaternPrior` now work in every solver (adaptive steps,
data, the iterated smoother).

- `A(h)`, `Q(h)` and `Q_sqr(h)` return the matrices in the **original**
  coordinates (they returned the preconditioned ones). The preconditioned form
  is `prior.transition(h)` (a `Transition` with `T_diag`). `A()`, `b()`, `Q()`
  and `Q_sqr()` without a step no longer exist.
- `PrecondIWP` is a subclass of `IWP`, and `PrecondMaternPrior` of
  `MaternPrior`: `isinstance(PrecondIWP(...), IWP)` is now `True`.
- `PrecondJointPrior` is removed: use `JointPrior(PrecondIWP(...), PrecondIWP(...))`.
  Plain and preconditioned blocks can be mixed.

## Models are equinox modules

Priors and information models are `equinox.Module`s: their arrays (`Xi`, a
length scale, a rate, the noise `R`) are pytree leaves, their orders,
dimensions and vector fields are static.

- `jax.jit(..., static_argnums=...)` on a prior or model fails (modules with
  array leaves are not hashable). Use `equinox.filter_jit`, which accepts any
  argument; likewise `equinox.filter_vmap` / `equinox.filter_grad` to batch or
  differentiate over a model.
- Modules are immutable: `measure.R = R` becomes `measure = measure.with_noise(R)`.
  Constructors take the noise directly: `ODEInformation(..., R=...)`,
  `BlackBoxMeasurement(..., noise=...)`.

### Priors

| 0.7 | now |
| --- | --- |
| subclass implements `A(h)` and `Q(h)` (with a dense-Cholesky `Q_sqr` fallback) | subclass implements `A(h)` and `Q_sqr(h)`; `Q(h)` is derived; `b(h)` defaults to zero |
| private `_dim`, `_E0`, `_D_x` | `d`, `state_dim`, `E0` / `E1` / `E2`, `JointPrior.prior_x.state_dim` |
| `MaternPrior._F`, `_L`, `_q`, `S`, `n`, `A_and_Q` | removed; `length_scale` is a field, `stationary_cov_sqr()` gives the stationary law |
| `IOUPPrior._F`, `_G` | `_drift_and_dispersion()`; the rate is the field `rate` |
| `JointPrior._prior_x`, `_prior_u` | `prior_x`, `prior_u` |
| a subclass calls `BasePrior.__init__(q, d, Xi)` | a subclass is an equinox module: it declares the fields `q`, `d` (static) and `xi`, sets them in its `__init__`, and implements `A(h)` and `Q_sqr(h)`; the filters call `transition(h)`, which `BasePrior` provides. Optionally set the static field `xi_diagonal` (see `IWP`), which lets a diagonal calibration policy check `Xi` under `jit` |

### Information models

| 0.7 | now |
| --- | --- |
| `ODEconservation(...)`, `SecondOrderODEconservation(...)` | `ODEInformation(vf, E0, E1, constraints=[Conservation(A, p)])` (and the second-order class with `E2`) |
| `measure.get_noise(t=t)` | `measure.noise_sqr(t=t)` |
| `measure.E_constraint` | `measure.E_lhs` |
| `Conservation.jacobian()` | `Conservation.A` |
| `Measurement.get_noise_matrix()`, the attribute `Measurement.noise` | `Measurement.R` (the constructor still takes `noise=`) |
| `Measurement.find_index`, `.residual`, `.jacobian` | removed; `prepare_observations` places the data |
| `TransformedMeasurement(base_model, sigma, use_autodiff_jacobian=..., sigma_jacobian=...)` | `TransformedMeasurement(base, sigma, sigma_jacobian=None)`: the keyword is `base`, autodiff when no Jacobian is given, and `base` is a public field |

A custom information model implements the `MeasurementModel` protocol
(`ode_filters.information.MeasurementModel`).

## Linearization

The `Correction` interface splits linearizing from conditioning.

| 0.7 | now |
| --- | --- |
| `correction.correct(measure, m_pred, P_pred_sqr, t=t)` returning a `CorrectionResult` | `correction.linearize(measure, m, t=t)` returning an `AffineModel`, and `correction.correct(measure, x, model, t=t)` with `x` a `Gaussian`, returning `(posterior, innovation)` |
| `CorrectionResult` | removed |
| `slr_linearize(..., t=0.0)`, `check_arg_projection(..., t=0.0)` | `t` is required (no default) |

## Gaussian algebra

| 0.7 | now |
| --- | --- |
| `sqr_marginalization(A, b, Q_sqr, m, P_sqr)` | `marginalize(A, b, Q_sqr, Gaussian(m, P_sqr))` |
| `sqr_inversion(...)` | `invert(A, Q_sqr, x, z)`, returning a `Conditional` |
| `compose_backward_conditionals(...)` | `compose(outer, inner)` |
| top-level `sqr_marginalization`, `sqr_inversion` | removed; import from `ode_filters.gaussian` |

New: the types `Gaussian`, `Conditional` and `AffineModel`, plus `update`
and `log_likelihood`.

## Data

- `prepare_observations` raises `ValueError` when a measurement time is not a
  grid node; 0.7 dropped such data without a warning. Put the measurement times
  into the grid with `merge_measurement_times(ts, measurements)`.
- Measurements at different times work: each row of the data has its own mask.

## Parameter inference

| 0.7 | now |
| --- | --- |
| `InferenceProblem(build, prior, tspan=..., N=..., min_sigma_sqr=...)` | `InferenceProblem(build, prior, ts=..., calibration=policy)` |
| `ODEFilter(..., tspan=..., N=...)` | `ODEFilter(..., ts=...)` (a tuple, so the model stays hashable) |
| `fit` trains every array leaf of the `ODEFilter` | `fit` trains `ode_params` only |
| `quasi_mle_sigma_sqr(m_z, P_z_sqr)`, `quasi_mle_sigma_sqr_from_Q(m_z, H, Q_sqr)` | the first argument is `mz` (and `Pz_sqr`), as everywhere else |

## Results that change

A valid 0.7 call that runs unchanged can still return slightly different
numbers in these cases:

- **EK0** (`TaylorCorrection(order=0)`) calibrates with its own linearization
  instead of the EK1 Jacobian, and evaluates no Jacobian of the vector field.
  `sigma_sqr` and dynamically calibrated results change slightly.
- **Data innovations** (`mz_obs`, `Pz_obs_sqr`) at rows without data are zero
  with unit variance; they were the residual of a zero observation.
- **`"diagonal_ekf0"` on second-order ODEs** used the wrong derivative and
  inflated `sigma_sqr` by orders of magnitude; the means were right.
- **Step-size controllers**: `PController.propose` / `PIController.propose`
  return a 0-d JAX array, and a custom controller passed to
  `gaussian_filter_adaptive` must be traceable.
- **`MaternPrior(length_scale=l)`** (and `PrecondMaternPrior`) returns
  different numbers for every `l != 1`: it uses the textbook length scale. The
  correlation of the process is the Matern kernel in `sqrt(2 nu) tau / l`
  (Rasmussen & Williams, *Gaussian Processes for Machine Learning*, 2006, Sec.
  4.2.1, eq. 4.14), and the rate of its SDE is
  `lambda = sqrt(2 nu) / l = sqrt(2q+1) / l` (Särkkä, Solin & Hartikainen,
  IEEE Signal Processing Magazine 30(4), 2013, Example 1). 0.7 used
  `lambda = sqrt((2q+1) / l)`, so its `length_scale` acted as the square of
  the usual one: an old `l_old` corresponds to `length_scale=sqrt(l_old)`.
  `length_scale=1` is unchanged.
- **Preconditioned priors** run in the original coordinates with a change of
  variables per step. Solutions agree with 0.7 to rounding. At order `q >= 8`
  on fine grids, rounding dominates the ODE-information log-likelihood in both
  versions, and the two can differ there.
