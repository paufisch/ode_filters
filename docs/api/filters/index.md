# Filters

The solvers all return a `FilterResult` in the original state coordinates:

- `gaussian_filter` — a given time grid (uniform or not), the ODE information
  linearized at each prediction by a [`Correction`](../../corrections.md).
- `gaussian_filter_adaptive` — adaptive steps between given save times.
- `affine_filter` — a frozen linearization (an `AffineSurrogate`).
- `rts_smoother` — smoothing of any result that carries backward conditionals.
- `iterated_smoother` — the IEKS and the IPLS: linearize along the smoothed
  trajectory, filter and smooth, repeat.

Every step calibrates the diffusion with the `calibration=` policy (`"dynamic"`
by default; see [Diffusion calibration](../../calibration.md)) and conditions on
the data of an `obs_model`, if any, after the ODE information. A preconditioned
prior changes how a step is computed, not its result. See
[How to choose](../../how-to-choose.md).

## The `FilterResult`

Recover a covariance with `P = P_sqr.T @ P_sqr` (see [notation](../../notation.md)).
On a grid of `K` nodes, per-step fields have `K - 1` entries; entry `n` belongs to
the step from `t[n]` to `t[n + 1]`.

| Field | Meaning |
| --- | --- |
| `t` | time grid (the save times on the adaptive path) |
| `m`, `P_sqr` | **filtered** mean and square-root covariance |
| `log_likelihood` | ODE-information evidence, after calibration (not comparable across calibration policies) |
| `log_likelihood_obs` | data evidence, each observation conditioned on the ODE information up to its time (`None` without data) |
| `sigma_sqr` | per-step diffusion estimate (`None` on the adaptive path) |
| `m_pred`, `P_pred_sqr` | predictions before each update (`None` on the adaptive path) |
| `G_back`, `d_back`, `P_back_sqr` | backward conditionals `p(x_n \| x_{n+1})`, consumed by `rts_smoother` |
| `mz`, `Pz_sqr` | ODE-information innovations |
| `mz_obs`, `Pz_obs_sqr` | data innovations; rows without data are zero with unit variance |
| `success` | adaptive: every save time reached with a finite evidence |
| `n_accepted`, `n_rejected` | adaptive: steps per save interval |
| `steps` | adaptive with `record_steps > 0`: every accepted step (a `StepRecord`) |

`result.filtered` and `result.backward` return the marginals and the backward
conditionals as a stacked `Gaussian` and `Conditional`.

`gaussian_filter_adaptive` is **filtering-only by default** (`smoother=False`):
its backward fields are then `None` and `rts_smoother` raises. With
`smoother=True` it composes one backward conditional per save interval
(`O(#save points)` memory), with or without data.

## Solvers

::: ode_filters.filters.gaussian_filter

::: ode_filters.filters.gaussian_filter_adaptive

::: ode_filters.filters.affine_filter

::: ode_filters.filters.rts_smoother

::: ode_filters.filters.FilterResult

::: ode_filters.filters.StepRecord

## Iterated smoothing

See [Linearization schemes](../../corrections.md#iterated-smoothing-iekf-ieks-and-ipls).

::: ode_filters.filters.iterated_smoother

::: ode_filters.filters.IteratedSmootherResult

::: ode_filters.filters.IterationState

## The filter step

The step every solver is built from, for building new drivers.

::: ode_filters.filters.kernel.filter_step

::: ode_filters.filters.kernel.StepOutput

::: ode_filters.filters.kernel.predict

::: ode_filters.filters.kernel.observe

## Step-size control

See [Adaptive step-size control](../../adaptive-steps.ipynb).

::: ode_filters.filters.StepSizeController

::: ode_filters.filters.PController

::: ode_filters.filters.PIController
