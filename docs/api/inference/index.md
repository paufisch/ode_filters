# Inference

The inference layer fits ODE parameters by maximizing the marginal
log-likelihood of external observations under the ODE-constrained Gauss-Markov
model. It consists of `ODEFilter`, `fit`, `fit_scipy`, `marginal_loglik` and
`InferenceProblem`, and of the parameter wrappers `Real`, `PositiveReal` and
`NonNegativeReal`. `fit` minimizes the negative log-likelihood with an Optax
optimizer supplied by the caller. `fit_scipy` minimizes it with SciPy's
quasi-Newton methods (L-BFGS-B by default), which on low-dimensional problems
usually need fewer evaluations. The diagnostics check a fitted model against its
data.

`marginal_loglik` takes the data as a list of `Measurement` objects, which it
converts with `prepare_observations(data, prior.E0_state, ts)`, or as an
`ObsModel` for `InferenceProblem.ts`. It solves on the fixed grid
`ts` by default. Data between the grid nodes become data-only nodes, where the
filter conditions on the data without an update on the ODE information, and
every channel works with them. With
`InferenceProblem(..., adaptive=AdaptiveSettings(atol=..., rtol=...))` it runs
`gaussian_filter_adaptive` with `ts` as the save grid, which lands on the data
times as well. The channels `"obs"`, `"ode"`, `"both"` and `"fenrir"` are
available there, and the result is differentiable in the parameters.

`fenrir_loglik` takes the data as an `ObsModel` from `prepare_observations`
and a pass without data, and `conditional_loglik` takes a pass with and one
without data. With data-only nodes the pass without data must run on the same
timeline as the data: pass `obs_model=obs_model.without_data()` to the solver,
which keeps the timeline and drops every row. `information_criteria` takes the
data like `marginal_loglik` and runs its passes with the data over the timeline
as well.

## Parameter inference

::: ode_filters.inference.model.ODEFilter

::: ode_filters.inference.model.fit

::: ode_filters.inference.model.fit_scipy

::: ode_filters.inference.likelihood.marginal_loglik

::: ode_filters.inference.likelihood.InferenceProblem

::: ode_filters.inference.likelihood.AdaptiveSettings

## Conditional evidence and information criteria

See [Selecting hyperparameters from data](../../calibration.md#selecting-hyperparameters-from-data).

::: ode_filters.inference.fenrir_loglik

::: ode_filters.inference.conditional_loglik

::: ode_filters.inference.information_criteria

::: ode_filters.inference.InformationCriteria

## Diagnostics and predictives

Under a consistent model the whitened data innovations of a filter pass
(`result.mz_obs`, `result.Pz_obs_sqr`) are independent standard normal.

::: ode_filters.inference.whitened_residuals

::: ode_filters.inference.nis

::: ode_filters.inference.residual_autocorrelation

::: ode_filters.inference.observation_predictive

## Parameters

::: ode_filters.inference.parameters.Real

::: ode_filters.inference.parameters.PositiveReal

::: ode_filters.inference.parameters.NonNegativeReal

::: ode_filters.inference.parameters.AbstractParameter

::: ode_filters.inference.parameters.unwrap
