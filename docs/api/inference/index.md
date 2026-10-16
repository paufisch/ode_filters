# Inference

A parameter-inference layer — `ODEFilter`, `fit`, `fit_scipy`, `marginal_loglik`,
and `InferenceProblem`, plus the `Real` / `PositiveReal` / `NonNegativeReal`
parameter wrappers — for fitting ODE parameters by maximizing the marginal
log-likelihood of external observations under the ODE-constrained Gauss-Markov
model. `fit` minimizes the negative log-likelihood with a caller-supplied Optax
optimizer; `fit_scipy` with SciPy's quasi-Newton methods (L-BFGS-B by default),
which usually need far fewer evaluations on low-dimensional problems. The
diagnostics check a fitted model against its data.

`marginal_loglik` solves on the fixed grid `InferenceProblem.ts` by default.
With `InferenceProblem(..., adaptive=AdaptiveSettings(atol=..., rtol=...))` it
runs `gaussian_filter_adaptive` with `ts` as the save grid, which carries the
data. The channels `"obs"`, `"ode"`, `"both"` and `"fenrir"` are available
there, and the result is differentiable in the parameters.

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
