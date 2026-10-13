# Inference

A parameter-inference layer — `ODEFilter`, `fit`, `marginal_loglik`, and
`InferenceProblem`, plus the `Real` / `PositiveReal` parameter wrappers — for fitting ODE parameters by
maximizing the marginal log-likelihood of external observations under the
ODE-constrained Gauss-Markov model. (`fit` minimizes the negative log-likelihood
via a caller-supplied Optax optimizer.)

## Parameter inference

::: ode_filters.inference.model.ODEFilter

::: ode_filters.inference.model.fit

::: ode_filters.inference.likelihood.marginal_loglik

::: ode_filters.inference.likelihood.InferenceProblem

## Conditional evidence and information criteria

See [Selecting hyperparameters from data](../../calibration.md#selecting-hyperparameters-from-data).

::: ode_filters.inference.fenrir_loglik

::: ode_filters.inference.conditional_loglik

::: ode_filters.inference.information_criteria

::: ode_filters.inference.InformationCriteria

## Parameters

::: ode_filters.inference.parameters.Real

::: ode_filters.inference.parameters.PositiveReal

::: ode_filters.inference.parameters.AbstractParameter

::: ode_filters.inference.parameters.unwrap
