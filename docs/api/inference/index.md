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

## Parameters

::: ode_filters.inference.parameters.Real

::: ode_filters.inference.parameters.PositiveReal

::: ode_filters.inference.parameters.AbstractParameter

::: ode_filters.inference.parameters.unwrap
