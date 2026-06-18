# Inference

Two things live here. The **square-root Gaussian algebra** (`sqr_marginalization`,
`sqr_inversion`) is the QR-based numerical core every filter step is built on: it
propagates covariances as upper-triangular factors `A_sqr` (with
`A = A_sqr.T @ A_sqr`), never forming a dense covariance and keeping it
positive-definite (see [notation](../../notation.md)).

The **parameter-inference layer** sits on top: the pure functional core
`marginal_loglik(theta, data, *, model)` (params-as-argument) and the ergonomic
`ODEFilter` + `fit` object API (params-in-model). See
[Parameter estimation](../../parameter-estimation.md).

::: ode_filters.inference
