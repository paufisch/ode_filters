# Inference

The **square-root Gaussian algebra** (`sqr_marginalization`, `sqr_inversion`) is the
QR-based numerical core every filter step is built on: it propagates covariances as
upper-triangular factors `A_sqr` (with `A = A_sqr.T @ A_sqr`), never forming a dense
covariance and keeping it positive-definite (see [notation](../../notation.md)).

::: ode_filters.inference
