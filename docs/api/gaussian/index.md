# Gaussian algebra

The QR-based numerical core every filter step is built on. Covariances are
propagated as upper-triangular factors `P_sqr` (with `P = P_sqr.T @ P_sqr`), never
as dense matrices, which keeps them symmetric and positive-*semi*-definite (no
indefiniteness from round-off; see [notation](../../notation.md)).

A filter step is `marginalize` (predict) and `invert` (the backward conditional),
then `update` (condition on the information, observed as zero), with
`log_likelihood` of the innovation as its evidence. The adaptive smoother uses
`compose`.

## Types

::: ode_filters.gaussian.Gaussian

::: ode_filters.gaussian.Conditional

::: ode_filters.gaussian.AffineModel

## Operations

::: ode_filters.gaussian.marginalize

::: ode_filters.gaussian.invert

::: ode_filters.gaussian.update

::: ode_filters.gaussian.compose

::: ode_filters.gaussian.log_likelihood
