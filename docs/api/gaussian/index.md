# Gaussian algebra

The QR-based numerical core every filter step is built on. Covariances are
propagated as upper-triangular factors `P_sqr` (with `P = P_sqr.T @ P_sqr`), never
as dense matrices, which keeps them symmetric and positive-*semi*-definite (no
indefiniteness from round-off; see [notation](../../notation.md)).

## Types

::: ode_filters.gaussian.Gaussian

::: ode_filters.gaussian.Conditional

::: ode_filters.gaussian.AffineModel

## Algebra

::: ode_filters.gaussian.algebra.sqr_marginalization

::: ode_filters.gaussian.algebra.sqr_inversion

::: ode_filters.gaussian.algebra.compose_backward_conditionals
