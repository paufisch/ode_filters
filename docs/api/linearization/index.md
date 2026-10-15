# Linearization

How the filters turn a nonlinear information operator into the affine model each
update conditions on: a Taylor expansion at a point (EK0 / EK1, iterated for the
IEKF) or a statistical linear regression over a Gaussian (SLR, iterated for the
IPLF). See [Linearization schemes](../../corrections.md) for when to use which.

## Corrections

::: ode_filters.linearization.correction.Correction

::: ode_filters.linearization.correction.TaylorCorrection

::: ode_filters.linearization.correction.IteratedTaylorCorrection

::: ode_filters.linearization.correction.QuadratureCorrection

::: ode_filters.linearization.taylor_linearize

## Frozen linearizations

One affine model per filter step, for `affine_filter` and `iterated_smoother`.

::: ode_filters.linearization.AffineSurrogate

::: ode_filters.linearization.taylor_surrogate

::: ode_filters.linearization.slr_surrogate

## Statistical linearization

The primitive behind `QuadratureCorrection` and `slr_surrogate`, and a check for
the argument selectors of a custom measurement model. The quadrature rule is
chosen by name (`rule="gauss_hermite"` or `"cubature"`, with `n_nodes`).

::: ode_filters.linearization.slr_linearize

::: ode_filters.linearization.SLRModel

::: ode_filters.linearization.check_arg_projection
