# Calibration

The diffusion `sigma^2` sets the *size* of the posterior uncertainty. The solvers
calibrate it online with the policy passed as `calibration=` (`"dynamic"` by
default, see [Diffusion calibration](../../calibration.md)). The estimators below
work on a finished run: estimate `sigma^2` from the innovations a result carries
(`result.mz`, `result.Pz_sqr`) and rescale its stored covariances.

## Online policies

::: ode_filters.calibration.NoCalibration

::: ode_filters.calibration.DynamicCalibration

::: ode_filters.calibration.DiagonalCalibration

::: ode_filters.calibration.Calibration

## Post-hoc estimators

::: ode_filters.calibration.sigma.posthoc_mle_sigma_sqr

::: ode_filters.calibration.sigma.quasi_mle_sigma_sqr

::: ode_filters.calibration.sigma.quasi_mle_sigma_sqr_from_Q

::: ode_filters.calibration.sigma.aggregate_sigma_sqr

## Rescaling

::: ode_filters.calibration.rescale.rescale_sqr

::: ode_filters.calibration.rescale.rescale_sqr_seq
