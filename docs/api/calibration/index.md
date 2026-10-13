# Calibration

The diffusion `sigma^2` sets the *size* of the posterior uncertainty: the filter's
mean does not depend on it, its covariance scales with it. The solvers calibrate it
online through their `calibration=` argument (`"dynamic"` by default, see
[Diffusion calibration](../../calibration.md)). The functions here are for working
with a finished run: estimate `sigma^2` from the innovations a result carries
(`result.mz`, `result.Pz_sqr`) and rescale its stored covariances.

## Estimators

::: ode_filters.calibration.sigma.posthoc_mle_sigma_sqr

::: ode_filters.calibration.sigma.quasi_mle_sigma_sqr

::: ode_filters.calibration.sigma.quasi_mle_sigma_sqr_from_Q

::: ode_filters.calibration.sigma.aggregate_sigma_sqr

## Rescaling

::: ode_filters.calibration.rescale.rescale_sqr

::: ode_filters.calibration.rescale.rescale_sqr_seq
