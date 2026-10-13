# Diffusion Calibration

Probabilistic ODE filters quantify uncertainty by propagating a Gaussian
covariance. The *magnitude* of that covariance is governed by a scalar
diffusion `sigma^2` on top of the prior's structural scale matrix `Xi`. The
filter is correct in the mean even with `sigma = 1`, but the posterior
uncertainty is scaled to match the residual magnitude only when `sigma^2` is
calibrated from the observed ODE residuals.

There are two ways to calibrate. **Online**, each filter step estimates
`sigma^2` from the ODE residual at its prediction and propagates with it; the
policy is the solvers' `calibration=` argument:

| Policy | Shorthand | Each step |
| --- | --- | --- |
| `NoCalibration()` | `"none"` | propagates the prior's noise unscaled; still reports the estimate |
| `DynamicCalibration(min_sigma_sqr=0.0)` | `"dynamic"` (default) | scales the step's noise by the scalar quasi-MLE |
| `DiagonalCalibration(jacobian="ek1", min_sigma_sqr=0.0)` | `"diagonal"` | one estimate per solution component |
| `DiagonalCalibration(jacobian="ek0")` | `"diagonal_ekf0"` | the same, with the ODE's left-hand-side selector in the denominator |

`min_sigma_sqr` floors the estimate, so a residual that vanishes identically (a
system at rest) cannot collapse the diffusion to zero. **Post hoc**, the
estimators below consume the innovations `(result.mz, result.Pz_sqr)` a finished
run carries, and rescale its covariances.

## Workflows

### Post-hoc MLE on a fixed-step run

Cheapest workflow: run any existing fixed-step filter, then rescale.
`posthoc_mle_sigma_sqr` is the joint MLE of `sigma^2` under the assumption
that one constant scale explains every step's whitened residual -- the same
Gaussian likelihood the per-step estimator maximises, but with a single
parameter rather than one per step. The closed form makes
`posthoc_mle_sigma_sqr` numerically identical to `aggregate_sigma_sqr(seq,
kind="mean")` applied to the per-step trace.

```python
import jax.numpy as np
from ode_filters import gaussian_filter
from ode_filters.calibration import posthoc_mle_sigma_sqr, rescale_sqr_seq
from ode_filters.information import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization


def vf(x, *, t):
    return x * (1 - x)


prior = IWP(q=2, d=1)
m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.1]), q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)
ts = np.linspace(0.0, 5.0, 51)

# Run an uncalibrated fixed-step filter.
result = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration="none")

sigma_sqr_hat = posthoc_mle_sigma_sqr(result.mz, result.Pz_sqr)

# Rescale stored covariances (returns a new array; result.P_sqr is unchanged).
P_seq_sqr_calibrated = rescale_sqr_seq(result.P_sqr, sigma_sqr_hat)
```

This is appropriate for smooth problems where a single global scalar captures
the residual size well.

!!! warning "Joint / latent-force priors: pass `prior=`"

    For a `JointPrior`, the diffusion `sigma^2` belongs to
    the ODE-**state** block only -- the hidden/input block must not be rescaled
    (Schmidt et al. 2021). The default `rescale_sqr` / `rescale_sqr_seq` scale
    the *whole* covariance, which is wrong for these priors. Pass the prior so
    the rescaling is routed through its block-aware
    `apply_state_sigma_to_cov_sqr`:

    ```python
    P_seq_sqr_calibrated = rescale_sqr_seq(result.P_sqr, sigma_sqr_hat, prior=prior)
    ```

    For non-joint priors (`IWP`, `MaternPrior`, ...) passing `prior=` is
    optional and gives the same result as the default whole-covariance scaling.
    The online policies are already block-aware, so this only matters for the
    post-hoc workflow above.

### Per-step quasi-MLE inside the adaptive solver

`gaussian_filter_adaptive` calibrates every attempted step the same way. With
`record_steps=` it also keeps each accepted step's estimate (see
[Adaptive Step-Size Control](adaptive-steps.ipynb)):

```python
from ode_filters import gaussian_filter_adaptive

result = gaussian_filter_adaptive(
    m_0, P_0_sqr, prior, measure, np.linspace(0.0, 5.0, 50),
    atol=1e-5, rtol=1e-3, record_steps=1000,
)
n = int(result.steps.count)
result.steps.sigma_sqr[:n]  # one estimate per accepted step
```

Pass `calibration="none"` to record the estimates without scaling the stored
covariances (diagnostics only).

### Manual per-step estimation

If you want the per-step number outside the adaptive solver (e.g., for
diagnostics on an existing fixed-step run):

```python
from ode_filters.calibration import quasi_mle_sigma_sqr

sigma_sqr_per_step = [
    float(quasi_mle_sigma_sqr(mz, Pz_sqr))
    for mz, Pz_sqr in zip(result.mz, result.Pz_sqr, strict=True)
]
```

The estimator is a one-line `solve_triangular` on the existing square-root
covariance.

## Aggregation

A per-step trace can be turned into a single number in three ways:

```python
from ode_filters.calibration import aggregate_sigma_sqr

mean_sigma = aggregate_sigma_sqr(seq, kind="mean")       # joint-MLE equivalent
last_sigma = aggregate_sigma_sqr(seq, kind="last")       # use the final estimate
running    = aggregate_sigma_sqr(seq, kind="running")    # cumulative mean trace
```

`"running"` returns an array of the same length as `seq` -- useful when you
want to display the calibrated uncertainty intermediate during integration
without waiting for the end.

## Note on measurement noise

The quasi-MLE assumes the predicted-observation covariance is proportional to
`sigma^2`. When the EKF measurement update adds a non-zero `R` (e.g. for
genuine observations under data assimilation), the full output covariance is
`H P_pred H.T + R` and feeding that into the quasi-MLE conflates the
diffusion scale with the measurement noise (it biases `sigma_hat^2` downward).

The online policies handle this transparently: they estimate from the
*noise-free* residual covariance `H Q(h) H.T`, matching what probnum and
probdiffeq do. With the default `ODEInformation` (`R = 0`) the distinction is
moot.

For the post-hoc workflow on a fixed-step run with non-zero `R`, the caller
must pass a noise-free `Pz_sqr` to `posthoc_mle_sigma_sqr` -- the
covariances returned by the filter include `R` and must be recomputed from the
saved `(result.m_pred, result.P_pred_sqr)` first.

## What calibration cannot fix

Calibration scales the posterior to match the size of the residuals. It does
not correct a prior with the wrong smoothness (`q` too low), a vector field
that is miscoded, or a Jacobian that the EKF linearisation cannot
approximate. To check that the calibrated filter is honest, plot the whitened
residuals: for each step the standardized innovation `z_n = P_z_sqr^{-T} m_z`
should look like standard normal noise, so `||z_n||^2 / d` hovers near 1 on a
well-specified problem (much larger means the uncertainty is too tight; much
smaller means it is too loose).

## See also

- [Adaptive Step-Size Control](adaptive-steps.ipynb) -- the controller that uses
  the per-step quasi-MLE.
- API reference for `ode_filters.calibration`.
