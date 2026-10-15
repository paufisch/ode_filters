# Diffusion calibration

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
that one constant scale explains every step's whitened residual — the same
Gaussian likelihood the per-step estimator maximises, but with a single
parameter rather than one per step. The closed form makes
`posthoc_mle_sigma_sqr` numerically identical to the mean of the per-step
terms `quasi_mle_sigma_sqr(mz, Pz_sqr)` (see
[below](#per-step-estimates-on-a-fixed-grid)).

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
    the ODE-**state** block only — the hidden/input block must not be rescaled
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

### Per-step estimates on a fixed grid

Two per-step traces measure different things:

- `result.sigma_sqr` is what each step's policy estimated: the local quasi-MLE
  `r.T (H Q H.T)^-1 r / d` of the residual `r` against the step's own process
  noise `Q` (Schober et al. 2019; Bosch et al. 2021). It is reported under every
  policy, also with `calibration="none"`, where it is not applied
  (`result.diffusion` is what each step propagated with).
- The global terms weigh the innovation `result.mz` against its full predicted
  covariance `result.Pz_sqr`. Their mean is the post-hoc MLE above:

```python
import jax
from ode_filters.calibration import quasi_mle_sigma_sqr

seq = jax.vmap(quasi_mle_sigma_sqr)(result.mz, result.Pz_sqr)
```

## Aggregation

A per-step trace can be turned into a single number in three ways:

```python
from ode_filters.calibration import aggregate_sigma_sqr

mean_sigma = aggregate_sigma_sqr(seq, kind="mean")       # == posthoc_mle_sigma_sqr
last_sigma = aggregate_sigma_sqr(seq, kind="last")       # use the final estimate
running    = aggregate_sigma_sqr(seq, kind="running")    # cumulative mean trace
```

`"running"` returns an array of the same length as `seq` — useful when you
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
must pass a noise-free `Pz_sqr` to `posthoc_mle_sigma_sqr` — the
covariances returned by the filter include `R` and must be recomputed from the
saved `(result.m_pred, result.P_pred_sqr)` first.

## Selecting hyperparameters from data

The diffusion is one hyperparameter; a prior's length scale, `Xi`, or the
parameters of a hidden input are others. With data, they are selected from the
evidence the data carry, and the library reports three versions of it. Write
`Z` for the ODE information (observed as zero), `Y` for the data and `U` for a
hidden input's trajectory.

**The sequential data evidence**, `result.log_likelihood_obs`
(`marginal_loglik(..., channel="obs")`), is `sum_n log p(y_n | z_<=n, y_<n)`:
each observation conditioned on the ODE information up to its own time. Together
with the ODE-information evidence `result.log_likelihood` it sums to
`log p(Z, Y)`.

**The conditional evidence** `log p(Y | Z) = log p(Z, Y) - log p(Z)` conditions
each observation on *all* of the ODE information, also the later one. It differs
from the sequential one whenever later ODE information says something about
earlier data, which is the usual case. Two estimators:

- `fenrir_loglik(data_free, obs_model)` (`channel="fenrir"`): one pass on the
  ODE information alone, then a Kalman filter backwards through its backward
  conditionals that updates on the data (Tronarp, Bosch & Hennig, 2022).
- `conditional_loglik(data, data_free)` (`channel="dalton"`): a pass with data
  minus a pass without. On one frozen linearization (two `affine_filter` runs
  of the same `AffineSurrogate`) it equals Fenrir's value exactly; with one-shot
  filters each pass linearizes at its own predictions, and it is Dalton's
  estimator (Wu & Lysy, 2024), which generally differs.

Use the conditional evidence to compare hyperparameters by how well the
ODE-constrained model explains the data; use the sequential one when the data
should be judged as they arrive (filtering, online use).

**Information criteria.** `information_criteria(m_0, P_0_sqr, prior, surrogate,
ts, obs_model)` returns, at a frozen linearization and for a `JointPrior` with a
hidden input `U`,

- `I_UZ = I(U; Z)`: what the ODE information says about the input;
- `I_UY_given_Z = I(U; Y | Z)`: what the data add to it;
- `D = sum_n I(U; y_n | z_<=n, y_<n)`: the directed information of the data
  about the input, as the filter sees it.

They come from the innovation log-determinants of four filter passes on the
frozen model (the joint model and the state block given `U`, each with and
without the data), so they are exact for it and differentiable in its
hyperparameters. The means drop out, so the passes carry only the square-root
covariances.

## What calibration cannot fix

Calibration scales the posterior to match the size of the residuals. It does
not correct a prior with the wrong smoothness (`q` too low), a vector field
that is miscoded, or a Jacobian that the EKF linearisation cannot
approximate. Since calibration fits the scale of the ODE residuals, their
average size matches by construction (exactly so for the post-hoc MLE); what
stays informative is their *structure*. The whitened residuals
`whitened_residuals(result.mz, result.Pz_sqr)` of an honest filter look like
white noise: a trend over time in `nis(result.mz, result.Pz_sqr)`, or a
`residual_autocorrelation` far from zero, points at a misspecified model that
no scale fixes. Against data with a known noise level (`result.mz_obs`,
`result.Pz_obs_sqr`) the NIS is a test of the scale too: under a consistent
model it averages the number of observed rows per step.

## See also

- [Adaptive Step-Size Control](adaptive-steps.ipynb) — the controller that uses
  the per-step quasi-MLE.
- API reference for `ode_filters.calibration`.
