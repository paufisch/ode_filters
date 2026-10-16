# Diffusion calibration

A probabilistic ODE filter quantifies its uncertainty with a Gaussian
covariance. The scale of this covariance is set by a scalar diffusion
`sigma^2`, which multiplies the prior's scale matrix `Xi`. With exact ODE
information (`R = 0`) and an exact initial state, a constant `sigma^2` scales
the posterior covariance and leaves the mean unchanged, so the filter computes
the same mean with `sigma = 1`. The covariance matches the size of the
residuals only when `sigma^2` is calibrated from the observed ODE residuals.

There are two ways to calibrate. In **online** calibration, each filter step
estimates `sigma^2` from the ODE residual at its prediction and propagates with
this estimate. The solvers' `calibration=` argument selects the policy:

| Policy | Shorthand | Each step |
| --- | --- | --- |
| `NoCalibration()` | `"none"` | propagates the prior's noise unscaled; still reports the estimate |
| `DynamicCalibration(min_sigma_sqr=0.0)` | `"dynamic"` (default) | scales the step's noise by the scalar quasi-MLE |
| `DiagonalCalibration(jacobian="ek1", min_sigma_sqr=0.0)` | `"diagonal"` | one estimate per solution component |
| `DiagonalCalibration(jacobian="ek0")` | `"diagonal_ekf0"` | deprecated, removed in 0.9: the same with the ODE's left-hand-side selector in the denominator; use `"diagonal"` |

`min_sigma_sqr` is a lower bound on the estimate, so a residual that vanishes
identically (a system at rest) does not set the diffusion to zero. In
**post-hoc** calibration, the estimators below take the innovations
`(result.mz, result.Pz_sqr)` of a finished run and rescale its covariances.

## Workflows

### Post-hoc MLE on a fixed-step run

This workflow needs no further filter pass: run a fixed-step filter, then
rescale its covariances. `posthoc_mle_sigma_sqr` is the joint MLE of `sigma^2`
under the assumption that one constant scale explains the whitened residuals of
all steps. It maximizes the same Gaussian likelihood as the per-step estimator,
with one parameter instead of one per step. In closed form,
`posthoc_mle_sigma_sqr` is numerically identical to the mean of the per-step
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

The workflow suits smooth problems, in which one constant scale describes the
size of the residuals on the whole grid.

!!! warning "Joint / latent-force priors: pass `prior=`"

    For a `JointPrior`, the diffusion `sigma^2` belongs to the block of the
    ODE state only, and the hidden or input block must not be rescaled
    (Schmidt et al. 2021). By default, `rescale_sqr` and `rescale_sqr_seq`
    scale the whole covariance, which is wrong for these priors. Pass the
    prior, so that the rescaling goes through its block-aware
    `apply_state_sigma_to_cov_sqr`:

    ```python
    P_seq_sqr_calibrated = rescale_sqr_seq(result.P_sqr, sigma_sqr_hat, prior=prior)
    ```

    For the other priors (`IWP`, `MaternPrior`, ...), `prior=` is optional and
    gives the same result as the default scaling of the whole covariance. The
    online policies are block-aware, so the argument concerns only the post-hoc
    workflow above.

### Per-step quasi-MLE inside the adaptive solver

`gaussian_filter_adaptive` calibrates every attempted step in the same way.
With `record_steps=` it also stores the estimate of each accepted step (see
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
covariances, for diagnostics.

### Per-step estimates on a fixed grid

Two per-step traces measure different quantities:

- `result.sigma_sqr` holds the estimate of each step's policy, the local
  quasi-MLE `r.T (H Q H.T)^-1 r / d` of the residual `r` against the step's own
  process noise `Q` (Schober et al. 2019; Bosch et al. 2021). It is reported
  under every policy, including `calibration="none"`, which does not apply it
  (`result.diffusion` holds the diffusion each step propagated with).
- The global terms weigh the innovation `result.mz` against its full predicted
  covariance `result.Pz_sqr`. Their mean is the post-hoc MLE above:

```python
import jax
from ode_filters.calibration import quasi_mle_sigma_sqr

seq = jax.vmap(quasi_mle_sigma_sqr)(result.mz, result.Pz_sqr)
```

## Aggregation

`aggregate_sigma_sqr` reduces a per-step trace in three ways:

```python
from ode_filters.calibration import aggregate_sigma_sqr

mean_sigma = aggregate_sigma_sqr(seq, kind="mean")       # == posthoc_mle_sigma_sqr
last_sigma = aggregate_sigma_sqr(seq, kind="last")       # use the final estimate
running    = aggregate_sigma_sqr(seq, kind="running")    # cumulative mean trace
```

`"running"` returns an array of the same length as `seq`, whose entry `n` is
the mean of the first `n + 1` estimates. It provides a calibrated uncertainty
at every step of the integration, before the run has ended.

## Note on measurement noise

The quasi-MLE assumes that the predicted-observation covariance is
proportional to `sigma^2`. When the EKF measurement update adds a nonzero `R`
(e.g. for observations of data), the full output covariance is
`H P_pred H.T + R`. In the quasi-MLE this covariance mixes the diffusion scale
with the measurement noise and biases `sigma_hat^2` downward.

The online policies estimate from the noise-free residual covariance
`H Q(h) H.T`, as probnum and probdiffeq do, so `R` does not enter the estimate.
With the default `ODEInformation` (`R = 0`), the covariance contains no
measurement noise.

For the post-hoc workflow on a fixed-step run with nonzero `R`, pass a
noise-free `Pz_sqr` to `posthoc_mle_sigma_sqr`. The covariances returned by
the filter include `R`, so recompute them from the saved
`(result.m_pred, result.P_pred_sqr)` first.

## The smoother's uncertainty

Every policy above estimates the diffusion from the filter's innovations, so
it calibrates the filter, and the smoother inherits the result. We measured
the consequences on four problems with known solutions (logistic growth, a
cubic decay, a linear oscillator in 2D and the Hopf normal form), with EK1,
`IWP(q)` and the step `h = 0.1` on `[0, 10]`. The ranges below span the four
problems. The measure of calibration is the RMS over the grid nodes of the
standardized error `z = (mean - truth) / std`. The error of a solver is not a
draw from its posterior, so an RMS `z` of exactly one is not expected, and a
value far below one means a band that is too wide.

**Constant diffusion.** Under a constant diffusion (`"none"`, also after a
post-hoc rescaling) and for `q = 2` to `5`, the smoothed mean is 8.5 to 3400
times more accurate than the filtered one at the grid nodes, and 15 to 46000
times more accurate midway between them. In a refinement study (`h = 0.2` to
`0.025`, `q = 2` to `4`) the smoother's error converges at the order `q + 1.4`
to `q + 2` and the filter's at `q + 1` (on three of the problems; the cubic
decay is not in its asymptotic regime at these steps). The smoother's
standard deviation is only 1.0 to 6.4 times smaller than the filter's. A
diffusion fitted to the filter's innovations, such as `posthoc_mle_sigma_sqr`,
therefore gives the filter an RMS `z` of 0.06 to 0.8 and the smoother one of
3e-4 to 0.03, which is a band too wide by one and a half to three and a half
orders of magnitude. At `q = 1` filter and smoother are equally accurate
within a factor of 1.4.

**Dynamic calibration.** The per-step estimate `r.T (H Q H.T)^-1 r / d`
assigns the whole residual `r` of a step to that step's fresh process noise
`Q`. It ignores the part `H A P A.T H.T` of the residual's predicted
covariance that the step carries over from the previous state, and it
overstates the diffusion for `q >= 2`. The global factor
`c = posthoc_mle_sigma_sqr(result.mz, result.Pz_sqr)` of a dynamic run is the
MLE of one constant multiplier on top of the per-step diffusions. It is 1.0 at
`q = 1`, 0.48 to 0.55 at `q = 2`, 0.10 to 0.26 at `q = 3`, 0.022 to 0.033 at
`q = 4` and 0.015 to 0.021 at `q = 5`.

**Changes from step to step.** A diffusion that changes from step to step
costs the smoother its accuracy. At `q = 3` to `5` the smoothed mean under
dynamic calibration is 5 to 2100 times less accurate than under a constant
diffusion, while the filtered mean loses a factor of 1.0 to 13. At `q = 3` the dynamic diffusion
alternates by a factor of 14 to 17 between neighboring steps on three of the
four problems, and on all four the smoother is only 1.1 to 1.7 times more
accurate than the filter. The smoother's RMS `z` (0.20 to 0.58) is then close
to the filter's (0.21 to 0.56). At `q = 2`, `4` and `5` the dynamic smoother
stays 3 to 23 times more accurate than the filter, and its RMS `z` is 0.02 to
0.08. The alternation of the diffusion at `q = 3` is therefore the reason that
`q = 3` appears the best-calibrated order for the smoother.

Two recipes follow from these measurements. Both rescale covariances and
leave the means of the run unchanged.

**Accurate smoothed means.** For smoothing on a fixed grid, solve with
`calibration="none"` and rescale with the post-hoc MLE. At `q = 3` to `5` the
smoothed means are 5 to 2100 times more accurate than under dynamic
calibration, and the smoother's band stays too wide by the margin above.

```python
from ode_filters import gaussian_filter, rts_smoother
from ode_filters.calibration import posthoc_mle_sigma_sqr, rescale_sqr_seq

result = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration="none")
sigma_sqr_hat = posthoc_mle_sigma_sqr(result.mz, result.Pz_sqr)
smoothed = rts_smoother(result)
P_smoothed_sqr = rescale_sqr_seq(smoothed.P_sqr, sigma_sqr_hat)
```

**Bands that match the errors** ("dynamic times `c`"). Solve with dynamic
calibration and rescale by the global factor `c`. Rescaling the filtered
covariances, the backward conditionals and the per-step diffusion makes
`rts_smoother` and `interpolate` return the rescaled posterior. At `q = 3` to
`5` the RMS `z` is then 0.51 to 1.09 for the filter and 0.19 to 1.14 for the
smoother.

```python
result = gaussian_filter(m_0, P_0_sqr, prior, measure, ts)  # "dynamic"
c = posthoc_mle_sigma_sqr(result.mz, result.Pz_sqr)
result = result._replace(  # other covariances, e.g. P_pred_sqr, stay unscaled
    P_sqr=rescale_sqr_seq(result.P_sqr, c),
    P_back_sqr=rescale_sqr_seq(result.P_back_sqr, c),
    diffusion=c * result.diffusion,
)
smoothed = rts_smoother(result)
```

The rescaling is exact only for an exact initial state
(`taylor_mode_initialization`); with an uncertain one, part of each
covariance does not scale with the diffusion. For a `JointPrior`, pass
`prior=` to `rescale_sqr_seq` (see the warning above). Neither recipe
calibrates the smoother at `q = 2`, where its RMS `z` stays between 0.001 and
0.06.

## Selecting hyperparameters from data

The diffusion is one hyperparameter. A prior's length scale, `Xi` and the
parameters of a hidden input are others. With data, they are selected by the
evidence of the data, of which the library reports three versions. Write
`Z` for the ODE information (observed as zero), `Y` for the data and `U` for a
hidden input's trajectory.

**The sequential data evidence**, `result.log_likelihood_obs`
(`marginal_loglik(..., channel="obs")`), is `sum_n log p(y_n | z_<=n, y_<n)`:
each observation conditioned on the ODE information up to its own time. Together
with the ODE-information evidence `result.log_likelihood` it sums to
`log p(Z, Y)`.

**The conditional evidence** `log p(Y | Z) = log p(Z, Y) - log p(Z)` conditions
each observation on all of the ODE information, including the later one. It
differs from the sequential one whenever later ODE information is informative
about earlier data, which is the usual case. It has two estimators:

- `fenrir_loglik(data_free, obs_model)` (`channel="fenrir"`): one pass on the
  ODE information alone, then a Kalman filter backwards through its backward
  conditionals that updates on the data (Tronarp, Bosch & Hennig, 2022).
- `conditional_loglik(data, data_free)` (`channel="dalton"`): a pass with data
  minus a pass without. On one frozen linearization (two `affine_filter` runs
  of the same `AffineSurrogate`) it equals Fenrir's value exactly. With
  one-shot filters each pass linearizes at its own predictions, and the
  difference is Dalton's estimator (Wu & Lysy, 2024), which in general differs
  from Fenrir's value.

With data between the grid nodes (data-only nodes, see
[sharp bits](sharp-bits.md#data-between-grid-nodes-add-nodes-without-ode-information)),
the pass without data runs on the timeline of the data, with
`obs_model=obs_model.without_data()`. Its ODE information is that of the grid,
and its backward conditionals reach the data-only nodes. `marginal_loglik` does
this for both channels.

With adaptive steps (`InferenceProblem(..., adaptive=AdaptiveSettings(...))`),
`marginal_loglik` computes Fenrir's estimator from the pass without data run
with `smoother=True`, whose backward conditionals span one save interval each.
At `atol = rtol = 1e-8` it agrees with a fixed grid of 2000 steps to a
relative difference of about `2e-8`. Dalton's estimator is not available
there: the passes with and without data choose different steps, so their ODE
evidences condition on ODE information at different times.

Use the conditional evidence to compare hyperparameters by the probability
that the ODE-constrained model assigns to the data. Use the sequential one when
the data are to be judged as they arrive (filtering, online use).

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
not correct a prior with the wrong smoothness (`q` too low), an incorrectly
coded vector field, or a Jacobian that the EKF linearization cannot
approximate. Since calibration fits the scale of the ODE residuals, their
average size matches by construction (exactly for the post-hoc MLE), and only
their structure remains informative. The whitened residuals
`whitened_residuals(result.mz, result.Pz_sqr)` of a consistent filter are
white noise. A trend over time in `nis(result.mz, result.Pz_sqr)`, or a
`residual_autocorrelation` outside about `±2 / sqrt(n)` for `n` residuals,
indicates a misspecified model, which no choice of scale corrects. Against
data with a known noise level (`result.mz_obs`, `result.Pz_obs_sqr`), the NIS
also tests the scale: under a consistent model its expected value is the
number of observed rows per step.

## See also

- [Adaptive Step-Size Control](adaptive-steps.ipynb): the controller that uses
  the per-step quasi-MLE.
- API reference for `ode_filters.calibration`.
