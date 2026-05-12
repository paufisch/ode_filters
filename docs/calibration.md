# Diffusion Calibration

Probabilistic ODE filters quantify uncertainty by propagating a Gaussian
covariance. The *magnitude* of that covariance is governed by a scalar
diffusion `sigma^2` on top of the prior's structural scale matrix `Xi`. The
filter is correct in the mean even with `sigma = 1`, but the posterior
uncertainty is honestly scaled only when `sigma^2` is calibrated from the
observed ODE residuals.

The `ode_filters.calibration` subpackage provides three estimators that all
consume the predicted-observation marginal `(m_z, P_z_sqr)` already returned
by every EKF step.

## Workflows

### Post-hoc MLE on a fixed-step run

Cheapest workflow: run any existing fixed-step loop, then rescale.
`posthoc_mle_sigma_sqr` is the joint MLE of `sigma^2` under the assumption
that one constant scale explains every step's whitened residual -- the same
Gaussian likelihood the per-step estimator maximises, but with a single
parameter rather than one per step. The closed form makes
`posthoc_mle_sigma_sqr` numerically identical to `aggregate_sigma_sqr(seq,
kind="mean")` applied to the per-step trace.

```python
import jax.numpy as np
from ode_filters.filters import ekf1_sqr_loop, rts_sqr_smoother_loop
from ode_filters.calibration import posthoc_mle_sigma_sqr, rescale_sqr_seq
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization


def vf(x, *, t):
    return x * (1 - x)


prior = IWP(q=2, d=1)
mu_0, S0 = taylor_mode_initialization(vf, np.array([0.1]), q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)

(m_seq, P_seq_sqr, m_pred_seq, P_pred_seq_sqr,
 G_back, d_back, P_back, mz_seq, Pz_seq_sqr, _) = ekf1_sqr_loop(
    mu_0, S0, prior, measure, (0.0, 5.0), N=50
)

sigma_sqr_hat = posthoc_mle_sigma_sqr(
    np.stack(list(mz_seq)), np.stack(list(Pz_seq_sqr)),
)

# Rescale stored covariances in-place.
P_seq_sqr_calibrated = rescale_sqr_seq(np.stack(list(P_seq_sqr)), sigma_sqr_hat)
```

This is appropriate for smooth problems where a single global scalar captures
the residual size well.

### Per-step quasi-MLE inside the adaptive loop

`ekf1_sqr_adaptive_loop` runs the per-step quasi-MLE by default and applies it
to the stored covariances on every accepted step. See
[Adaptive Step-Size Control](adaptive-steps.md).

```python
from ode_filters.filters import ekf1_sqr_adaptive_loop

result = ekf1_sqr_adaptive_loop(
    mu_0, S0, prior, measure, (0.0, 5.0), atol=1e-5, rtol=1e-3,
)
# result.sigma_sqr_seq holds the per-step estimates.
```

Pass `calibrate=False` to record the estimates without scaling the stored
covariances (diagnostics only).

### Manual per-step estimation

If you want the per-step number outside the adaptive loop (e.g., for
diagnostics on an existing fixed-step run):

```python
from ode_filters.calibration import quasi_mle_sigma_sqr

sigma_sqr_per_step = [
    float(quasi_mle_sigma_sqr(mz, Pz_sqr))
    for mz, Pz_sqr in zip(mz_seq, Pz_seq_sqr, strict=True)
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

The adaptive loop handles this transparently: internally it recomputes the
*noise-free* predicted residual covariance and uses that for calibration,
matching what probnum and probdiffeq do. With the default `ODEInformation`
(no measurements, `R = 0`) the distinction is moot.

For the post-hoc workflow on a fixed-step loop with non-zero `R`, the caller
must pass a noise-free `Pz_seq_sqr` to `posthoc_mle_sigma_sqr` -- the
covariances returned by the loop include `R` and must be recomputed from the
saved `(m_pred_seq, P_pred_seq_sqr)` first.

## What calibration cannot fix

Calibration scales the posterior to match the size of the residuals. It does
not correct a prior with the wrong smoothness (`q` too low), a vector field
that is miscoded, or a Jacobian that the EKF linearisation cannot
approximate. To check that the calibrated filter is honest, plot the whitened
residual diagnostics described in the wiki note
*whitened-residual-diagnostics* -- `||z_n||^2 / d` should hover near 1 on a
well-specified problem.

## See also

- [Adaptive Step-Size Control](adaptive-steps.md) -- the controller that uses
  the per-step quasi-MLE.
- API reference for `ode_filters.calibration`.
