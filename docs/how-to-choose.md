# How to choose

A solve in `ode_filters` combines a prior, a measurement model, a correction, a
stepping mode and a calibration policy. For each of these choices, this page
gives the default, the conditions for a different choice, and a table of the
options. The defaults are the recommended starting point.

Every prior (plain or preconditioned) works with every stepping mode,
correction and calibration policy, with or without data. Three restrictions
apply. The per-component policies (`"diagonal"`, `DiagonalCalibration`) need a
diagonal `Xi`, `interpolate` needs a fixed-grid result, and the iterated
smoother keeps the diffusion fixed. See [Sharp bits](sharp-bits.md).

## Defaults

```python
from ode_filters.priors import IWP, taylor_mode_initialization
from ode_filters.information import ODEInformation
from ode_filters import gaussian_filter, TaylorCorrection

prior   = IWP(q=2, d=d)                       # smoothness order 2
measure = ODEInformation(vf, prior.E0, prior.E1)
# EK1 correction and dynamic calibration are the defaults of gaussian_filter.
```

## Prior family

The prior encodes the assumptions about the solution before the ODE enters.

| Prior | Assumes | Use when |
| --- | --- | --- |
| **`IWP(q, d)`** | a `q`-times integrated Wiener process (polynomial-like smoothness) | the default, and the starting point for most problems |
| **`MaternPrior`** | a stationary Matern process of smoothness `nu = q + 1/2` with the textbook length scale `l` (for `q = 1` the correlation at lag `tau` is `(1 + sqrt(3) tau / l) exp(-sqrt(3) tau / l)`) | the solution has a characteristic correlation length, or the prior should be stationary |
| **`IOUPPrior(q, d, rate)`** | like `IWP`, but the highest derivative decays towards zero at `rate` instead of drifting freely | the ODE has a known linear part, which, passed as `rate`, makes the solver a probabilistic exponential integrator |
| **`JointPrior(prior_x, prior_u)`** | a state block and a hidden block | joint inference of state and parameters, or latent-force inference (`*WithHidden` models) |

The preconditioned variants (`PrecondIWP`, `PrecondMaternPrior`) compute each
step in rescaled coordinates, which improves the conditioning at high `q` and
small step sizes.

`IOUPPrior` is intended for stiff or semi-linear problems. With the vector field
written as `f(x) = R x + g(x)`, passing `rate=R` puts the linear part into the
prior. The prior mean then integrates the linear part exactly, and the filter
corrects only the nonlinear remainder `g` (Bosch, Hennig & Tronarp, NeurIPS
2023). `rate` may be a scalar, a length-`d` vector (one rate per dimension), or
a `d × d` matrix (the linear part or a reference Jacobian). `IWP` is the case
`rate=0`, so replacing an `IWP` by an `IOUPPrior` leaves the measurement model
and the filter recursion unchanged. Under `"dynamic"` or `"diagonal"`
calibration, the diffusion is then estimated on the remainder instead of the
full field, so the fitted `sigma^2` is not on the same scale as under `IWP`.

## Smoothness order `q`

`q` is the number of derivatives in the state. It sets the order of accuracy of
the method, which is comparable to that of a classical solver of order `q`.

| `q` | Character | Use when |
| --- | --- | --- |
| 1 | low order, best-conditioned covariances | non-smooth dynamics, debugging |
| **2–3** | balance of accuracy and conditioning | most problems (default `q=2`) |
| 4+ | high order, high accuracy | smooth problems and tight tolerances, preferably with a preconditioned prior |

A higher `q` is more accurate per step. A step costs more, however, since it
costs `O(D^3)` operations in the state dimension `D = d (q + 1)`, and the
covariances are worse conditioned on coarse grids. If the covariances lose
positive definiteness or become `NaN` at high `q`, lower `q` or switch to a
preconditioned prior. Initialize with `taylor_mode_initialization(vf, x0, q)`.

## Correction (linearization)

The correction sets how the update linearizes the nonlinear vector field. It is
a pluggable [`Correction`](corrections.md), passed as `correction=`.

| Correction | Cost | Use when |
| --- | --- | --- |
| **`TaylorCorrection(order=1)` (EK1)** | one Jacobian per step | the default; uses the full Jacobian of the vector field and is generally more accurate per step |
| **`TaylorCorrection(order=0)` (EK0)** | no Jacobian | the Jacobian is unavailable or dominates the cost of a step, e.g. in high dimensions |
| **`IteratedTaylorCorrection` (IEKF)** | `max_iters` Jacobians per step | the linearization at the prediction is inaccurate (large steps, strongly nonlinear or stiff fields) |
| **`QuadratureCorrection` (SLR)** | `n_nodes ** len(args)` field evaluations per step | the linearization error should enter the uncertainty, or the field has no useful Jacobian |

A correction acts within one step. To relinearize over the whole trajectory,
iterate the smoother:

| Smoother | Converges to | Use when |
| --- | --- | --- |
| **`iterated_smoother(linearization="taylor")` (IEKS)** | the MAP trajectory (Gauss-Newton) | a point estimate that conditions on all the information, e.g. with data or a hidden input |
| **`iterated_smoother(linearization="slr")` (IPLS)** | a moment-matched Gaussian | as the IEKS, with the linearization error in the uncertainty |

## Stepping

| Mode | Function | Use when |
| --- | --- | --- |
| **Fixed grid** | `gaussian_filter` | a suitable number of steps is known |
| **Adaptive** | `gaussian_filter_adaptive` (jit/vmap/grad-able) | the step sizes should follow from a tolerance |

Both paths are `jit`/`grad`/`vmap`-compatible (see [Sharp bits](sharp-bits.md)).
See [Adaptive step-size control](adaptive-steps.ipynb) for the controller.

## Calibration

The calibration sets the scale of the uncertainty. See
[Diffusion calibration](calibration.md) for the details.

| Policy | Shorthand | Use when |
| --- | --- | --- |
| **`DynamicCalibration()`** | `"dynamic"` | per-step quasi-MLE, the default of the solvers |
| **`NoCalibration()`** | `"none"` | the prior's scale is fixed or fitted, e.g. inside a likelihood |
| **`DiagonalCalibration()`** | `"diagonal"` | per-component scales (needs a diagonal `Xi`) |
| **`DiagonalCalibration(jacobian="ek0")`** | `"diagonal_ekf0"` | per-component scales without a Jacobian; pair with EK0 |
| post-hoc (`posthoc_mle_sigma_sqr`) | | a single global scale after a `"none"` run |

The policy objects take a lower bound on the estimate, e.g.
`DynamicCalibration(min_sigma_sqr=1e-12)`.

## Fitting to data

The table lists the tools for an ODE with unknown parameters or inputs. The
[Inference from data](examples/parameter-estimation.ipynb) tutorials
demonstrate each of them.

| Goal | Use |
| --- | --- |
| a point estimate of ODE parameters | `marginal_loglik` and `fit_scipy` (L-BFGS-B), or `fit` with an Optax optimizer |
| a posterior over the parameters | `marginal_loglik` inside NumPyro or BlackJAX |
| the likelihood with adaptive steps between the data times | `InferenceProblem(..., adaptive=AdaptiveSettings(atol=..., rtol=...))` |
| data off the solver's grid | `measurements=[Measurement(...)]` or `prepare_observations(measurements, prior.E0_state, ts)`: data-only nodes (see [sharp bits](sharp-bits.md#data-between-grid-nodes-add-nodes-without-ode-information)) |
| a hidden input `u(t)` | a `JointPrior` with a `*WithHidden` model; `rts_smoother` / `sample_posterior` |
| hyperparameters (length scale, `Xi`, an input's magnitude) | the conditional evidence: `channel="fenrir"` or `"dalton"` (see [calibration](calibration.md#selecting-hyperparameters-from-data) and the [comparison of the channels](examples/parameter-estimation.ipynb#comparing-the-evidence-channels)) |
| the information that the data carry about a hidden input | `information_criteria` |
| whether the fitted model is consistent with the data | `nis`, `whitened_residuals`, `residual_autocorrelation` |
| predictions between grid nodes, or of new observations | `interpolate`, `observation_predictive` |

## Measurement model

The measurement model defines what the filter conditions on. See the
[information API](api/information/index.md).

| Model | Use when |
| --- | --- |
| **`ODEInformation`** | a first-order ODE `dx/dt = f(x, t)` |
| **`SecondOrderODEInformation`** | a second-order ODE `d²x/dt² = f(x, v, t)` |
| **`*WithHidden`** | a hidden parameter enters the dynamics (with `JointPrior`) |
| **`Conservation` / `Measurement`** | enforce a conservation law / assimilate data at given times |
| **`BlackBoxMeasurement` / `TransformedMeasurement`** | a custom or transformed observation operator |

## See also

- [What is a probabilistic ODE solver?](probabilistic-ode-solvers.md)
- [Linearization schemes](corrections.md) · [Diffusion calibration](calibration.md) · [Adaptive steps](adaptive-steps.ipynb)
