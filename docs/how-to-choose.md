# How to choose

`ode_filters` is a small menu of components. This page turns that menu into
decisions: for each axis, a default, when to deviate, and a comparison table.
When in doubt, the defaults are good.

The components combine freely: every prior (plain or preconditioned) works with
every stepping mode, correction and calibration policy, with or without data.
The few restrictions: the per-component policies (`"diagonal"`,
`DiagonalCalibration`) need a diagonal `Xi`; `interpolate` needs a fixed-grid
result; and the iterated smoother keeps the diffusion fixed. See
[Sharp bits](sharp-bits.md).

## TL;DR defaults

```python
from ode_filters.priors import IWP, taylor_mode_initialization
from ode_filters.information import ODEInformation
from ode_filters import gaussian_filter, TaylorCorrection

prior   = IWP(q=2, d=d)                       # smoothness order 2
measure = ODEInformation(vf, prior.E0, prior.E1)
# EK1 correction and dynamic calibration are the defaults of gaussian_filter.
```

## Prior family

The prior encodes what you assume about the solution before seeing the ODE.

| Prior | Assumes | Use when |
| --- | --- | --- |
| **`IWP(q, d)`** | a `q`-times integrated Wiener process (polynomial-like smoothness) | the default — almost always start here |
| **`MaternPrior`** | a stationary Matern process of smoothness `nu = q + 1/2` with the textbook length scale `l` (for `q = 1` the correlation at lag `tau` is `(1 + sqrt(3) tau / l) exp(-sqrt(3) tau / l)`) | you have a characteristic correlation length / want stationary behaviour |
| **`IOUPPrior(q, d, rate)`** | like `IWP`, but the highest derivative decays towards zero at `rate` instead of drifting freely | the ODE has a known linear part — pass it as `rate` and the solver becomes a probabilistic *exponential integrator* |
| **`JointPrior(prior_x, prior_u)`** | a state block **and** a hidden block | joint state–parameter or latent-force inference (`*WithHidden` models) |

Preconditioned variants (`PrecondIWP`, `PrecondMaternPrior`) trade a coordinate
change for better conditioning at high `q` / small step size.

`IOUPPrior` is worth reaching for on **stiff or semi-linear** problems. Writing
the field as `f(x) = R x + g(x)`, passing `rate=R` bakes the linear part into the
prior, so the prior mean integrates it *exactly* and the filter only has to
correct the nonlinear remainder `g` (Bosch, Hennig & Tronarp, NeurIPS 2023).
`rate` may be a scalar, a length-`d` vector (per-dimension), or a `d × d` matrix
(the linear part or a reference Jacobian). `IWP` is exactly the `rate=0` case, so
it is a drop-in swap — the measurement model and the filter recursion are
unchanged. One caveat: `"dynamic"` / `"diagonal"` calibration then estimates the
diffusion on the *remainder* rather than the full field, so the fitted `sigma^2`
is not on the same scale as under `IWP`.

## Smoothness order `q`

`q` is the number of derivatives tracked; it sets the method's order of accuracy
(roughly like a classical solver of order `q`).

| `q` | Character | Use when |
| --- | --- | --- |
| 1 | low order, very robust | rough/non-smooth dynamics, debugging |
| **2–3** | the sweet spot | most problems (default `q=2`) |
| 4+ | high order, high accuracy | smooth problems, tight tolerances — watch conditioning; prefer the preconditioned priors |

Higher `q` is more accurate per step but more expensive and more ill-conditioned on
coarse grids. If covariances misbehave at high `q`, lower it or switch to a
preconditioned prior. Initialize with `taylor_mode_initialization(vf, x0, q)`.

## Correction (linearization)

How the nonlinear vector field is linearized in the update — a pluggable
[`Correction`](corrections.md), passed as `correction=`.

| Correction | Cost | Use when |
| --- | --- | --- |
| **`TaylorCorrection(order=1)` (EK1)** | one Jacobian per step | the default — generally more accurate per step (uses the full vector-field Jacobian) |
| **`TaylorCorrection(order=0)` (EK0)** | no Jacobian | high-dimensional or expensive Jacobians; very robust |
| **`IteratedTaylorCorrection` (IEKF)** | a few Jacobians per step | single steps are badly linearized (large steps, strongly nonlinear or stiff fields) |
| **`QuadratureCorrection` (SLR)** | `n_nodes ** len(args)` field evaluations per step | the linearization error should enter the uncertainty, or the field has no useful Jacobian |

A correction acts within one step. To re-linearize over the whole trajectory,
iterate the smoother:

| Smoother | Converges to | Use when |
| --- | --- | --- |
| **`iterated_smoother(linearization="taylor")` (IEKS)** | the MAP trajectory (Gauss-Newton) | a point estimate that uses all the information, e.g. with data or a hidden input |
| **`iterated_smoother(linearization="slr")` (IPLS)** | a moment-matched Gaussian | as the IEKS, with the linearization error in the uncertainty |

## Stepping

| Mode | Function | Use when |
| --- | --- | --- |
| **Fixed grid** | `gaussian_filter` | you know a good step count |
| **Adaptive** | `gaussian_filter_adaptive` (jit/vmap/grad-able) | you want tolerance-based efficiency |

Both paths are `jit`/`grad`/`vmap`-compatible (see [Sharp bits](sharp-bits.md)).
See [Adaptive step-size control](adaptive-steps.ipynb) for the controller.

## Calibration

Sizes the uncertainty. Full treatment in [Diffusion calibration](calibration.md).

| Policy | Shorthand | Use when |
| --- | --- | --- |
| **`DynamicCalibration()`** | `"dynamic"` | per-step quasi-MLE — the default for a plain solve |
| **`NoCalibration()`** | `"none"` | the prior's scale is fixed or fitted, e.g. inside a likelihood |
| **`DiagonalCalibration()`** | `"diagonal"` | per-component scales (needs a diagonal `Xi`) |
| **`DiagonalCalibration(jacobian="ek0")`** | `"diagonal_ekf0"` | per-component scales without a Jacobian; pair with EK0 |
| post-hoc (`posthoc_mle_sigma_sqr`) | — | a single global scale after a `"none"` run |

The policy objects take a floor, e.g. `DynamicCalibration(min_sigma_sqr=1e-12)`.

## Fitting to data

What to reach for when the ODE has unknowns. The
[Inference from data](examples/parameter-estimation.ipynb) tutorials walk
through each.

| Goal | Reach for |
| --- | --- |
| a point estimate of ODE parameters | `marginal_loglik` and `fit_scipy` (L-BFGS-B), or `fit` with an Optax optimizer |
| a posterior over the parameters | `marginal_loglik` inside NumPyro or BlackJAX |
| a hidden input `u(t)` | a `JointPrior` with a `*WithHidden` model; `rts_smoother` / `sample_posterior` |
| hyperparameters (length scale, `Xi`, an input's magnitude) | the conditional evidence: `channel="fenrir"` or `"dalton"` (see [calibration](calibration.md#selecting-hyperparameters-from-data) and the [comparison of the channels](examples/parameter-estimation.ipynb#comparing-the-evidence-channels)) |
| how much the data say about a hidden input | `information_criteria` |
| whether the fitted model is consistent with the data | `nis`, `whitened_residuals`, `residual_autocorrelation` |
| predictions between grid nodes, or of new observations | `interpolate`, `observation_predictive` |

## Measurement model

What the filter conditions on — see the [information API](api/information/index.md).

| Model | Use when |
| --- | --- |
| **`ODEInformation`** | a plain first-order ODE `dx/dt = f(x, t)` |
| **`SecondOrderODEInformation`** | a second-order ODE `d²x/dt² = f(x, v, t)` |
| **`*WithHidden`** | a hidden parameter enters the dynamics (with `JointPrior`) |
| **`Conservation` / `Measurement`** | enforce a conservation law / assimilate data at times |
| **`BlackBoxMeasurement` / `TransformedMeasurement`** | a custom or transformed observation operator |

## See also

- [What is a probabilistic ODE solver?](probabilistic-ode-solvers.md)
- [Linearization schemes](corrections.md) · [Diffusion calibration](calibration.md) · [Adaptive steps](adaptive-steps.ipynb)
