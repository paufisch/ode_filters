# How to choose

`ode_filters` is a small menu of components. This page turns that menu into
decisions: for each axis, a default, when to deviate, and a comparison table.
When in doubt, the defaults are good.

Not every combination is supported: preconditioned priors (`PrecondIWP` /
`PrecondMaternPrior`) and the adaptive smoother (`smoother=True`) do not yet accept
external observations (`obs_model`), and the `"diagonal"` / `"diagonal_ekf0"`
calibration modes require the prior's `Xi` to be diagonal. See [Sharp bits](sharp-bits.md).

## TL;DR defaults

```python
from ode_filters.priors import IWP, taylor_mode_initialization
from ode_filters.measurement import ODEInformation
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
| **`MaternPrior`** | Matern smoothness with a length scale | you have a characteristic correlation length / want stationary behaviour |
| **`JointPrior(prior_x, prior_u)`** | a state block **and** a hidden block | joint state–parameter or latent-force inference (`*WithHidden` models) |

Preconditioned variants (`PrecondIWP`, `PrecondMaternPrior`) trade a coordinate
change for better conditioning at high `q` / small step size.

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
[`Correction`](corrections.md).

| Correction | Cost | Use when |
| --- | --- | --- |
| **`TaylorCorrection(order=1)` (EK1)** | Jacobian per step | the default — generally more accurate per step (uses the full vector-field Jacobian) |
| **`TaylorCorrection(order=0)` (EK0)** | no Jacobian | high-dimensional or expensive Jacobians; very robust |

## Stepping

| Mode | Function | Use when |
| --- | --- | --- |
| **Fixed grid** | `gaussian_filter` | you know a good step count |
| **Adaptive** | `gaussian_filter_adaptive` (jit/vmap/grad-able) | you want tolerance-based efficiency |

Both paths are `jit`/`grad`/`vmap`-compatible (see [Sharp bits](sharp-bits.md)).
See [Adaptive step-size control](adaptive-steps.md) for the controller.

## Calibration

Sizes the uncertainty. Full treatment in [Diffusion calibration](calibration.md).

| Mode | Use when |
| --- | --- |
| **`"dynamic"`** | per-step quasi-MLE — the default for a plain solve |
| **`"none"`** | a fixed `Xi` is wanted, so the likelihood is not confounded by dynamic calibration |
| **`"diagonal"` / `"diagonal_ekf0"`** | per-component scales; pair `diagonal_ekf0` with an EK0 correction |
| post-hoc (`posthoc_mle_sigma_sqr`) | a single global scale after a fixed-step run |

## Measurement model

What the filter conditions on — see the [measurement API](api/measurement/index.md).

| Model | Use when |
| --- | --- |
| **`ODEInformation`** | a plain first-order ODE `dx/dt = f(x, t)` |
| **`SecondOrderODEInformation`** | a second-order ODE `d²x/dt² = f(x, v, t)` |
| **`*WithHidden`** | a hidden parameter enters the dynamics (with `JointPrior`) |
| **`Conservation` / `Measurement`** | enforce a conservation law / assimilate data at times |
| **`BlackBoxMeasurement` / `TransformedMeasurement`** | a custom or transformed observation operator |

## See also

- [What is a probabilistic ODE solver?](probabilistic-ode-solvers.md)
- [Linearization schemes](corrections.md) · [Diffusion calibration](calibration.md) · [Adaptive steps](adaptive-steps.md)
