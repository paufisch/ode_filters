# API reference

The API mirrors the package layout. New here? Read
[What is a probabilistic ODE solver?](../probabilistic-ode-solvers.md) and
[How to choose](../how-to-choose.md) first — this reference is for looking things
up, and assumes the [notation](../notation.md).

## Which function do I need?

| Task | Reach for |
| --- | --- |
| solve an ODE (fixed step) | [`gaussian_filter`](filters/index.md) + `rts_smoother` |
| solve with tolerance-based stepping | [`gaussian_filter_adaptive`](filters/index.md) (jit/vmap/grad-able) |
| pick the linearization (EK0 / EK1 / IEKF / SLR) | [`TaylorCorrection` and friends](linearization/index.md) |
| define the ODE / observations | [Information](information/index.md) operators |
| choose a prior | [`IWP` / `MaternPrior` / `IOUPPrior` / `JointPrior`](priors/index.md) |
| initialize the state | [`taylor_mode_initialization`](priors/index.md) |
| calibrate posterior uncertainty (diffusion `sigma^2`) | the `calibration=` argument, or [post-hoc](calibration/index.md) ([guide](../calibration.md)) |
| fit ODE parameters to data (max-likelihood) | [`ODEFilter` / `fit` / `fit_scipy` / `marginal_loglik`](inference/index.md) |
| check a fit against its data | [`nis` / `whitened_residuals` / `observation_predictive`](inference/index.md) |
| draw posterior paths, evaluate between nodes | [`sample_posterior` / `interpolate`](filters/index.md) |

## Subpackages

- **[Filters](filters/index.md)** — the solvers and smoothers, and step-size control.
- **[Linearization](linearization/index.md)** — Taylor and statistical linearization (the corrections).
- **[Information](information/index.md)** — ODE-information operators and data observations.
- **[Priors](priors/index.md)** — Gauss–Markov process priors.
- **[Calibration](calibration/index.md)** — post-hoc diffusion (`sigma^2`) estimation and covariance rescaling.
- **[Inference](inference/index.md)** — parameter inference (`fit`, `marginal_loglik`, `ODEFilter`).
- **[Gaussian algebra](gaussian/index.md)** — the square-root Gaussian types and algebra everything builds on.

The reference sections are generated from the docstrings, so they stay in sync with the source.
