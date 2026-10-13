# API Reference

The API mirrors the package layout. New here? Read
[What is a probabilistic ODE solver?](../probabilistic-ode-solvers.md) and
[How to choose](../how-to-choose.md) first — this reference is for looking things
up, and assumes the [notation](../notation.md).

## Which function do I need?

| Task | Reach for |
| --- | --- |
| solve an ODE (fixed step) | [`gaussian_filter`](filters/index.md) + `rts_smoother` |
| solve with tolerance-based stepping | [`gaussian_filter_adaptive`](filters/index.md) (jit/vmap/grad-able) |
| pick the linearization (EK0 / EK1 / IEKF / SLR) | [`TaylorCorrection` and friends](filters/index.md#linearization-corrections) |
| define the ODE / observations | [Measurement](measurement/index.md) models |
| choose a prior | [`IWP` / `MaternPrior` / `IOUPPrior` / `JointPrior`](priors/index.md) |
| initialize the state | [`taylor_mode_initialization`](priors/index.md) |
| calibrate posterior uncertainty (diffusion `sigma^2`) | the `calibration=` argument, or [post-hoc](calibration/index.md) ([guide](../calibration.md)) |
| fit ODE parameters to data (max-likelihood) | [`ODEFilter` / `fit` / `marginal_loglik`](inference/index.md) |

## Subpackages

- **[Filters](filters/index.md)** — the solvers and smoothers, linearizations (corrections), and step-size control.
- **[Calibration](calibration/index.md)** — post-hoc diffusion (`sigma^2`) estimation and covariance rescaling.
- **[Inference](inference/index.md)** — ODE-parameter inference (`fit`, `marginal_loglik`, `ODEFilter`) and the square-root Gaussian algebra it builds on.
- **[Measurement](measurement/index.md)** — ODE-information and observation models.
- **[Priors](priors/index.md)** — Gauss–Markov process priors.

The reference sections are generated from the docstrings, so they stay in sync with the source.
