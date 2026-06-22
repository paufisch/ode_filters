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
| pick the linearization (EK0 / EK1) | [`TaylorCorrection`](filters/index.md) |
| define the ODE / observations | [Measurement](measurement/index.md) models |
| choose a prior | [`IWP` / `MaternPrior` / `JointPrior`](priors/index.md) |
| initialize the state | [`taylor_mode_initialization`](priors/index.md) |
| calibrate posterior uncertainty (diffusion `sigma^2`) | [calibration guide](../calibration.md) |
| fit ODE parameters to data (max-likelihood) | [`ODEFilter` / `fit` / `marginal_loglik`](inference/index.md) |

## Subpackages

- **[Filters](filters/index.md)** — filtering / smoothing loops, corrections, adaptive control.
- **[Inference](inference/index.md)** — ODE-parameter inference (`fit`, `marginal_loglik`, `ODEFilter`) and the square-root Gaussian algebra it builds on.
- **[Measurement](measurement/index.md)** — ODE-information and observation models.
- **[Priors](priors/index.md)** — Gauss–Markov process priors.

Each page is generated with mkdocstrings and stays in sync with the source.
