# Filters

The consolidated solver API: `gaussian_filter` (fixed grid),
`gaussian_filter_adaptive` (adaptive, save-at-grid), and `rts_smoother` for the
backward smoothing pass — all returning a single `FilterResult`. The
linearization is a pluggable [`Correction`](../../corrections.md) (EK0 / EK1 /
IEKF), and adaptive stepping uses a step-size controller (`PController` /
`PIController`). Diffusion is calibrated per step by default
(`calibration="dynamic"`, a scalar quasi-MLE), so the process noise is rescaled
each step; pass `calibration="none"` for a fixed diffusion. See
[How to choose](../../how-to-choose.md).

Dispatch is automatic: a `PrecondIWP` / `PrecondMaternPrior` prior selects the
preconditioned square-root recursion, and passing `obs_model=` adds a masked
observation update.

## The `FilterResult`

Both `gaussian_filter` and `gaussian_filter_adaptive` return a `FilterResult` with
named fields (recover a covariance with `P = P_sqr.T @ P_sqr`; see
[notation](../../notation.md)):

| Field | Meaning |
| --- | --- |
| `t` | time grid |
| `m`, `P_sqr` | **filtered** posterior mean and square-root covariance |
| `log_likelihood` | scalar **post-calibration** ODE-information log-marginal-likelihood (not comparable across `calibration` modes — each defines a different generative model) |
| `log_likelihood_obs` | observation log-likelihood (`None` without `obs_model`) |
| `m_pred`, `P_pred_sqr` | one-step predictions before each update (`None` for the adaptive save-at solver) |
| `G_back`, `d_back`, `P_back_sqr` | backward transitions consumed by `rts_smoother` |
| `sigma_sqr` | per-step calibrated diffusion `sigma_hat^2` (`None` for the adaptive solver) |
| `success` | adaptive only: whether sub-stepping reached every save time **and** the log-likelihood is finite (`None` for the fixed grid) |
| `m_bar`, `P_bar_sqr`, `T` | preconditioned-space internals (`None` for a plain prior) |

!!! warning "Preconditioned priors: some fields are in preconditioned coordinates"
    With `PrecondIWP` / `PrecondMaternPrior` / `PrecondJointPrior`, `m`, `P_sqr`,
    `mz` and `Pz_sqr` are in the original coordinates, but `m_pred`,
    `P_pred_sqr`, `G_back`, `d_back` and `P_back_sqr` are in the preconditioned
    (bar) coordinates the recursion runs in, with `x = T @ x_bar`. Map them back
    before using them directly, e.g. `m_pred = result.m_pred @ result.T.T` and
    `P_pred_sqr = result.P_pred_sqr @ result.T.T`. `rts_smoother` handles this
    for you.

`gaussian_filter_adaptive` is **filtering-only by default** (`smoother=False`):
its backward-pass fields are then `None` and `rts_smoother` raises. Pass
`smoother=True` to additionally compute a fixed-point smoothing backward pass
(one composite conditional per save interval, `O(#save points)` memory); the
result then carries `G_back` / `d_back` / `P_back_sqr` and `rts_smoother(prior,
result)` works directly. `smoother=True` is not supported together with
`obs_model`.

## Solvers

::: ode_filters.filters.gaussian_filter.gaussian_filter

::: ode_filters.filters.gaussian_filter.gaussian_filter_adaptive

::: ode_filters.filters.gaussian_filter.rts_smoother

::: ode_filters.filters.gaussian_filter.FilterResult

## Linearization (corrections)

See [Linearization schemes](../../corrections.md) for when to use which.

::: ode_filters.filters.correction.Correction

::: ode_filters.filters.correction.CorrectionResult

::: ode_filters.filters.correction.TaylorCorrection

::: ode_filters.filters.correction.IteratedTaylorCorrection

::: ode_filters.filters.correction.QuadratureCorrection

## Iterated posterior-linearization smoother

::: ode_filters.filters.ipls.ipls_smoother

::: ode_filters.filters.ipls.IPLSResult

::: ode_filters.filters.ipls.affine_filter_scan

## Statistical linearization

::: ode_filters.filters.statistical_linearization.slr_linearize

::: ode_filters.filters.statistical_linearization.SLRModel

::: ode_filters.filters.statistical_linearization.gauss_hermite_rule

::: ode_filters.filters.statistical_linearization.cubature_rule

::: ode_filters.filters.statistical_linearization.quadrature_nodes

::: ode_filters.filters.statistical_linearization.check_arg_projection

## Step-size control

See [Adaptive step-size control](../../adaptive-steps.ipynb).

::: ode_filters.filters.adaptive_controller.StepSizeController

::: ode_filters.filters.adaptive_controller.PController

::: ode_filters.filters.adaptive_controller.PIController
