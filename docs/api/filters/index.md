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

`gaussian_filter_adaptive` is **filtering-only by default** (`smoother=False`):
its backward-pass fields are then `None` and `rts_smoother` raises. Pass
`smoother=True` to additionally compute a fixed-point smoothing backward pass
(one composite conditional per save interval, `O(#save points)` memory); the
result then carries `G_back` / `d_back` / `P_back_sqr` and `rts_smoother(prior,
result)` works directly. `smoother=True` is not supported together with
`obs_model`.

The low-level scan loops and step functions that `gaussian_filter` wraps live in
the `ode_filters.filters.ode_filter_loop` / `ode_filter_step` / `ode_filter_adaptive`
submodules; they are implementation detail, not part of the public API.

::: ode_filters.filters
