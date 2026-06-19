# Filters

The consolidated solver API: `gaussian_filter` (fixed grid),
`gaussian_filter_adaptive` (adaptive, save-at-grid), and `rts_smoother` for the
backward smoothing pass — all returning a single `FilterResult`. The
linearization is a pluggable [`Correction`](../../corrections.md) (EK0 / EK1 /
IEKF), and adaptive stepping uses a step-size controller (`PController` /
`PIController`). See [How to choose](../../how-to-choose.md).

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
| `log_likelihood` | scalar ODE-information log-marginal-likelihood |
| `log_likelihood_obs` | observation log-likelihood (`None` without `obs_model`) — the parameter-inference target |
| `m_pred`, `P_pred_sqr` | one-step predictions (before each update) |
| `G_back`, `d_back`, `P_back_sqr` | backward transitions consumed by `rts_smoother` |
| `sigma_sqr` | per-step calibrated diffusion `sigma_hat^2` |
| `m_bar`, `P_bar_sqr`, `T` | preconditioned-space internals (`None` for a plain prior) |

`gaussian_filter_adaptive` is filtering-only, so its backward-pass fields are
`None` and `rts_smoother` does not apply to it.

The low-level loop variants (`ekf1_sqr_loop*`, `*_step*`, the adaptive trajectory
driver) live in the `ode_filters.filters.ode_filter_loop` / `ode_filter_step` /
`ode_filter_adaptive` submodules; `gaussian_filter` wraps them.

::: ode_filters.filters
