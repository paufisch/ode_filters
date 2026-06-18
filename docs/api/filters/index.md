# Filters

Filtering and smoothing loops, the pluggable [`Correction`](../../corrections.md)
strategies (EK0 / EK1), and adaptive step-size control. See
[How to choose](../../how-to-choose.md) for which loop to call.

## The filter return tuple

The fixed-step loops (`ekf1_sqr_loop` and the `*_scan` variants) return a stacked
sequence of square-root Gaussians — ten fields, in this order:

| Index | Field | Meaning |
| --- | --- | --- |
| 0–1 | `m_seq`, `P_seq_sqr` | **filtered** posterior mean and square-root covariance |
| 2–3 | `m_pred_seq`, `P_pred_seq_sqr` | one-step **predictions** (before each update) |
| 4–6 | `G_back_seq`, `d_back_seq`, `P_back_seq_sqr` | **backward transitions** consumed by `rts_sqr_smoother_loop` |
| 7–8 | `mz_seq`, `Pz_seq_sqr` | **predicted-observation** marginals (the ODE-defect innovations) |
| 9 | `log_likelihood` | scalar log-marginal-likelihood |

Recover a covariance with `P = P_sqr.T @ P_sqr` (see [notation](../../notation.md)).
The adaptive loop returns an `AdaptiveLoopResult` named tuple instead. Calibration
adds a per-step `sigma_sqr_seq`; the observation variants split the likelihood into
`(log_likelihood_ode, log_likelihood_obs)`.

::: ode_filters.filters
