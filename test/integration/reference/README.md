# Reference outputs

`reference_outputs.npz` holds solver outputs recorded with `generate.py` from the
scenarios in `scenarios.py`. `test_reference_outputs.py` compares the current
solvers against them, so a restructure of the internals that changes a result
fails loudly.

Regenerate only for an intended change of results, and record it below: which
scenarios changed, by how much, and why.

## Changes

- 2026-10-13: first recording (library state of release 0.7.3 plus the October
  correctness fixes). Against release 0.7.3 itself only
  `second_order_diagonal_ekf0` differs: `sigma_sqr` is about 300 times smaller,
  because `"diagonal_ekf0"` now takes the EK0 Jacobian from the model's
  left-hand-side selector (`E2` for a second-order ODE) instead of the prior's
  `E1` (CHANGELOG, Fixed).
- 2026-10-13, restructure onto one step kernel. Everything else agreed with the
  first recording to better than `1e-9` (relative to each array's scale),
  including the preconditioned scenarios, which now run as a change of variables
  inside each step. Two intended changes:
  - `logistic_dynamic_ek0`, `logistic_none_ek0`: the diffusion estimate of an
    EK0 filter now uses the EK0 linearization (the one its update uses) instead
    of an EK1 Jacobian computed only for the estimate. `sigma_sqr` changes by
    about 1.5 %, and with dynamic calibration the solution by about `4e-6`.
  - `*_obs*`, `adaptive_obs`: `mz_obs` / `Pz_obs` at steps without data. They
    used to hold the residual of a zero observation (`H m`, `H P H.T + R`); data
    rows without data are now masked to zero mean and unit variance. Steps with
    data, the states and the evidence are unchanged. In `adaptive_obs` the
    recorded `log_likelihood` is the sum of the ODE and data evidence, which the
    adaptive solver used to report together.
- 2026-10-14, textbook Matern length scale. Only `matern_dynamic`
  (`MaternPrior(q=2, length_scale=1.5)`) changes: its SDE rate is now
  `sqrt(5) / 1.5 = 1.49` instead of `sqrt(5 / 1.5) = 1.83`, a longer
  correlation (CHANGELOG, Changed). The solution component of `m` moves by at
  most `5e-5` (its error against the exact logistic solution drops from
  `1.2e-4` to `7.2e-5`), the second derivative by up to `0.066`; `sigma_sqr`
  changes by up to 15 %, the covariances by up to 70 % of their scale, and
  `log_likelihood` goes from 37.7 to 48.1. `joint_hidden_obs` uses
  `length_scale=1.0`, where both conventions agree, and is unchanged. Only the
  `matern_dynamic` arrays were re-recorded; all others are kept bit for bit (a
  fresh run differs from them by rounding, below `1e-13`).
