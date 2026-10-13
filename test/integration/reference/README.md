# Reference outputs

`reference_outputs.npz` holds solver outputs recorded with `generate.py` from the
scenarios in `scenarios.py`. `test_reference_outputs.py` compares the current
solvers against them, so a restructure of the internals that changes a result
fails loudly.

Regenerate only for an intended change of results, and record it below: which
scenarios changed, by how much, and why.

## Changes

- 2026-10-13: first recording (library state of release 0.7.3 plus the October
  correctness fixes).
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
