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
