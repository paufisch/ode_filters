<!--
Thanks for contributing to ode_filters! PRs target `development` (not `main`).
See CONTRIBUTING.md for the full workflow. Fill in the sections below.
-->

## Summary

<!-- What does this change do, and why? -->

## Breaking change?

<!-- Pick one. Pre-1.0, breaking changes ship in a minor bump, non-breaking in a
     patch bump -- the maintainer decides the actual version at release time. -->

- [ ] No -- backward compatible.
- [ ] Yes -- describe what breaks and how users should migrate:

## Checklist

- [ ] Tests added/updated under `test/` (mirroring the package layout).
- [ ] `uv run pre-commit run --all-files` passes (ruff lint + format, pyright).
- [ ] `uv run pytest` passes (suite is gated at 100% line coverage).
- [ ] If public API changed: the `docs/examples/` notebooks still execute.
- [ ] **Added a `CHANGELOG.md` entry under the top `... - Unreleased` section**
      in the right category (Added / Changed / Deprecated / Removed / Fixed /
      Security), and marked it if it is breaking. Do **not** bump the version in
      `pyproject.toml` -- the maintainer assigns the version at release time.
