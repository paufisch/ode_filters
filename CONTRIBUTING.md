# Contributing to ode_filters

Contributions are very welcome! This guide covers the development setup and the
conventions enforced by CI. The authoritative style rules live in
[`CLAUDE.md`](CLAUDE.md); this file focuses on *how to get set up and submit a
change*.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for environment and
dependency management.

```bash
git clone https://github.com/paufisch/ode_filters.git
cd ode_filters
uv sync --group dev          # create the venv and install runtime + dev deps
uv run pre-commit install    # install the git hook so checks run on every commit
```

That's it -- `uv run <cmd>` always executes inside the managed environment, so
you never need to activate a venv manually.

## Branching workflow

CI runs on `development` and `main`. Open pull requests against `development`:

```
feature/your-change  ->  development  ->  main
```

Merging to `main` triggers an automated release (see [Releasing](#releasing-maintainers)).

## Making a change

1. Branch from `development`.
2. Make your change and add tests under `test/` (mirroring the package layout).
3. Run the checks locally (CI runs the same ones):

   ```bash
   uv run pre-commit run --all-files   # ruff (lint+format), pyright, nbstripout, ...
   uv run pytest                       # tests + coverage
   ```

4. Add a `CHANGELOG.md` entry under the top `... - Unreleased` section (see below).
5. Push and open a PR against `development`. Note whether the change is breaking.

The pull-request template prompts for each of these.

## Changelog and versioning

The split is deliberate: **contributors describe the change, maintainers assign
the version.**

- **Contributors:** add a line to `CHANGELOG.md` under the top, not-yet-released
  section (the `## [...] - Unreleased` heading), in the appropriate
  [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) category (`Added`,
  `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`), and flag it if it is
  breaking. **Do not edit `version` in `pyproject.toml`** -- the right bump
  depends on everything else queued for the release, so it is the maintainer's
  call.
- **Maintainers (at release time):** decide the version under the project's
  pre-1.0 rule (breaking changes -> minor bump, otherwise patch), stamp the date
  on the top section (`## [X.Y.Z] - YYYY-MM-DD`, with the version matching
  `pyproject.toml`), refresh the compare links at the bottom of the file, and
  bump `version` in `pyproject.toml` -- all in the release PR (see below).

## Releasing (maintainers)

Releases are automated: publishing to PyPI and creating the GitHub Release happen
on merge to `main`, gated on green CI. The manual surface is just the release PR.

1. On `development`, in one commit: bump `version` in `pyproject.toml`, and in
   `CHANGELOG.md` date the top section (`## [X.Y.Z] - YYYY-MM-DD`, version matching
   `pyproject.toml`) and update the compare links.
2. Open a PR `development -> main` and merge once CI is green.
3. That's it. On the `main` push, after CI succeeds, the release workflow
   (`.github/workflows/pypi-publish.yml`) reads the version, and if there is no
   `vX.Y.Z` tag yet it builds, smoke-tests, publishes to PyPI (trusted publishing),
   and creates the `vX.Y.Z` tag + GitHub Release with notes taken from the
   CHANGELOG. Ordinary pushes to `main` that do not change the version are skipped.

## Documentation

Docs are built with MkDocs; example notebooks in `docs/examples/` are executed
during the build, so they must run end-to-end.

```bash
uv run --group docs mkdocs serve         # live preview at http://localhost:8000
uv run --group docs mkdocs build --strict  # what CI checks
```

When you change the public API, verify the notebooks still execute:

```bash
uv run jupyter nbconvert --to notebook --execute \
  docs/examples/<notebook>.ipynb --output-dir /tmp/nb_test
```

## Type checking

Static checking uses **pyright**, which runs as a pre-commit hook (and therefore
in CI, since CI runs `pre-commit run --all-files`). The tree is pyright-clean, so
please keep it that way -- add type hints (`Array` for outputs, `ArrayLike` for
scalar/array inputs) rather than introducing new errors. Run it directly with
`uv run pyright` for a fast type-only check.

There is also opt-in *runtime* shape checking via
[jaxtyping](https://github.com/patrick-kidger/jaxtyping) + beartype. Enabling it
wraps every `ode_filters` function so that its annotations -- including jaxtyping
array types like `Float[Array, "n n"]` -- are validated on each call:

```bash
ODE_FILTERS_TYPECHECK=1 uv run pytest
```

It is off by default because a few negative-path tests intentionally pass
invalid inputs (to assert the library's own `ValueError`s), which the runtime
checker rejects earlier with a `TypeCheckError`. Use it locally to check new
code. Migrating annotations from plain `Array` to jaxtyping shapes -- and
reconciling those tests -- is an ongoing, incremental effort; new code is
encouraged to use jaxtyping types.

## Quick reference

| Task            | Command                                  |
| --------------- | ---------------------------------------- |
| Install hooks   | `uv run pre-commit install`              |
| Lint + format   | `uv run pre-commit run --all-files`      |
| Tests           | `uv run pytest`                          |
| Type check      | `uv run pyright`                         |
| Docs preview    | `uv run --group docs mkdocs serve`       |
