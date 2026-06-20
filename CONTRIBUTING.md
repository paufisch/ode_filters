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

## Making a change

1. Branch from `development`.
2. Make your change and add tests under `test/` (mirroring the package layout).
3. Run the checks locally (CI runs the same ones):

   ```bash
   uv run pre-commit run --all-files   # ruff lint + format, nbstripout, ...
   uv run pytest                       # tests + coverage
   uv run pyright                      # static type checking (advisory)
   ```

4. Push and open a PR. Note in the description whether the change is breaking.

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

Static checking uses **pyright** (`uv run pyright`), currently advisory in CI
while an existing baseline of findings is burned down -- please don't add new
errors.

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
