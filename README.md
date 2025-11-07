# ODE Filters

Probabilistic filtering and smoothing algorithms for ordinary differential equation solvers.

## Overview

The package implements square-root Gaussian inference routines along with filter and smoother loops for probabilistic ODE solvers. It targets research-grade experimentation while staying close to practical applications.

## Installation

1. Install [uv](https://github.com/astral-sh/uv) if you have not already:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync the project dependencies (this installs the package in editable mode and pulls in development extras):

   ```bash
   uv sync --group dev
   ```

This creates a virtual environment under `.venv/` and makes the `uv run` command available for executing project tools.

## Quickstart

- Run the full test suite:

  ```bash
  uv run pytest --cov=ode_filters --cov-report=term-missing
  ```

- Execute a specific test module (for example, the logistic consistency checks):

  ```bash
  uv run pytest test/test_filter_loop/test_preconditioned_logistic_consistency.py -k consistency
  ```

- Launch the tutorial notebook showcased in the documentation:

  ```bash
  uv run jupyter notebook examples.ipynb
  ```

## Documentation

Lightweight documentation lives in `docs/`. Start with `docs/index.md` for an outline of available modules and concepts. The primary tutorial is the `examples.ipynb` notebook, which walks through the filtering workflow step by step.

## Contributing

We welcome issues and pull requests. Please read `CONTRIBUTING.md` for guidance on local setup, coding standards, and pre-commit hooks. Running `uv run pre-commit run --all-files` before opening a pull request keeps the linters and notebook cleaner in sync with CI.
