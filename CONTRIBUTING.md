# Contributing

Thank you for your interest in improving `ode-filters`. This guide outlines the recommended development workflow.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) for dependency management
- Python 3.11 or newer (the project targets bleeding-edge Python features)
- Git and a GitHub account for submitting pull requests

## Environment Setup

1. Clone the repository and move into the project root.
2. Install the development environment:

   ```bash
   uv sync --group dev
   ```

3. Activate pre-commit hooks (runs formatters, linters, and notebook cleaners automatically):

   ```bash
   uv run pre-commit install
   ```

## Development Workflow

1. Create a feature branch off `development`:

   ```bash
   git checkout -b feature/<short-description>
   ```

2. Make your changes and keep commits focused on a single concern.

3. Run quality checks before pushing:

   ```bash
   uv run pre-commit run --all-files
   uv run pytest --cov=ode_filters --cov-report=term-missing
   ```

4. Push your branch and open a pull request. Describe the change, testing performed, and any follow-up work.

## Documentation and Tutorials

- Add or update conceptual docs in `docs/` when introducing new features or behaviour.
- The primary tutorial is the `examples.ipynb` notebook. Keep it executable and clean; the `jupyter-nb-cleaner` pre-commit hook will strip metadata automatically.

## Release Notes and Versioning

Release versions follow semantic versioning. When preparing a release:

1. Update `pyproject.toml` with the new version.
2. Document noteworthy changes in `CHANGELOG.md` (create the file if it does not yet exist).
3. Tag the release with `git tag vX.Y.Z` and push the tag.
4. Coordinate with the maintainer to publish to PyPI via the CI release workflow (coming soon).

For questions or discussion, open an issue or start a thread in the project discussions.
