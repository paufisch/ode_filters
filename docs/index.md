# ODE Filters Documentation

Welcome to the lightweight documentation hub for `ode-filters`. The goal is to provide a concise map of the codebase and point you toward hands-on resources.

## Project Layout

- `ode_filters/`: Core filtering, smoothing, and inference routines.
- `test/`: Pytest suite covering inference primitives and filter loops.
- `examples.ipynb`: End-to-end walkthrough of the probabilistic ODE solver pipeline.

## Tutorials

Start with the tutorial notebook to see the filtering workflow in action:

```bash
uv run jupyter notebook examples.ipynb
```

The notebook is kept clean by the `jupyter-nb-cleaner` pre-commit hook; please re-run all cells before committing updates.

## Further Reading

- Check inline docstrings throughout `ode_filters/` for API-level guidance.
- Review the tests for usage patterns that exercise edge cases.

As the project matures, this directory will grow with narrative guides and API references.
