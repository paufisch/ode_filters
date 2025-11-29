# Contributing to ode_filters

## Code Style

- Follow **PEP 8** (enforced by Ruff)
- Run `uv run pre-commit run --all-files` before committing
- Lowercase module names with underscores (`gmp_priors.py`)
- PascalCase for classes (`BasePrior`, `ODEInformation`)
- Prefix private attributes with `_` (e.g., `_R`, `_jacobian_vf`)

## Type Hints

Always use type hints. Use `Array` from JAX, `ArrayLike` for flexible inputs:

```python
def A(self, h: float) -> Array:
```

## Imports

```python
# stdlib
from collections.abc import Callable

# third-party
import jax.numpy as np
from jax import Array

# local (relative)
from .gmp_priors import BasePrior
```

## Docstrings

Google-style, ASCII only (no Unicode math symbols):

```python
def function(arg: int) -> str:
    """Short description.

    Args:
        arg: Description.

    Returns:
        Description.
    """
```

## Testing

```bash
uv run pytest              # run tests
uv run pytest --cov        # with coverage
```

Tests go in `test/` mirroring package structure. Use `hypothesis` for property-based tests.

## Before Committing

```bash
uv run pre-commit run --all-files
uv run pytest
```

## API Design

- Prefer `E0`/`E1` matrices over `q`/`d` integers
- Use keyword-only arguments: `def f(x, *, t):`
- Export public API through `__init__.py`
- JAX arrays are immutable; use `.at[].set()` for updates
