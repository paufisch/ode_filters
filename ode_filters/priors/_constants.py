"""A cache for the priors' NumPy constants that hands out fresh copies."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import cast

import numpy as onp


def cached_constants[**P, T: tuple](builder: Callable[P, T]) -> Callable[P, T]:
    """``functools.lru_cache`` for a builder of NumPy constants, returning copies.

    JAX keys its device copy of a NumPy constant captured under ``jit`` by the
    array object, and from 0.9.2 on (still in 0.11.2) keeps it across a change
    of ``jax_enable_x64``: a cached array traced once in float32 came back as
    float32 in a later float64 trace, eight digits short, and one traced in
    float64 broke a later float32 trace. A copy per call is a new object, so
    every trace converts it afresh. The arrays have a few dozen entries.

    Args:
        builder: A function of hashable arguments returning a tuple; the NumPy
            arrays in the tuple are copied on every call.

    Returns:
        The cached builder.
    """
    cached = functools.lru_cache(builder)

    @functools.wraps(builder)
    def copies(*args: P.args, **kwargs: P.kwargs) -> T:
        value = cached(*args, **kwargs)
        return cast(
            T, tuple(v.copy() if isinstance(v, onp.ndarray) else v for v in value)
        )

    return copies
