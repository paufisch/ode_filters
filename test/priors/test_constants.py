"""Tests for ``ode_filters.priors._constants``: cached constants as fresh copies."""

from __future__ import annotations

import numpy as onp
import pytest

from ode_filters.priors._constants import cached_constants
from ode_filters.priors._lti import _leggauss
from ode_filters.priors.iwp import _iwp_constants
from ode_filters.priors.matern import _matern_constants


def test_a_cached_builder_runs_once_and_returns_new_arrays():
    calls = []

    @cached_constants
    def build(n: int) -> tuple[onp.ndarray, float]:
        calls.append(n)
        return onp.arange(float(n)), 0.5

    first, scale = build(3)
    first[0] = 99.0  # a caller mutating its copy leaves the cache alone
    second, _ = build(3)
    assert calls == [3]
    assert scale == 0.5
    assert second is not first
    assert onp.array_equal(second, onp.arange(3.0))


@pytest.mark.parametrize(
    "builder", [_iwp_constants, _matern_constants, _leggauss], ids=lambda f: f.__name__
)
def test_the_priors_constants_are_new_objects_on_every_call(builder):
    """JAX keys a captured constant's device copy by the array object."""
    first, second = builder(3), builder(3)
    for a, b in zip(first, second, strict=True):
        if isinstance(a, onp.ndarray):
            assert a is not b
            assert onp.array_equal(a, b)
        else:
            assert a == b
