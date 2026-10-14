"""Pytest configuration for ode_filters tests."""

import os

# Opt-in runtime type/shape checking via jaxtyping + beartype.
#
# Set ODE_FILTERS_TYPECHECK=1 to wrap every function in `ode_filters` with a
# beartype runtime checker, so jaxtyping array annotations (e.g.
# ``Float[Array, "n n"]``) are validated on each call. It is OFF by default so
# the suite stays green while annotations are migrated incrementally; see
# CONTRIBUTING.md ("Type checking"). The import hook MUST run before any
# `ode_filters` import, hence it lives at the top of conftest.
if os.environ.get("ODE_FILTERS_TYPECHECK") == "1":
    from jaxtyping import install_import_hook

    install_import_hook("ode_filters", "beartype.beartype")

import jax
import pytest
from hypothesis import settings

# Hypothesis profiles: "dev" (default) keeps local runs quick, "ci" runs more
# examples. Select with HYPOTHESIS_PROFILE=ci. Deadlines are off because the
# first call of a jitted function includes its compilation.
settings.register_profile("dev", max_examples=25, deadline=None)
settings.register_profile("ci", max_examples=100, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

# Enable 64-bit precision for all tests
# This allows tests to use float64 without truncation warnings
# See: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
jax.config.update("jax_enable_x64", True)


@pytest.fixture(autouse=True)
def _standard_promotion_for_marked_tests(request):
    """Run tests marked ``standard_promotion`` under standard dtype promotion.

    optax, NumPyro and BlackJAX fail under JAX_NUMPY_DTYPE_PROMOTION=strict in
    their own code (optax's Adam raises a Python float to an int32 step count),
    so the tests that drive them keep the default promotion rules there.
    """
    if request.node.get_closest_marker("standard_promotion") is None:
        yield
        return
    with jax.numpy_dtype_promotion("standard"):
        yield


def pytest_collection_modifyitems(config, items):
    """Under runtime type checking, skip the tests of wrongly typed input.

    The checker rejects such input before the library's own validation runs,
    so these tests cannot see the error they check for.
    """
    if os.environ.get("ODE_FILTERS_TYPECHECK") != "1":
        return
    skip = pytest.mark.skip(reason="the runtime type checker rejects the input first")
    for item in items:
        if "wrong_input_types" in item.keywords:
            item.add_marker(skip)
