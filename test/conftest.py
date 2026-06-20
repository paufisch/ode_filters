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

# Enable 64-bit precision for all tests
# This allows tests to use float64 without truncation warnings
# See: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
jax.config.update("jax_enable_x64", True)
