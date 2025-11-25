"""Pytest configuration for ode_filters tests."""

import jax

# Enable 64-bit precision for all tests
# This allows tests to use float64 without truncation warnings
# See: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision
jax.config.update("jax_enable_x64", True)
