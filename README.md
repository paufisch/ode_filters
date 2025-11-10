# ODE Filters

[![PyPI](https://img.shields.io/pypi/v/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![Python](https://img.shields.io/pypi/pyversions/ode-filters.svg)](https://pypi.org/project/ode-filters/)
[![CI](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/paufisch/ode_filters/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-latest-brightgreen)](https://paufisch.github.io/ode_filters/)
[![Coverage](https://codecov.io/gh/paufisch/ode_filters/branch/main/graph/badge.svg)](https://codecov.io/gh/paufisch/ode_filters)

The `ode-filters` package is an experimental implementation of basic ODE filtering and smoothing functionalities. Its main purpose is educational and research-oriented, providing a simple starting point for ODE filtering in Python. As such, it uses NumPy for the most part and JAX where autodiff is needed. Currently, only constant step sizes and time-invariant observation and dynamics are supported.

## Installation

1. Install the latest release from PyPI:

   ```
   pip install ode-filters
   ```

## Quickstart

- Run the full test suite:

  ```
  uv run pytest --cov=ode_filters --cov-report=term-missing
  ```
