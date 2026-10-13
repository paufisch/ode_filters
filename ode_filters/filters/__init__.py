"""Filtering and smoothing routines for ODE models.

The public solver API is ``gaussian_filter`` / ``gaussian_filter_adaptive`` /
``rts_smoother`` (returning ``FilterResult``) and the adaptive step-size
controllers. The linearization strategies live in ``ode_filters.linearization``.
"""

from .adaptive_controller import PController, PIController, StepSizeController
from .gaussian_filter import (
    FilterResult,
    gaussian_filter,
    gaussian_filter_adaptive,
    rts_smoother,
)
from .ipls import IPLSResult, affine_filter_scan, ipls_smoother

__all__ = [
    "FilterResult",
    "IPLSResult",
    "PController",
    "PIController",
    "StepSizeController",
    "affine_filter_scan",
    "gaussian_filter",
    "gaussian_filter_adaptive",
    "ipls_smoother",
    "rts_smoother",
]
