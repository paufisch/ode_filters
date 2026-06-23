"""Filtering and smoothing routines for ODE models.

The public solver API is the consolidated :func:`gaussian_filter` /
:func:`gaussian_filter_adaptive` / :func:`rts_smoother` (returning
:class:`FilterResult`), plus the pluggable :class:`Correction` strategies and the
adaptive step-size controllers. The historical ``sqr_loop*`` / ``*_step*``
matrix is now an implementation detail of those entry points; import it from the
``ode_filters.filters.ode_filter_loop`` / ``ode_filter_step`` /
``ode_filter_adaptive`` submodules if you need the low-level variants directly.
"""

from .adaptive_controller import PController, PIController, StepSizeController
from .correction import (
    Correction,
    CorrectionResult,
    IteratedTaylorCorrection,
    TaylorCorrection,
)
from .gaussian_filter import (
    FilterResult,
    gaussian_filter,
    gaussian_filter_adaptive,
    rts_smoother,
)

__all__ = [
    "Correction",
    "CorrectionResult",
    "FilterResult",
    "IteratedTaylorCorrection",
    "PController",
    "PIController",
    "StepSizeController",
    "TaylorCorrection",
    "gaussian_filter",
    "gaussian_filter_adaptive",
    "rts_smoother",
]
