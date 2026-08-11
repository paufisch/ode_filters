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
    QuadratureCorrection,
    TaylorCorrection,
)
from .gaussian_filter import (
    FilterResult,
    gaussian_filter,
    gaussian_filter_adaptive,
    rts_smoother,
)
from .ipls import IPLSResult, affine_filter_scan, ipls_smoother
from .statistical_linearization import (
    SLRModel,
    check_arg_projection,
    cubature_rule,
    gauss_hermite_rule,
    quadrature_nodes,
    slr_linearize,
)

__all__ = [
    "Correction",
    "CorrectionResult",
    "FilterResult",
    "IPLSResult",
    "IteratedTaylorCorrection",
    "PController",
    "PIController",
    "QuadratureCorrection",
    "SLRModel",
    "StepSizeController",
    "TaylorCorrection",
    "affine_filter_scan",
    "check_arg_projection",
    "cubature_rule",
    "gauss_hermite_rule",
    "gaussian_filter",
    "gaussian_filter_adaptive",
    "ipls_smoother",
    "quadrature_nodes",
    "rts_smoother",
    "slr_linearize",
]
