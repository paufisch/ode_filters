"""Linearization of the information operators: Taylor and statistical."""

from .correction import (
    Correction,
    CorrectionResult,
    IteratedTaylorCorrection,
    QuadratureCorrection,
    TaylorCorrection,
)
from .statistical import (
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
    "IteratedTaylorCorrection",
    "QuadratureCorrection",
    "SLRModel",
    "TaylorCorrection",
    "check_arg_projection",
    "cubature_rule",
    "gauss_hermite_rule",
    "quadrature_nodes",
    "slr_linearize",
]
