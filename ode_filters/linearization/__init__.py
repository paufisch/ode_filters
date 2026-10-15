"""Linearization of the information operators: Taylor and statistical."""

from .correction import (
    Correction,
    IteratedTaylorCorrection,
    QuadratureCorrection,
    TaylorCorrection,
    taylor_linearize,
)
from .statistical import SLRModel, check_arg_projection, slr_linearize
from .surrogate import AffineSurrogate, slr_surrogate, taylor_surrogate

__all__ = [
    "AffineSurrogate",
    "Correction",
    "IteratedTaylorCorrection",
    "QuadratureCorrection",
    "SLRModel",
    "TaylorCorrection",
    "check_arg_projection",
    "slr_linearize",
    "slr_surrogate",
    "taylor_linearize",
    "taylor_surrogate",
]
