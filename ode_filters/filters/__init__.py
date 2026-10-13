"""Filters and smoothers.

- ``gaussian_filter``: fixed grid (uniform or not), online linearization.
- ``gaussian_filter_adaptive``: adaptive steps, solution at given save times.
- ``affine_filter``: a frozen linearization (``AffineSurrogate``).
- ``rts_smoother``: smoothing of any of their results; ``sample_posterior``
  draws joint posterior paths and ``interpolate`` gives marginals between nodes.
- ``iterated_smoother``: the IEKS / IPLS, iterating linearization and smoothing.

All return named results in the original state coordinates. The steps they are
built from are in ``ode_filters.filters.kernel``.
"""

from .adaptive import gaussian_filter_adaptive
from .adaptive_controller import PController, PIController, StepSizeController
from .fixed_grid import affine_filter, gaussian_filter
from .iterated import IteratedSmootherResult, IterationState, iterated_smoother
from .result import FilterResult, StepRecord
from .smoothing import interpolate, rts_smoother, sample_posterior

__all__ = [
    "FilterResult",
    "IteratedSmootherResult",
    "IterationState",
    "PController",
    "PIController",
    "StepRecord",
    "StepSizeController",
    "affine_filter",
    "gaussian_filter",
    "gaussian_filter_adaptive",
    "interpolate",
    "iterated_smoother",
    "rts_smoother",
    "sample_posterior",
]
