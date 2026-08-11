"""ODE Filters: Kalman filtering and smoothing for differential equations.

This package provides implementations of Extended Kalman Filters (EKF), Kalman
smoothers, and related utilities for inference in ordinary differential equation
(ODE) systems. The public API is organized into the following subpackages:

- ``filters``: Filtering and smoothing routines (fixed-step, adaptive, and
  preconditioned variants; ``PController`` / ``PIController`` step-size
  controllers).
- ``calibration``: Online and post-hoc diffusion (sigma^2) calibration.
- ``inference``: Square-root Gaussian inference utilities.
- ``measurement``: Measurement model helpers.
- ``priors``: Gaussian Markov process prior models.
"""

from importlib.metadata import PackageNotFoundError, version

from .calibration import (
    aggregate_sigma_sqr,
    posthoc_mle_sigma_sqr,
    quasi_mle_sigma_sqr,
    quasi_mle_sigma_sqr_from_Q,
    rescale_sqr,
    rescale_sqr_seq,
)
from .filters import (
    Correction,
    CorrectionResult,
    FilterResult,
    IPLSResult,
    IteratedTaylorCorrection,
    PController,
    PIController,
    QuadratureCorrection,
    SLRModel,
    StepSizeController,
    TaylorCorrection,
    gaussian_filter,
    gaussian_filter_adaptive,
    ipls_smoother,
    rts_smoother,
    slr_linearize,
)
from .inference import (
    AbstractParameter,
    InferenceProblem,
    ODEFilter,
    PositiveReal,
    Real,
    fit,
    marginal_loglik,
    sqr_inversion,
    sqr_marginalization,
    unwrap,
)
from .measurement import (
    ObsModel,
    ODEconservation,
    ODEInformation,
    prepare_observations,
)
from .priors import (
    IWP,
    JointPrior,
    MaternPrior,
    PrecondIWP,
    PrecondJointPrior,
    PrecondMaternPrior,
    taylor_mode_initialization,
)

try:
    __version__ = version("ode-filters")
except PackageNotFoundError:  # pragma: no cover - source checkout without metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "IWP",
    "AbstractParameter",
    "Correction",
    "CorrectionResult",
    "FilterResult",
    "IPLSResult",
    "InferenceProblem",
    "IteratedTaylorCorrection",
    "JointPrior",
    "MaternPrior",
    "ODEFilter",
    "ODEInformation",
    "ODEconservation",
    "ObsModel",
    "PController",
    "PIController",
    "PositiveReal",
    "PrecondIWP",
    "PrecondJointPrior",
    "PrecondMaternPrior",
    "QuadratureCorrection",
    "Real",
    "SLRModel",
    "StepSizeController",
    "TaylorCorrection",
    "__version__",
    "aggregate_sigma_sqr",
    "fit",
    "gaussian_filter",
    "gaussian_filter_adaptive",
    "ipls_smoother",
    "marginal_loglik",
    "posthoc_mle_sigma_sqr",
    "prepare_observations",
    "quasi_mle_sigma_sqr",
    "quasi_mle_sigma_sqr_from_Q",
    "rescale_sqr",
    "rescale_sqr_seq",
    "rts_smoother",
    "slr_linearize",
    "sqr_inversion",
    "sqr_marginalization",
    "taylor_mode_initialization",
    "unwrap",
]
