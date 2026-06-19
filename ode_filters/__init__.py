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
    IteratedTaylorCorrection,
    PController,
    PIController,
    StepSizeController,
    TaylorCorrection,
    gaussian_filter,
    gaussian_filter_adaptive,
    rts_smoother,
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
    ODEconservationmeasurement,
    ODEInformation,
    ODEmeasurement,
    build_obs_at_time,
    prepare_observations,
)
from .priors import (
    IWP,
    JointPrior,
    MaternPrior,
    PrecondIWP,
    PrecondMaternPrior,
    taylor_mode_initialization,
)

__all__ = [
    "IWP",
    "AbstractParameter",
    "Correction",
    "CorrectionResult",
    "FilterResult",
    "InferenceProblem",
    "IteratedTaylorCorrection",
    "JointPrior",
    "MaternPrior",
    "ODEFilter",
    "ODEInformation",
    "ODEconservation",
    "ODEconservationmeasurement",
    "ODEmeasurement",
    "ObsModel",
    "PController",
    "PIController",
    "PositiveReal",
    "PrecondIWP",
    "PrecondMaternPrior",
    "Real",
    "StepSizeController",
    "TaylorCorrection",
    "aggregate_sigma_sqr",
    "build_obs_at_time",
    "fit",
    "gaussian_filter",
    "gaussian_filter_adaptive",
    "marginal_loglik",
    "posthoc_mle_sigma_sqr",
    "prepare_observations",
    "quasi_mle_sigma_sqr",
    "quasi_mle_sigma_sqr_from_Q",
    "rescale_sqr",
    "rescale_sqr_seq",
    "rts_smoother",
    "sqr_inversion",
    "sqr_marginalization",
    "taylor_mode_initialization",
    "unwrap",
]
