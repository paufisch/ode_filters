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
    AdaptiveLoopResult,
    PController,
    PIController,
    StepSizeController,
    ekf1_sqr_adaptive_loop,
    ekf1_sqr_filter_step,
    ekf1_sqr_filter_step_preconditioned,
    ekf1_sqr_filter_step_preconditioned_sequential,
    ekf1_sqr_filter_step_preconditioned_sequential_scan,
    ekf1_sqr_filter_step_sequential,
    ekf1_sqr_filter_step_sequential_scan,
    ekf1_sqr_loop,
    ekf1_sqr_loop_dynamic,
    ekf1_sqr_loop_dynamic_scan,
    ekf1_sqr_loop_preconditioned,
    ekf1_sqr_loop_preconditioned_dynamic,
    ekf1_sqr_loop_preconditioned_dynamic_scan,
    ekf1_sqr_loop_preconditioned_sequential,
    ekf1_sqr_loop_preconditioned_sequential_scan,
    ekf1_sqr_loop_sequential,
    ekf1_sqr_loop_sequential_scan,
    rts_sqr_smoother_loop,
    rts_sqr_smoother_loop_preconditioned,
    rts_sqr_smoother_step,
    rts_sqr_smoother_step_preconditioned,
)
from .inference import sqr_inversion, sqr_marginalization
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
    "AdaptiveLoopResult",
    "JointPrior",
    "MaternPrior",
    "ODEInformation",
    "ODEconservation",
    "ODEconservationmeasurement",
    "ODEmeasurement",
    "ObsModel",
    "PController",
    "PIController",
    "PrecondIWP",
    "PrecondMaternPrior",
    "StepSizeController",
    "aggregate_sigma_sqr",
    "build_obs_at_time",
    "ekf1_sqr_adaptive_loop",
    "ekf1_sqr_filter_step",
    "ekf1_sqr_filter_step_preconditioned",
    "ekf1_sqr_filter_step_preconditioned_sequential",
    "ekf1_sqr_filter_step_preconditioned_sequential_scan",
    "ekf1_sqr_filter_step_sequential",
    "ekf1_sqr_filter_step_sequential_scan",
    "ekf1_sqr_loop",
    "ekf1_sqr_loop_dynamic",
    "ekf1_sqr_loop_dynamic_scan",
    "ekf1_sqr_loop_preconditioned",
    "ekf1_sqr_loop_preconditioned_dynamic",
    "ekf1_sqr_loop_preconditioned_dynamic_scan",
    "ekf1_sqr_loop_preconditioned_sequential",
    "ekf1_sqr_loop_preconditioned_sequential_scan",
    "ekf1_sqr_loop_sequential",
    "ekf1_sqr_loop_sequential_scan",
    "posthoc_mle_sigma_sqr",
    "prepare_observations",
    "quasi_mle_sigma_sqr",
    "quasi_mle_sigma_sqr_from_Q",
    "rescale_sqr",
    "rescale_sqr_seq",
    "rts_sqr_smoother_loop",
    "rts_sqr_smoother_loop_preconditioned",
    "rts_sqr_smoother_step",
    "rts_sqr_smoother_step_preconditioned",
    "sqr_inversion",
    "sqr_marginalization",
    "taylor_mode_initialization",
]
