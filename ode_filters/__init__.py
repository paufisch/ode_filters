"""ODE Filters: Kalman filtering and smoothing for differential equations.

This package provides implementations of Extended Kalman Filters (EKF), Kalman
smoothers, and related utilities for inference in ordinary differential equation
(ODE) systems. The public API is organized into the following subpackages:

- ``filters``: Filtering and smoothing routines.
- ``inference``: Square-root Gaussian inference utilities.
- ``measurement``: Measurement model helpers.
- ``priors``: Gaussian Markov process prior models.
"""

from .filters import (
    ekf1_sqr_filter_step,
    ekf1_sqr_filter_step_preconditioned,
    ekf1_sqr_loop,
    ekf1_sqr_loop_preconditioned,
    rts_sqr_smoother_loop,
    rts_sqr_smoother_loop_preconditioned,
    rts_sqr_smoother_step,
    rts_sqr_smoother_step_preconditioned,
)
from .inference import sqr_inversion, sqr_marginalization
from .measurement import (
    ODEconservation,
    ODEconservationmeasurement,
    ODEInformation,
    ODEmeasurement,
)
from .priors import IWP, JointPrior, MaternPrior, PrecondIWP, taylor_mode_initialization

__all__ = [
    "IWP",
    "JointPrior",
    "MaternPrior",
    "ODEInformation",
    "ODEconservation",
    "ODEconservationmeasurement",
    "ODEmeasurement",
    "PrecondIWP",
    "ekf1_sqr_filter_step",
    "ekf1_sqr_filter_step_preconditioned",
    "ekf1_sqr_loop",
    "ekf1_sqr_loop_preconditioned",
    "rts_sqr_smoother_loop",
    "rts_sqr_smoother_loop_preconditioned",
    "rts_sqr_smoother_step",
    "rts_sqr_smoother_step_preconditioned",
    "sqr_inversion",
    "sqr_marginalization",
    "taylor_mode_initialization",
]
