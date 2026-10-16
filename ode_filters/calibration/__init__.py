"""Diffusion (sigma^2) calibration for probabilistic ODE filters.

Online: the policies ``NoCalibration``, ``DynamicCalibration`` and
``DiagonalCalibration`` decide how each filter step estimates and applies
``sigma^2`` (see ``ode_filters.calibration.policies``).

Post hoc, from a finished run's innovations:

- ``quasi_mle_sigma_sqr``: the per-step quasi-MLE (Bosch et al. 2021).
- ``posthoc_mle_sigma_sqr``: the closed-form MLE over a full trajectory.
- ``aggregate_sigma_sqr``: aggregates a per-step trace into a global
  estimate or its running cumulative mean.
- ``rescale_sqr`` / ``rescale_sqr_seq``: apply an estimate to stored
  covariances.
"""

from .policies import (
    Calibration,
    CalibrationMode,
    DiagonalCalibration,
    DynamicCalibration,
    NoCalibration,
    as_calibration,
)
from .rescale import rescale_sqr, rescale_sqr_seq
from .sigma import (
    aggregate_sigma_sqr,
    posthoc_mle_sigma_sqr,
    quasi_mle_sigma_sqr,
    quasi_mle_sigma_sqr_from_Q,
)

__all__ = [
    "Calibration",
    "CalibrationMode",
    "DiagonalCalibration",
    "DynamicCalibration",
    "NoCalibration",
    "aggregate_sigma_sqr",
    "as_calibration",
    "posthoc_mle_sigma_sqr",
    "quasi_mle_sigma_sqr",
    "quasi_mle_sigma_sqr_from_Q",
    "rescale_sqr",
    "rescale_sqr_seq",
]
