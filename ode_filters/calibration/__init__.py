"""Online and post-hoc diffusion (sigma^2) calibration for probabilistic ODE filters.

Three calibration recipes are provided, all consuming the predicted-observation
marginal ``(m_z, P_z_sqr)`` already returned by the EKF step:

- :func:`quasi_mle_sigma_sqr` -- per-step quasi-MLE (Bosch et al. 2021).
- :func:`posthoc_mle_sigma_sqr` -- closed-form MLE over a full trajectory.
- :func:`aggregate_sigma_sqr` -- aggregate a per-step trace into a global
  estimate or its running cumulative mean.

Covariance rescaling helpers live in :mod:`ode_filters.calibration.rescale`.
"""

from .rescale import rescale_sqr, rescale_sqr_seq
from .sigma import aggregate_sigma_sqr, posthoc_mle_sigma_sqr, quasi_mle_sigma_sqr

__all__ = [
    "aggregate_sigma_sqr",
    "posthoc_mle_sigma_sqr",
    "quasi_mle_sigma_sqr",
    "rescale_sqr",
    "rescale_sqr_seq",
]
