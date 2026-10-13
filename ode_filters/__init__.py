"""ode_filters: probabilistic ODE filters for inference from data, in JAX.

Gaussian (Kalman) filtering and smoothing for ordinary differential equations,
in square-root form throughout. The public API is re-exported here; the
subpackages are organized by layer:

- ``priors``: Gauss-Markov priors over the solution and its derivatives.
- ``information``: what the filters condition on -- the ODE information
  (plus conservation laws) and data observations.
- ``linearization``: Taylor and statistical linearization (the corrections),
  and frozen linearizations along a trajectory.
- ``calibration``: online diffusion calibration policies and post-hoc rescaling.
- ``filters``: the filters, the smoother and the iterated smoother.
- ``inference``: marginal likelihoods and parameter fitting.
- ``gaussian``: the square-root Gaussian types and algebra.
"""

from importlib.metadata import PackageNotFoundError, version

from .calibration import (
    Calibration,
    DiagonalCalibration,
    DynamicCalibration,
    NoCalibration,
    aggregate_sigma_sqr,
    posthoc_mle_sigma_sqr,
    quasi_mle_sigma_sqr,
    quasi_mle_sigma_sqr_from_Q,
    rescale_sqr,
    rescale_sqr_seq,
)
from .filters import (
    FilterResult,
    IteratedSmootherResult,
    IterationState,
    PController,
    PIController,
    StepRecord,
    StepSizeController,
    affine_filter,
    gaussian_filter,
    gaussian_filter_adaptive,
    iterated_smoother,
    rts_smoother,
)
from .gaussian import AffineModel, Conditional, Gaussian
from .inference import (
    AbstractParameter,
    InferenceProblem,
    ODEFilter,
    PositiveReal,
    Real,
    fit,
    marginal_loglik,
    unwrap,
)
from .information import (
    BlackBoxMeasurement,
    Conservation,
    Measurement,
    ObsModel,
    ODEconservation,
    ODEInformation,
    ODEInformationWithHidden,
    SecondOrderODEInformation,
    SecondOrderODEInformationWithHidden,
    TransformedMeasurement,
    prepare_observations,
)
from .linearization import (
    AffineSurrogate,
    Correction,
    IteratedTaylorCorrection,
    QuadratureCorrection,
    SLRModel,
    TaylorCorrection,
    slr_linearize,
    slr_surrogate,
    taylor_surrogate,
)
from .priors import (
    IWP,
    BasePrior,
    IOUPPrior,
    JointPrior,
    MaternPrior,
    PrecondIWP,
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
    "AffineModel",
    "AffineSurrogate",
    "BasePrior",
    "BlackBoxMeasurement",
    "Calibration",
    "Conditional",
    "Conservation",
    "Correction",
    "DiagonalCalibration",
    "DynamicCalibration",
    "FilterResult",
    "Gaussian",
    "IOUPPrior",
    "InferenceProblem",
    "IteratedSmootherResult",
    "IteratedTaylorCorrection",
    "IterationState",
    "JointPrior",
    "MaternPrior",
    "Measurement",
    "NoCalibration",
    "ODEFilter",
    "ODEInformation",
    "ODEInformationWithHidden",
    "ODEconservation",
    "ObsModel",
    "PController",
    "PIController",
    "PositiveReal",
    "PrecondIWP",
    "PrecondMaternPrior",
    "QuadratureCorrection",
    "Real",
    "SLRModel",
    "SecondOrderODEInformation",
    "SecondOrderODEInformationWithHidden",
    "StepRecord",
    "StepSizeController",
    "TaylorCorrection",
    "TransformedMeasurement",
    "__version__",
    "affine_filter",
    "aggregate_sigma_sqr",
    "fit",
    "gaussian_filter",
    "gaussian_filter_adaptive",
    "iterated_smoother",
    "marginal_loglik",
    "posthoc_mle_sigma_sqr",
    "prepare_observations",
    "quasi_mle_sigma_sqr",
    "quasi_mle_sigma_sqr_from_Q",
    "rescale_sqr",
    "rescale_sqr_seq",
    "rts_smoother",
    "slr_linearize",
    "slr_surrogate",
    "taylor_mode_initialization",
    "taylor_surrogate",
    "unwrap",
]
