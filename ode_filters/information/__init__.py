"""Information operators: what the filters condition on.

The ODE information (the residual of the ODE, plus conservation laws) and the
data observations. Every measurement model implements ``MeasurementModel``;
the optional protocols ``EK0Linearizable``, ``HasNoiseCovariance`` and
``NoiseSettable`` declare the members that single features call.
"""

from .blackbox import BlackBoxMeasurement, TransformedMeasurement
from .observations import Measurement, ObsModel, prepare_observations
from .ode_information import (
    BaseODEInformation,
    Conservation,
    ODEInformation,
    ODEInformationWithHidden,
    SecondOrderODEInformation,
    SecondOrderODEInformationWithHidden,
)
from .protocol import (
    EK0Linearizable,
    HasNoiseCovariance,
    MeasurementModel,
    NoiseSettable,
)

__all__ = [
    "BaseODEInformation",
    "BlackBoxMeasurement",
    "Conservation",
    "EK0Linearizable",
    "HasNoiseCovariance",
    "Measurement",
    "MeasurementModel",
    "NoiseSettable",
    "ODEInformation",
    "ODEInformationWithHidden",
    "ObsModel",
    "SecondOrderODEInformation",
    "SecondOrderODEInformationWithHidden",
    "TransformedMeasurement",
    "prepare_observations",
]
