"""The interface every measurement model offers the filters, and optional members.

``MeasurementModel`` is the interface that every solver requires. Three further
members are optional. One feature calls each of them, and each has its own
protocol:

- ``EK0Linearizable``: ``linearize_ek0``, for ``TaylorCorrection(order=0)``.
- ``HasNoiseCovariance``: ``R``, for ``TransformedMeasurement.R``.
- ``NoiseSettable``: ``with_noise``, for ``TransformedMeasurement.with_noise``.

The optional protocols do not extend ``MeasurementModel``. An ``isinstance``
check against one of them therefore tests only the member that its feature
calls. ``HasLHSSelector`` (``E_lhs``) serves only the deprecated
``DiagonalCalibration(jacobian="ek0")`` and is internal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from jax import Array
from jax.typing import ArrayLike


@runtime_checkable
class MeasurementModel(Protocol):
    """A residual ``g(state, t)`` that the filters observe as zero.

    The ODE-information models, ``BlackBoxMeasurement`` and
    ``TransformedMeasurement`` implement it; so can a custom model. The first
    ``ode_dim`` rows of ``g`` are the ODE residual (they drive the diffusion
    calibration); further rows, e.g. conservation laws, only update the state.

    A feature that calls one of the optional members (``EK0Linearizable``,
    ``HasNoiseCovariance``, ``NoiseSettable``) checks for it and raises an
    error that names the member if the model lacks it.
    """

    @property
    def ode_dim(self) -> int:
        """Number of leading rows of ``g`` that are ODE residual."""
        ...

    def g(self, state: Array, *, t: ArrayLike) -> Array:
        """The residual at ``state``, shape ``[k]``."""
        ...

    def jacobian_g(self, state: Array, *, t: ArrayLike) -> Array:
        """The Jacobian of the residual, shape ``[k, D]``."""
        ...

    def linearize(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
        """``(H, c)`` with ``H x + c`` the first-order expansion of ``g`` at ``state``."""
        ...

    def noise_sqr(self, *, t: ArrayLike) -> Array:
        """Upper-triangular square root of the residual's noise covariance."""
        ...


@runtime_checkable
class HasLHSSelector(Protocol):
    """A model that declares the selector ``E_lhs`` of its ODE's left-hand side.

    The ODE rows of the residual are ``E_lhs @ state - f(...)``, so ``E_lhs``
    is their Jacobian with the vector field held constant (the EK0 Jacobian).
    The deprecated ``DiagonalCalibration(jacobian="ek0")`` estimates the
    diffusion with it. The ODE-information models implement it. Internal: it
    goes with that option in 0.9.
    """

    @property
    def E_lhs(self) -> Array:
        """Selector of the ODE's left-hand side, shape ``[ode_dim, D]``."""
        ...


@runtime_checkable
class EK0Linearizable(Protocol):
    """A model with a zeroth-order (EK0) linearization.

    ``TaylorCorrection(order=0)`` linearizes with ``linearize_ek0``. The
    ODE-information models implement it.
    """

    def linearize_ek0(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
        """``(H, c)`` without the vector-field Jacobian in the ODE rows of ``H``.

        ``H state + c`` equals the residual at ``state``.
        """
        ...


@runtime_checkable
class HasNoiseCovariance(Protocol):
    """A model that stores its noise covariance as ``R``.

    ``TransformedMeasurement.R`` returns the ``R`` of its base model. The
    ODE-information models, ``BlackBoxMeasurement`` and
    ``TransformedMeasurement`` implement it.
    """

    @property
    def R(self) -> Array:
        """Noise covariance of the rows that carry noise.

        These are the ODE rows of an ODE-information model and all rows of a
        ``BlackBoxMeasurement``.
        """
        ...


@runtime_checkable
class NoiseSettable(Protocol):
    """A model that returns a copy of itself with a new noise.

    ``TransformedMeasurement.with_noise`` calls ``with_noise`` of its base
    model. The ODE-information models, ``BlackBoxMeasurement`` and
    ``TransformedMeasurement`` implement it.
    """

    def with_noise(self, noise: ArrayLike, /) -> MeasurementModel:
        """A copy with the noise ``noise``.

        ``noise`` is a variance, per-row variances or a covariance of the rows
        that carry noise (see ``HasNoiseCovariance``).
        """
        ...
