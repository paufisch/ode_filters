"""The interface every measurement model offers the filters."""

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
