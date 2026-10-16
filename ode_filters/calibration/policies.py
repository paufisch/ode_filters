"""Online diffusion calibration: how each filter step sets ``sigma^2``.

The prior's process noise is ``sigma^2 Q(h)`` for an unknown diffusion
``sigma^2``. Before each prediction a calibration policy estimates it from the
ODE residual at the predicted mean, the per-step quasi-MLE of Bosch, Tronarp &
Hennig (2021, Eq. 32),

    sigma_hat^2 = r.T (H Q(h) H.T)^-1 r / d,

and decides whether the step propagates with it:

- ``NoCalibration`` propagates ``Q(h)`` unscaled (a fixed diffusion, carried
  by the prior's ``Xi``). The estimate is still reported, for post-hoc
  rescaling.
- ``DynamicCalibration`` scales the step's process noise by the scalar
  estimate.
- ``DiagonalCalibration`` uses one estimate per solution component,
  ``r_i^2 / (H Q H.T)_ii``. Its option ``jacobian="ek0"`` (shorthand
  ``"diagonal_ekf0"``) is deprecated and will be removed in 0.9.

Only the ODE-residual rows (the first ``ode_dim`` rows of the information)
drive the estimate; conservation rows and data update the posterior but live
on other scales. For joint priors only the state block of the noise is scaled.

The strings ``"none"``, ``"dynamic"``, ``"diagonal"`` and ``"diagonal_ekf0"``
(deprecated) are accepted wherever a policy is, as shorthands for the default
policies.
"""

from __future__ import annotations

import abc
import warnings
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
from jax import Array

from .sigma import quasi_mle_sigma_sqr_from_Q

CalibrationMode = Literal["none", "dynamic", "diagonal", "diagonal_ekf0"]

#: Raised as a ``FutureWarning`` by ``DiagonalCalibration(jacobian="ek0")``.
EK0_JACOBIAN_DEPRECATION = (
    'DiagonalCalibration(jacobian="ek0") and its shorthand "diagonal_ekf0" are '
    'deprecated and will be removed in ode-filters 0.9. Use "diagonal" '
    "(DiagonalCalibration()): with an EK0 correction it is identical, and with "
    "EK1 it calibrates to within a few percent."
)


class Calibration(eqx.Module):
    """Base class of the diffusion-calibration policies."""

    @abc.abstractmethod
    def calibrate(
        self,
        residual: Array,
        H: Array,
        Q_sqr: Array,
        *,
        H_ek0: Callable[[], Array],
    ) -> tuple[Array, Array]:
        """Estimate ``sigma^2`` for one step and choose the diffusion to apply.

        Args:
            residual: ODE residual at the predicted mean, shape ``[d]``.
            H: Its Jacobian (the linearization's ODE rows), shape ``[d, D]``.
            Q_sqr: Square-root process noise of the step, shape ``[D, D]``.
            H_ek0: Returns the ODE's left-hand-side selector in the same
                coordinates as ``H``; only called by policies that need it.

        Returns:
            ``(sigma_sqr, diffusion)``: the estimate (scalar, or one per
            component) and the diffusion the step propagates with, which the
            step applies to the prior's state block
            (``prior.apply_state_sigma_sqr``).
        """
        raise NotImplementedError

    def validate(self, prior) -> None:
        """Reject a prior this policy cannot calibrate (called once per solve)."""


def _scalar_estimate(residual: Array, H: Array, Q_sqr: Array) -> Array:
    return quasi_mle_sigma_sqr_from_Q(residual, H, Q_sqr)


class NoCalibration(Calibration):
    """Fixed diffusion: propagate the prior's noise unscaled.

    The scalar quasi-MLE is still computed and reported as ``sigma_sqr``, so a
    run can be rescaled post hoc (see ``posthoc_mle_sigma_sqr``).
    """

    def calibrate(self, residual, H, Q_sqr, *, H_ek0):
        """Return the scalar quasi-MLE and a unit diffusion."""
        estimate = _scalar_estimate(residual, H, Q_sqr)
        return estimate, np.ones_like(estimate)


class DynamicCalibration(Calibration):
    """Scalar per-step diffusion, baked into the step's process noise.

    Attributes:
        min_sigma_sqr: Lower bound on the estimate. A positive floor keeps a
            residual that vanishes identically (a system at rest) from
            collapsing the diffusion to zero.
    """

    min_sigma_sqr: float = eqx.field(static=True, default=0.0)

    def calibrate(self, residual, H, Q_sqr, *, H_ek0):
        """Return the floored scalar quasi-MLE, also as the diffusion."""
        sigma_sqr = np.maximum(_scalar_estimate(residual, H, Q_sqr), self.min_sigma_sqr)
        return sigma_sqr, sigma_sqr


class DiagonalCalibration(Calibration):
    """One diffusion per solution component.

    ``sigma_i^2 = r_i^2 / (H Q H.T)_ii``. Requires the state block of ``Xi`` to
    be diagonal; checked when a solve starts.

    Attributes:
        jacobian: ``"ek1"`` (default) takes ``H`` from the linearization; for
            a decoupled system this is the scalar estimate of each component,
            for a coupled Jacobian the diagonal is a heuristic. ``"ek0"`` takes
            the ODE's left-hand-side selector (``E1`` for ``x' = f``, ``E2`` for
            ``x'' = f``). It is deprecated and will be removed in 0.9: with an
            EK0 correction it equals ``"ek1"``, and with EK1 both calibrate to
            within a few percent on the problems we measured.
        min_sigma_sqr: Lower bound on each component's estimate.
    """

    jacobian: Literal["ek1", "ek0"] = eqx.field(static=True, default="ek1")
    min_sigma_sqr: float = eqx.field(static=True, default=0.0)

    def __check_init__(self):
        if self.jacobian not in ("ek1", "ek0"):
            raise ValueError(f'jacobian must be "ek1" or "ek0", got {self.jacobian!r}.')
        if self.jacobian == "ek0":
            warnings.warn(EK0_JACOBIAN_DEPRECATION, FutureWarning, stacklevel=2)

    def calibrate(self, residual, H, Q_sqr, *, H_ek0):
        """Return the floored per-component estimate, also as the diffusion."""
        H_calib = H_ek0() if self.jacobian == "ek0" else H
        # diag(H Q H.T)_i = ||(H @ Q_sqr.T)[i]||^2, since Q = Q_sqr.T @ Q_sqr.
        HQ_sqr = H_calib @ Q_sqr.T
        denom = np.sum(HQ_sqr * HQ_sqr, axis=1)
        # A degenerate row (denom == 0) would give NaN; the double ``where``
        # floors it on both the forward and the reverse pass.
        positive = denom > 0.0
        safe_denom = np.where(positive, denom, 1.0)
        estimate = np.where(positive, residual**2 / safe_denom, self.min_sigma_sqr)
        sigma_sqr = np.maximum(estimate, self.min_sigma_sqr)
        return sigma_sqr, sigma_sqr

    def validate(self, prior) -> None:
        """Reject a prior whose state block of ``Xi`` is not diagonal."""
        try:
            xi = onp.asarray(prior.xi_state)
            diagonal = bool(onp.allclose(xi - onp.diag(onp.diag(xi)), 0.0))
        except jax.errors.TracerArrayConversionError:
            # Under jit Xi is traced: use what the prior recorded when it was
            # built. None (built from a traced Xi, or a custom prior) is trusted.
            diagonal = getattr(prior, "xi_state_diagonal", None)
        if diagonal is False:
            raise ValueError(
                "DiagonalCalibration requires the state block of Xi to be diagonal "
                "(prior.xi_state; for a joint prior, the state prior's Xi)."
            )


_ALIASES: dict[str, Callable[[], Calibration]] = {
    "none": NoCalibration,
    "dynamic": DynamicCalibration,
    "diagonal": DiagonalCalibration,
    "diagonal_ekf0": lambda: DiagonalCalibration(jacobian="ek0"),
}


def as_calibration(calibration: Calibration | str) -> Calibration:
    """Resolve a policy or one of the string shorthands to a policy.

    Raises:
        ValueError: For an unknown string.
        TypeError: For anything else that is not a ``Calibration``.
    """
    if isinstance(calibration, Calibration):
        return calibration
    if isinstance(calibration, str):
        if calibration not in _ALIASES:
            raise ValueError(
                f"calibration must be a Calibration or one of {tuple(_ALIASES)}; "
                f"got {calibration!r}."
            )
        return _ALIASES[calibration]()
    raise TypeError(
        f"calibration must be a Calibration or a string, got {type(calibration)}."
    )
