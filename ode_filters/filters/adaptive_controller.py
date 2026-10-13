"""Step-size controllers for the adaptive EKF loop.

A controller's job is to propose a new step size ``h_new`` given the current
step ``h`` and a *normalised* error estimate ``err`` (one means "at tolerance",
greater than one means "rejected, shrink"). Accept/reject is decided by the
loop driver, not by the controller.

Two controllers are provided:

- :class:`PController` -- proportional-only on the current error. Sometimes
  called the *I-controller* or *elementary controller* in the
  adaptive-ODE-solver literature (Hairer-Wanner-Norsett, Soederlind) because
  ``h`` itself is the integral of the log-step adjustment.
- :class:`PIController` -- adds a Gustafsson PI term using the previous
  accepted step's error to damp oscillations around the tolerance.

Both implement the :class:`StepSizeController` protocol. ``propose`` is
written with ``jax.numpy`` so the jittable adaptive solver can call it inside
its traced loop; a custom controller must be traceable in the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

# Floor on the normalised error, so ``err ** (-alpha)`` stays finite.
_MIN_ERR = 1e-12


@runtime_checkable
class StepSizeController(Protocol):
    """Duck-typed interface for step-size controllers.

    The adaptive solvers call :meth:`propose` after every attempted step, from
    inside ``jax`` tracing in the jittable solver, so it must be written with
    ``jax.numpy`` (no ``float()`` / Python ``min`` / ``max`` on its inputs).

    ``err_prev`` is the normalised error of the previously accepted step. The
    absence of that memory (first step, or right after a reject) is signalled
    by ``None`` or, inside traced code, by a non-positive value.
    """

    def propose(
        self, h: ArrayLike, err: ArrayLike, err_prev: ArrayLike | None
    ) -> Array:
        """Return the proposed next step size."""
        ...


def _sanitize_error(err: ArrayLike) -> Array:
    """Map a non-finite error to ``+inf`` and floor it at ``_MIN_ERR``.

    A NaN/inf error (a blow-up) then collapses the proposed factor to
    ``min_factor`` (a shrink) instead of propagating NaN into ``h``, which
    would turn the next trial step NaN and stall the adaptive loop.
    """
    err = np.asarray(err)
    err = np.where(np.isfinite(err), err, np.inf)
    return np.maximum(err, _MIN_ERR)


@dataclass(frozen=True)
class PController:
    """Proportional (single-error) step-size controller.

    The proposed step is

        h_new = h * safety * err^(-alpha)

    clipped to ``[min_factor, max_factor] * h``. The previous error is
    ignored, so this controller has no memory.

    Default ``alpha = 1 / order`` matches the standard "I-controller" of
    Hairer-Wanner-Norsett (1993) and the elementary controller in scipy's
    ``solve_ivp``. For an EKF1 with IWP(q) prior, ``order = q``.

    Args:
        order: Convergence order of the local error estimate.
        safety: Safety factor on the proposed step.
        alpha: Gain on the current error; defaults to ``1.0 / order``.
        min_factor: Lower clip on ``h_new / h``.
        max_factor: Upper clip on ``h_new / h``.
    """

    order: int
    safety: float = 0.9
    alpha: float | None = None
    min_factor: float = 0.2
    max_factor: float = 5.0

    @property
    def _alpha(self) -> float:
        return 1.0 / self.order if self.alpha is None else self.alpha

    def propose(
        self, h: ArrayLike, err: ArrayLike, err_prev: ArrayLike | None = None
    ) -> Array:
        """Return the next proposed step size.

        Args:
            h: Current step size.
            err: Normalised error of the current step (1.0 = at tolerance).
            err_prev: Ignored; accepted for protocol compatibility with
                :class:`PIController`.
        """
        del err_prev  # unused by design
        factor = self.safety * _sanitize_error(err) ** (-self._alpha)
        return np.asarray(h) * np.clip(factor, self.min_factor, self.max_factor)


@dataclass(frozen=True)
class PIController:
    """Gustafsson-style proportional-integral controller.

    The proposed step is

        h_new = h * safety * err^(-alpha) * (err_prev / err)^(beta)

    clipped to ``[min_factor, max_factor] * h``. When ``err_prev`` is unknown
    (e.g. on the first step or right after a reject) the I-term is dropped
    and the update reduces to the :class:`PController` form
    ``h_new = h * safety * err^(-alpha)``.

    Defaults follow Gustafsson (1991): ``alpha = 0.7 / order``, ``beta = 0.4 /
    order``, which are also the values used in Bosch et al. (2021). For an
    EKF1 with IWP(q) prior, ``order = q``.

    Args:
        order: Convergence order of the local error estimate.
        safety: Safety factor on the proposed step.
        alpha: Proportional gain; defaults to ``0.7 / order``.
        beta: Integral gain; defaults to ``0.4 / order``. Set ``beta=0`` for a
            pure proportional controller -- :class:`PController` is the
            standalone equivalent with the more natural default ``alpha``.
        min_factor: Lower clip on ``h_new / h``.
        max_factor: Upper clip on ``h_new / h``.
    """

    order: int
    safety: float = 0.9
    alpha: float | None = None
    beta: float | None = None
    min_factor: float = 0.2
    max_factor: float = 5.0

    @property
    def _alpha(self) -> float:
        return 0.7 / self.order if self.alpha is None else self.alpha

    @property
    def _beta(self) -> float:
        return 0.4 / self.order if self.beta is None else self.beta

    def propose(
        self, h: ArrayLike, err: ArrayLike, err_prev: ArrayLike | None
    ) -> Array:
        """Return the next proposed step size.

        Args:
            h: Current step size.
            err: Normalised error of the current step (1.0 = at tolerance).
            err_prev: Normalised error of the previously accepted step, or
                ``None`` (or a non-positive value, inside traced code) if
                unavailable.
        """
        err = _sanitize_error(err)
        factor = self.safety * err ** (-self._alpha)
        if err_prev is not None:
            err_prev = np.asarray(err_prev)
            integral = np.where(
                err_prev > 0.0,
                (np.maximum(err_prev, _MIN_ERR) / err) ** self._beta,
                1.0,
            )
            factor = factor * integral
        return np.asarray(h) * np.clip(factor, self.min_factor, self.max_factor)
