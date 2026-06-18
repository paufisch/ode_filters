"""Pluggable linearization/correction strategies for the Gaussian ODE filter.

A :class:`Correction` owns the predicted-mean -> updated-posterior transition
of a filter step. The measurement model declares *what* to observe (the
residual ``g``, its Jacobian, and the observation noise); the Correction
decides *how* to linearize and update. Separating these two axes lets any
measurement model (ODE information, conservation laws, hidden/joint models)
compose with any linearization scheme, so the choice of scheme is no longer
baked into the filter.

This module ships the Taylor-series corrections:

- ``TaylorCorrection(order=1)`` -- EK1, the first-order extended Kalman
  linearization (full vector-field Jacobian). This reproduces the historical
  behavior of :func:`ekf1_sqr_filter_step` exactly.
- ``TaylorCorrection(order=0)`` -- EK0, the zeroth-order linearization that
  treats the vector field as locally constant, so the ODE-defect rows of the
  Jacobian reduce to the selection matrix ``E_constraint`` (``E1``/``E2``).

Both are one-shot affine updates and reuse the existing square-root Gaussian
algebra (:func:`sqr_marginalization`, :func:`sqr_inversion`); this module adds
no new numerics. Iterated (IEKF) and sigma-point (UKF/SLR) corrections are
planned and will fit the same :class:`CorrectionResult` seam (those schemes
must own the update loop, which is why ``correct`` returns a finished posterior
rather than a one-shot ``(H, c)``).
"""

from __future__ import annotations

import abc
from typing import NamedTuple

import equinox as eqx
from jax import Array

from ..inference.sqr_gaussian_inference import sqr_inversion, sqr_marginalization
from ..measurement.measurement_models import BaseODEInformation


class CorrectionResult(NamedTuple):
    """Output of a single :meth:`Correction.correct` call.

    Attributes:
        m: Updated posterior mean (shape ``[state_dim]``).
        P_sqr: Updated posterior covariance in square-root form
            (``P = P_sqr.T @ P_sqr``).
        mz: Innovation (predicted-observation) mean (shape ``[obs_dim]``).
        Pz_sqr: Innovation covariance in square-root form -- consumed by the
            Gaussian log-marginal-likelihood.
        mz_ode: The ODE-defect rows of ``mz`` (``mz[:ode_dim]``); the only rows
            that should drive diffusion calibration (Bosch, Tronarp, Hennig
            2022, sec. 3).
        H_eff: The ODE-defect rows of the effective observation Jacobian
            (``E_constraint`` for EK0, ``E1 - J_f E0`` for EK1); used by the
            per-component "diagonal" calibration denominator.
    """

    m: Array
    P_sqr: Array
    mz: Array
    Pz_sqr: Array
    mz_ode: Array
    H_eff: Array


class Correction(eqx.Module):
    """Base class for linearization/correction strategies.

    A Correction is an :class:`equinox.Module` (a registered pytree carrying
    only static configuration here), so it can be captured by ``jax.jit`` /
    ``jax.lax.scan`` without retracing on identity. Subclasses own the full
    predicted -> posterior update and return a :class:`CorrectionResult`, so
    schemes that must control the update loop (iterated EKF, sigma-point moment
    matching) fit the same interface as the one-shot Taylor corrections.
    """

    @abc.abstractmethod
    def correct(
        self,
        measure: BaseODEInformation,
        m_pred: Array,
        P_pred_sqr: Array,
        *,
        t: float = 0.0,
    ) -> CorrectionResult:
        """Update a predicted Gaussian with the measurement model at time ``t``.

        Args:
            measure: ODE-information measurement model exposing
                ``g`` / ``jacobian_g`` / ``linearize`` / ``get_noise`` /
                ``ode_dim`` / ``E_constraint``.
            m_pred: Predicted (prior) state mean.
            P_pred_sqr: Predicted state covariance, square-root form.
            t: Current time.

        Returns:
            A :class:`CorrectionResult` with the updated posterior and the
            quantities needed for calibration and the log-marginal-likelihood.
        """
        raise NotImplementedError


class TaylorCorrection(Correction):
    """Taylor-series correction: EK1 (``order=1``) or EK0 (``order=0``).

    Attributes:
        order: ``1`` for the first-order (EK1) linearization with the full
            vector-field Jacobian (default; bit-identical to the historical
            filter step). ``0`` for the zeroth-order (EK0) linearization, which
            treats the vector field as locally constant so the ODE-defect rows
            of the Jacobian are just the selection matrix ``E_constraint``.
    """

    order: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        if self.order not in (0, 1):
            raise ValueError(
                f"TaylorCorrection.order must be 0 (EK0) or 1 (EK1), "
                f"got {self.order!r}."
            )

    def correct(
        self,
        measure: BaseODEInformation,
        m_pred: Array,
        P_pred_sqr: Array,
        *,
        t: float = 0.0,
    ) -> CorrectionResult:
        H, c = self._linearize(measure, m_pred, t=t)
        R_sqr = measure.get_noise(t=t)
        mz, Pz_sqr = sqr_marginalization(H, c, R_sqr, m_pred, P_pred_sqr)
        _, m_new, P_new_sqr = sqr_inversion(H, m_pred, P_pred_sqr, mz, Pz_sqr, R_sqr)
        d_ode = measure.ode_dim
        return CorrectionResult(
            m=m_new,
            P_sqr=P_new_sqr,
            mz=mz,
            Pz_sqr=Pz_sqr,
            mz_ode=mz[:d_ode],
            H_eff=H[:d_ode],
        )

    def _linearize(
        self, measure: BaseODEInformation, m_pred: Array, *, t: float
    ) -> tuple[Array, Array]:
        """Return the effective affine observation model ``(H, c)``.

        EK1 uses the measurement model's own linearization. EK0 drops the
        vector-field Jacobian on the ODE-defect rows -- replacing them with the
        linear selection ``E_constraint`` -- and keeps the (already linear)
        conservation/observation rows unchanged. ``c`` is recomputed from the
        exact residual ``g`` so that ``H @ m_pred + c = g(m_pred)``.
        """
        if self.order == 1:
            return measure.linearize(m_pred, t=t)
        H, _ = measure.linearize(m_pred, t=t)
        d_ode = measure.ode_dim
        H = H.at[:d_ode].set(measure.E_constraint)
        c = measure.g(m_pred, t=t) - H @ m_pred
        return H, c
