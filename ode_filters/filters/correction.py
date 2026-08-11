"""Pluggable linearization/correction strategies for the Gaussian ODE filter.

A :class:`Correction` owns the predicted-mean -> updated-posterior transition of a
filter step. The measurement model declares *what* to observe (the residual ``g``,
its Jacobian, and the observation noise); the Correction decides *how* to linearize
and update. Separating these two axes lets any measurement model (ODE information,
conservation laws, hidden/joint models) compose with any linearization scheme.

Shipped schemes:

- ``TaylorCorrection(order=1)`` -- EK1, the first-order extended Kalman
  linearization (full vector-field Jacobian); reproduces the historical behavior.
- ``TaylorCorrection(order=0)`` -- EK0, the zeroth-order linearization (vector field
  treated as locally constant, so the ODE-defect rows of the Jacobian reduce to the
  selection matrix ``E_constraint``).
- ``IteratedTaylorCorrection`` -- IEKF, the iterated EKF: a single forward pass in
  which *each step's update* relinearizes at the updated mean to a fixed point
  (Gauss-Newton on the per-step MAP). This reduces local linearization error on
  nonlinear problems. (The whole-trajectory iterated *smoother*, IEKS, is a separate
  outer-loop construct and is not a Correction.)
- ``QuadratureCorrection`` -- statistical linearization (SLF / sigma-point family):
  the affine surrogate is fitted over the predictive *spread* rather than tangent
  to a point, with the required Gaussian expectations evaluated by quadrature.
  With ``max_iters > 1`` it becomes the iterated posterior-linearization filter
  (IPLF), which relinearizes with respect to the updated *density* -- the axis-2
  counterpart of ``IteratedTaylorCorrection``'s axis-3 iteration.

The Taylor schemes reuse the existing square-root algebra;
``QuadratureCorrection`` adds only the quadrature of
:mod:`~ode_filters.filters.statistical_linearization` on top of the same update.
"""

from __future__ import annotations

import abc
from typing import NamedTuple

import equinox as eqx
import jax
from jax import Array
from jax.typing import ArrayLike

from ..inference.sqr_gaussian_inference import sqr_inversion, sqr_marginalization
from ..measurement.measurement_models import BaseODEInformation
from .statistical_linearization import QuadratureRule, quadrature_nodes, slr_linearize


class CorrectionResult(NamedTuple):
    """Output of a single :meth:`Correction.correct` call.

    Attributes:
        m: Updated posterior mean (shape ``[state_dim]``).
        P_sqr: Updated posterior covariance in square-root form
            (``P = P_sqr.T @ P_sqr``).
        mz: Innovation (predicted-observation) mean (shape ``[obs_dim]``).
        Pz_sqr: Innovation covariance in square-root form -- consumed by the
            Gaussian log-marginal-likelihood.
    """

    m: Array
    P_sqr: Array
    mz: Array
    Pz_sqr: Array


def _linearize(
    measure: BaseODEInformation, state: Array, *, t: ArrayLike, order: int
) -> tuple[Array, Array]:
    """Effective affine measurement model ``(H, c)`` with ``H x + c ~= g(x)``.

    ``order=0`` drops the vector-field Jacobian on the ODE-defect rows (EK0);
    ``order=1`` keeps it (EK1).
    """
    H, c = measure.linearize(state, t=t)
    if order == 0:
        g = c + H @ state  # reconstruct the residual g(state)
        d = measure.ode_dim
        H = H.at[:d].set(measure.E_constraint)
        c = g - H @ state
    return H, c


def _affine_correct(
    H: Array, c: Array, R_sqr: Array, m_pred: Array, P_pred_sqr: Array
) -> tuple[Array, Array, Array, Array]:
    """One affine square-root update; returns ``(m, P_sqr, mz, Pz_sqr)``."""
    mz, Pz_sqr = sqr_marginalization(H, c, R_sqr, m_pred, P_pred_sqr)
    _, m_new, P_new_sqr = sqr_inversion(H, m_pred, P_pred_sqr, mz, Pz_sqr, R_sqr)
    return m_new, P_new_sqr, mz, Pz_sqr


def _noise(measure: BaseODEInformation, t: ArrayLike) -> Array:
    return measure.get_noise(t=t)


class Correction(eqx.Module):
    """Base class for linearization/correction strategies.

    A Correction is an :class:`equinox.Module` (static configuration), so it can be
    captured by ``jax.jit`` / ``jax.lax.scan`` without retracing. Subclasses own the
    full predicted -> posterior update and return a :class:`CorrectionResult`.
    """

    @abc.abstractmethod
    def correct(
        self,
        measure: BaseODEInformation,
        m_pred: Array,
        P_pred_sqr: Array,
        *,
        t: ArrayLike = 0.0,
    ) -> CorrectionResult:
        """Update a predicted Gaussian with the measurement model at time ``t``.

        Args:
            measure: ODE-information measurement model.
            m_pred: Predicted (prior) state mean.
            P_pred_sqr: Predicted state covariance, square-root form.
            t: Current time.
        """
        raise NotImplementedError  # pragma: no cover - abstract method


class TaylorCorrection(Correction):
    """Taylor-series correction: EK1 (``order=1``) or EK0 (``order=0``).

    Attributes:
        order: ``1`` for the first-order (EK1) linearization with the full
            vector-field Jacobian (default; bit-identical to the historical filter
            step). ``0`` for the zeroth-order (EK0) linearization.
    """

    order: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        if self.order not in (0, 1):
            raise ValueError(
                f"TaylorCorrection.order must be 0 (EK0) or 1 (EK1), "
                f"got {self.order!r}."
            )

    def correct(self, measure, m_pred, P_pred_sqr, *, t=0.0) -> CorrectionResult:
        H, c = _linearize(measure, m_pred, t=t, order=self.order)
        R_sqr = _noise(measure, t)
        m, P_sqr, mz, Pz_sqr = _affine_correct(H, c, R_sqr, m_pred, P_pred_sqr)
        return CorrectionResult(m, P_sqr, mz, Pz_sqr)


class IteratedTaylorCorrection(Correction):
    """Iterated EKF (IEKF): per-step Gauss-Newton relinearization.

    A single forward pass; within each update, the EK1 linearization is recomputed
    at the *updated* mean and the update redone, for a fixed ``max_iters`` passes
    (``max_iters=1`` reproduces EK1). A fixed iteration count -- rather than a
    convergence-based ``while_loop`` -- keeps the correction reverse-mode
    differentiable, so it can be used inside gradient-based parameter inference.

    Attributes:
        max_iters: Number of relinearizations per step (>= 1).
    """

    max_iters: int = eqx.field(static=True, default=3)

    def __check_init__(self):
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters!r}.")

    def correct(self, measure, m_pred, P_pred_sqr, *, t=0.0) -> CorrectionResult:
        R_sqr = _noise(measure, t)

        def update_from(m_lin):
            H, c = _linearize(measure, m_lin, t=t, order=1)
            m, P_sqr, mz, Pz_sqr = _affine_correct(H, c, R_sqr, m_pred, P_pred_sqr)
            return m, P_sqr, mz, Pz_sqr

        # Iteration 0 linearizes at the prediction (this alone == EK1); each
        # further pass relinearizes at the current updated mean.
        init = update_from(m_pred)
        carry = jax.lax.fori_loop(
            1, self.max_iters, lambda _i, c: update_from(c[0]), init
        )
        m, P_sqr, mz, Pz_sqr = carry
        return CorrectionResult(m, P_sqr, mz, Pz_sqr)


class QuadratureCorrection(Correction):
    """Statistical linearization by quadrature (sigma-point / SLF family).

    Replaces the Taylor tangent at the predicted mean with the affine map that
    best fits the residual *over the predictive spread*, computing the required
    Gaussian expectations by quadrature on the projected marginal (see
    :mod:`~ode_filters.filters.statistical_linearization`). Two things change
    relative to ``TaylorCorrection(order=1)``:

    - ``H`` becomes the *expected* Jacobian rather than the Jacobian at the mean;
    - the linearization-residual covariance ``Omega`` is added to the measurement
      noise, so the innovation covariance is ``H P H^T + Omega + R`` -- exactly
      the term a sigma-point filter's ``S`` carries and the EKF's omits.

    Both differences vanish for an affine vector field, so this correction is a
    no-op (up to floating point) on linear problems.

    Where it matters: the closure error scales with the predictive spread of the
    vector field's arguments. In a pure ODE solve that spread vanishes as
    ``h -> 0`` and this reduces to EK1, but with a latent force it has a floor
    set by the irreducible force uncertainty, and the resulting Jensen gap does
    not vanish under grid refinement.

    Attributes:
        rule: ``"gauss_hermite"`` (default; tensor product, spectrally accurate,
            ``n_nodes ** p`` nodes) or ``"cubature"`` (third-degree spherical,
            ``2 p`` nodes, derivative-free-equivalent and dimension-linear).
            ``p`` is the ODE dimension, not the state dimension.
        n_nodes: Nodes per dimension for the Gauss-Hermite rule; exactness is
            ``2 * n_nodes - 1``. A polynomial vector field of degree ``k`` needs
            ``n_nodes >= ceil((k + 1) / 2)`` for ``H`` and ``c``, and
            ``n_nodes >= ceil((2 k + 1) / 2)`` for ``Omega`` too -- so a cubic
            (Duffing, van der Pol) is exactly reproduced at ``n_nodes = 4``, and
            a quadratic (Lotka-Volterra, SIRD) at ``n_nodes = 3``. Ignored by
            the cubature rule.
        max_iters: Number of posterior-linearization passes. ``1`` (default) is
            the one-shot filter; ``> 1`` is the iterated posterior-linearization
            filter (IPLF), which refits the surrogate over the *updated* density.
            Contrast :class:`IteratedTaylorCorrection`, which iterates the
            linearization *point* at zero spread. The whole-trajectory version is
            :func:`~ode_filters.filters.ipls.ipls_smoother`.
    """

    rule: QuadratureRule = eqx.field(static=True, default="gauss_hermite")
    n_nodes: int = eqx.field(static=True, default=4)
    max_iters: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters!r}.")
        # Validated eagerly so a bad rule name fails at construction rather than
        # inside a traced scan body. dim=1 is enough to exercise the dispatch.
        quadrature_nodes(self.rule, 1, self.n_nodes)

    def correct(self, measure, m_pred, P_pred_sqr, *, t=0.0) -> CorrectionResult:
        def update_from(m_lin, P_lin_sqr):
            model = slr_linearize(
                measure,
                m_lin,
                P_lin_sqr,
                t=t,
                rule=self.rule,
                n_nodes=self.n_nodes,
            )
            return _affine_correct(
                model.H, model.c, model.R_eff_sqr, m_pred, P_pred_sqr
            )

        # Pass 0 linearizes over the *predictive* density (the one-shot
        # sigma-point update); each further pass refits over the updated one.
        init = update_from(m_pred, P_pred_sqr)
        m, P_sqr, mz, Pz_sqr = jax.lax.fori_loop(
            1, self.max_iters, lambda _i, c: update_from(c[0], c[1]), init
        )
        return CorrectionResult(m, P_sqr, mz, Pz_sqr)
