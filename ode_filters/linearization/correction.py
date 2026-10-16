"""Pluggable linearization/correction strategies for the Gaussian ODE filter.

A ``Correction`` owns the update of a filter step. The measurement model declares
*what* to observe (the residual ``g``, its Jacobian, and the observation noise);
the Correction decides *how* to linearize and update. Separating these two axes
lets any measurement model (ODE information, conservation laws, hidden/joint
models) compose with any linearization scheme.

A step uses a correction twice. ``linearize`` gives the affine model at the
predicted mean, before the predicted covariance exists: the diffusion
calibration estimates ``sigma^2`` from it. ``correct`` then updates the
predicted Gaussian; the Taylor schemes reuse that same model, so each step
linearizes once.

Shipped schemes:

- ``TaylorCorrection(order=1)`` is EK1, the first-order extended Kalman
  linearization (full vector-field Jacobian).
- ``TaylorCorrection(order=0)`` is EK0, the zeroth-order linearization. It
  treats the vector field as locally constant, so the ODE-defect rows of the
  Jacobian reduce to the selection matrix of the ODE's left-hand side, and it
  never differentiates the vector field.
- ``IteratedTaylorCorrection`` is the iterated EKF (IEKF). It runs a single
  forward pass in which the update of each step relinearizes at the updated
  mean, in a fixed number of Gauss-Newton passes on the MAP objective of the
  step. (The iterated smoother over the whole trajectory, the IEKS, is
  ``iterated_smoother``.)
- ``QuadratureCorrection`` is statistical linearization, the linearization of
  the statistically linearized filter (SLF) and the sigma-point filters. It
  fits the affine surrogate over the predictive distribution instead of taking
  the tangent at a point, and evaluates the Gaussian expectations by
  quadrature. With ``max_iters > 1`` it becomes the iterated
  posterior-linearization filter (IPLF), which relinearizes with respect to the
  updated density.
"""

from __future__ import annotations

import abc

import equinox as eqx
import jax
from jax import Array
from jax.typing import ArrayLike

from ..gaussian import AffineModel, Gaussian, update
from ..information.protocol import EK0Linearizable, MeasurementModel
from .statistical import QuadratureRule, quadrature_nodes, slr_linearize


def taylor_linearize(
    measure: MeasurementModel, state: Array, *, t: ArrayLike, order: int = 1
) -> AffineModel:
    """Taylor linearization ``H x + c ~= g(x)`` of ``measure`` at ``state``.

    Args:
        measure: Measurement model.
        state: Linearization point, shape ``[D]``.
        t: Time.
        order: ``1`` for the full Jacobian (EK1); ``0`` drops the vector-field
            Jacobian from the ODE rows (EK0), which needs a model with a
            ``linearize_ek0`` method (``EK0Linearizable``, which the
            ODE-information models implement).

    Returns:
        The affine model, with the measurement noise.

    Raises:
        TypeError: If ``order`` is ``0`` and ``measure`` has no
            ``linearize_ek0``.
    """
    if order == 0:
        if not isinstance(measure, EK0Linearizable):
            raise TypeError(
                f"An EK0 linearization needs an ODE-information model; "
                f"{type(measure).__name__} has no linearize_ek0."
            )
        H, c = measure.linearize_ek0(state, t=t)
    else:
        H, c = measure.linearize(state, t=t)
    return AffineModel(H, c, measure.noise_sqr(t=t))


class Correction(eqx.Module):
    """Base class of the linearization/correction strategies.

    A Correction is an ``equinox.Module`` (static configuration), so it can be
    captured by ``jax.jit`` / ``jax.lax.scan`` without retracing.
    """

    def linearize(
        self, measure: MeasurementModel, m: Array, *, t: ArrayLike
    ) -> AffineModel:
        """The affine model at the predicted mean ``m`` (EK1 by default).

        The diffusion calibration estimates ``sigma^2`` from it before the
        predicted covariance is known, and ``correct`` receives it.
        """
        return taylor_linearize(measure, m, t=t, order=1)

    @abc.abstractmethod
    def correct(
        self,
        measure: MeasurementModel,
        x: Gaussian,
        model: AffineModel,
        *,
        t: ArrayLike,
    ) -> tuple[Gaussian, Gaussian]:
        """Update the predicted Gaussian ``x`` with the measurement model.

        Args:
            measure: Measurement model.
            x: Predicted Gaussian.
            model: ``self.linearize(measure, x.m, t=t)``.
            t: Time.

        Returns:
            ``(posterior, innovation)``; the innovation's density at zero is
            the step's evidence.
        """
        raise NotImplementedError


class TaylorCorrection(Correction):
    """Taylor-series correction: EK1 (``order=1``) or EK0 (``order=0``).

    Attributes:
        order: ``1`` for the first-order (EK1) linearization with the full
            vector-field Jacobian (default). ``0`` for the zeroth-order (EK0)
            linearization, which never evaluates the vector-field Jacobian.
    """

    order: int = eqx.field(static=True, default=1)

    def __check_init__(self):
        if self.order not in (0, 1):
            raise ValueError(
                f"TaylorCorrection.order must be 0 (EK0) or 1 (EK1), "
                f"got {self.order!r}."
            )

    def linearize(self, measure, m, *, t):
        """Taylor-linearize ``measure`` at ``m`` (order ``self.order``)."""
        return taylor_linearize(measure, m, t=t, order=self.order)

    def correct(self, measure, x, model, *, t):
        """A Kalman update of ``x`` with the linearization at the prediction."""
        return update(model, x)


class IteratedTaylorCorrection(Correction):
    """Iterated EKF (IEKF): per-step Gauss-Newton relinearization.

    A single forward pass. Within each update, the EK1 linearization is
    recomputed at the updated mean and the update is redone, for a fixed number
    ``max_iters`` of passes (``max_iters=1`` reproduces EK1). The fixed
    iteration count, in place of a ``while_loop`` with a convergence test,
    keeps the correction reverse-mode differentiable, so it can be used inside
    gradient-based parameter inference.

    Attributes:
        max_iters: Number of linearizations per step (>= 1).
    """

    max_iters: int = eqx.field(static=True, default=3)

    def __check_init__(self):
        if self.max_iters < 1:
            raise ValueError(f"max_iters must be >= 1, got {self.max_iters!r}.")

    def correct(self, measure, x, model, *, t):
        """Update ``x``, relinearizing at the updated mean ``max_iters - 1`` times."""

        def relinearize(_i, carry):
            posterior, _ = carry
            H, c = measure.linearize(posterior.m, t=t)
            return update(AffineModel(H, c, model.R_sqr), x)

        # Iteration 0 uses the linearization at the prediction (this alone is
        # EK1); each further pass relinearizes at the current updated mean.
        return jax.lax.fori_loop(1, self.max_iters, relinearize, update(model, x))


class QuadratureCorrection(Correction):
    """Statistical linearization by quadrature (sigma-point / SLF family).

    Replaces the Taylor tangent at the predicted mean with the affine map that
    best fits the residual over the predictive distribution, and computes the
    Gaussian expectations by quadrature on the projected marginal (see
    ``ode_filters.linearization.statistical``). Two things change relative to
    ``TaylorCorrection(order=1)``:

    - ``H`` becomes the expected Jacobian instead of the Jacobian at the mean;
    - the linearization-residual covariance ``Omega`` is added to the measurement
      noise, so the innovation covariance is ``H P H^T + Omega + R``. ``Omega``
      is the term that the ``S`` of a sigma-point filter carries and that of the
      EKF omits.

    Both differences vanish for an affine vector field, so on linear problems
    this correction coincides with EK1 up to floating point. The diffusion
    calibration still uses the EK1 linearization at the predicted mean, because
    the statistical one needs the predicted covariance it is calibrating.

    Where it matters: the closure error scales with the predictive spread of the
    vector field's arguments. In a pure ODE solve that spread vanishes as
    ``h -> 0`` and this reduces to EK1, but with a latent force it has a floor
    set by the irreducible force uncertainty, and the resulting Jensen gap does
    not vanish under grid refinement.

    Attributes:
        rule: ``"gauss_hermite"`` (default; tensor product, spectrally accurate,
            ``n_nodes ** p`` nodes) or ``"cubature"`` (third-degree spherical,
            ``2 p`` nodes, derivative-free-equivalent and dimension-linear).
            ``p`` is the dimension of the vector field's arguments, not the
            state dimension.
        n_nodes: Nodes per dimension for the Gauss-Hermite rule; exactness is
            ``2 * n_nodes - 1``. A polynomial vector field of degree ``k`` needs
            ``n_nodes >= ceil((k + 1) / 2)`` for ``H`` and ``c``, and
            ``n_nodes >= ceil((2 k + 1) / 2)`` for ``Omega`` as well. A cubic
            field (Duffing, van der Pol) is therefore reproduced exactly at
            ``n_nodes = 4``, and a quadratic one (Lotka-Volterra, SIRD) at
            ``n_nodes = 3``. Ignored by the cubature rule.
        max_iters: Number of posterior-linearization passes. ``1`` (default) is
            the one-shot filter; ``> 1`` is the iterated posterior-linearization
            filter (IPLF), which refits the surrogate over the *updated* density.
            Contrast ``IteratedTaylorCorrection``, which iterates the
            linearization *point* at zero spread. The whole-trajectory version is
            ``iterated_smoother(linearization="slr")``.
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

    def correct(self, measure, x, model, *, t):
        """Update ``x`` with the statistical linearization over the density."""

        def update_over(density: Gaussian):
            slr = slr_linearize(
                measure,
                density.m,
                density.P_sqr,
                t=t,
                rule=self.rule,
                n_nodes=self.n_nodes,
            )
            return update(AffineModel(slr.H, slr.c, slr.R_eff_sqr), x)

        # Pass 0 linearizes over the *predictive* density (the one-shot
        # sigma-point update); each further pass refits over the updated one.
        return jax.lax.fori_loop(
            1, self.max_iters, lambda _i, c: update_over(c[0]), update_over(x)
        )
