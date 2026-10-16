"""Trajectory-level iterated smoothing: the IEKS and the IPLS.

A one-shot filter linearizes the ODE information at each step's *prediction*,
a choice made before the later information is seen. An iterated smoother
instead linearizes along the *smoothed* trajectory of the previous pass, then
filters and smooths the resulting linear-Gaussian model again, for a fixed
number of passes:

- ``linearization="taylor"`` linearizes at the smoothed means: the iterated
  extended Kalman smoother (IEKS), Gauss-Newton on the MAP objective (Bell
  1994; for ODE filters, Tronarp, Saerkkae & Hennig 2021).
- ``linearization="slr"`` statistically linearizes over the smoothed
  marginals: the iterated posterior-linearization smoother (IPLS), which
  converges to a moment-matched Gaussian rather than to the MAP
  (Garcia-Fernandez, Svensson & Saerkkae 2017).

Contrast the IEKF (``IteratedTaylorCorrection``) and the IPLF
(``QuadratureCorrection(max_iters > 1)``), which iterate within each step of a
single forward pass.

The diffusion is not recalibrated between passes, because a ``sigma^2``
estimated in each pass would depend on the linearization of that pass, on
which it is meant to be conditioned. The prior must therefore carry its own
scale, as with ``calibration="none"``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian import Gaussian
from ..information import MeasurementModel, ObsModel
from ..linearization.correction import QuadratureCorrection, TaylorCorrection
from ..linearization.statistical import QuadratureRule
from ..linearization.surrogate import AffineSurrogate, slr_surrogate, taylor_surrogate
from ..priors.base import BasePrior
from .fixed_grid import affine_filter, check_grid, gaussian_filter
from .result import FilterResult
from .smoothing import rts_smoother


class IterationState(NamedTuple):
    """What one pass of ``iterated_smoother`` produced (passed to ``diagnostics``).

    Attributes:
        surrogate: The linearization of the pass.
        filter_result: Its filter result.
        m_smooth: Its smoothed means, shape ``[N + 1, D]``.
        P_smooth_sqr: Its smoothed square-root covariances.
        previous_point: The means it linearized at (nodes ``1..N``).
    """

    surrogate: AffineSurrogate
    filter_result: FilterResult
    m_smooth: Array
    P_smooth_sqr: Array
    previous_point: Array


class IteratedSmootherResult(NamedTuple):
    """Result of ``iterated_smoother``.

    Attributes:
        t: Time grid, shape ``[N + 1]``.
        m: Smoothed means of the final pass, shape ``[N + 1, D]``.
        P_sqr: Smoothed square-root covariances of the final pass.
        filter: ``FilterResult`` of the final pass.
        surrogate: The linearization the final pass used (``None`` when
            ``n_iters == 0``, where the one-shot filter linearized online).
        n_iters: Number of passes.
        step: RMS change of the smoothed means at nodes ``1..N`` in each pass,
            shape ``[n_iters]``: a convergence diagnostic.
        diagnostics: The ``diagnostics`` outputs of each pass, stacked along a
            leading axis of length ``n_iters`` (``None`` without a callable).
    """

    t: Array
    m: Array
    P_sqr: Array
    filter: FilterResult
    surrogate: AffineSurrogate | None
    n_iters: int
    step: Array
    diagnostics: Any


def iterated_smoother(
    m_0: ArrayLike,
    P_0_sqr: ArrayLike,
    prior: BasePrior,
    measure: MeasurementModel,
    ts: ArrayLike,
    *,
    linearization: Literal["taylor", "slr"] = "taylor",
    n_iters: int = 3,
    init: ArrayLike | None = None,
    init_P_sqr: ArrayLike | None = None,
    obs_model: ObsModel | None = None,
    rule: QuadratureRule = "gauss_hermite",
    n_nodes: int = 4,
    diagnostics: Callable[[IterationState], Any] | None = None,
    checkpoint: bool = False,
) -> IteratedSmootherResult:
    """Iterated smoother (IEKS / IPLS) over a time grid.

    Without ``init``, the first linearization point is the smoothed trajectory
    of a one-shot filter with the matching correction (EK1 for Taylor, a
    ``QuadratureCorrection`` for SLR). Each of the ``n_iters`` passes then
    linearizes along the current point (``taylor_surrogate`` /
    ``slr_surrogate``), runs ``affine_filter`` and ``rts_smoother``, and takes
    the smoothed marginals as the next point. ``n_iters=0`` returns the
    one-shot filter and smoother.

    Args:
        m_0: Initial mean, shape ``[D]``.
        P_0_sqr: Initial square-root covariance, shape ``[D, D]``.
        prior: Gauss-Markov prior (plain or preconditioned) carrying the
            diffusion scale.
        measure: ODE-information model.
        ts: Time grid, shape ``[N + 1]``.
        linearization: ``"taylor"`` (IEKS) or ``"slr"`` (IPLS).
        n_iters: Number of passes (``>= 0``; ``>= 1`` with ``init``).
        init: Warm start: the means at nodes ``1..N`` to linearize at first,
            shape ``[N, D]``. Skips the one-shot filter.
        init_P_sqr: With ``linearization="slr"`` and ``init``: the square-root
            covariances at nodes ``1..N``, shape ``[N, D, D]``. Not accepted for
            Taylor.
        obs_model: Data on the same grid, from ``prepare_observations``.
        rule: Quadrature rule for ``"slr"``.
        n_nodes: Gauss-Hermite nodes per dimension for ``"slr"``.
        diagnostics: Optional traceable ``IterationState -> PyTree``, called in
            every pass; its outputs are stacked in ``result.diagnostics``.
        checkpoint: Wrap each filter step, of the one-shot filter and of every
            pass, in ``jax.checkpoint`` (see ``gaussian_filter``).

    Returns:
        An ``IteratedSmootherResult``.

    Raises:
        ValueError: For an unknown ``linearization``, ``n_iters < 0``, ``init``
            with ``n_iters=0``, or an inconsistent ``init`` / ``init_P_sqr``.
    """
    if linearization not in ("taylor", "slr"):
        raise ValueError(
            f'linearization must be "taylor" or "slr", got {linearization!r}.'
        )
    if n_iters < 0:
        raise ValueError(f"n_iters must be >= 0, got {n_iters!r}.")
    ts = check_grid(ts)
    m_0 = np.asarray(m_0)
    n_steps = ts.shape[0] - 1
    taylor = linearization == "taylor"

    if init is None:
        correction = (
            TaylorCorrection()
            if taylor
            else QuadratureCorrection(rule=rule, n_nodes=n_nodes)
        )
        one_shot = gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            ts,
            correction=correction,
            calibration="none",
            obs_model=obs_model,
            checkpoint=checkpoint,
        )
        smoothed = rts_smoother(one_shot)
        if n_iters == 0:
            return IteratedSmootherResult(
                t=np.asarray(ts),
                m=smoothed.m,
                P_sqr=smoothed.P_sqr,
                filter=one_shot,
                surrogate=None,
                n_iters=0,
                step=np.zeros(0),
                diagnostics=None,
            )
        point = Gaussian(smoothed.m[1:], smoothed.P_sqr[1:])
    else:
        if n_iters == 0:
            raise ValueError("A warm start (init) needs n_iters >= 1.")
        init = np.asarray(init)
        if init.shape != (n_steps, m_0.shape[0]):
            raise ValueError(
                f"init must have shape {(n_steps, m_0.shape[0])} (the means at "
                f"nodes 1..N), got {init.shape}."
            )
        if taylor and init_P_sqr is not None:
            raise ValueError("A Taylor linearization takes no init_P_sqr.")
        if not taylor and init_P_sqr is None:
            raise ValueError("A statistical linearization needs init_P_sqr with init.")
        P_init = (
            np.zeros((n_steps, m_0.shape[0], m_0.shape[0]))
            if init_P_sqr is None
            else np.asarray(init_P_sqr)
        )
        point = Gaussian(init, P_init)

    def one_pass(point: Gaussian):
        if taylor:
            surrogate = taylor_surrogate(measure, point.m, ts=ts[1:])
        else:
            surrogate = slr_surrogate(
                measure, point.m, point.P_sqr, ts=ts[1:], rule=rule, n_nodes=n_nodes
            )
        result = affine_filter(
            m_0,
            P_0_sqr,
            prior,
            surrogate,
            ts,
            obs_model=obs_model,
            checkpoint=checkpoint,
        )
        smoothed = rts_smoother(result)
        step = np.sqrt(np.mean((smoothed.m[1:] - point.m) ** 2))
        extra = None
        if diagnostics is not None:
            extra = diagnostics(
                IterationState(surrogate, result, smoothed.m, smoothed.P_sqr, point.m)
            )
        next_point = Gaussian(smoothed.m[1:], smoothed.P_sqr[1:])
        return next_point, (result, surrogate, smoothed), (step, extra)

    # The final pass's filter result and surrogate ride in the carry; it starts
    # from zeros of the right structure.
    _, last_shape, _ = jax.eval_shape(one_pass, point)
    last_init = jax.tree.map(lambda s: np.zeros(s.shape, s.dtype), last_shape)

    def body(carry, _):
        next_point, last, record = one_pass(carry[0])
        return (next_point, last), record

    (_, (result, surrogate, smoothed)), (steps, extra) = jax.lax.scan(
        body, (point, last_init), None, length=n_iters
    )
    return IteratedSmootherResult(
        t=np.asarray(ts),
        m=smoothed.m,
        P_sqr=smoothed.P_sqr,
        filter=result,
        surrogate=surrogate,
        n_iters=n_iters,
        step=steps,
        diagnostics=extra,
    )
