"""Adaptive-step EKF loop with online sigma calibration.

The trajectory length is data-dependent (steps may be rejected and retried at
a smaller ``h``), so this loop cannot use ``jax.lax.scan``. Instead it is a
Python ``while`` driver around a jitted per-step body. The per-step body does
the prediction + update + per-step quasi-MLE sigma and returns the normalised
local-error estimate; the Python layer makes the accept/reject decision.

Four calibration schemes are exposed via the ``calibration`` parameter:

- ``"dynamic"`` (default, Bosch et al. 2021 Eq. 32): scalar
  ``sigma_hat^2_n = m_z.T (H Q(h) H.T)^{-1} m_z / d``, baked into the
  current step's ``Q_h`` *before* propagation. Past steps keep their own
  ``sigma_hat^2`` -- statistically honest per-step, Markov-preserving.
  **Fails on multi-scale ODEs** (the scalar average is dominated by the
  larger-residual component, which starves smaller-scale components).

- ``"diagonal_ekf0"``: per-component
  ``sigma_hat^2_i = m_z[i]^2 / (E_1 Q E_1.T)_ii``. Because ``E_1`` selects
  only the first-derivative block, ``E_1 Q E_1.T`` is *exactly* diagonal
  when ``xi`` is diagonal -- so ``sigma_hat^2_i`` is the genuine MLE in a
  block-diagonal sub-model for component ``i``. This is the standard
  per-component recipe (Bosch et al. 2021 §4.2, ``DynamicMVDiffusion``)
  and is the recommended choice for multi-component problems. Requires
  diagonal ``xi``.

- ``"diagonal"`` (EK1-flavoured heuristic): same formula as
  ``"diagonal_ekf0"`` but with the EK1 Jacobian ``H_t = E_1 - J_f E_0`` in
  the denominator. ``H_t Q H_t.T`` is then generally dense, and taking
  its diagonal ignores cross-component coupling introduced by ``J_f``.
  When ``J_f`` is block-diagonal in components (e.g. fully decoupled RHS)
  this collapses to ``"diagonal_ekf0"``; when ``J_f`` couples components
  strongly, the diagonal-of-dense recipe biases the per-component MLE.
  Use only when you understand the trade-off; prefer ``"diagonal_ekf0"``
  by default. Requires diagonal ``xi``.

- ``"none"``: propagate uncalibrated, still report ``sigma_hat^2`` for
  diagnostics / post-hoc rescaling.

A separate ``sigma_in_error`` parameter controls how ``sigma_hat^2`` enters
the local-error estimate used for step-size selection. The quasi-MLE
estimator has chi-squared noise (relative std ``sqrt(2/d)``) which
propagates into ``h`` decisions, causing visible oscillations. The default
``sigma_in_error="running_mean"`` substitutes the cumulative running mean
in the error formula -- noise vanishes after a few accepted steps, ``h``
becomes much smoother, reject counts drop. Per-step ``sigma_hat^2`` is
still used in the actual Q calibration. Pass ``sigma_in_error="per_step"``
to recover the original Bosch et al. 2021 recipe. *Caveat:* the running
mean lags real changes in problem stiffness; in regimes where the local
diffusion changes abruptly (entering a stiff transition), ``"per_step"``
is more responsive.

The accepted-step outputs are stored in lists; the result can be passed
directly to :func:`rts_sqr_smoother_loop`.

The returned ``log_likelihood`` is the *post-calibration* marginal
likelihood: each step's contribution uses the calibrated ``Pz_sqr`` (which
includes ``sigma_hat^2`` in the dynamic/diagonal modes). It is therefore the right
quantity for inference *given* the chosen calibration model, but is **not**
directly comparable across calibration modes -- different ``calibration``
settings define different generative models.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, NamedTuple, cast

import equinox.internal as eqxi
import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..inference.sqr_gaussian_inference import sqr_inversion, sqr_marginalization
from ..measurement.measurement_models import (
    BaseODEInformation,
    ObsModel,
)
from ..priors.gmp_priors import BasePrior
from .adaptive_controller import PIController, StepSizeController
from .correction import Correction
from .ode_filter_loop import (
    _calibrate_diffusion,
    _check_state_xi_diagonal,
    _log_likelihood_contrib,
)
from .ode_filter_step import ekf1_sqr_filter_step

CalibrationMode = Literal["dynamic", "diagonal", "diagonal_ekf0", "none"]
SigmaInError = Literal["per_step", "running_mean"]
_VALID_CALIBRATIONS = ("dynamic", "diagonal", "diagonal_ekf0", "none")


class AdaptiveLoopResult(NamedTuple):
    """Output of :func:`ekf1_sqr_adaptive_loop`.

    All sequences are aligned to *accepted* steps. ``t_seq`` has length
    ``N_accepted + 1`` (initial time plus one entry per accepted step); the
    per-step sequences (``m_pred_seq``, ``Pz_seq_sqr``, ...) and the smoothing
    quantities (``G_back_seq``, ``d_back_seq``, ``P_back_seq_sqr``) have length
    ``N_accepted``. Filtered means/covariances (``m_seq``, ``P_seq_sqr``) have
    length ``N_accepted + 1`` and include the initial state.
    """

    t_seq: Array
    m_seq: list[Array]
    P_seq_sqr: list[Array]
    m_pred_seq: list[Array]
    P_pred_seq_sqr: list[Array]
    G_back_seq: list[Array]
    d_back_seq: list[Array]
    P_back_seq_sqr: list[Array]
    mz_seq: list[Array]
    Pz_seq_sqr: list[Array]
    sigma_sqr_seq: list[float]
    h_seq: list[float]
    n_rejected: int
    log_likelihood: float


def _local_error_norm(
    D: Array,
    m_pred_value: Array,
    atol: float,
    rtol: float,
) -> Array:
    """Tolerance-weighted RMS norm of the local error estimate.

    ``err = sqrt(mean((D / (atol + rtol * |m|))^2))`` -- the same form used by
    OrdinaryDiffEq and Bosch et al. (2021).
    """
    scale = atol + rtol * np.abs(m_pred_value)
    return np.sqrt(np.mean((D / scale) ** 2))


def _make_step_body(
    prior: BasePrior,
    measure: BaseODEInformation,
    atol: float,
    rtol: float,
    *,
    calibration: CalibrationMode = "dynamic",
    min_sigma_sqr: float = 0.0,
    correction: Correction | None = None,
) -> Callable[[ArrayLike, ArrayLike, Array, Array], tuple]:
    """Construct and jit the per-step body of the adaptive loop.

    Args:
        prior: Gauss-Markov prior; supplies ``A(h)``, ``b(h)``, ``Q(h)``.
        measure: Measurement model.
        atol: Absolute tolerance for the normalised local-error estimate.
        rtol: Relative tolerance for the normalised local-error estimate.
        calibration: ``"dynamic"`` (default), ``"diagonal"``, ``"diagonal_ekf0"``,
            or ``"none"``. See module docstring for the model each scheme
            corresponds to.
        correction: Linearization/correction strategy for the measurement update
            (default ``TaylorCorrection(order=1)``, i.e. EK1). Enables EK0 /
            IEKF in the adaptive solver.

    The returned ``step_body`` produces (in order):
      ``m_pred, P_pred_sqr, G_back, d_back, P_back_sqr, mz, Pz_sqr, m, P_sqr,
      sigma_sqr_vec, err_per_step, loglik_step, D_unscaled_sqr_diag, m_value``.

    The last two outputs let a driver recompute ``err`` with a smoothed sigma
    (e.g. ``sigma_in_error="running_mean"``). ``sigma_sqr_vec`` is always a
    length-``d`` array (broadcast from the scalar in non-diagonal modes), so
    downstream code has a uniform shape.
    """
    E0_state = prior.E0_state
    E1 = prior.E1
    d = E0_state.shape[0]
    d_ode = measure.ode_dim

    def step_body(
        h: ArrayLike,
        t_next: ArrayLike,
        m_prev: Array,
        P_prev_sqr: Array,
    ) -> tuple:
        A_h = prior.A(h)
        b_h = prior.b(h)
        Q_h_sqr = prior.Q_sqr(h)

        # Mean is independent of the prior covariance, so it can be used to
        # linearise the measurement model up front.
        m_pred_provisional = A_h @ m_prev + b_h
        H_t, c_t = measure.linearize(m_pred_provisional, t=t_next)
        mz_pred = H_t @ m_pred_provisional + c_t

        # Calibration uses only the ODE-defect rows of the stacked residual
        # (Bosch, Tronarp, Hennig 2022 sec. 3). Conservation and observation
        # rows still update the posterior and contribute to the
        # log-likelihood; they just do not drive sigma.
        H_ode = H_t[:d_ode]
        mz_ode = mz_pred[:d_ode]

        sigma_sqr, Q_step_sqr = _calibrate_diffusion(
            calibration, mz_ode, H_ode, E1, Q_h_sqr, prior, min_sigma_sqr
        )
        # The adaptive driver stores/uses a per-component vector; broadcast the
        # scalar (dynamic / none) modes to length d.
        sigma_vec = (
            sigma_sqr
            if calibration in ("diagonal", "diagonal_ekf0")
            else sigma_sqr * np.ones(d)
        )

        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz, Pz_sqr),
            (m, P_sqr),
        ) = ekf1_sqr_filter_step(
            A_h,
            b_h,
            Q_step_sqr,
            m_prev,
            P_prev_sqr,
            measure,
            t=t_next,
            correction=correction,
        )

        # Local error estimate (Bosch et al. 2021 Eq. 49; Bosch et al. 2022
        # sec. 3: error vector has the dimension of the ODE solution).
        # diag(E0_state Q E0_state.T) is the per-component, uncalibrated
        # state-value variance; the driver multiplies it by whatever
        # sigma_for_err it chooses. Computed from the square-root Q_h_sqr
        # (Q = Q_sqr.T @ Q_sqr) as the row-norms of E0_state @ Q_h_sqr.T, so the
        # dense Q is never formed.
        E0Q_sqr = E0_state @ Q_h_sqr.T
        D_unscaled_sqr_diag = np.sum(E0Q_sqr * E0Q_sqr, axis=1)
        D = np.sqrt(np.maximum(sigma_vec * D_unscaled_sqr_diag, 0.0))
        m_value = E0_state @ m_pred
        err = _local_error_norm(D, m_value, atol, rtol)

        # Per-step log-likelihood (reflects whatever scaling was applied).
        loglik_step = _log_likelihood_contrib(mz, Pz_sqr)

        return (
            m_pred,
            P_pred_sqr,
            G_back,
            d_back,
            P_back_sqr,
            mz,
            Pz_sqr,
            m,
            P_sqr,
            sigma_vec,
            err,
            loglik_step,
            D_unscaled_sqr_diag,
            m_value,
        )

    return jax.jit(step_body)


def ekf1_sqr_adaptive_loop(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    *,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    h_init: float | None = None,
    h_min: float = 1e-10,
    h_max: float | None = None,
    controller: StepSizeController | None = None,
    calibration: CalibrationMode = "dynamic",
    sigma_in_error: SigmaInError = "running_mean",
    min_sigma_sqr: float = 0.0,
    max_steps: int = 100_000,
    correction: Correction | None = None,
) -> AdaptiveLoopResult:
    """Adaptive-step square-root EKF with per-step diffusion calibration.

    Python ``while`` driver around a jitted per-step body. Every accepted
    step contributes to the returned sequences and the result carries the
    smoothing-relevant outputs, so it can be fed directly to
    :func:`rts_sqr_smoother_loop`.

    Args:
        mu_0: Initial state mean.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (e.g. :class:`IWP`). The Kronecker structure
            ``Q(h) = kron(_Q(h), xi)`` is used to build the per-step process
            noise. ``prior.q`` sets the default controller order.
        measure: Measurement model (e.g. :class:`ODEInformation`).
        tspan: Time interval ``(t_start, t_end)``. Must be a tuple (hashable
            for jit-compatible inner step).
        atol: Absolute tolerance on the function-value local error.
        rtol: Relative tolerance.
        h_init: Initial step. Defaults to ``(t_end - t_start) / 100``.
        h_min: Minimum step. Steps below this raise ``RuntimeError``.
        h_max: Maximum step. Defaults to ``t_end - t_start``.
        controller: Step controller implementing
            :class:`~ode_filters.filters.adaptive_controller.StepSizeController`.
            Defaults to ``PIController(order=prior.q)``; pass
            :class:`~ode_filters.filters.adaptive_controller.PController` for a
            memoryless proportional-only controller.
        calibration: How the per-step ``sigma_hat^2`` enters the stored
            posterior. See the module docstring for the four available modes:

            - ``"dynamic"`` (default) -- Bosch et al. 2021 Eq. 32. Scalar
              ``sigma_hat^2`` from ``H Q(h) H.T`` baked into ``Q_h``.
              Honest per-step but **scalar**: fails on multi-scale ODE
              systems (one component's residual dominates ``sigma_hat^2``
              and starves the others).
            - ``"diagonal_ekf0"`` -- per-component
              ``sigma_hat^2_i = m_z[i]^2 / (E_1 Q E_1.T)_ii`` using the
              EK0 observation matrix ``H_0 = E_1``. The denominator is
              *exactly* diagonal when ``xi`` is diagonal, so the
              per-component estimator is the genuine MLE in a
              block-diagonal sub-model. The recommended choice for
              multi-component problems. Requires ``prior.xi`` diagonal.
            - ``"diagonal"`` -- same formula, but with the EK1 Jacobian
              ``H_t = E_1 - J_f E_0`` in the denominator. Heuristic:
              taking the diagonal of a generally-dense
              ``H_t Q(h) H_t.T`` ignores cross-component coupling from
              ``J_f``. Equivalent to ``"diagonal_ekf0"`` when ``J_f`` is
              block-diagonal (e.g. fully decoupled RHS); biased in
              proportion to the off-diagonal entries of ``J_f``
              otherwise. Requires ``prior.xi`` diagonal.
            - ``"none"`` -- propagate ``sigma=1``, still report
              ``sigma_hat^2`` for post-hoc rescaling / diagnostics.

            All four modes work with :class:`JointPrior` /
            :class:`PrecondJointPrior`: the diagonal modes scale only the
            state block per-component and require ``prior._prior_x.xi`` to
            be diagonal.

            For the diagonal modes the per-step entries of
            ``result.sigma_sqr_seq`` are length-``d`` arrays; for the
            scalar modes they are Python floats.
        sigma_in_error: How ``sigma_hat^2`` enters the local-error estimate.

            - ``"running_mean"`` (default) -- use the cumulative running
              mean of past accepted-step ``sigma_hat^2`` values in the
              error formula (``sigma_hat^2`` is still per-step in the Q
              calibration). The running mean has variance ``2/(nd)``, so
              estimator noise vanishes after a few accepted steps -- ``h``
              becomes much smoother and reject counts typically drop by
              several times. For ``"diagonal"`` modes the running mean is
              per-component.
            - ``"per_step"`` -- use the just-computed ``sigma_hat^2`` (Bosch
              et al. 2021 original recipe). Honest but inherits the
              chi-squared estimator noise (relative std ``sqrt(2/d)``),
              which propagates into step-size decisions and causes visible
              h oscillations in flat regions.
        min_sigma_sqr: Lower bound applied to the per-step ``sigma_hat^2``
            (and to each component of the per-component vector in diagonal
            modes) before it is baked into ``Q_step_sqr``. Default ``0.0``
            preserves the unclamped behavior. Use a small positive value
            (e.g. ``1e-30``) on problems where the trivial zero-residual
            fixed point would otherwise collapse the state-block diffusion
            to zero and propagate NaN.
        max_steps: Hard cap on iterations (rejected + accepted) as a safety
            valve against infinite loops.

    Returns:
        :class:`AdaptiveLoopResult` with the accepted trajectory.

    Raises:
        RuntimeError: If the controller proposes a step below ``h_min`` or the
            iteration cap is exceeded.
        ValueError: If ``tspan`` is non-increasing or ``calibration`` /
            ``sigma_in_error`` is unknown.
    """
    if calibration not in _VALID_CALIBRATIONS:
        raise ValueError(
            f"calibration must be one of {_VALID_CALIBRATIONS}; got {calibration!r}."
        )
    if sigma_in_error not in ("per_step", "running_mean"):
        raise ValueError(
            f"sigma_in_error must be 'per_step' or 'running_mean'; "
            f"got {sigma_in_error!r}."
        )
    if calibration in ("diagonal", "diagonal_ekf0"):
        _check_state_xi_diagonal(prior, calibration)
    t_start, t_end = float(tspan[0]), float(tspan[1])
    if t_end <= t_start:
        raise ValueError(f"tspan must be increasing; got {tspan!r}")
    span = t_end - t_start
    if h_init is None:
        h_init = span / 100.0
    if h_max is None:
        h_max = span
    if controller is None:
        controller = PIController(order=max(int(prior.q), 1))

    step_body = _make_step_body(
        prior,
        measure,
        atol,
        rtol,
        calibration=calibration,
        min_sigma_sqr=min_sigma_sqr,
        correction=correction,
    )

    t = t_start
    h = float(min(h_init, h_max))
    m_curr = mu_0
    P_curr_sqr = Sigma_0_sqr
    err_prev: float | None = None
    d_components = int(prior.E0_state.shape[0])
    is_diagonal = calibration in ("diagonal", "diagonal_ekf0")

    t_list: list[float] = [t_start]
    m_seq: list[Array] = [mu_0]
    P_seq_sqr: list[Array] = [Sigma_0_sqr]
    m_pred_seq: list[Array] = []
    P_pred_seq_sqr: list[Array] = []
    G_back_seq: list[Array] = []
    d_back_seq: list[Array] = []
    P_back_seq_sqr: list[Array] = []
    mz_seq: list[Array] = []
    Pz_seq_sqr: list[Array] = []
    sigma_sqr_seq: list = []  # list[float] in scalar modes, list[Array] in diagonal
    h_seq: list[float] = []
    n_rejected = 0
    log_likelihood = 0.0
    iters = 0

    # Running-mean accumulator for sigma_in_error="running_mean".
    sigma_running_sum = onp.zeros(d_components, dtype=float)
    n_accepted_so_far = 0

    # Tolerance for "we are essentially at the endpoint". Accumulating
    # ``t = t + h_try`` in float64 typically drifts by O(span * 2^-52) over
    # many steps; allowing a small endpoint slack avoids a spurious
    # sub-h_min residual step at termination when ``span / h`` does not
    # divide evenly in fp64.
    endpoint_tol = max(1e-12 * span, 1e-14)

    while t_end - t > endpoint_tol:
        if iters >= max_steps:
            raise RuntimeError(
                f"Adaptive loop exceeded max_steps={max_steps} "
                f"(at t={t:.6g}, h={h:.3g})."
            )
        iters += 1

        h_try = min(h, t_end - t)
        if h_try < h_min:
            raise RuntimeError(
                f"Proposed step h={h_try:.3g} below h_min={h_min:.3g} "
                f"at t={t:.6g} (clamped to t_end)."
            )

        t_next = t + h_try
        (
            m_pred,
            P_pred_sqr,
            G_back,
            d_back,
            P_back_sqr,
            mz,
            Pz_sqr,
            m_new,
            P_new_sqr,
            sigma_vec,
            err_per_step,
            loglik_step,
            D_unscaled_sqr_diag,
            m_value,
        ) = step_body(h_try, t_next, m_curr, P_curr_sqr)

        # Decide which sigma feeds the local-error / step-size decision.
        if sigma_in_error == "running_mean" and n_accepted_so_far > 0:
            sigma_for_err = sigma_running_sum / n_accepted_so_far
            scale = atol + rtol * onp.abs(onp.asarray(m_value))
            D = onp.sqrt(
                onp.maximum(sigma_for_err * onp.asarray(D_unscaled_sqr_diag), 0.0)
            )
            err_val = float(onp.sqrt(onp.mean((D / scale) ** 2)))
        else:
            err_val = float(err_per_step)

        if err_val <= 1.0:
            # Calibration was baked into Q_step_sqr inside the step body;
            # the returned covariances are already correct.
            sigma_arr = onp.asarray(sigma_vec)
            sigma_running_sum = sigma_running_sum + sigma_arr
            n_accepted_so_far += 1

            m_pred_seq.append(m_pred)
            P_pred_seq_sqr.append(P_pred_sqr)
            G_back_seq.append(G_back)
            d_back_seq.append(d_back)
            P_back_seq_sqr.append(P_back_sqr)
            mz_seq.append(mz)
            Pz_seq_sqr.append(Pz_sqr)
            m_seq.append(m_new)
            P_seq_sqr.append(P_new_sqr)
            # Scalar modes store a float (back-compat); diagonal stores array.
            sigma_sqr_seq.append(sigma_arr if is_diagonal else float(sigma_arr[0]))
            h_seq.append(h_try)
            t_list.append(t_next)
            log_likelihood += float(loglik_step)
            m_curr = m_new
            P_curr_sqr = P_new_sqr
            t = t_next
            h = min(h_max, controller.propose(h_try, err_val, err_prev))
            err_prev = err_val
        else:
            n_rejected += 1
            h = controller.propose(h_try, err_val, err_prev=None)
            # err_prev unchanged: only successful steps update the I-term memory.

    return AdaptiveLoopResult(
        t_seq=np.asarray(t_list),
        m_seq=m_seq,
        P_seq_sqr=P_seq_sqr,
        m_pred_seq=m_pred_seq,
        P_pred_seq_sqr=P_pred_seq_sqr,
        G_back_seq=G_back_seq,
        d_back_seq=d_back_seq,
        P_back_seq_sqr=P_back_seq_sqr,
        mz_seq=mz_seq,
        Pz_seq_sqr=Pz_seq_sqr,
        sigma_sqr_seq=sigma_sqr_seq,
        h_seq=h_seq,
        n_rejected=n_rejected,
        log_likelihood=log_likelihood,
    )


class AdaptiveSolveResult(NamedTuple):
    """Output of :func:`ekf1_sqr_adaptive_solve` (fixed-shape, save-at-grid).

    Attributes:
        t: The save grid (the ``save_at`` times), shape ``[K]``.
        m: Filtered state means at the save times, shape ``[K, state_dim]``.
        P_sqr: Square-root covariances at the save times, shape
            ``[K, state_dim, state_dim]`` (``P = P_sqr.T @ P_sqr``).
        log_likelihood: Accumulated Gaussian log-marginal-likelihood (scalar),
            summed over every accepted step.
        success: Scalar boolean -- ``True`` if the adaptive sub-stepping reached
            the final save time with a finite log-likelihood. ``False`` signals a
            failed solve (e.g. a blow-up exhausted ``max_steps`` before the
            endpoint); the returned ``m`` / ``P_sqr`` then hold the last accepted
            (finite) state rather than NaN. Because the solver is jittable it
            cannot raise -- check this flag instead.
    """

    t: Array
    m: Array
    P_sqr: Array
    log_likelihood: Array
    success: Array


def _controller_coeffs(
    controller: StepSizeController | None, order: int
) -> tuple[float, float, float, float, float]:
    """Static ``(safety, alpha, beta, min_factor, max_factor)`` for the jnp PI law.

    The dataclass controllers cast through Python ``float`` / ``min`` / ``max`` and
    so cannot run under tracing; here we extract their (static) coefficients and
    re-implement the proposal in ``jax.numpy`` inside the loop. ``beta = 0``
    recovers the proportional-only :class:`PController`.
    """
    if controller is None:
        controller = PIController(order=order)
    beta = float(controller._beta) if isinstance(controller, PIController) else 0.0
    return (
        float(controller.safety),
        float(controller._alpha),
        beta,
        float(controller.min_factor),
        float(controller.max_factor),
    )


def ekf1_sqr_adaptive_solve(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    save_at: Array,
    *,
    obs_model: ObsModel | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    h_init: float | None = None,
    calibration: CalibrationMode = "dynamic",
    controller: StepSizeController | None = None,
    min_sigma_sqr: float = 0.0,
    max_steps: int = 4096,
    correction: Correction | None = None,
) -> AdaptiveSolveResult:
    """``jit`` / ``vmap`` / ``grad``-able adaptive EKF1, saved on a fixed grid.

    Unlike :func:`ekf1_sqr_adaptive_loop` (a Python ``while`` driver that returns
    *every* accepted step and feeds the smoother), this returns the **filtered
    solution at a fixed array of query times** ``save_at`` -- the shape is known at
    trace time, so the whole solve is jittable, vmappable, and reverse-mode
    differentiable. It is built as a ``jax.lax.scan`` over ``save_at`` whose body is
    a *checkpointed* ``equinox`` while-loop (``eqx.internal.while_loop``); the
    checkpointing is what makes reverse-mode autodiff work (plain ``lax.while_loop``
    does not support it).

    Adaptive accept/reject sub-stepping happens *between* consecutive save times;
    the final sub-step of each interval is clamped to land exactly on the next save
    time, so no interpolation is needed (adaptivity within an interval is preserved
    -- only that last sub-step is shortened). Filtering only: the RTS smoother stays
    on the fixed-grid path.

    **Observations at fixed locations.** Pass ``obs_model`` to assimilate linear
    observations: the adaptive solver integrates to each save time and then applies
    a masked affine observation update there. Because the save times are already
    mandatory landing points, observations must be aligned to ``save_at`` -- build
    the model with ``prepare_observations(measurements, prior.E0, save_at)`` (its
    per-step offset/mask sequences then index ``save_at[1:]``). The update is exact
    (observations are linear, so no linearization/correction is needed) and the
    post-observation state is what propagates onward (proper filtering). Data
    observations always go through ``obs_model``; ``measure`` carries only the ODE
    information and Conservation constraints.

    The per-step calibration, local-error estimate and log-likelihood are shared
    with :func:`ekf1_sqr_adaptive_loop` (same ``_make_step_body``); the controller
    uses the per-step error (the ``sigma_in_error="running_mean"`` variant of the
    Python loop is not reproduced here).

    Args:
        mu_0: Initial state mean.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Gauss-Markov prior (e.g. :class:`IWP`); supplies ``A``/``b``/``Q``
            and the default controller order ``prior.q``.
        measure: Measurement model (ODE + Conservation only).
        save_at: Strictly increasing 1-D array of save times; ``save_at[0]`` is the
            initial time (the initial state is returned there unchanged).
        obs_model: Optional linear observations to assimilate at the save times,
            built via ``prepare_observations(measurements, prior.E0, save_at)`` (so
            its ``c_seq``/``mask`` index ``save_at[1:]``). ``None`` for a pure solve.
        atol: Absolute tolerance for the normalised local-error estimate.
        rtol: Relative tolerance.
        h_init: Initial step. Defaults to ``(save_at[-1] - save_at[0]) / 100``.
        calibration: Diffusion calibration mode (see :func:`ekf1_sqr_adaptive_loop`).
        controller: Step-size controller; defaults to ``PIController(order=prior.q)``.
        min_sigma_sqr: Lower bound on the per-step ``sigma_hat^2``.
        max_steps: Hard cap on sub-steps per save interval (bounds the checkpointed
            while-loop). Raise it (or loosen tolerances) if a solve fails to reach a
            save time.
        correction: Linearization/correction strategy for the ODE measurement
            update (default ``TaylorCorrection(order=1)``, i.e. EK1); pass
            ``TaylorCorrection(order=0)`` for EK0 or ``IteratedTaylorCorrection``
            for IEKF. The observation update (if ``obs_model`` is given) is linear
            and unaffected.

    Returns:
        An :class:`AdaptiveSolveResult` with the solution sampled at ``save_at``.
    """
    save_at = np.asarray(save_at, dtype=float)
    safety, alpha, beta, min_factor, max_factor = _controller_coeffs(
        controller, prior.q
    )
    step_body = _make_step_body(
        prior,
        measure,
        atol,
        rtol,
        calibration=calibration,
        min_sigma_sqr=min_sigma_sqr,
        correction=correction,
    )
    span = save_at[-1] - save_at[0]
    h0 = span / 100.0 if h_init is None else np.asarray(h_init, dtype=float)

    def propose(h, err, err_prev):
        # Gustafsson PI law (jax form of adaptive_controller.PIController.propose);
        # err_prev < 0 signals "no memory" (first step / right after a reject) and
        # drops the integral term.
        # A non-finite error (NaN/inf from a blow-up) is mapped to +inf so the
        # proposed factor collapses to ``min_factor`` (a shrink) instead of
        # propagating NaN into ``h`` -- which would otherwise turn ``h_try`` NaN
        # and spin the checkpointed while-loop to ``max_steps`` with no progress.
        # (cast: jnp.where's overloads widen to Array | tuple under pyright.)
        err = cast(Array, np.where(np.isfinite(err), err, np.inf))
        err = np.maximum(err, 1e-12)
        proportional = err ** (-alpha)
        integral = np.where(
            err_prev > 0.0, (np.maximum(err_prev, 1e-12) / err) ** beta, 1.0
        )
        factor = np.clip(safety * proportional * integral, min_factor, max_factor)
        return h * factor

    def integrate_to(target, carry):
        rel_tol = 1e-10 * np.abs(target) + 1e-12

        def cond(c):
            t = c[0]
            return t < target - rel_tol

        def body(c):
            t, m, P_sqr, h, ll, err_prev = c
            h_try = np.minimum(h, target - t)  # clamp so we land on `target`
            t_next = t + h_try
            out = step_body(h_try, t_next, m, P_sqr)
            m_new, P_new_sqr = out[7], out[8]
            err, loglik_step = out[10], out[11]
            # Reject non-finite errors (NaN/inf): never accept a NaN state, and
            # ``propose`` shrinks ``h`` so the loop can recover instead of
            # silently emitting NaN.
            accept = (err <= 1.0) & np.isfinite(err)
            t2 = np.where(accept, t_next, t)
            m2 = np.where(accept, m_new, m)
            P2 = np.where(accept, P_new_sqr, P_sqr)
            ll2 = np.where(accept, ll + loglik_step, ll)
            h2 = propose(h_try, err, np.where(accept, err_prev, -1.0))
            err_prev2 = np.where(accept, err, err_prev)
            return (t2, m2, P2, h2, ll2, err_prev2)

        return eqxi.while_loop(
            cond, body, carry, max_steps=max_steps, kind="checkpointed"
        )

    init = (save_at[0], mu_0, Sigma_0_sqr, h0, np.array(0.0), np.array(-1.0))

    if obs_model is None:

        def scan_body_plain(carry, target):
            carry = integrate_to(target, carry)
            return carry, (carry[1], carry[2])

        final, (m_seq, P_seq_sqr) = jax.lax.scan(scan_body_plain, init, save_at[1:])
    else:
        n_obs_steps = obs_model.c_seq.shape[0]
        if n_obs_steps != save_at.shape[0] - 1:
            raise ValueError(
                f"obs_model has {n_obs_steps} steps but save_at has "
                f"{save_at.shape[0]} points; build it with "
                f"prepare_observations(measurements, prior.E0, save_at) so its "
                f"per-step sequences index save_at[1:]."
            )
        H_obs = obs_model.H
        R_obs_sqr = obs_model.R_sqr

        def scan_body_obs(carry, step_data):
            target, c_obs, mask = step_data
            t, m, P_sqr, h, ll, err_prev = integrate_to(target, carry)
            # Exact (linear) observation update at the save time, masked off when
            # no observation is active there (same all-or-nothing convention as the
            # dynamic-observation scan loop).
            obs_active = mask.any()
            mz_obs, Pz_obs_sqr = sqr_marginalization(H_obs, c_obs, R_obs_sqr, m, P_sqr)
            _, m_obs, P_obs_sqr = sqr_inversion(
                H_obs, m, P_sqr, mz_obs, Pz_obs_sqr, R_obs_sqr
            )
            m = np.where(obs_active, m_obs, m)
            P_sqr = np.where(obs_active, P_obs_sqr, P_sqr)
            ll = ll + np.where(
                obs_active, _log_likelihood_contrib(mz_obs, Pz_obs_sqr), 0.0
            )
            return (t, m, P_sqr, h, ll, err_prev), (m, P_sqr)

        final, (m_seq, P_seq_sqr) = jax.lax.scan(
            scan_body_obs, init, (save_at[1:], obs_model.c_seq, obs_model.mask)
        )

    m_seq = cast(Array, m_seq)
    P_seq_sqr = cast(Array, P_seq_sqr)
    m_out = np.concatenate([mu_0[None], m_seq], axis=0)
    P_out = np.concatenate([Sigma_0_sqr[None], P_seq_sqr], axis=0)
    # The solve succeeded iff the sub-stepping reached the final save time
    # (``final[0]`` is the running time after the last interval) with a finite
    # log-likelihood. A blow-up that exhausts ``max_steps`` leaves ``final[0]``
    # short of ``save_at[-1]``.
    end_tol = 1e-10 * np.abs(save_at[-1]) + 1e-12
    success = (final[0] >= save_at[-1] - end_tol) & np.isfinite(final[4])
    return AdaptiveSolveResult(
        t=save_at, m=m_out, P_sqr=P_out, log_likelihood=final[4], success=success
    )


__all__ = [
    "AdaptiveLoopResult",
    "AdaptiveSolveResult",
    "ekf1_sqr_adaptive_loop",
    "ekf1_sqr_adaptive_solve",
]
