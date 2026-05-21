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

- ``"cumulative"`` (legacy multiplicative scheme): scalar ``sigma_hat^2_n``
  from the *full* noise-free predicted-obs covariance ``H P_pred H.T``;
  post-multiplies the whole step's covariance, so past contributions
  inherit every subsequent ``sigma_hat^2``. Non-Markovian by design; an
  online empirical-Bayes update of a single global unknown ``sigma``. *Not*
  per-step honest, but stays robust on multi-scale problems (the inflated
  carried ``P`` accidentally compensates). ``"diagonal_ekf0"`` is strictly
  better when applicable; keep ``"cumulative"`` for non-diagonal Xi or for
  backward compatibility with earlier runs.

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

The accepted-step outputs are stored in lists matching the shape conventions of
:func:`ekf1_sqr_loop`, so the result can be passed directly to
:func:`rts_sqr_smoother_loop`.

The returned ``log_likelihood`` is the *post-calibration* marginal
likelihood: each step's contribution uses the calibrated ``Pz_sqr`` (which
includes ``sigma_hat^2`` in the dynamic/diagonal modes and the
post-multiplied scale in cumulative mode). It is therefore the right
quantity for inference *given* the chosen calibration model, but is **not**
directly comparable across calibration modes -- different ``calibration``
settings define different generative models.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal, NamedTuple

import jax
import jax.numpy as np
import numpy as onp
from jax import Array

from ..calibration.sigma import quasi_mle_sigma_sqr, quasi_mle_sigma_sqr_from_Q
from ..inference.sqr_gaussian_inference import sqr_marginalization
from ..measurement.measurement_models import BaseODEInformation
from ..priors.gmp_priors import BasePrior
from .adaptive_controller import PIController, StepSizeController
from .ode_filter_step import ekf1_sqr_filter_step

CalibrationMode = Literal["dynamic", "cumulative", "diagonal", "diagonal_ekf0", "none"]
SigmaInError = Literal["per_step", "running_mean"]
_VALID_CALIBRATIONS = ("dynamic", "cumulative", "diagonal", "diagonal_ekf0", "none")


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
) -> Callable[[float, float, Array, Array], tuple]:
    """Construct and jit the per-step body of the adaptive loop.

    Args:
        prior: Gauss-Markov prior; supplies ``A(h)``, ``b(h)``, ``Q(h)``.
        measure: Measurement model.
        atol: Absolute tolerance for the normalised local-error estimate.
        rtol: Relative tolerance for the normalised local-error estimate.
        calibration: ``"dynamic"`` (default), ``"cumulative"``, ``"diagonal"``,
            or ``"none"``. See module docstring for the model each scheme
            corresponds to.

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

    @jax.jit
    def step_body(
        h: float,
        t_next: float,
        m_prev: Array,
        P_prev_sqr: Array,
    ) -> tuple:
        A_h = prior.A(h)
        b_h = prior.b(h)
        Q_h = prior.Q(h)
        Q_h_sqr = np.linalg.cholesky(Q_h).T

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

        if calibration == "cumulative":
            # Cumulative: sigma from the full noise-free predicted-obs
            # covariance H P_pred H.T (P_pred includes A P_prev A.T plus Q),
            # restricted to the ODE rows.
            _m_pred_pre, P_pred_pre_sqr = sqr_marginalization(
                A_h, b_h, Q_h_sqr, m_prev, P_prev_sqr
            )
            zero_R_sqr = np.zeros((d_ode, d_ode))
            _, Pz_calib_sqr = sqr_marginalization(
                H_ode, c_t[:d_ode], zero_R_sqr, _m_pred_pre, P_pred_pre_sqr
            )
            sigma_scalar = quasi_mle_sigma_sqr(mz_ode, Pz_calib_sqr)
            sigma_scalar = np.maximum(sigma_scalar, min_sigma_sqr)
            sigma_vec = sigma_scalar * np.ones(d)
            Q_step_sqr = Q_h_sqr
        elif calibration in ("diagonal", "diagonal_ekf0"):
            # Per-component sigma_hat^2_i.
            #   "diagonal":      denom = (H1 Q H1.T)_ii  (EKF1 H, dense
            #                    matrix's diagonal; uses info the step
            #                    already has, marginally better on
            #                    single-component problems).
            #   "diagonal_ekf0": denom = (E1  Q  E1.T)_ii (EKF0 H = E1,
            #                    the matrix is *exactly* diagonal when xi
            #                    is diagonal -- mathematically clean; can
            #                    be slightly better on multi-scale at
            #                    loose tolerance).
            H_for_calib = E1 if calibration == "diagonal_ekf0" else H_ode
            denom = np.einsum("ij,jk,ik->i", H_for_calib, Q_h, H_for_calib)
            sigma_vec = mz_ode**2 / denom
            sigma_vec = np.maximum(sigma_vec, min_sigma_sqr)
            Q_step_sqr = prior.apply_state_sigma_sqr(Q_h_sqr, sigma_vec)
        else:
            # Dynamic / none: sigma from H_ode Q(h) H_ode.T only (process noise).
            sigma_scalar = quasi_mle_sigma_sqr_from_Q(mz_ode, H_ode, Q_h_sqr)
            sigma_scalar = np.maximum(sigma_scalar, min_sigma_sqr)
            sigma_vec = sigma_scalar * np.ones(d)
            if calibration == "dynamic":
                Q_step_sqr = prior.apply_state_sigma_sqr(Q_h_sqr, sigma_scalar)
            else:  # "none"
                Q_step_sqr = Q_h_sqr

        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz, Pz_sqr),
            (m, P_sqr),
        ) = ekf1_sqr_filter_step(
            A_h, b_h, Q_step_sqr, m_prev, P_prev_sqr, measure, t=t_next
        )

        if calibration == "cumulative":
            # Post-multiply the carried covariance. For joint priors only
            # the state block is scaled (delegated to the prior). Pz_sqr
            # lives in measurement-row space, so we column-scale just the
            # ODE rows there.
            sigma_scalar = sigma_vec[0]
            P_pred_sqr = prior.apply_state_sigma_to_cov_sqr(P_pred_sqr, sigma_scalar)
            P_back_sqr = prior.apply_state_sigma_to_cov_sqr(P_back_sqr, sigma_scalar)
            P_sqr = prior.apply_state_sigma_to_cov_sqr(P_sqr, sigma_scalar)
            Pz_sqr = Pz_sqr.at[:, :d_ode].multiply(np.sqrt(sigma_scalar))

        # Local error estimate (Bosch et al. 2021 Eq. 49; Bosch et al. 2022
        # sec. 3: error vector has the dimension of the ODE solution).
        # diag(E0_state Q E0_state.T) is the per-component, uncalibrated
        # state-value variance; the driver multiplies it by whatever
        # sigma_for_err it chooses.
        D_unscaled_sqr_diag = np.diag(E0_state @ Q_h @ E0_state.T)
        D = np.sqrt(np.maximum(sigma_vec * D_unscaled_sqr_diag, 0.0))
        m_value = E0_state @ m_pred
        err = _local_error_norm(D, m_value, atol, rtol)

        # Per-step log-likelihood (reflects whatever scaling was applied).
        log_det = 2.0 * np.sum(np.log(np.abs(np.diag(Pz_sqr))))
        v = jax.scipy.linalg.solve_triangular(Pz_sqr.T, mz, lower=True)
        maha = v @ v
        obs_dim = mz.shape[0]
        loglik_step = -0.5 * (obs_dim * np.log(2 * np.pi) + log_det + maha)

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

    return step_body


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
    calibrate: bool | None = None,
    max_steps: int = 100_000,
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
            - ``"cumulative"`` -- legacy multiplicative scheme; scalar sigma
              from the full ``H P_pred H.T`` post-multiplies the whole step.
              Non-Markovian; robust on multi-scale by accident (inflated P
              inherited from earlier transitions). Prefer
              ``"diagonal_ekf0"``.
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
        calibrate: Deprecated. ``True`` maps to ``calibration="dynamic"``,
            ``False`` maps to ``calibration="none"``. Pass ``calibration``
            directly instead.
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
    if calibrate is not None:
        warnings.warn(
            '`calibrate` is deprecated; use `calibration="dynamic"` '
            '(or "none" for the old `calibrate=False` behaviour).',
            DeprecationWarning,
            stacklevel=2,
        )
        calibration = "dynamic" if calibrate else "none"
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
        # Host-numpy on a host-converted xi so the bool stays concrete and
        # this entire validation can survive being traced by jit / grad.
        xi_state = onp.asarray(prior.xi_state)
        if not onp.allclose(xi_state - onp.diag(onp.diag(xi_state)), 0.0):
            raise ValueError(
                f"calibration={calibration!r} requires the state-block "
                "Xi to be diagonal (joint priors: this is "
                "``prior._prior_x.xi``)."
            )
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
                f"Proposed step h={h_try:.3g} below h_min={h_min:.3g} at t={t:.6g}."
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
            # Calibration was baked into Q_step_sqr inside the step body (or
            # post-multiplied for cumulative mode); the returned covariances
            # are already correct.
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


__all__ = [
    "AdaptiveLoopResult",
    "ekf1_sqr_adaptive_loop",
]
