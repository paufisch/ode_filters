"""Adaptive-step EKF loop with online sigma calibration.

The trajectory length is data-dependent (steps may be rejected and retried at
a smaller ``h``), so this loop cannot use ``jax.lax.scan``. Instead it is a
Python ``while`` driver around a jitted per-step body. The per-step body does
the prediction + update + per-step quasi-MLE sigma and returns the normalised
local-error estimate; the Python layer makes the accept/reject decision.

The accepted-step outputs are stored in lists matching the shape conventions of
:func:`ekf1_sqr_loop`, so the result can be passed directly to
:func:`rts_sqr_smoother_loop`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as np
from jax import Array

from ..calibration.sigma import quasi_mle_sigma_sqr
from ..inference.sqr_gaussian_inference import sqr_marginalization
from ..measurement.measurement_models import BaseODEInformation
from ..priors.gmp_priors import BasePrior
from .adaptive_controller import PIController, StepSizeController
from .ode_filter_step import ekf1_sqr_filter_step


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
) -> Callable[[float, float, Array, Array], tuple]:
    """Construct and jit the per-step body of the adaptive loop."""
    E0 = prior.E0

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

        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz, Pz_sqr),
            (m, P_sqr),
        ) = ekf1_sqr_filter_step(
            A_h, b_h, Q_h_sqr, m_prev, P_prev_sqr, measure, t=t_next
        )

        # Calibration target: the *prior-propagated* residual covariance
        # H P_pred H.T, excluding any measurement noise R that
        # ``measure.get_noise`` might add. Mixing R into the quasi-MLE would
        # conflate sigma^2 with the measurement noise (cf. probnum's
        # ``meas_rv_error_free`` and probdiffeq's noise-free observed RV).
        H_t, c_t = measure.linearize(m_pred, t=t_next)
        R_dim = mz.shape[0]
        zero_R_sqr = np.zeros((R_dim, R_dim))
        _, Pz_calib_sqr = sqr_marginalization(H_t, c_t, zero_R_sqr, m_pred, P_pred_sqr)
        sigma_sqr = quasi_mle_sigma_sqr(mz, Pz_calib_sqr)

        # Local error estimate (Bosch et al. 2021, Eq. 49):
        # D_i = sqrt(sigma^2 * (E0 Q(h) E0.T)_{ii})
        D_sqr = sigma_sqr * np.diag(E0 @ Q_h @ E0.T)
        D = np.sqrt(np.maximum(D_sqr, 0.0))
        m_value = E0 @ m_pred
        err = _local_error_norm(D, m_value, atol, rtol)

        # Per-step log-likelihood contribution (uncalibrated).
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
            sigma_sqr,
            err,
            loglik_step,
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
    calibrate: bool = True,
    max_steps: int = 100_000,
) -> AdaptiveLoopResult:
    """Adaptive-step square-root EKF with per-step diffusion calibration.

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
        calibrate: If ``True``, the stored covariances of an accepted step are
            multiplied by ``sqrt(sigma_hat^2)`` so the posterior is honestly
            scaled. If ``False``, sigma_hat is still computed and returned for
            diagnostics but covariances are stored uncalibrated.
        max_steps: Hard cap on iterations (rejected + accepted) as a safety
            valve against infinite loops.

    Returns:
        :class:`AdaptiveLoopResult` with the accepted trajectory.

    Raises:
        RuntimeError: If the controller proposes a step below ``h_min`` or the
            iteration cap is exceeded.
    """
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

    step_body = _make_step_body(prior, measure, atol, rtol)

    t = t_start
    h = float(min(h_init, h_max))
    m_curr = mu_0
    P_curr_sqr = Sigma_0_sqr
    err_prev: float | None = None

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
    sigma_sqr_seq: list[float] = []
    h_seq: list[float] = []
    n_rejected = 0
    log_likelihood = 0.0
    iters = 0

    while t < t_end:
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
            sigma_sqr,
            err,
            loglik_step,
        ) = step_body(h_try, t_next, m_curr, P_curr_sqr)

        err_val = float(err)
        sigma_val = float(sigma_sqr)

        if err_val <= 1.0:
            scale = float(np.sqrt(sigma_sqr)) if calibrate else 1.0
            m_pred_seq.append(m_pred)
            P_pred_seq_sqr.append(scale * P_pred_sqr)
            G_back_seq.append(G_back)
            d_back_seq.append(d_back)
            P_back_seq_sqr.append(scale * P_back_sqr)
            mz_seq.append(mz)
            Pz_seq_sqr.append(scale * Pz_sqr)
            m_seq.append(m_new)
            P_seq_sqr.append(scale * P_new_sqr)
            sigma_sqr_seq.append(sigma_val)
            h_seq.append(h_try)
            t_list.append(t_next)
            log_likelihood += float(loglik_step)
            m_curr = m_new
            P_curr_sqr = scale * P_new_sqr
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
