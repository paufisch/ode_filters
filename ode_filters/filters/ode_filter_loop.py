from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as np
from jax import Array

from ..measurement.measurement_models import BaseODEInformation, ScanData
from ..priors.gmp_priors import BasePrior
from .ode_filter_step import (
    ekf1_sqr_filter_step,
    ekf1_sqr_filter_step_preconditioned,
    ekf1_sqr_filter_step_scan,
    rts_sqr_smoother_step,
    rts_sqr_smoother_step_preconditioned,
)

StateFunction = Callable[[Array], Array]
JacobianFunction = Callable[[Array], Array]

LoopResult = tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    float,
]


def ekf1_sqr_loop(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
) -> LoopResult:
    """Run a square-root EKF over ``N`` observation steps.

    Args:
        mu_0: Initial state mean estimate.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (e.g., IWP or PrecondIWP).
        measure: Measurement model (e.g., ODEInformation or subclass).
        tspan: Time interval (t_start, t_end).
        N: Number of filter steps.

    Returns:
        Tuple of 9 arrays and a scalar log marginal likelihood.
    """

    m_seq = [mu_0]
    P_seq_sqr = [Sigma_0_sqr]
    m_pred_seq = []
    P_pred_seq_sqr = []
    G_back_seq = []
    d_back_seq = []
    P_back_seq_sqr = []
    mz_seq = []
    Pz_seq_sqr = []

    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_h = prior.A(h)
    b_h = prior.b(h)
    Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T

    log_likelihood = 0.0

    for i in range(N):
        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz, Pz_sqr),
            (m, P_sqr),
        ) = ekf1_sqr_filter_step(
            A_h,
            b_h,
            Q_h_sqr,
            m_seq[-1],
            P_seq_sqr[-1],
            measure,
            t=ts[i + 1],
        )

        log_det = 2.0 * np.sum(np.log(np.abs(np.diag(Pz_sqr))))
        v = jax.scipy.linalg.solve_triangular(Pz_sqr.T, mz, lower=True)
        maha = v @ v
        obs_dim = mz.shape[0]
        log_likelihood += -0.5 * (obs_dim * np.log(2 * np.pi) + log_det + maha)

        m_pred_seq.append(m_pred)
        P_pred_seq_sqr.append(P_pred_sqr)
        G_back_seq.append(G_back)
        d_back_seq.append(d_back)
        P_back_seq_sqr.append(P_back_sqr)
        mz_seq.append(mz)
        Pz_seq_sqr.append(Pz_sqr)
        m_seq.append(m)
        P_seq_sqr.append(P_sqr)

    return (
        m_seq,
        P_seq_sqr,
        m_pred_seq,
        P_pred_seq_sqr,
        G_back_seq,
        d_back_seq,
        P_back_seq_sqr,
        mz_seq,
        Pz_seq_sqr,
        log_likelihood,
    )


def rts_sqr_smoother_loop(
    m_N: Array,
    P_N_sqr: Array,
    G_back_seq: Array,
    d_back_seq: Array,
    P_back_seq_sqr: Array,
    N: int,
) -> tuple[Array, Array]:
    """Run a Rauch-Tung-Striebel smoother over ``N`` steps.

    Args:
        m_N: Final filtered state mean.
        P_N_sqr: Final filtered state covariance (square-root form).
        G_back_seq: Backward pass gain sequence from filter.
        d_back_seq: Backward pass offset sequence from filter.
        P_back_seq_sqr: Backward pass covariance sequence (square-root form) from filter.
        N: Number of smoothing steps.

    Returns:
        Tuple of smoothed state means and covariances (square-root form).
    """

    state_dim = m_N.shape[0]
    m_smooth = np.zeros((N + 1, state_dim))
    P_smooth_sqr = np.zeros((N + 1, state_dim, state_dim))
    m_smooth = m_smooth.at[-1].set(m_N)
    P_smooth_sqr = P_smooth_sqr.at[-1].set(P_N_sqr)

    for j in range(N - 1, -1, -1):
        m_j, P_j = rts_sqr_smoother_step(
            G_back_seq[j],
            d_back_seq[j],
            P_back_seq_sqr[j],
            m_smooth[j + 1],
            P_smooth_sqr[j + 1],
        )
        m_smooth = m_smooth.at[j].set(m_j)
        P_smooth_sqr = P_smooth_sqr.at[j].set(P_j)

    return m_smooth, P_smooth_sqr


def ekf1_sqr_loop_preconditioned(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
) -> tuple[
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    Array,
    float,
]:
    """Run a preconditioned square-root EKF over ``N`` observation steps.

    Args:
        mu_0: Initial state mean estimate.
        P_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (typically PrecondIWP).
        measure: Measurement model (e.g., ODEInformation or subclass).
        tspan: Time interval (t_start, t_end).
        N: Number of filter steps.

    Returns:
        Tuple of 11 arrays, preconditioning matrix, and scalar log marginal
        likelihood.
    """

    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    # For PrecondIWP, A() and b() don't use h, but accept it for API compatibility
    A_bar = prior.A()
    b_bar = prior.b()
    Q_sqr_bar = np.linalg.cholesky(prior.Q()).T
    T_h = prior.T(h)

    m_seq = [mu_0]
    P_seq_sqr = [P_0_sqr]
    m_seq_bar = [np.linalg.solve(T_h, mu_0)]
    P_seq_sqr_bar = [np.linalg.solve(T_h, P_0_sqr.T).T]

    m_pred_seq_bar = []
    P_pred_seq_sqr_bar = []
    G_back_seq_bar = []
    d_back_seq_bar = []
    P_back_seq_sqr_bar = []
    mz_seq = []
    Pz_seq_sqr = []

    log_likelihood = 0.0

    for i in range(N):
        (
            (m_pred_seq_bar_i, P_pred_seq_sqr_bar_i),
            (G_back_seq_bar_i, d_back_seq_bar_i, P_back_seq_sqr_bar_i),
            (mz_seq_i, Pz_seq_sqr_i),
            (m_seq_bar_next, P_seq_sqr_bar_next),
            (m_seq_next, P_seq_sqr_next),
        ) = ekf1_sqr_filter_step_preconditioned(
            A_bar,
            b_bar,
            Q_sqr_bar,
            T_h,
            m_seq_bar[-1],
            P_seq_sqr_bar[-1],
            measure,
            t=ts[i + 1],
        )

        log_det = 2.0 * np.sum(np.log(np.abs(np.diag(Pz_seq_sqr_i))))
        v = jax.scipy.linalg.solve_triangular(Pz_seq_sqr_i.T, mz_seq_i, lower=True)
        maha = v @ v
        obs_dim = mz_seq_i.shape[0]
        log_likelihood += -0.5 * (obs_dim * np.log(2 * np.pi) + log_det + maha)

        m_pred_seq_bar.append(m_pred_seq_bar_i)
        P_pred_seq_sqr_bar.append(P_pred_seq_sqr_bar_i)
        G_back_seq_bar.append(G_back_seq_bar_i)
        d_back_seq_bar.append(d_back_seq_bar_i)
        P_back_seq_sqr_bar.append(P_back_seq_sqr_bar_i)
        mz_seq.append(mz_seq_i)
        Pz_seq_sqr.append(Pz_seq_sqr_i)
        m_seq_bar.append(m_seq_bar_next)
        P_seq_sqr_bar.append(P_seq_sqr_bar_next)
        m_seq.append(m_seq_next)
        P_seq_sqr.append(P_seq_sqr_next)

    return (
        m_seq,
        P_seq_sqr,
        m_seq_bar,
        P_seq_sqr_bar,
        m_pred_seq_bar,
        P_pred_seq_sqr_bar,
        G_back_seq_bar,
        d_back_seq_bar,
        P_back_seq_sqr_bar,
        mz_seq,
        Pz_seq_sqr,
        T_h,
        log_likelihood,
    )


def rts_sqr_smoother_loop_preconditioned(
    m_N: Array,
    P_N_sqr: Array,
    m_N_bar: Array,
    P_N_sqr_bar: Array,
    G_back_seq_bar: Array,
    d_back_seq_bar: Array,
    P_back_seq_sqr_bar: Array,
    N: int,
    T_h: Array,
) -> tuple[Array, Array]:
    """Run a preconditioned Rauch-Tung-Striebel smoother over ``N`` steps.

    Args:
        m_N: Final filtered state mean (original space).
        P_N_sqr: Final filtered state covariance (square-root form, original space).
        m_N_bar: Final filtered state mean (preconditioned space).
        P_N_sqr_bar: Final filtered state covariance (square-root form, preconditioned space).
        G_back_seq_bar: Backward pass gain sequence (preconditioned).
        d_back_seq_bar: Backward pass offset sequence (preconditioned).
        P_back_seq_sqr_bar: Backward pass covariance sequence (square-root form, preconditioned).
        N: Number of smoothing steps.
        T_h: Preconditioning transformation matrix.

    Returns:
        Smoothed state means and covariances (square-root form, original space).
    """

    state_dim = m_N.shape[0]

    m_smooth = np.zeros((N + 1, state_dim))
    P_smooth_sqr = np.zeros((N + 1, state_dim, state_dim))
    m_smooth = m_smooth.at[-1].set(m_N)
    P_smooth_sqr = P_smooth_sqr.at[-1].set(P_N_sqr)
    m_smooth_bar = np.zeros((N + 1, state_dim))
    P_smooth_sqr_bar = np.zeros((N + 1, state_dim, state_dim))
    m_smooth_bar = m_smooth_bar.at[-1].set(m_N_bar)
    P_smooth_sqr_bar = P_smooth_sqr_bar.at[-1].set(P_N_sqr_bar)

    for j in range(N - 1, -1, -1):
        (m_bar_j, P_bar_j), (m_j, P_j) = rts_sqr_smoother_step_preconditioned(
            G_back_seq_bar[j],
            d_back_seq_bar[j],
            P_back_seq_sqr_bar[j],
            m_smooth_bar[j + 1],
            P_smooth_sqr_bar[j + 1],
            T_h,
        )
        m_smooth_bar = m_smooth_bar.at[j].set(m_bar_j)
        P_smooth_sqr_bar = P_smooth_sqr_bar.at[j].set(P_bar_j)
        m_smooth = m_smooth.at[j].set(m_j)
        P_smooth_sqr = P_smooth_sqr.at[j].set(P_j)

    return m_smooth, P_smooth_sqr


# =============================================================================
# Scan-based loop functions (jax.lax.scan compatible)
# =============================================================================


class ScanLoopResult(NamedTuple):
    """Result from ekf1_sqr_loop_scan.

    Attributes:
        m_seq: Filtered state means, shape [N+1, state_dim].
        P_seq_sqr: Filtered state covariances (sqrt), shape [N+1, state_dim, state_dim].
        m_pred_seq: Predicted state means, shape [N, state_dim].
        P_pred_seq_sqr: Predicted state covariances (sqrt), shape [N, state_dim, state_dim].
        G_back_seq: Backward pass gains, shape [N, state_dim, state_dim].
        d_back_seq: Backward pass offsets, shape [N, state_dim].
        P_back_seq_sqr: Backward pass covariances (sqrt), shape [N, state_dim, state_dim].
        mz_seq: Observation marginal means (padded), shape [N, max_obs_dim].
        Pz_seq_sqr: Observation marginal covariances (sqrt, padded),
            shape [N, max_obs_dim, max_obs_dim].
        log_likelihood: Log marginal likelihood (scalar).
        scan_data: Pre-computed scan data (includes obs_mask for extracting active dims).
    """

    m_seq: Array
    P_seq_sqr: Array
    m_pred_seq: Array
    P_pred_seq_sqr: Array
    G_back_seq: Array
    d_back_seq: Array
    P_back_seq_sqr: Array
    mz_seq: Array
    Pz_seq_sqr: Array
    log_likelihood: Array
    scan_data: ScanData


def ekf1_sqr_loop_scan(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
) -> ScanLoopResult:
    """Run a square-root EKF over ``N`` observation steps using jax.lax.scan.

    This is a scan-compatible version of ekf1_sqr_loop. It pre-computes
    measurement data once, then uses jax.lax.scan for the main loop.
    The observation marginal outputs (mz_seq, Pz_seq_sqr) have fixed shapes
    padded to max_obs_dim.

    Args:
        mu_0: Initial state mean estimate.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (e.g., IWP or PrecondIWP).
        measure: Measurement model (e.g., ODEInformation or subclass).
        tspan: Time interval (t_start, t_end).
        N: Number of filter steps.

    Returns:
        ScanLoopResult containing filtered states, predictions, backward pass
        data, observation marginals (padded), log likelihood, and scan_data
        (which includes obs_mask for extracting active observation dimensions).
    """
    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_h = prior.A(h)
    b_h = prior.b(h)
    Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T

    # Pre-compute measurement data for all time steps
    scan_data = measure.prepare_scan_data(ts)

    def scan_body(carry, step_idx):
        m, P_sqr, log_lik = carry

        # Run one filter step
        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz, Pz_sqr),
            (m_new, P_new_sqr),
        ) = ekf1_sqr_filter_step_scan(
            A_h, b_h, Q_h_sqr, m, P_sqr, measure, step_idx, scan_data
        )

        # Compute log-likelihood contribution (masked for active observations)
        obs_mask = scan_data.obs_mask[step_idx]
        log_diag = np.log(np.abs(np.diag(Pz_sqr)))
        log_det = 2.0 * np.sum(np.where(obs_mask, log_diag, 0.0))

        v = jax.scipy.linalg.solve_triangular(Pz_sqr.T, mz, lower=True)
        v_masked = np.where(obs_mask, v, 0.0)
        maha = v_masked @ v_masked

        active_dim = np.sum(obs_mask.astype(np.float32))
        log_lik_step = -0.5 * (active_dim * np.log(2 * np.pi) + log_det + maha)

        new_carry = (m_new, P_new_sqr, log_lik + log_lik_step)
        outputs = (
            m_pred,
            P_pred_sqr,
            G_back,
            d_back,
            P_back_sqr,
            mz,
            Pz_sqr,
            m_new,
            P_new_sqr,
        )

        return new_carry, outputs

    # Initial carry
    init_carry = (mu_0, Sigma_0_sqr, 0.0)

    # Run scan
    (_, _, log_likelihood), outputs = jax.lax.scan(scan_body, init_carry, np.arange(N))

    # Unpack outputs
    (
        m_pred_seq,
        P_pred_seq_sqr,
        G_back_seq,
        d_back_seq,
        P_back_seq_sqr,
        mz_seq,
        Pz_seq_sqr,
        m_seq_body,
        P_seq_sqr_body,
    ) = outputs

    # Prepend initial state to sequences
    m_seq = np.concatenate([mu_0[None, :], m_seq_body], axis=0)
    P_seq_sqr = np.concatenate([Sigma_0_sqr[None, :, :], P_seq_sqr_body], axis=0)

    return ScanLoopResult(
        m_seq=m_seq,
        P_seq_sqr=P_seq_sqr,
        m_pred_seq=m_pred_seq,
        P_pred_seq_sqr=P_pred_seq_sqr,
        G_back_seq=G_back_seq,
        d_back_seq=d_back_seq,
        P_back_seq_sqr=P_back_seq_sqr,
        mz_seq=mz_seq,
        Pz_seq_sqr=Pz_seq_sqr,
        log_likelihood=log_likelihood,
        scan_data=scan_data,
    )


def rts_sqr_smoother_loop_scan(
    m_N: Array,
    P_N_sqr: Array,
    G_back_seq: Array,
    d_back_seq: Array,
    P_back_seq_sqr: Array,
) -> tuple[Array, Array]:
    """Run a Rauch-Tung-Striebel smoother using jax.lax.scan.

    This is a scan-compatible version of rts_sqr_smoother_loop. Since all
    shapes are already fixed (state_dim is constant), this is straightforward.

    Args:
        m_N: Final filtered state mean.
        P_N_sqr: Final filtered state covariance (square-root form).
        G_back_seq: Backward pass gain sequence from filter, shape [N, state_dim, state_dim].
        d_back_seq: Backward pass offset sequence from filter, shape [N, state_dim].
        P_back_seq_sqr: Backward pass covariance sequence (sqrt), shape [N, state_dim, state_dim].

    Returns:
        Tuple of smoothed state means and covariances (square-root form),
        both with shape [N+1, state_dim, ...].
    """

    def scan_body(carry, inputs):
        m_smooth_next, P_smooth_sqr_next = carry
        G_back, d_back, P_back_sqr = inputs

        # RTS smoother step
        m_smooth_j, P_smooth_sqr_j = rts_sqr_smoother_step(
            G_back, d_back, P_back_sqr, m_smooth_next, P_smooth_sqr_next
        )

        return (m_smooth_j, P_smooth_sqr_j), (m_smooth_j, P_smooth_sqr_j)

    # Initial carry is the final filtered state
    init_carry = (m_N, P_N_sqr)

    # Stack inputs for scan (reverse order for backward pass)
    xs = (G_back_seq, d_back_seq, P_back_seq_sqr)

    # Run scan in reverse
    _, outputs = jax.lax.scan(scan_body, init_carry, xs, reverse=True)

    m_smooth_body, P_smooth_sqr_body = outputs

    # Append final state
    m_smooth = np.concatenate([m_smooth_body, m_N[None, :]], axis=0)
    P_smooth_sqr = np.concatenate([P_smooth_sqr_body, P_N_sqr[None, :, :]], axis=0)

    return m_smooth, P_smooth_sqr
