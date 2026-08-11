"""Iterated posterior-linearization smoother (IPLS).

The one-shot filters -- EK1, and :class:`~ode_filters.filters.correction.\
QuadratureCorrection` -- fit their affine surrogate over the *predictive*
density, a choice made before the measurement is seen. IPLS instead iterates:
having smoothed the trajectory once, it refits the surrogate at every step over
the *smoothed* marginal and re-runs filter and smoother, to a fixed number of
passes.

Two axes are being separated here (Garcia-Fernandez, Svensson & Sarkka 2017):

- **which point** the linearization sits at -- iterating that alone is the
  iterated EKF/EKS (:class:`~ode_filters.filters.correction.\
IteratedTaylorCorrection`), Gauss-Newton on the MAP objective;
- **which spread** it is fitted over -- iterating that is IPLS, and it converges
  to a moment-matched Gaussian rather than to the MAP.

Why the *smoother* rather than the filter: the linearization error of a Gaussian
ODE filter scales with the spread of the vector field's arguments, and a
single ODE-residual update removes variance mainly in the highest-derivative
direction -- so the per-step posterior spread of the vector-field arguments is
close to the predictive one, and per-step iteration (IPLF,
``QuadratureCorrection(max_iters > 1)``) has little to work with. Pooling
observations across the whole trajectory does shrink that spread, so the
trajectory-level iteration is where the mechanism has room.

Scope: fixed grid, plain (non-preconditioned) priors, fixed process noise. The
diffusion is *not* recalibrated between passes -- a per-pass ``sigma^2`` would
chase the linearization it is supposed to be conditioned on -- so the prior must
carry its own scale, matching ``calibration="none"`` on
:func:`~ode_filters.filters.gaussian_filter.gaussian_filter`.

References:
    Garcia-Fernandez, Svensson, Morelande & Sarkka (2015), *Posterior
    linearization filter: principles and implementation using sigma points*.
    Garcia-Fernandez, Svensson & Sarkka (2017), *Iterated posterior linearization
    smoother*.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as np
from jax import Array

from ..inference.sqr_gaussian_inference import sqr_inversion, sqr_marginalization
from ..measurement.measurement_models import BaseODEInformation, ObsModel
from ..priors.gmp_priors import BasePrior
from .correction import QuadratureCorrection
from .gaussian_filter import _is_preconditioned, gaussian_filter, rts_smoother
from .ode_filter_loop import _log_likelihood_contrib, rts_sqr_smoother_loop
from .statistical_linearization import QuadratureRule, slr_linearize


class IPLSResult(NamedTuple):
    """Output of :func:`ipls_smoother`.

    Attributes:
        t: Time grid, shape ``[N+1]``.
        m: Smoothed means of the final pass, shape ``[N+1, state_dim]``.
        P_sqr: Smoothed square-root covariances of the final pass,
            ``[N+1, state_dim, state_dim]``.
        m_filt: Filtered means of the final pass (same shapes as ``m``); the
            forward marginals under the last affine surrogate.
        P_filt_sqr: Filtered square-root covariances of the final pass.
        log_likelihood: ODE-channel log-marginal-likelihood of the final pass's
            affine model. Comparable across IPLS iterations, but *not* against a
            one-shot EK1 run: each pass linearizes differently, so this is the
            evidence under a different surrogate model.
        log_likelihood_obs: Observation-channel log-marginal-likelihood of the
            final pass (``None`` when no ``obs_model`` was given).
        n_iters: Number of relinearization passes actually run.
    """

    t: Array
    m: Array
    P_sqr: Array
    m_filt: Array
    P_filt_sqr: Array
    log_likelihood: Array
    log_likelihood_obs: Array | None
    n_iters: int


def affine_filter_scan(
    mu_0: Array,
    P_0_sqr: Array,
    A_h: Array,
    b_h: Array,
    Q_sqr: Array,
    H_seq: Array,
    c_seq: Array,
    R_seq_sqr: Array,
    *,
    obs_model: ObsModel | None = None,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    """Filter a linear-Gaussian model whose measurement is given per step.

    The measurement model is supplied as a *precomputed* affine surrogate
    ``(H_seq[i], c_seq[i], R_seq_sqr[i])`` rather than linearized from the
    running state. That is what makes an outer relinearization loop possible: the
    surrogate can be fitted over any density, including a smoothed one.

    An ``obs_model`` adds the usual masked observation update after the
    (surrogate) ODE update, matching the sequential ordering of
    :func:`~ode_filters.filters.ode_filter_step.sqr_filter_step_sequential_scan`.

    Args:
        mu_0: Initial mean.
        P_0_sqr: Initial square-root covariance.
        A_h: State transition matrix (constant over the fixed grid).
        b_h: Drift vector.
        Q_sqr: Square-root process noise (already carrying any diffusion scale).
        H_seq: Per-step surrogate observation matrices, ``[N, obs_dim, state_dim]``.
        c_seq: Per-step surrogate offsets, ``[N, obs_dim]``.
        R_seq_sqr: Per-step effective noise square roots,
            ``[N, obs_dim, obs_dim]`` (includes the linearization residual
            ``Omega``).
        obs_model: Optional external observations.

    Returns:
        Tuple ``(m_seq, P_seq_sqr, G_back, d_back, P_back_sqr, ll, ll_obs)``.
        ``m_seq`` / ``P_seq_sqr`` span ``[N+1]`` (the initial condition
        prepended); the backward-pass arrays span ``[N]``. ``ll_obs`` is zero
        when no ``obs_model`` was given.
    """
    n_steps = H_seq.shape[0]
    # The observation branch is a Python-level choice (traced once), so an
    # ODE-only run pays nothing for it. Placeholders keep the scan's input
    # pytree structure fixed either way.
    has_obs = obs_model is not None
    if obs_model is None:
        c_obs_seq = np.zeros((n_steps, 1))
        mask_seq = np.zeros((n_steps, 1), dtype=bool)
    else:
        c_obs_seq = obs_model.c_seq
        mask_seq = obs_model.mask

    def scan_body(carry, step_data):
        m_prev, P_prev_sqr, ll, ll_obs = carry
        H_i, c_i, R_i_sqr, c_obs_i, mask_i = step_data

        m_pred, P_pred_sqr = sqr_marginalization(A_h, b_h, Q_sqr, m_prev, P_prev_sqr)
        G_back, d_back, P_back_sqr = sqr_inversion(
            A_h, m_prev, P_prev_sqr, m_pred, P_pred_sqr, Q_sqr
        )

        mz, Pz_sqr = sqr_marginalization(H_i, c_i, R_i_sqr, m_pred, P_pred_sqr)
        _, m_ode, P_ode_sqr = sqr_inversion(
            H_i, m_pred, P_pred_sqr, mz, Pz_sqr, R_i_sqr
        )
        ll = ll + _log_likelihood_contrib(mz, Pz_sqr)

        if has_obs:
            assert obs_model is not None  # narrowed by has_obs
            obs_active = mask_i.any()
            mz_obs, Pz_obs_sqr = sqr_marginalization(
                obs_model.H, c_obs_i, obs_model.R_sqr, m_ode, P_ode_sqr
            )
            _, m_obs, P_obs_sqr = sqr_inversion(
                obs_model.H, m_ode, P_ode_sqr, mz_obs, Pz_obs_sqr, obs_model.R_sqr
            )
            m_new = jax.lax.select(obs_active, m_obs, m_ode)
            P_new_sqr = jax.lax.select(obs_active, P_obs_sqr, P_ode_sqr)
            ll_obs = ll_obs + jax.lax.select(
                obs_active, _log_likelihood_contrib(mz_obs, Pz_obs_sqr), np.array(0.0)
            )
        else:
            m_new, P_new_sqr = m_ode, P_ode_sqr

        outputs = (G_back, d_back, P_back_sqr, m_new, P_new_sqr)
        return (m_new, P_new_sqr, ll, ll_obs), outputs

    init = (mu_0, P_0_sqr, np.array(0.0), np.array(0.0))
    (_, _, ll, ll_obs), outputs = jax.lax.scan(
        scan_body, init, (H_seq, c_seq, R_seq_sqr, c_obs_seq, mask_seq)
    )
    G_back, d_back, P_back_sqr, m_updates, P_updates_sqr = outputs
    m_seq = np.concatenate([mu_0[None, :], m_updates], axis=0)
    P_seq_sqr = np.concatenate([P_0_sqr[None, :, :], P_updates_sqr], axis=0)
    return m_seq, P_seq_sqr, G_back, d_back, P_back_sqr, ll, ll_obs


def ipls_smoother(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    *,
    n_iters: int = 3,
    rule: QuadratureRule = "gauss_hermite",
    n_nodes: int = 4,
    obs_model: ObsModel | None = None,
) -> IPLSResult:
    """Iterated posterior-linearization smoother over a fixed grid.

    Pass 0 is the one-shot statistical-linearization filter (a
    :class:`~ode_filters.filters.correction.QuadratureCorrection` run) followed
    by RTS smoothing. Each subsequent pass refits the affine surrogate at every
    step over the *smoothed* marginal at that step and re-runs filter and
    smoother. ``n_iters=0`` therefore returns exactly the one-shot
    filter-and-smooth baseline, which is the comparison to make.

    Args:
        mu_0: Initial state mean.
        P_0_sqr: Initial state covariance (square-root form).
        prior: Gauss-Markov prior. Must **not** be preconditioned.
        measure: ODE-information measurement model.
        tspan: Time interval ``(t0, t1)`` as a tuple (jit-static hashing).
        N: Number of fixed-grid steps.
        n_iters: Number of relinearization passes after the initial one.
        rule: Quadrature rule for the statistical linearization.
        n_nodes: Nodes per dimension for the Gauss-Hermite rule (see
            :class:`~ode_filters.filters.correction.QuadratureCorrection`).
        obs_model: Optional external observations from
            :func:`~ode_filters.measurement.measurement_models.prepare_observations`.

    Note:
        Differentiating any field of the result requires a **positive-definite**
        ``P_0_sqr``. The result carries smoothed covariances, and the smoother's
        covariance recursion already has a NaN reverse-mode derivative when the
        initial covariance is exactly singular (as
        :func:`~ode_filters.priors.taylor_mode_initialization` returns) -- a
        pre-existing property of :func:`~ode_filters.filters.gaussian_filter.\
rts_smoother`, not of the linearization. Since a zero cotangent times a NaN
        derivative is still NaN, that poisons the *mean* gradient too. Add a
        small jitter to ``P_0_sqr`` when differentiating.

    Returns:
        An :class:`IPLSResult`.

    Raises:
        ValueError: If ``n_iters < 0``.
        NotImplementedError: If ``prior`` is preconditioned. The bar-space
            recursion would need the surrogate expressed in bar coordinates per
            pass; use a plain ``IWP`` / ``MaternPrior`` / ``JointPrior``.
    """
    if n_iters < 0:
        raise ValueError(f"n_iters must be >= 0, got {n_iters!r}.")
    if _is_preconditioned(prior):
        raise NotImplementedError(
            "ipls_smoother does not support preconditioned priors; pass a plain "
            "IWP / MaternPrior / JointPrior."
        )

    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_h = prior.A(h)
    b_h = prior.b(h)
    Q_sqr = prior.Q_sqr(h)
    t_steps = ts[1:]

    # Pass 0: one-shot SLR filter (linearized over the predictive density) plus
    # RTS. This is the only pass that needs a nonlinear forward recursion; every
    # later pass consumes a precomputed surrogate.
    init_result = gaussian_filter(
        mu_0,
        P_0_sqr,
        prior,
        measure,
        tspan,
        N,
        correction=QuadratureCorrection(rule=rule, n_nodes=n_nodes),
        calibration="none",
        obs_model=obs_model,
    )
    m_smooth, P_smooth_sqr = rts_smoother(prior, init_result)

    def relinearize(m_lin_seq: Array, P_lin_seq_sqr: Array):
        """Refit the surrogate at every step over the given per-step densities."""
        models = jax.vmap(
            lambda m_i, P_i, t_i: slr_linearize(
                measure, m_i, P_i, t=t_i, rule=rule, n_nodes=n_nodes
            )
        )(m_lin_seq, P_lin_seq_sqr, t_steps)
        return models.H, models.c, models.R_eff_sqr

    def pass_body(_i, carry):
        m_lin, P_lin_sqr, _m_filt, _P_filt_sqr, _ll, _ll_obs = carry
        # Step i updates at time ts[i+1], so its linearization density is the
        # smoothed marginal at index i+1.
        H_seq, c_seq, R_seq_sqr = relinearize(m_lin[1:], P_lin_sqr[1:])
        m_filt, P_filt_sqr, G_back, d_back, P_back_sqr, ll, ll_obs = affine_filter_scan(
            mu_0,
            P_0_sqr,
            A_h,
            b_h,
            Q_sqr,
            H_seq,
            c_seq,
            R_seq_sqr,
            obs_model=obs_model,
        )
        m_new, P_new_sqr = rts_sqr_smoother_loop(
            m_filt[-1], P_filt_sqr[-1], G_back, d_back, P_back_sqr, N
        )
        return (m_new, P_new_sqr, m_filt, P_filt_sqr, ll, ll_obs)

    ll_obs_init = (
        np.array(0.0)
        if init_result.log_likelihood_obs is None
        else init_result.log_likelihood_obs
    )
    carry = (
        m_smooth,
        P_smooth_sqr,
        init_result.m,
        init_result.P_sqr,
        init_result.log_likelihood,
        ll_obs_init,
    )
    m_out, P_out_sqr, m_filt, P_filt_sqr, ll, ll_obs = jax.lax.fori_loop(
        0, n_iters, pass_body, carry
    )

    return IPLSResult(
        t=ts,
        m=m_out,
        P_sqr=P_out_sqr,
        m_filt=m_filt,
        P_filt_sqr=P_filt_sqr,
        log_likelihood=ll,
        log_likelihood_obs=None if obs_model is None else ll_obs,
        n_iters=n_iters,
    )
