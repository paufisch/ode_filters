"""Consolidated public solver API: ``gaussian_filter`` + smoother.

This is the recommended entry point for filtering/smoothing an ODE with a
Gaussian (Markov) prior. It replaces the historical matrix of ``ekf1_sqr_loop*``
functions: the ``{plain/preconditioned} x {joint/sequential} x {fixed/adaptive}``
choices are now *dispatched automatically* rather than encoded in the function
name --

- **preconditioning** is selected from the prior type (``PrecondIWP`` /
  ``PrecondMaternPrior`` -> preconditioned path);
- **sequential** observation handling is selected by whether ``obs_model`` is given;
- the **correction** (EK0/EK1/IEKF) is an object passed as ``correction=``;
- **calibration** is a string (``"none"`` = static diffusion).

All variants return a single :class:`FilterResult` with named fields, instead of
the old 9-to-16-wide positional tuples. Smoothing is a separate call,
:func:`rts_smoother`, which consumes a result's backward pass.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as np
from jax import Array

from ..measurement.measurement_models import BaseODEInformation, ObsModel
from ..priors.gmp_priors import BasePrior, PrecondIWP, PrecondMaternPrior
from .correction import Correction
from .ode_filter_adaptive import ekf1_sqr_adaptive_solve
from .ode_filter_loop import (
    ekf1_sqr_loop_dynamic_scan,
    ekf1_sqr_loop_preconditioned_dynamic_scan,
    rts_sqr_smoother_loop,
    rts_sqr_smoother_loop_preconditioned,
)


class FilterResult(NamedTuple):
    """Filtered solution of a Gaussian ODE filter.

    Attributes:
        t: Time grid, shape ``[K]``.
        m: Filtered state means at ``t``, shape ``[K, state_dim]``.
        P_sqr: Square-root covariances at ``t``, shape
            ``[K, state_dim, state_dim]`` (``P = P_sqr.T @ P_sqr``).
        log_likelihood: Marginal log-likelihood of the ODE-information residuals.
        m_pred: Predicted (prior) means per step, ``[K-1, state_dim]`` (``None`` for
            the adaptive save-at solver, which keeps no backward pass).
        P_pred_sqr: Predicted square-root covariances per step.
        G_back: Backward-pass gains per step (smoother input).
        d_back: Backward-pass offsets per step.
        P_back_sqr: Backward-pass square-root covariances per step.
        mz: Predicted-observation (ODE-defect) innovation means per step; the input
            to post-hoc diffusion calibration. ``None`` for the adaptive solver.
        Pz_sqr: Predicted-observation innovation square-root covariances per step.
        sigma_sqr: Per-step calibrated diffusion ``sigma_hat^2``.
        log_likelihood_obs: Marginal log-likelihood of the external observations
            (``None`` when no ``obs_model`` was given) -- the quantity to maximize
            for data-driven parameter inference.
        m_bar: Preconditioned-space means (``None`` unless the prior is
            preconditioned); internal, consumed by :func:`rts_smoother`.
        P_bar_sqr: Preconditioned-space square-root covariances (``None`` for plain).
        T: Preconditioner matrix (``None`` for plain); presence selects the
            preconditioned smoother.
    """

    t: Array
    m: Array
    P_sqr: Array
    log_likelihood: Array
    m_pred: Array | None
    P_pred_sqr: Array | None
    G_back: Array | None
    d_back: Array | None
    P_back_sqr: Array | None
    mz: Array | None
    Pz_sqr: Array | None
    sigma_sqr: Array | None
    log_likelihood_obs: Array | None
    m_bar: Array | None
    P_bar_sqr: Array | None
    T: Array | None


def _is_preconditioned(prior: BasePrior) -> bool:
    return isinstance(prior, (PrecondIWP, PrecondMaternPrior))


def gaussian_filter(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    *,
    correction: Correction | None = None,
    calibration: str = "dynamic",
    obs_model: ObsModel | None = None,
    min_sigma_sqr: float = 0.0,
) -> FilterResult:
    """Fixed-grid Gaussian (EKF) filter over ``N`` steps on ``tspan``.

    Dispatches automatically: a preconditioned prior selects the preconditioned
    square-root recursion; an ``obs_model`` adds a masked observation update.

    Args:
        mu_0: Initial state mean.
        P_0_sqr: Initial state covariance (square-root form).
        prior: Gauss-Markov prior. ``PrecondIWP`` / ``PrecondMaternPrior`` select
            the preconditioned path.
        measure: ODE-information measurement model (ODE + Conservation).
        tspan: Time interval ``(t0, t1)`` (a tuple, for jit-static hashing).
        N: Number of fixed-grid steps.
        correction: Linearization strategy (``TaylorCorrection`` /
            ``IteratedTaylorCorrection``); ``None`` defaults to EK1.
        calibration: Diffusion calibration mode -- ``"dynamic"`` (default),
            ``"none"`` (static), ``"diagonal"``, or ``"diagonal_ekf0"``.
        obs_model: Optional external observations (see ``prepare_observations``).
        min_sigma_sqr: Lower bound on the per-step ``sigma_hat^2``.

    Returns:
        A :class:`FilterResult`.
    """
    t = np.linspace(tspan[0], tspan[1], N + 1)

    if _is_preconditioned(prior):
        if obs_model is not None or correction is not None:
            raise NotImplementedError(
                "Preconditioned priors do not yet support observations or a "
                "non-default correction. Use a plain IWP / Matern prior for those."
            )
        out = ekf1_sqr_loop_preconditioned_dynamic_scan(
            mu_0,
            P_0_sqr,
            prior,
            measure,
            tspan,
            N,
            calibration=calibration,
            min_sigma_sqr=min_sigma_sqr,
        )
        (
            m_seq,
            P_seq_sqr,
            m_bar,
            P_bar_sqr,
            m_pred_bar,
            P_pred_bar,
            G_back_bar,
            d_back_bar,
            P_back_bar,
            mz,
            Pz_sqr,
            sigma_sqr,
            T_h,
            ll,
        ) = out
        return FilterResult(
            t=t,
            m=m_seq,
            P_sqr=P_seq_sqr,
            log_likelihood=ll,
            m_pred=m_pred_bar,
            P_pred_sqr=P_pred_bar,
            G_back=G_back_bar,
            d_back=d_back_bar,
            P_back_sqr=P_back_bar,
            mz=mz,
            Pz_sqr=Pz_sqr,
            sigma_sqr=sigma_sqr,
            log_likelihood_obs=None,
            m_bar=m_bar,
            P_bar_sqr=P_bar_sqr,
            T=T_h,
        )

    out = ekf1_sqr_loop_dynamic_scan(
        mu_0,
        P_0_sqr,
        prior,
        measure,
        tspan,
        N,
        calibration=calibration,
        min_sigma_sqr=min_sigma_sqr,
        obs_model=obs_model,
        correction=correction,
    )
    if obs_model is None:
        (
            m_seq,
            P_seq_sqr,
            m_pred,
            P_pred,
            G_back,
            d_back,
            P_back,
            mz,
            Pz_sqr,
            sigma_sqr,
            ll,
        ) = out
        ll_obs = None
    else:
        (
            m_seq,
            P_seq_sqr,
            m_pred,
            P_pred,
            G_back,
            d_back,
            P_back,
            mz,
            Pz_sqr,
            _mz_obs,
            _Pz_obs,
            sigma_sqr,
            ll,
            ll_obs,
        ) = out
    return FilterResult(
        t=t,
        m=m_seq,
        P_sqr=P_seq_sqr,
        log_likelihood=ll,
        m_pred=m_pred,
        P_pred_sqr=P_pred,
        G_back=G_back,
        d_back=d_back,
        P_back_sqr=P_back,
        mz=mz,
        Pz_sqr=Pz_sqr,
        sigma_sqr=sigma_sqr,
        log_likelihood_obs=ll_obs,
        m_bar=None,
        P_bar_sqr=None,
        T=None,
    )


def gaussian_filter_adaptive(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    save_at: Array,
    *,
    obs_model: ObsModel | None = None,
    atol: float = 1e-4,
    rtol: float = 1e-2,
    h_init: float | None = None,
    calibration: str = "dynamic",
    controller=None,
    min_sigma_sqr: float = 0.0,
    max_steps: int = 4096,
) -> FilterResult:
    """Adaptive-step Gaussian filter, returning the solution at ``save_at``.

    ``jit`` / ``vmap`` / reverse-``grad``-able (checkpointed adaptive loop).
    Filtering only -- the result carries no backward pass, so :func:`rts_smoother`
    does not apply. See :func:`ekf1_sqr_adaptive_solve` for the full argument docs.
    """
    res = ekf1_sqr_adaptive_solve(
        mu_0,
        P_0_sqr,
        prior,
        measure,
        save_at,
        obs_model=obs_model,
        atol=atol,
        rtol=rtol,
        h_init=h_init,
        calibration=calibration,
        controller=controller,
        min_sigma_sqr=min_sigma_sqr,
        max_steps=max_steps,
    )
    return FilterResult(
        t=res.t,
        m=res.m,
        P_sqr=res.P_sqr,
        log_likelihood=res.log_likelihood,
        m_pred=None,
        P_pred_sqr=None,
        G_back=None,
        d_back=None,
        P_back_sqr=None,
        mz=None,
        Pz_sqr=None,
        sigma_sqr=None,
        log_likelihood_obs=None,
        m_bar=None,
        P_bar_sqr=None,
        T=None,
    )


def rts_smoother(prior: BasePrior, result: FilterResult) -> tuple[Array, Array]:
    """Rauch-Tung-Striebel smoothing of a fixed-grid :func:`gaussian_filter` result.

    Args:
        prior: The prior used for the forward filter (selects the preconditioned
            smoother when preconditioned).
        result: A :class:`FilterResult` from :func:`gaussian_filter` (must carry a
            backward pass -- adaptive results do not).

    Returns:
        Tuple ``(m_smooth, P_smooth_sqr)`` of smoothed means and square-root
        covariances, shapes ``[K, state_dim]`` and ``[K, state_dim, state_dim]``.
    """
    if result.G_back is None:
        raise ValueError(
            "FilterResult carries no backward pass (e.g. from "
            "gaussian_filter_adaptive); smoothing requires a fixed-grid "
            "gaussian_filter result."
        )
    N = result.m.shape[0] - 1
    if result.T is not None:
        return rts_sqr_smoother_loop_preconditioned(
            result.m[-1],
            result.P_sqr[-1],
            result.m_bar[-1],
            result.P_bar_sqr[-1],
            result.G_back,
            result.d_back,
            result.P_back_sqr,
            N,
            result.T,
        )
    return rts_sqr_smoother_loop(
        result.m[-1],
        result.P_sqr[-1],
        result.G_back,
        result.d_back,
        result.P_back_sqr,
        N,
    )
