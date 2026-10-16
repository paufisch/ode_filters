"""Adaptive step-size filtering: ``gaussian_filter_adaptive``.

Between consecutive save times the solver takes as many accepted and rejected
steps as its error control needs. Each attempt is one ``filter_step`` with the
trial step size; its local-error estimate (Bosch et al. 2021, Eq. 49; the error
vector has the dimension of the ODE solution, Bosch et al. 2022, Sec. 3) is

    err = sqrt(mean((D / (atol + rtol |x|))^2)),   D_i^2 = sigma_i^2 (E0 Q(h) E0.T)_ii,

the standard deviation the step's calibrated process noise adds to each
solution component, weighted by the tolerances. A step is accepted when
``err <= 1``; the controller proposes the next step size either way, capped at
``max_step`` if one is given. The last step of each save interval is shortened
to land on the save time. The first trial step is
``(save_at[-1] - save_at[0]) / 100`` by default; with ``h_init="auto"`` it is
the starting step size of Hairer, Norsett & Wanner (1993, Sec. II.4), computed
from the initial state and the tolerances.

The solve is a ``jax.lax.scan`` over the save times whose body is a
checkpointed while-loop (``equinox.internal.while_loop``), so it is ``jit``-,
``vmap``- and reverse-mode differentiable, and cannot raise: ``success``
reports whether every save time was reached.

With ``smoother=True`` each accepted step's backward conditional is composed
into one conditional per save interval (Kraemer 2025, "Adaptive Probabilistic
ODE Solvers Without Adaptive Memory Requirements"), so ``rts_smoother`` works
on the result with memory independent of the number of steps.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import equinox.internal as eqxi
import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..calibration.policies import Calibration, CalibrationMode, as_calibration
from ..gaussian import AffineModel, Conditional, Gaussian, compose
from ..gaussian.algebra import _safe_sqrt
from ..information import MeasurementModel, ObsModel
from ..linearization.correction import Correction, TaylorCorrection
from ..priors.base import BasePrior, Transition
from .adaptive_controller import PIController, StepSizeController
from .fixed_grid import check_grid, check_obs_model
from .kernel import OnlineInformation, filter_step, observe
from .result import FilterResult, StepRecord

SigmaInError = Literal["per_step", "running_mean"]

# A previous error of -1 tells the controller there is none (the first step of
# a solve, or right after a rejection), which drops its integral term.
_NO_PREVIOUS_ERROR = -1.0

# Guards of h_init="auto", with the values of Hairer, Norsett & Wanner's code.
# With d0 or d1 below _TINY_NORM, h0 = _FALLBACK_STEP replaces 0.01 d0 / d1.
# With max(d1, d2) at or below _FLAT_NORM, h1 = max(_FALLBACK_STEP, 1e-3 h0)
# replaces (0.01 / max(d1, d2))^(1 / (q + 1)).
_TINY_NORM = 1e-5
_FLAT_NORM = 1e-15
_FALLBACK_STEP = 1e-6


def _landing_tolerance(target: Array) -> Array:
    """How close to a save time counts as having reached it."""
    return 1e-10 * np.abs(target) + 1e-12


def _check_max_step(max_step: ArrayLike | None) -> None:
    """Raise unless ``max_step`` is ``None`` or a positive scalar (where concrete)."""
    if max_step is None:
        return
    if onp.ndim(max_step) != 0:
        raise ValueError(f"max_step must be a scalar, got shape {onp.shape(max_step)}.")
    try:
        value = float(onp.asarray(max_step))
    except jax.errors.TracerArrayConversionError:
        return  # traced (e.g. a jit argument): trusted, like a traced save_at
    if not value > 0.0:
        raise ValueError(f"max_step must be > 0, got {value!r}.")


def _check_h_init(
    h_init: ArrayLike | Literal["auto"] | None,
    prior: BasePrior,
    measure: MeasurementModel,
) -> None:
    """Raise for a string other than ``"auto"``, or ``"auto"`` without its inputs."""
    if not isinstance(h_init, str):
        return
    if h_init != "auto":
        raise ValueError(f'h_init must be None, a step size or "auto", got {h_init!r}.')
    if prior.q < 1:
        raise ValueError('h_init="auto" needs a prior with q >= 1, got q = 0.')
    d = prior.E0_state.shape[0]
    if prior.q == 1 and measure.ode_dim != d:
        raise ValueError(
            "h_init=\"auto\" with q = 1 estimates x'' from the ODE rows of the "
            f"residual, which must have one row per solution component ({d}); "
            f"the measurement model has {measure.ode_dim}."
        )


def _rms(v: Array) -> Array:
    """Root-mean-square norm, with a finite reverse-mode derivative at zero."""
    return _safe_sqrt(np.mean(v * v))


def _initial_step(
    m_0: Array,
    prior: BasePrior,
    measure: MeasurementModel,
    *,
    t: Array,
    atol: ArrayLike,
    rtol: ArrayLike,
) -> Array:
    """The first trial step of ``h_init="auto"``.

    The starting step size of Hairer, Norsett & Wanner (1993, Sec. II.4) for the
    order ``q``. With ``x0``, ``x0'`` and ``x0''`` the solution and its first two
    derivatives at ``t``, the scale ``sc = atol + rtol |x0|`` and RMS norms,

        d0 = ||x0 / sc||,   d1 = ||x0' / sc||,   d2 = ||x0'' / sc||,
        h0 = 0.01 d0 / d1,  or 1e-6 if d0 < 1e-5 or d1 < 1e-5,
        h1 = (0.01 / max(d1, d2))^(1 / (q + 1)),
                            or max(1e-6, 1e-3 h0) if max(d1, d2) <= 1e-15,
        h  = min(100 h0, h1).

    ``x0`` and ``x0'`` are read from ``m_0`` with ``E0_state`` and ``E1``, and
    for ``q >= 2`` also ``x0''`` with ``E2``. For ``q = 1``, ``x0''`` is the
    explicit-Euler difference of the ODE rows of the residual. The solution
    part of ``m_0`` moves by ``h0 x0'`` and the time by ``h0``, which changes
    the rows ``x' - f(x, t)`` of a first-order ODE by
    ``-(f(x0 + h0 x0', t + h0) - f(x0, t))``. ``np.where`` computes both
    branches of every guard, with the division of the unused branch kept off
    zero, so the step is traceable and has finite derivatives.
    """
    q = prior.q
    E0 = prior.E0_state
    x0 = E0 @ m_0
    dx0 = prior.E1 @ m_0
    scale = atol + rtol * np.abs(x0)
    d0 = _rms(x0 / scale)
    d1 = _rms(dx0 / scale)
    tiny = (d0 < _TINY_NORM) | (d1 < _TINY_NORM)
    h0 = np.where(tiny, _FALLBACK_STEP, 0.01 * d0 / np.where(tiny, 1.0, d1))
    E2 = prior.E2
    if E2 is not None:
        d2 = _rms(E2 @ m_0 / scale)
    else:
        k = measure.ode_dim
        m_1 = m_0 + E0.T @ (h0 * dx0)
        r_0 = measure.g(m_0, t=t)[:k]
        r_1 = measure.g(m_1, t=t + h0)[:k]
        d2 = _rms((r_1 - r_0) / scale) / h0
    d_max = np.maximum(d1, d2)
    flat = d_max <= _FLAT_NORM
    h1 = np.where(
        flat,
        np.maximum(_FALLBACK_STEP, 1e-3 * h0),
        (0.01 / np.where(flat, 1.0, d_max)) ** (1.0 / (q + 1)),
    )
    return np.minimum(100.0 * h0, h1)


class _Carry(NamedTuple):
    """State of the adaptive loop between attempts."""

    t: Array
    x: Gaussian
    h: Array
    log_likelihood: Array
    err_prev: Array
    innovation: Gaussian  # ODE innovation of the last accepted step
    sigma_sum: Array  # sum of the accepted steps' sigma^2 (running mean)
    n_total: Array  # accepted steps of the whole solve
    n_accepted: Array  # accepted steps of the current save interval
    n_rejected: Array  # rejected steps of the current save interval
    composite: Conditional | None  # backward conditional of the interval so far
    record: StepRecord | None


def _local_error(
    transition: Transition,
    sigma_sqr: Array,
    m_pred: Array,
    E0: Array,
    atol: ArrayLike,
    rtol: ArrayLike,
) -> Array:
    """Tolerance-weighted RMS norm of the step's local error estimate."""
    s = transition.T_diag
    Q_sqr = transition.Q_sqr if s is None else transition.Q_sqr * s[None, :]
    # diag(E0 Q E0.T) as the row norms of E0 @ Q_sqr.T: the dense Q is never formed.
    E0Q_sqr = E0 @ Q_sqr.T
    variance = np.sum(E0Q_sqr * E0Q_sqr, axis=1)
    D = _safe_sqrt(np.maximum(sigma_sqr * variance, 0.0))
    scale = atol + rtol * np.abs(E0 @ m_pred)
    # _safe_sqrt: an exactly-zero error (e.g. at an equilibrium) would otherwise
    # give a NaN reverse-mode derivative.
    return _safe_sqrt(np.mean((D / scale) ** 2))


def gaussian_filter_adaptive(
    m_0: ArrayLike,
    P_0_sqr: ArrayLike,
    prior: BasePrior,
    measure: MeasurementModel,
    save_at: ArrayLike,
    *,
    correction: Correction | None = None,
    calibration: Calibration | CalibrationMode = "dynamic",
    obs_model: ObsModel | None = None,
    atol: ArrayLike = 1e-4,
    rtol: ArrayLike = 1e-2,
    h_init: ArrayLike | Literal["auto"] | None = None,
    max_step: ArrayLike | None = None,
    controller: StepSizeController | None = None,
    sigma_in_error: SigmaInError = "per_step",
    max_steps: int = 4096,
    smoother: bool = False,
    record_steps: int = 0,
) -> FilterResult:
    """Adaptive-step Gaussian ODE filter, returning the solution at ``save_at``.

    Args:
        m_0: Initial mean, shape ``[D]``.
        P_0_sqr: Initial square-root covariance, shape ``[D, D]``.
        prior: Gauss-Markov prior (plain or preconditioned). Its ``q`` sets the
            default controller order.
        measure: ODE-information model (the ODE plus conservation laws).
        save_at: Strictly increasing save times; ``save_at[0]`` is the initial
            time, where the initial state is returned unchanged.
        correction: Linearization; ``None`` is EK1.
        calibration: Diffusion calibration policy, or one of ``"dynamic"``
            (default), ``"none"``, ``"diagonal"``, ``"diagonal_ekf0"``.
        obs_model: Data at the save times, from
            ``prepare_observations(measurements, E0, save_at)``; conditioned on
            after landing on each save time.
        atol: Absolute tolerance of the local-error estimate.
        rtol: Relative tolerance of the local-error estimate.
        h_init: First trial step. ``None`` (default) takes
            ``(save_at[-1] - save_at[0]) / 100``. ``"auto"`` computes it from
            ``m_0`` and the tolerances by the starting step-size rule of
            Hairer, Norsett & Wanner (1993, Sec. II.4). With the scale
            ``sc = atol + rtol |x0|`` and the RMS norms ``d0 = ||x0 / sc||``,
            ``d1 = ||x0' / sc||`` and ``d2 = ||x0'' / sc||``, the step is
            ``min(100 h0, h1)`` with ``h0 = 0.01 d0 / d1`` and
            ``h1 = (0.01 / max(d1, d2))^(1 / (q + 1))``. If ``d0`` or ``d1`` is
            below ``1e-5``, ``h0`` is ``1e-6``; if ``max(d1, d2) <= 1e-15``,
            ``h1`` is ``max(1e-6, 1e-3 h0)``. Both fallbacks are in the time
            unit of ``save_at``; a zero ``x0`` or ``x0'`` limits the step to at
            most ``1e-4``.
            The rule reads ``x0'`` (``E1``) and, for ``q >= 2``, ``x0''``
            (``E2``) from ``m_0``, which must therefore carry the derivatives,
            as ``taylor_mode_initialization`` provides. For ``q = 1``, ``x0''``
            is the explicit-Euler difference
            ``(f(x0 + h0 x0', t0 + h0) - f(x0, t0)) / h0`` of the first-order
            ODE rows of ``measure``. ``max_step`` and the next save time bound
            the first step as they bound every step.
        max_step: Upper bound on every step, the first trial step included;
            ``None`` (default) leaves the next save time as the only bound. A
            loose tolerance can step over a feature much shorter than its
            steps, such as a brief pulse in a forcing term, without the error
            estimate seeing it; a ``max_step`` shorter than the feature
            resolves it. There is no lower bound: a solve that needs too many
            steps reports ``success=False`` (see ``max_steps``).
        controller: Step-size controller; defaults to
            ``PIController(order=prior.q)``. Its ``propose`` runs inside the
            traced loop, so a custom controller must use ``jax.numpy``.
        sigma_in_error: The diffusion in the error estimate. ``"per_step"``
            (default) uses the step's own estimate (Bosch et al. 2021);
            ``"running_mean"`` the mean over the accepted steps so far. The
            noise of this mean decreases with the number of steps, so the step
            sizes vary more smoothly, but the mean lags an abrupt change in the
            problem's stiffness.
        max_steps: Cap on attempts per save interval (it bounds the
            checkpointed while-loop). Raise it, or loosen the tolerances, if
            ``result.success`` is ``False``.
        smoother: Also compose a backward conditional per save interval, so
            ``rts_smoother`` applies to the result.
        record_steps: Record up to this many accepted steps in
            ``result.steps`` (see ``StepRecord``); ``0`` records none. The
            record is carried through the loop, so keep it off where memory or
            speed matters, e.g. inside an optimization.

    Returns:
        A ``FilterResult`` on ``save_at``. ``log_likelihood`` and
        ``log_likelihood_obs`` are the ODE-information and data evidence;
        ``n_accepted`` / ``n_rejected`` count the steps per save interval;
        ``m_pred``, ``P_pred_sqr`` and ``sigma_sqr`` are ``None``.

    Raises:
        ValueError: For an invalid ``save_at``, ``sigma_in_error``,
            ``record_steps`` or ``max_step`` (not a positive scalar), an
            ``obs_model`` of another length, a calibration the prior does not
            support, or an ``h_init`` string other than ``"auto"``. Also for
            ``h_init="auto"`` with a prior of order ``q = 0``, or of order
            ``q = 1`` with a ``measure`` whose number of ODE rows differs from
            the dimension of the solution.
    """
    save_at = np.asarray(check_grid(save_at))
    calibration = as_calibration(calibration)
    calibration.validate(prior)
    if sigma_in_error not in ("per_step", "running_mean"):
        raise ValueError(
            f"sigma_in_error must be 'per_step' or 'running_mean'; got {sigma_in_error!r}."
        )
    if record_steps < 0:
        raise ValueError(f"record_steps must be >= 0, got {record_steps!r}.")
    _check_max_step(max_step)
    _check_h_init(h_init, prior, measure)
    if obs_model is not None:
        check_obs_model(obs_model, save_at.shape[0] - 1)
    if controller is None:
        controller = PIController(order=max(int(prior.q), 1))
    information = OnlineInformation(
        measure, TaylorCorrection() if correction is None else correction
    )
    E0 = prior.E0_state
    d = E0.shape[0]

    x_0 = Gaussian(np.asarray(m_0, dtype=float), np.asarray(P_0_sqr, dtype=float))
    D = x_0.m.shape[0]
    span = save_at[-1] - save_at[0]
    if isinstance(h_init, str):
        h_0 = _initial_step(x_0.m, prior, measure, t=save_at[0], atol=atol, rtol=rtol)
    else:
        h_0 = span / 100.0 if h_init is None else np.asarray(h_init, dtype=float)

    def bounded(h: Array) -> Array:
        return h if max_step is None else np.minimum(h, max_step)

    # The innovation carried between intervals needs its static shape only.
    k = information.linearize(x_0.m, t=save_at[0]).H.shape[0]
    identity = Conditional(np.eye(D), np.zeros(D), np.zeros((D, D)))
    record = None
    if record_steps:
        nan = np.full(record_steps, np.nan)
        record = StepRecord(
            t=nan,
            h=nan,
            m=np.full((record_steps, D), np.nan),
            sigma_sqr=np.full((record_steps, d), np.nan),
            count=np.array(0),
        )
    zero = np.array(0)
    init = _Carry(
        t=save_at[0],
        x=x_0,
        h=bounded(h_0),
        log_likelihood=np.array(0.0),
        err_prev=np.array(_NO_PREVIOUS_ERROR),
        innovation=Gaussian(np.zeros(k), np.zeros((k, k))),
        sigma_sum=np.zeros(d),
        n_total=zero,
        n_accepted=zero,
        n_rejected=zero,
        composite=identity if smoother else None,
        record=record,
    )

    def attempt(c: _Carry, target: Array) -> _Carry:
        h = np.minimum(c.h, target - c.t)  # land exactly on the save time
        t_next = c.t + h
        transition = prior.transition(h)
        out = filter_step(
            c.x,
            transition,
            information,
            t=t_next,
            calibration=calibration,
            scale_noise=prior.apply_state_sigma_sqr,
        )
        sigma = np.broadcast_to(out.sigma_sqr, (d,))
        if sigma_in_error == "running_mean":
            mean = c.sigma_sum / np.maximum(c.n_total, 1)
            sigma_err = np.where(c.n_total > 0, mean, sigma)
        else:
            sigma_err = sigma
        err = _local_error(transition, sigma_err, out.predicted.m, E0, atol, rtol)
        # A non-finite error is a rejection: the controller maps it to its
        # minimum factor, so the loop shrinks h instead of accepting NaN.
        accept = (err <= 1.0) & np.isfinite(err)
        n_new = accept.astype(c.n_total.dtype)  # counter increments, no bool + int

        def keep(new, old):
            return jax.tree.map(lambda a, b: np.where(accept, a, b), new, old)

        composite = c.composite
        if composite is not None:
            composite = keep(compose(composite, out.backward), composite)
        record = c.record
        if record is not None:
            i = np.minimum(c.n_total, record.t.shape[0] - 1)
            write = accept & (c.n_total < record.t.shape[0])
            record = StepRecord(
                t=record.t.at[i].set(np.where(write, t_next, record.t[i])),
                h=record.h.at[i].set(np.where(write, h, record.h[i])),
                m=record.m.at[i].set(np.where(write, out.filtered.m, record.m[i])),
                sigma_sqr=record.sigma_sqr.at[i].set(
                    np.where(write, sigma, record.sigma_sqr[i])
                ),
                count=c.n_total + n_new,
            )
        return _Carry(
            t=np.where(accept, t_next, c.t),
            x=keep(out.filtered, c.x),
            h=bounded(
                controller.propose(
                    h, err, np.where(accept, c.err_prev, _NO_PREVIOUS_ERROR)
                )
            ),
            log_likelihood=np.where(
                accept, c.log_likelihood + out.log_likelihood, c.log_likelihood
            ),
            err_prev=np.where(accept, err, c.err_prev),
            innovation=keep(out.innovation, c.innovation),
            sigma_sum=np.where(accept, c.sigma_sum + sigma, c.sigma_sum),
            n_total=c.n_total + n_new,
            n_accepted=c.n_accepted + n_new,
            n_rejected=c.n_rejected + (1 - n_new),
            composite=composite,
            record=record,
        )

    def integrate_to(c: _Carry, target: Array) -> _Carry:
        return eqxi.while_loop(
            lambda c: c.t < target - _landing_tolerance(target),
            lambda c: attempt(c, target),
            c,
            max_steps=max_steps,
            kind="checkpointed",
        )

    def save_interval(c: _Carry, inputs):
        target, data = inputs
        c = integrate_to(c, target)
        reached = c.t >= target - _landing_tolerance(target)
        innovation_obs = ll_obs = None
        x = c.x
        if data is not None:
            assert obs_model is not None
            y, mask = data
            x, innovation_obs, ll_obs = observe(
                x, AffineModel(obs_model.H, y, obs_model.R_sqr), mask
            )
        out = (
            x,
            reached,
            c.innovation,
            innovation_obs,
            ll_obs,
            c.composite,
            c.n_accepted,
            c.n_rejected,
        )
        reset = c._replace(
            x=x,
            n_accepted=zero,
            n_rejected=zero,
            composite=identity if smoother else None,
        )
        return reset, out

    data = None if obs_model is None else (obs_model.c_seq, obs_model.mask)
    final, out = jax.lax.scan(save_interval, init, (save_at[1:], data))
    x, reached, innovation, innovation_obs, ll_obs, composite, n_acc, n_rej = out

    def with_initial(first: Array, rest: Array) -> Array:
        return np.concatenate([first[None], rest], axis=0)

    ll_obs_total = None if ll_obs is None else np.sum(ll_obs)
    # The solve succeeded iff every save interval (not just the last) landed on
    # its save time and the evidence is finite.
    finite = np.isfinite(final.log_likelihood)
    if ll_obs_total is not None:
        finite = finite & np.isfinite(ll_obs_total)
    return FilterResult(
        t=save_at,
        m=with_initial(x_0.m, x.m),
        P_sqr=with_initial(x_0.P_sqr, x.P_sqr),
        log_likelihood=final.log_likelihood,
        log_likelihood_obs=ll_obs_total,
        G_back=None if composite is None else composite.G,
        d_back=None if composite is None else composite.d,
        P_back_sqr=None if composite is None else composite.P_sqr,
        mz=innovation.m,
        Pz_sqr=innovation.P_sqr,
        mz_obs=None if innovation_obs is None else innovation_obs.m,
        Pz_obs_sqr=None if innovation_obs is None else innovation_obs.P_sqr,
        success=np.all(reached) & finite,
        n_accepted=n_acc,
        n_rejected=n_rej,
        steps=final.record,
    )
