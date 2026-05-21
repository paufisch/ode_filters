from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as np
import numpy as onp
from jax import Array

from ..calibration.sigma import quasi_mle_sigma_sqr_from_Q
from ..measurement.measurement_models import (
    BaseODEInformation,
    Measurement,
    ObsModel,
    build_obs_at_time,
)
from ..priors.gmp_priors import BasePrior
from .ode_filter_step import (
    ekf1_sqr_filter_step,
    ekf1_sqr_filter_step_preconditioned,
    ekf1_sqr_filter_step_preconditioned_sequential,
    ekf1_sqr_filter_step_preconditioned_sequential_scan,
    ekf1_sqr_filter_step_sequential,
    ekf1_sqr_filter_step_sequential_scan,
    rts_sqr_smoother_step,
    rts_sqr_smoother_step_preconditioned,
)

StateFunction = Callable[[Array], Array]
JacobianFunction = Callable[[Array], Array]


def _check_state_xi_diagonal(prior: BasePrior, calibration: str) -> None:
    """Verify ``prior.xi_state`` is diagonal for diagonal-mode calibration.

    Per-component diagonal calibration estimates ``sigma_hat^2_i`` from a
    per-row formula that is only correct when ``Xi`` is diagonal -- the
    estimator silently mis-allocates variance across components otherwise.

    The check is gated on whether ``xi`` is concretely materializable: if
    we're under outer-loop tracing (e.g. ``jax.grad`` over ``Xi`` itself)
    we trust the caller's invariant and skip the check. This is the only
    pattern that lets the loop be both ``jit``-traceable end-to-end and
    diff-traceable through ``Xi`` while still catching the common
    misconfiguration of passing a non-diagonal ``Xi`` to a diagonal-mode
    call.
    """
    try:
        xi_h = onp.asarray(prior.xi_state)
    except jax.errors.TracerArrayConversionError:
        return
    if not onp.allclose(xi_h - onp.diag(onp.diag(xi_h)), 0.0):
        raise ValueError(
            f"calibration={calibration!r} requires the state-block "
            "Xi to be diagonal (joint priors: this is "
            "``prior._prior_x.xi``)."
        )


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

    ts, h = onp.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    h = float(h)
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
            t=float(ts[i + 1]),
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


DynamicLoopResult = tuple[
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    float,
]


def ekf1_sqr_loop_dynamic(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    *,
    calibration: str = "dynamic",
    min_sigma_sqr: float = 0.0,
    calibrate: bool | None = None,
) -> DynamicLoopResult:
    """Fixed-step square-root EKF with online per-step diffusion calibration.

    Same fixed grid as :func:`ekf1_sqr_loop`, but at every step the per-step
    quasi-MLE diffusion is estimated and -- depending on ``calibration`` --
    baked into the current step's ``Q_h`` before propagation. See
    :func:`~ode_filters.filters.ekf1_sqr_adaptive_loop` for the full
    description of the modes; this loop supports the same names:

    - ``"dynamic"`` (default, scalar): Bosch et al. 2021 Eq. 32.
      ``P_pred = A P_prev A.T + sigma_hat^2_n * Q_h``.
    - ``"diagonal_ekf0"``: per-component ``sigma_hat^2_i`` from
      ``(E_1 Q E_1.T)_ii`` (exact-diagonal denominator). Genuine MLE in
      a block-diagonal sub-model for component ``i``. Recommended default
      for multi-component problems.
    - ``"diagonal"``: same per-component formula but with the EK1
      Jacobian ``H_t = E_1 - J_f E_0`` in the denominator. Heuristic --
      equivalent to ``"diagonal_ekf0"`` when ``J_f`` is block-diagonal in
      components, biased in proportion to ``J_f``'s off-diagonal entries
      otherwise.
    - ``"none"``: propagate ``sigma=1``, report ``sigma_hat^2`` for post-hoc
      rescaling (combine with :func:`posthoc_mle_sigma_sqr` /
      :func:`rescale_sqr_seq`).

    ``"cumulative"`` is intentionally *not* exposed on the fixed-step loop:
    the cumulative scheme's appeal is robust adaptive-step control, and at
    fixed step there's nothing to gain over post-hoc rescaling.

    The returned ``log_likelihood`` reflects the *calibrated* posterior --
    each step's contribution uses the calibrated ``Pz_sqr``. With
    ``calibration="none"`` it matches the log-likelihood from
    :func:`ekf1_sqr_loop`; in the calibrated modes it is the log-likelihood
    under the corresponding generative model and is not directly comparable
    across modes.

    Args:
        mu_0: Initial state mean.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (e.g. ``IWP``).
        measure: Measurement model (e.g. ``ODEInformation``).
        tspan: Time interval ``(t_start, t_end)``.
        N: Number of filter steps.
        calibration: ``"dynamic"`` (default), ``"diagonal"``, or ``"none"``.
            All modes work with :class:`JointPrior` /
            :class:`PrecondJointPrior`; diagonal modes scale the state block
            only and require ``prior._prior_x.xi`` to be diagonal.
        min_sigma_sqr: Lower bound on the per-step ``sigma_hat^2`` (and on
            each component in diagonal modes) before it is baked into
            ``Q_step_sqr``. Default ``0.0`` preserves unclamped behavior.
        calibrate: Deprecated; ``True`` -> ``"dynamic"``, ``False`` -> ``"none"``.

    Returns:
        Tuple of 10 sequences plus the scalar log-marginal-likelihood:
        ``(m_seq, P_seq_sqr, m_pred_seq, P_pred_seq_sqr, G_back_seq,
        d_back_seq, P_back_seq_sqr, mz_seq, Pz_seq_sqr, sigma_sqr_seq,
        log_likelihood)``. In the diagonal modes ``sigma_sqr_seq``
        entries are length-``d_ode`` arrays; otherwise Python floats.
    """
    import warnings

    if calibrate is not None:
        warnings.warn(
            "`calibrate` is deprecated; use `calibration='dynamic'` or 'none'.",
            DeprecationWarning,
            stacklevel=2,
        )
        calibration = "dynamic" if calibrate else "none"
    _valid_fixed = ("dynamic", "diagonal", "diagonal_ekf0", "none")
    if calibration not in _valid_fixed:
        raise ValueError(
            f"calibration must be one of {_valid_fixed} for the "
            f"fixed-step dynamic loop; got {calibration!r}."
        )
    if calibration in ("diagonal", "diagonal_ekf0"):
        _check_state_xi_diagonal(prior, calibration)

    m_seq = [mu_0]
    P_seq_sqr = [Sigma_0_sqr]
    m_pred_seq = []
    P_pred_seq_sqr = []
    G_back_seq = []
    d_back_seq = []
    P_back_seq_sqr = []
    mz_seq = []
    Pz_seq_sqr = []
    sigma_sqr_seq: list = []

    ts, h = onp.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    h = float(h)
    A_h = prior.A(h)
    b_h = prior.b(h)
    Q_h = prior.Q(h)
    Q_h_sqr = np.linalg.cholesky(Q_h).T
    E1 = prior.E1
    d_ode = measure.ode_dim

    log_likelihood = 0.0

    for i in range(N):
        t = float(ts[i + 1])
        m_prev = m_seq[-1]
        P_prev_sqr = P_seq_sqr[-1]

        m_pred_provisional = A_h @ m_prev + b_h
        H_t, c_t = measure.linearize(m_pred_provisional, t=t)
        mz_pred = H_t @ m_pred_provisional + c_t

        # Calibration uses only the ODE-defect rows of the stacked residual
        # (Bosch, Tronarp, Hennig 2022 sec. 3 / ProbNumDiffEq.jl
        # `perform_step!`). Conservation and observation rows still update
        # the posterior and contribute to the log-likelihood; they just do
        # not drive sigma.
        H_ode = H_t[:d_ode]
        mz_ode = mz_pred[:d_ode]

        if calibration in ("diagonal", "diagonal_ekf0"):
            H_for_calib = E1 if calibration == "diagonal_ekf0" else H_ode
            denom = np.einsum("ij,jk,ik->i", H_for_calib, Q_h, H_for_calib)
            sigma_vec = mz_ode**2 / denom
            sigma_vec = np.maximum(sigma_vec, min_sigma_sqr)
            Q_step_sqr = prior.apply_state_sigma_sqr(Q_h_sqr, sigma_vec)
            sigma_to_store: float | Array = onp.asarray(sigma_vec)
        else:
            sigma_scalar = quasi_mle_sigma_sqr_from_Q(mz_ode, H_ode, Q_h_sqr)
            sigma_scalar = np.maximum(sigma_scalar, min_sigma_sqr)
            Q_step_sqr = (
                prior.apply_state_sigma_sqr(Q_h_sqr, sigma_scalar)
                if calibration == "dynamic"
                else Q_h_sqr
            )
            sigma_to_store = float(sigma_scalar)

        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz, Pz_sqr),
            (m, P_sqr),
        ) = ekf1_sqr_filter_step(A_h, b_h, Q_step_sqr, m_prev, P_prev_sqr, measure, t=t)

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
        sigma_sqr_seq.append(sigma_to_store)
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
        sigma_sqr_seq,
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

    ts, h = onp.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    h = float(h)
    A_bar = prior.A(h)
    b_bar = prior.b(h)
    Q_sqr_bar = np.linalg.cholesky(prior.Q(h)).T
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
            t=float(ts[i + 1]),
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


PrecondDynamicLoopResult = tuple[
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    Array,
    float,
]


def ekf1_sqr_loop_preconditioned_dynamic(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    *,
    calibration: str = "dynamic",
    min_sigma_sqr: float = 0.0,
    calibrate: bool | None = None,
) -> PrecondDynamicLoopResult:
    """Preconditioned fixed-step square-root EKF with online calibration.

    Same algorithm as :func:`ekf1_sqr_loop_dynamic`, executed in the
    preconditioned (bar) space. The calibration estimators are invariant
    under preconditioning: ``H Q H.T = (H T) Q_bar (H T).T``, so
    ``sigma_hat^2`` computed in bar space matches the original-space value
    exactly.

    Supported ``calibration`` modes: ``"dynamic"``, ``"diagonal"``,
    ``"diagonal_ekf0"``, ``"none"`` -- see
    :func:`ekf1_sqr_adaptive_loop` for the full description. (``"cumulative"``
    is not exposed for fixed-step loops.)

    For the diagonal modes the bar-space time-block ``Q_bar_time`` is
    obtained from ``prior._Q_bar`` if present (the ``PrecondIWP``
    convention, where ``Q_bar_time`` is h-independent); otherwise this
    function raises ``AttributeError``. Other preconditioned priors that
    expose a different bar-space convention are not supported in the
    diagonal modes.

    Args:
        mu_0: Initial state mean (original space).
        P_0_sqr: Initial state covariance (square-root form, original space).
        prior: Preconditioned prior (typically ``PrecondIWP``).
        measure: Measurement model (e.g., ``ODEInformation``).
        tspan: Time interval ``(t_start, t_end)``.
        N: Number of filter steps.
        calibration: ``"dynamic"`` (default), ``"diagonal"``,
            ``"diagonal_ekf0"``, or ``"none"``. All modes work with
            :class:`JointPrior` / :class:`PrecondJointPrior`; diagonal modes
            scale the state block only and require ``prior._prior_x.xi`` to
            be diagonal.
        min_sigma_sqr: Lower bound applied to per-step ``sigma_hat^2``
            before baking into ``Q_step_sqr_bar``. Default ``0.0``
            preserves unclamped behavior.
        calibrate: Deprecated; ``True`` -> ``"dynamic"``, ``False`` -> ``"none"``.

    Returns:
        Tuple of 12 sequences, the preconditioning matrix ``T_h``, and the
        scalar log-marginal-likelihood. ``sigma_sqr_seq`` entries are
        length-``d_ode`` arrays for the diagonal modes; floats otherwise.
    """
    import warnings

    if calibrate is not None:
        warnings.warn(
            "`calibrate` is deprecated; use `calibration='dynamic'` or 'none'.",
            DeprecationWarning,
            stacklevel=2,
        )
        calibration = "dynamic" if calibrate else "none"
    _valid = ("dynamic", "diagonal", "diagonal_ekf0", "none")
    if calibration not in _valid:
        raise ValueError(f"calibration must be one of {_valid}; got {calibration!r}.")
    if calibration in ("diagonal", "diagonal_ekf0"):
        _check_state_xi_diagonal(prior, calibration)

    ts, h = onp.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    h = float(h)
    A_bar = prior.A(h)
    b_bar = prior.b(h)
    Q_sqr_bar = np.linalg.cholesky(prior.Q(h)).T
    T_h = prior.T(h)
    E1 = prior.E1
    d_ode = measure.ode_dim

    m_seq = [mu_0]
    P_seq_sqr = [P_0_sqr]
    m_seq_bar = [np.linalg.solve(T_h, mu_0)]
    P_seq_sqr_bar = [np.linalg.solve(T_h, P_0_sqr.T).T]

    m_pred_seq_bar: list = []
    P_pred_seq_sqr_bar: list = []
    G_back_seq_bar: list = []
    d_back_seq_bar: list = []
    P_back_seq_sqr_bar: list = []
    mz_seq: list = []
    Pz_seq_sqr: list = []
    sigma_sqr_seq: list = []

    log_likelihood = 0.0

    for i in range(N):
        t = float(ts[i + 1])
        m_prev_bar = m_seq_bar[-1]
        P_prev_sqr_bar = P_seq_sqr_bar[-1]

        # Provisional predicted mean in bar space, then linearise in original
        # space and build the bar-space observation Jacobian H_bar = H @ T.
        m_pred_prov_bar = A_bar @ m_prev_bar + b_bar
        m_pred_prov = T_h @ m_pred_prov_bar
        H_t, c_t = measure.linearize(m_pred_prov, t=t)
        H_t_bar = H_t @ T_h
        mz_pred = H_t_bar @ m_pred_prov_bar + c_t

        # Calibration uses only the ODE-defect rows (Bosch et al. 2022 sec. 3).
        H_ode_bar = H_t_bar[:d_ode]
        mz_ode = mz_pred[:d_ode]

        if calibration in ("diagonal", "diagonal_ekf0"):
            # Note: E1 in original space; E1_bar = E1 @ T.
            E1_bar = E1 @ T_h
            H_for_calib = E1_bar if calibration == "diagonal_ekf0" else H_ode_bar
            Q_bar_full = Q_sqr_bar.T @ Q_sqr_bar
            denom = np.einsum("ij,jk,ik->i", H_for_calib, Q_bar_full, H_for_calib)
            sigma_vec = mz_ode**2 / denom
            sigma_vec = np.maximum(sigma_vec, min_sigma_sqr)
            Q_step_sqr_bar = prior.apply_state_sigma_sqr(Q_sqr_bar, sigma_vec)
            sigma_to_store: float | Array = onp.asarray(sigma_vec)
        else:
            sigma_scalar = quasi_mle_sigma_sqr_from_Q(mz_ode, H_ode_bar, Q_sqr_bar)
            sigma_scalar = np.maximum(sigma_scalar, min_sigma_sqr)
            Q_step_sqr_bar = (
                prior.apply_state_sigma_sqr(Q_sqr_bar, sigma_scalar)
                if calibration == "dynamic"
                else Q_sqr_bar
            )
            sigma_to_store = float(sigma_scalar)

        (
            (m_pred_bar, P_pred_sqr_bar),
            (G_back_bar, d_back_bar, P_back_sqr_bar),
            (mz, Pz_sqr),
            (m_bar_next, P_sqr_bar_next),
            (m_next, P_sqr_next),
        ) = ekf1_sqr_filter_step_preconditioned(
            A_bar,
            b_bar,
            Q_step_sqr_bar,
            T_h,
            m_prev_bar,
            P_prev_sqr_bar,
            measure,
            t=t,
        )

        log_det = 2.0 * np.sum(np.log(np.abs(np.diag(Pz_sqr))))
        v = jax.scipy.linalg.solve_triangular(Pz_sqr.T, mz, lower=True)
        maha = v @ v
        obs_dim = mz.shape[0]
        log_likelihood += -0.5 * (obs_dim * np.log(2 * np.pi) + log_det + maha)

        m_pred_seq_bar.append(m_pred_bar)
        P_pred_seq_sqr_bar.append(P_pred_sqr_bar)
        G_back_seq_bar.append(G_back_bar)
        d_back_seq_bar.append(d_back_bar)
        P_back_seq_sqr_bar.append(P_back_sqr_bar)
        mz_seq.append(mz)
        Pz_seq_sqr.append(Pz_sqr)
        sigma_sqr_seq.append(sigma_to_store)
        m_seq_bar.append(m_bar_next)
        P_seq_sqr_bar.append(P_sqr_bar_next)
        m_seq.append(m_next)
        P_seq_sqr.append(P_sqr_next)

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
        sigma_sqr_seq,
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
# Sequential update loop functions
# =============================================================================


def _log_likelihood_contrib(mz: Array, Pz_sqr: Array) -> float:
    """Compute log-likelihood contribution from an observation marginal."""
    log_det = 2.0 * np.sum(np.log(np.abs(np.diag(Pz_sqr))))
    v = jax.scipy.linalg.solve_triangular(Pz_sqr.T, mz, lower=True)
    maha = v @ v
    obs_dim = mz.shape[0]
    return -0.5 * (obs_dim * np.log(2 * np.pi) + log_det + maha)


SeqLoopResult = tuple[
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    list,
    float,
    float,
]


def ekf1_sqr_loop_sequential(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    *,
    observations: list[Measurement] | None = None,
) -> SeqLoopResult:
    """Run a sequential square-root EKF over ``N`` steps.

    At each step the ODE + Conservation update is applied first.  Then,
    if *observations* are provided, an observation update is applied
    using the ODE-updated state as the prior.

    Args:
        mu_0: Initial state mean estimate.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (e.g., IWP or PrecondIWP).
        measure: Measurement model (e.g., ODEInformation or subclass).
        tspan: Time interval (t_start, t_end).
        N: Number of filter steps.
        observations: Optional list of :class:`Measurement` constraints.
            When provided, ``build_obs_at_time`` is called at each step
            to construct an ``(H, c, R_sqr)`` tuple for the observation
            update.

    Returns:
        Tuple of 11 lists and two scalars
        ``(log_likelihood_ode, log_likelihood_obs)``.
    """
    m_seq = [mu_0]
    P_seq_sqr = [Sigma_0_sqr]
    m_pred_seq = []
    P_pred_seq_sqr = []
    G_back_seq = []
    d_back_seq = []
    P_back_seq_sqr = []
    mz_ode_seq = []
    Pz_ode_seq_sqr = []
    mz_obs_seq = []
    Pz_obs_seq_sqr = []

    ts, h = onp.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    h = float(h)
    A_h = prior.A(h)
    b_h = prior.b(h)
    Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T

    log_likelihood_ode = 0.0
    log_likelihood_obs = 0.0

    for i in range(N):
        t = float(ts[i + 1])

        # Build observation tuple for this step (or None)
        obs = None
        if observations is not None:
            obs = build_obs_at_time(observations, measure._E0, t)

        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz_ode, Pz_ode_sqr),
            (mz_obs, Pz_obs_sqr),
            (m, P_sqr),
        ) = ekf1_sqr_filter_step_sequential(
            A_h,
            b_h,
            Q_h_sqr,
            m_seq[-1],
            P_seq_sqr[-1],
            measure,
            t=t,
            obs=obs,
        )

        # ODE log-likelihood (always active)
        log_likelihood_ode += _log_likelihood_contrib(mz_ode, Pz_ode_sqr)

        # Observation log-likelihood (only when active)
        if mz_obs.shape[0] > 0:
            log_likelihood_obs += _log_likelihood_contrib(mz_obs, Pz_obs_sqr)

        m_pred_seq.append(m_pred)
        P_pred_seq_sqr.append(P_pred_sqr)
        G_back_seq.append(G_back)
        d_back_seq.append(d_back)
        P_back_seq_sqr.append(P_back_sqr)
        mz_ode_seq.append(mz_ode)
        Pz_ode_seq_sqr.append(Pz_ode_sqr)
        mz_obs_seq.append(mz_obs)
        Pz_obs_seq_sqr.append(Pz_obs_sqr)
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
        mz_ode_seq,
        Pz_ode_seq_sqr,
        mz_obs_seq,
        Pz_obs_seq_sqr,
        log_likelihood_ode,
        log_likelihood_obs,
    )


def ekf1_sqr_loop_preconditioned_sequential(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    *,
    observations: list[Measurement] | None = None,
) -> tuple:
    """Run a preconditioned sequential square-root EKF over ``N`` steps.

    Args:
        mu_0: Initial state mean estimate.
        P_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (typically PrecondIWP).
        measure: Measurement model (e.g., ODEInformation or subclass).
        tspan: Time interval (t_start, t_end).
        N: Number of filter steps.
        observations: Optional list of :class:`Measurement` constraints.

    Returns:
        Tuple of arrays, preconditioning matrix, and two scalars
        ``(log_likelihood_ode, log_likelihood_obs)``.
    """
    ts, h = onp.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    h = float(h)
    A_bar = prior.A(h)
    b_bar = prior.b(h)
    Q_sqr_bar = np.linalg.cholesky(prior.Q(h)).T
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
    mz_ode_seq = []
    Pz_ode_seq_sqr = []
    mz_obs_seq = []
    Pz_obs_seq_sqr = []

    log_likelihood_ode = 0.0
    log_likelihood_obs = 0.0

    for i in range(N):
        t = float(ts[i + 1])

        # Build observation tuple for this step (or None)
        obs = None
        if observations is not None:
            obs = build_obs_at_time(observations, measure._E0, t)

        (
            (m_pred_bar, P_pred_sqr_bar),
            (G_back_bar, d_back_bar, P_back_sqr_bar),
            (mz_ode, Pz_ode_sqr),
            (mz_obs, Pz_obs_sqr),
            (m_bar_next, P_sqr_bar_next),
            (m_next, P_sqr_next),
        ) = ekf1_sqr_filter_step_preconditioned_sequential(
            A_bar,
            b_bar,
            Q_sqr_bar,
            T_h,
            m_seq_bar[-1],
            P_seq_sqr_bar[-1],
            measure,
            t=t,
            obs=obs,
        )

        # ODE log-likelihood
        log_likelihood_ode += _log_likelihood_contrib(mz_ode, Pz_ode_sqr)

        # Observation log-likelihood
        if mz_obs.shape[0] > 0:
            log_likelihood_obs += _log_likelihood_contrib(mz_obs, Pz_obs_sqr)

        m_pred_seq_bar.append(m_pred_bar)
        P_pred_seq_sqr_bar.append(P_pred_sqr_bar)
        G_back_seq_bar.append(G_back_bar)
        d_back_seq_bar.append(d_back_bar)
        P_back_seq_sqr_bar.append(P_back_sqr_bar)
        mz_ode_seq.append(mz_ode)
        Pz_ode_seq_sqr.append(Pz_ode_sqr)
        mz_obs_seq.append(mz_obs)
        Pz_obs_seq_sqr.append(Pz_obs_sqr)
        m_seq_bar.append(m_bar_next)
        P_seq_sqr_bar.append(P_sqr_bar_next)
        m_seq.append(m_next)
        P_seq_sqr.append(P_sqr_next)

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
        mz_ode_seq,
        Pz_ode_seq_sqr,
        mz_obs_seq,
        Pz_obs_seq_sqr,
        T_h,
        log_likelihood_ode,
        log_likelihood_obs,
    )


# =============================================================================
# Scan-based sequential update loop functions
# =============================================================================


SeqScanLoopResult = tuple[
    Array,  # m_seq [N+1, state_dim]
    Array,  # P_seq_sqr [N+1, state_dim, state_dim]
    Array,  # m_pred_seq [N, state_dim]
    Array,  # P_pred_seq_sqr [N, state_dim, state_dim]
    Array,  # G_back_seq [N, state_dim, state_dim]
    Array,  # d_back_seq [N, state_dim]
    Array,  # P_back_seq_sqr [N, state_dim, state_dim]
    Array,  # mz_ode_seq [N, fixed_dim]
    Array,  # Pz_ode_seq_sqr [N, fixed_dim, fixed_dim]
    Array,  # mz_obs_seq [N, obs_dim]
    Array,  # Pz_obs_seq_sqr [N, obs_dim, obs_dim]
    Array,  # log_likelihood_ode
    Array,  # log_likelihood_obs
]


def ekf1_sqr_loop_sequential_scan(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    obs_model: ObsModel | None = None,
) -> SeqScanLoopResult:
    """Run a sequential square-root EKF using ``jax.lax.scan``.

    At each step the ODE + Conservation update is applied first.  Then,
    if *obs_model* is provided, an observation update is applied using
    the ODE-updated state as the prior.  Inactive observation steps are
    masked via ``jax.lax.select`` so that the state and log-likelihood
    are unaffected.

    Args:
        mu_0: Initial state mean estimate.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (e.g., IWP).
        measure: Measurement model (ODE + Conservation only, no
            :class:`Measurement` constraints bundled in).
        tspan: Time interval ``(t_start, t_end)``.
        N: Number of filter steps.
        obs_model: Pre-computed observation data from
            :func:`prepare_observations`, or ``None`` for ODE-only.

    Returns:
        Tuple of 11 arrays and two scalar log-likelihoods
        ``(log_likelihood_ode, log_likelihood_obs)``.
    """
    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_h = prior.A(h)
    b_h = prior.b(h)
    Q_h_sqr = np.linalg.cholesky(prior.Q(h)).T

    state_dim = mu_0.shape[0]

    has_obs = obs_model is not None
    if has_obs:
        obs_dim = obs_model.H.shape[0]
        H_obs = obs_model.H
        R_obs_sqr = obs_model.R_sqr
        c_obs_seq = obs_model.c_seq  # [N, obs_dim]
        mask_seq = obs_model.mask  # [N, obs_dim]
    else:
        obs_dim = 1
        H_obs = np.zeros((obs_dim, state_dim))
        R_obs_sqr = np.eye(obs_dim)
        c_obs_seq = np.zeros((N, obs_dim))
        mask_seq = np.zeros((N, obs_dim), dtype=bool)

    def scan_body(carry, step_data):
        m, P_sqr, ll_ode, ll_obs = carry
        t_i, c_obs_i, mask_i = step_data

        obs_active = mask_i.any()

        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz_ode, Pz_ode_sqr),
            (mz_obs, Pz_obs_sqr),
            (m_new, P_new_sqr),
        ) = ekf1_sqr_filter_step_sequential_scan(
            A_h,
            b_h,
            Q_h_sqr,
            m,
            P_sqr,
            measure,
            t_i,
            H_obs,
            c_obs_i,
            R_obs_sqr,
            obs_active,
        )

        ll_ode = ll_ode + _log_likelihood_contrib(mz_ode, Pz_ode_sqr)
        ll_obs = ll_obs + jax.lax.select(
            obs_active,
            _log_likelihood_contrib(mz_obs, Pz_obs_sqr),
            np.array(0.0),
        )

        outputs = (
            m_pred,
            P_pred_sqr,
            G_back,
            d_back,
            P_back_sqr,
            mz_ode,
            Pz_ode_sqr,
            mz_obs,
            Pz_obs_sqr,
            m_new,
            P_new_sqr,
        )
        return (m_new, P_new_sqr, ll_ode, ll_obs), outputs

    init_carry = (mu_0, Sigma_0_sqr, np.array(0.0), np.array(0.0))
    step_data = (ts[1:], c_obs_seq, mask_seq)

    (_, _, ll_ode, ll_obs), outputs = jax.lax.scan(scan_body, init_carry, step_data)

    (
        m_pred_seq,
        P_pred_seq_sqr,
        G_back_seq,
        d_back_seq,
        P_back_seq_sqr,
        mz_ode_seq,
        Pz_ode_seq_sqr,
        mz_obs_seq,
        Pz_obs_seq_sqr,
        m_updates,
        P_updates_sqr,
    ) = outputs

    # Prepend initial state
    m_seq = np.concatenate([mu_0[None, :], m_updates], axis=0)
    P_seq_sqr = np.concatenate([Sigma_0_sqr[None, :, :], P_updates_sqr], axis=0)

    return (
        m_seq,
        P_seq_sqr,
        m_pred_seq,
        P_pred_seq_sqr,
        G_back_seq,
        d_back_seq,
        P_back_seq_sqr,
        mz_ode_seq,
        Pz_ode_seq_sqr,
        mz_obs_seq,
        Pz_obs_seq_sqr,
        ll_ode,
        ll_obs,
    )


SeqPrecondScanLoopResult = tuple[
    Array,  # m_seq [N+1, state_dim]
    Array,  # P_seq_sqr [N+1, state_dim, state_dim]
    Array,  # m_seq_bar [N+1, state_dim]
    Array,  # P_seq_sqr_bar [N+1, state_dim, state_dim]
    Array,  # m_pred_seq_bar [N, state_dim]
    Array,  # P_pred_seq_sqr_bar [N, state_dim, state_dim]
    Array,  # G_back_seq_bar [N, state_dim, state_dim]
    Array,  # d_back_seq_bar [N, state_dim]
    Array,  # P_back_seq_sqr_bar [N, state_dim, state_dim]
    Array,  # mz_ode_seq [N, fixed_dim]
    Array,  # Pz_ode_seq_sqr [N, fixed_dim, fixed_dim]
    Array,  # mz_obs_seq [N, obs_dim]
    Array,  # Pz_obs_seq_sqr [N, obs_dim, obs_dim]
    Array,  # T_h [state_dim, state_dim]
    Array,  # log_likelihood_ode
    Array,  # log_likelihood_obs
]


def ekf1_sqr_loop_preconditioned_sequential_scan(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    obs_model: ObsModel | None = None,
) -> SeqPrecondScanLoopResult:
    """Run a preconditioned sequential square-root EKF using ``jax.lax.scan``.

    Args:
        mu_0: Initial state mean estimate.
        P_0_sqr: Initial state covariance (square-root form).
        prior: Prior model (typically PrecondIWP).
        measure: Measurement model (ODE + Conservation only).
        tspan: Time interval ``(t_start, t_end)``.
        N: Number of filter steps.
        obs_model: Pre-computed observation data from
            :func:`prepare_observations`, or ``None`` for ODE-only.

    Returns:
        Tuple of 13 arrays, preconditioning matrix, and two scalar
        log-likelihoods ``(log_likelihood_ode, log_likelihood_obs)``.
    """
    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_bar = prior.A(h)
    b_bar = prior.b(h)
    Q_sqr_bar = np.linalg.cholesky(prior.Q(h)).T
    T_h = prior.T(h)

    state_dim = mu_0.shape[0]
    m_0_bar = np.linalg.solve(T_h, mu_0)
    P_0_sqr_bar = np.linalg.solve(T_h, P_0_sqr.T).T

    has_obs = obs_model is not None
    if has_obs:
        obs_dim = obs_model.H.shape[0]
        H_obs = obs_model.H
        R_obs_sqr = obs_model.R_sqr
        c_obs_seq = obs_model.c_seq
        mask_seq = obs_model.mask
    else:
        obs_dim = 1
        H_obs = np.zeros((obs_dim, state_dim))
        R_obs_sqr = np.eye(obs_dim)
        c_obs_seq = np.zeros((N, obs_dim))
        mask_seq = np.zeros((N, obs_dim), dtype=bool)

    def scan_body(carry, step_data):
        m_bar, P_sqr_bar, ll_ode, ll_obs = carry
        t_i, c_obs_i, mask_i = step_data

        obs_active = mask_i.any()

        (
            (m_pred_bar, P_pred_sqr_bar),
            (G_back_bar, d_back_bar, P_back_sqr_bar),
            (mz_ode, Pz_ode_sqr),
            (mz_obs, Pz_obs_sqr),
            (m_new_bar, P_new_sqr_bar),
            (m_new, P_new_sqr),
        ) = ekf1_sqr_filter_step_preconditioned_sequential_scan(
            A_bar,
            b_bar,
            Q_sqr_bar,
            T_h,
            m_bar,
            P_sqr_bar,
            measure,
            t_i,
            H_obs,
            c_obs_i,
            R_obs_sqr,
            obs_active,
        )

        ll_ode = ll_ode + _log_likelihood_contrib(mz_ode, Pz_ode_sqr)
        ll_obs = ll_obs + jax.lax.select(
            obs_active,
            _log_likelihood_contrib(mz_obs, Pz_obs_sqr),
            np.array(0.0),
        )

        outputs = (
            m_pred_bar,
            P_pred_sqr_bar,
            G_back_bar,
            d_back_bar,
            P_back_sqr_bar,
            mz_ode,
            Pz_ode_sqr,
            mz_obs,
            Pz_obs_sqr,
            m_new_bar,
            P_new_sqr_bar,
            m_new,
            P_new_sqr,
        )
        return (m_new_bar, P_new_sqr_bar, ll_ode, ll_obs), outputs

    init_carry = (m_0_bar, P_0_sqr_bar, np.array(0.0), np.array(0.0))
    step_data = (ts[1:], c_obs_seq, mask_seq)

    (_, _, ll_ode, ll_obs), outputs = jax.lax.scan(scan_body, init_carry, step_data)

    (
        m_pred_seq_bar,
        P_pred_seq_sqr_bar,
        G_back_seq_bar,
        d_back_seq_bar,
        P_back_seq_sqr_bar,
        mz_ode_seq,
        Pz_ode_seq_sqr,
        mz_obs_seq,
        Pz_obs_seq_sqr,
        m_updates_bar,
        P_updates_sqr_bar,
        m_updates,
        P_updates_sqr,
    ) = outputs

    # Prepend initial state
    m_seq = np.concatenate([mu_0[None, :], m_updates], axis=0)
    P_seq_sqr = np.concatenate([P_0_sqr[None, :, :], P_updates_sqr], axis=0)
    m_seq_bar = np.concatenate([m_0_bar[None, :], m_updates_bar], axis=0)
    P_seq_sqr_bar = np.concatenate([P_0_sqr_bar[None, :, :], P_updates_sqr_bar], axis=0)

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
        mz_ode_seq,
        Pz_ode_seq_sqr,
        mz_obs_seq,
        Pz_obs_seq_sqr,
        T_h,
        ll_ode,
        ll_obs,
    )


# =============================================================================
# Scan-based dynamic-diffusion loops (joint EKF update, online calibration)
# =============================================================================


DynamicScanLoopResult = tuple[
    Array,  # m_seq [N+1, state_dim]
    Array,  # P_seq_sqr [N+1, state_dim, state_dim]
    Array,  # m_pred_seq [N, state_dim]
    Array,  # P_pred_seq_sqr [N, state_dim, state_dim]
    Array,  # G_back_seq [N, state_dim, state_dim]
    Array,  # d_back_seq [N, state_dim]
    Array,  # P_back_seq_sqr [N, state_dim, state_dim]
    Array,  # mz_seq [N, obs_dim]
    Array,  # Pz_seq_sqr [N, obs_dim, obs_dim]
    Array,  # sigma_sqr_seq [N]
    Array,  # log_likelihood (scalar)
]


def ekf1_sqr_loop_dynamic_scan(
    mu_0: Array,
    Sigma_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    *,
    calibration: str = "dynamic",
    min_sigma_sqr: float = 0.0,
    calibrate: bool | None = None,
) -> DynamicScanLoopResult:
    """``jax.lax.scan`` variant of :func:`ekf1_sqr_loop_dynamic`.

    Same per-step semantics, executed inside a single ``jax.lax.scan`` for
    full JIT/compile-once efficiency. Returns stacked ``Array`` sequences
    rather than lists. Supports ``calibration`` in ``{"dynamic", "diagonal",
    "diagonal_ekf0", "none"}``; in the diagonal modes ``sigma_sqr_seq`` has
    shape ``(N, d)`` rather than ``(N,)``.

    Args:
        mu_0: Initial state mean.
        Sigma_0_sqr: Initial state covariance (square-root form).
        prior: Prior (e.g. ``IWP``).
        measure: Measurement model.
        tspan: Time interval ``(t_start, t_end)``.
        N: Number of filter steps.
        calibration: ``"dynamic"`` (default), ``"diagonal"``,
            ``"diagonal_ekf0"``, or ``"none"``. All modes work with
            :class:`JointPrior` / :class:`PrecondJointPrior`; diagonal modes
            require ``prior._prior_x.xi`` to be diagonal.
        min_sigma_sqr: Lower bound applied to per-step ``sigma_hat^2`` (or
            each component in diagonal modes) before baking into
            ``Q_step_sqr``. Default ``0.0`` preserves unclamped behavior.
        calibrate: Deprecated; ``True`` -> ``"dynamic"``, ``False`` -> ``"none"``.

    Returns:
        :class:`DynamicScanLoopResult` -- 10 arrays plus the scalar
        log-marginal-likelihood. ``sigma_sqr_seq`` has shape ``(N, d_ode)``
        in the diagonal modes, ``(N,)`` otherwise.
    """
    import warnings

    if calibrate is not None:
        warnings.warn(
            "`calibrate` is deprecated; use `calibration='dynamic'` or 'none'.",
            DeprecationWarning,
            stacklevel=2,
        )
        calibration = "dynamic" if calibrate else "none"
    _valid = ("dynamic", "diagonal", "diagonal_ekf0", "none")
    if calibration not in _valid:
        raise ValueError(f"calibration must be one of {_valid}; got {calibration!r}.")
    is_diagonal = calibration in ("diagonal", "diagonal_ekf0")
    if is_diagonal:
        _check_state_xi_diagonal(prior, calibration)

    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_h = prior.A(h)
    b_h = prior.b(h)
    Q_h = prior.Q(h)
    Q_h_sqr = np.linalg.cholesky(Q_h).T
    E1 = prior.E1
    d_ode = measure.ode_dim

    def scan_body(carry, step_data):
        m_prev, P_prev_sqr, ll = carry
        t_i = step_data

        m_pred_prov = A_h @ m_prev + b_h
        H_t, c_t = measure.linearize(m_pred_prov, t=t_i)
        mz_pred = H_t @ m_pred_prov + c_t

        # Calibration uses only the ODE-defect rows of the stacked residual
        # (Bosch, Tronarp, Hennig 2022 sec. 3). Conservation / observation
        # rows still update the posterior but do not drive sigma.
        H_ode = H_t[:d_ode]
        mz_ode = mz_pred[:d_ode]

        if is_diagonal:
            H_for_calib = E1 if calibration == "diagonal_ekf0" else H_ode
            denom = np.einsum("ij,jk,ik->i", H_for_calib, Q_h, H_for_calib)
            sigma_sqr = mz_ode**2 / denom  # shape (d_ode,)
            sigma_sqr = np.maximum(sigma_sqr, min_sigma_sqr)
            Q_step_sqr = prior.apply_state_sigma_sqr(Q_h_sqr, sigma_sqr)
        else:
            sigma_scalar = quasi_mle_sigma_sqr_from_Q(mz_ode, H_ode, Q_h_sqr)
            sigma_scalar = np.maximum(sigma_scalar, min_sigma_sqr)
            sigma_sqr = sigma_scalar  # scalar
            if calibration == "dynamic":
                Q_step_sqr = prior.apply_state_sigma_sqr(Q_h_sqr, sigma_scalar)
            else:  # "none"
                Q_step_sqr = Q_h_sqr

        (
            (m_pred, P_pred_sqr),
            (G_back, d_back, P_back_sqr),
            (mz, Pz_sqr),
            (m_new, P_new_sqr),
        ) = ekf1_sqr_filter_step(
            A_h, b_h, Q_step_sqr, m_prev, P_prev_sqr, measure, t=t_i
        )

        ll_step = _log_likelihood_contrib(mz, Pz_sqr)
        ll = ll + ll_step

        outputs = (
            m_pred,
            P_pred_sqr,
            G_back,
            d_back,
            P_back_sqr,
            mz,
            Pz_sqr,
            sigma_sqr,
            m_new,
            P_new_sqr,
        )
        return (m_new, P_new_sqr, ll), outputs

    init_carry = (mu_0, Sigma_0_sqr, np.array(0.0))
    (_, _, ll), outputs = jax.lax.scan(scan_body, init_carry, ts[1:])

    (
        m_pred_seq,
        P_pred_seq_sqr,
        G_back_seq,
        d_back_seq,
        P_back_seq_sqr,
        mz_seq,
        Pz_seq_sqr,
        sigma_sqr_seq,
        m_updates,
        P_updates_sqr,
    ) = outputs

    m_seq = np.concatenate([mu_0[None, :], m_updates], axis=0)
    P_seq_sqr = np.concatenate([Sigma_0_sqr[None, :, :], P_updates_sqr], axis=0)

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
        sigma_sqr_seq,
        ll,
    )


PrecondDynamicScanLoopResult = tuple[
    Array,  # m_seq [N+1, state_dim]
    Array,  # P_seq_sqr [N+1, state_dim, state_dim]
    Array,  # m_seq_bar [N+1, state_dim]
    Array,  # P_seq_sqr_bar [N+1, state_dim, state_dim]
    Array,  # m_pred_seq_bar [N, state_dim]
    Array,  # P_pred_seq_sqr_bar [N, state_dim, state_dim]
    Array,  # G_back_seq_bar [N, state_dim, state_dim]
    Array,  # d_back_seq_bar [N, state_dim]
    Array,  # P_back_seq_sqr_bar [N, state_dim, state_dim]
    Array,  # mz_seq [N, obs_dim]
    Array,  # Pz_seq_sqr [N, obs_dim, obs_dim]
    Array,  # sigma_sqr_seq [N]
    Array,  # T_h [state_dim, state_dim]
    Array,  # log_likelihood (scalar)
]


def ekf1_sqr_loop_preconditioned_dynamic_scan(
    mu_0: Array,
    P_0_sqr: Array,
    prior: BasePrior,
    measure: BaseODEInformation,
    tspan: tuple[float, float],
    N: int,
    *,
    calibration: str = "dynamic",
    min_sigma_sqr: float = 0.0,
    calibrate: bool | None = None,
) -> PrecondDynamicScanLoopResult:
    """``jax.lax.scan`` variant of :func:`ekf1_sqr_loop_preconditioned_dynamic`.

    Operates in preconditioned (bar) space. Calibration is invariant under
    preconditioning: ``H Q H.T = (HT) Q_bar (HT).T``, so all four modes
    behave identically to the original-space variants.

    Args:
        mu_0: Initial state mean (original space).
        P_0_sqr: Initial state covariance (square-root form, original space).
        prior: Preconditioned prior (e.g. ``PrecondIWP``).
        measure: Measurement model.
        tspan: Time interval ``(t_start, t_end)``.
        N: Number of filter steps.
        calibration: ``"dynamic"`` (default), ``"diagonal"``,
            ``"diagonal_ekf0"``, or ``"none"``. All modes work with
            :class:`PrecondJointPrior`; diagonal modes require
            ``prior._prior_x.xi`` to be diagonal.
        min_sigma_sqr: Lower bound applied to per-step ``sigma_hat^2``
            before baking into ``Q_step_sqr_bar``. Default ``0.0``.
        calibrate: Deprecated; ``True`` -> ``"dynamic"``, ``False`` -> ``"none"``.

    Returns:
        :class:`PrecondDynamicScanLoopResult` -- 12 arrays, ``T_h``, and the
        scalar log-marginal-likelihood. ``sigma_sqr_seq`` shape is
        ``(N, d_ode)`` in diagonal modes, ``(N,)`` otherwise.
    """
    import warnings

    if calibrate is not None:
        warnings.warn(
            "`calibrate` is deprecated; use `calibration='dynamic'` or 'none'.",
            DeprecationWarning,
            stacklevel=2,
        )
        calibration = "dynamic" if calibrate else "none"
    _valid = ("dynamic", "diagonal", "diagonal_ekf0", "none")
    if calibration not in _valid:
        raise ValueError(f"calibration must be one of {_valid}; got {calibration!r}.")
    is_diagonal = calibration in ("diagonal", "diagonal_ekf0")
    if is_diagonal:
        _check_state_xi_diagonal(prior, calibration)

    ts, h = np.linspace(tspan[0], tspan[1], N + 1, retstep=True)
    A_bar = prior.A(h)
    b_bar = prior.b(h)
    Q_bar_full = prior.Q(h)
    Q_sqr_bar = np.linalg.cholesky(Q_bar_full).T
    T_h = prior.T(h)
    E1 = prior.E1
    d_ode = measure.ode_dim

    m_0_bar = np.linalg.solve(T_h, mu_0)
    P_0_sqr_bar = np.linalg.solve(T_h, P_0_sqr.T).T

    def scan_body(carry, step_data):
        m_prev_bar, P_prev_sqr_bar, ll = carry
        t_i = step_data

        m_pred_prov_bar = A_bar @ m_prev_bar + b_bar
        m_pred_prov = T_h @ m_pred_prov_bar
        H_t, c_t = measure.linearize(m_pred_prov, t=t_i)
        H_t_bar = H_t @ T_h
        mz_pred = H_t_bar @ m_pred_prov_bar + c_t

        # Calibration uses only the ODE-defect rows (Bosch et al. 2022 sec. 3).
        H_ode_bar = H_t_bar[:d_ode]
        mz_ode = mz_pred[:d_ode]

        if is_diagonal:
            E1_bar = E1 @ T_h
            H_for_calib = E1_bar if calibration == "diagonal_ekf0" else H_ode_bar
            denom = np.einsum("ij,jk,ik->i", H_for_calib, Q_bar_full, H_for_calib)
            sigma_sqr = mz_ode**2 / denom
            sigma_sqr = np.maximum(sigma_sqr, min_sigma_sqr)
            Q_step_sqr_bar = prior.apply_state_sigma_sqr(Q_sqr_bar, sigma_sqr)
        else:
            sigma_scalar = quasi_mle_sigma_sqr_from_Q(mz_ode, H_ode_bar, Q_sqr_bar)
            sigma_scalar = np.maximum(sigma_scalar, min_sigma_sqr)
            sigma_sqr = sigma_scalar
            if calibration == "dynamic":
                Q_step_sqr_bar = prior.apply_state_sigma_sqr(Q_sqr_bar, sigma_scalar)
            else:  # "none"
                Q_step_sqr_bar = Q_sqr_bar

        (
            (m_pred_bar, P_pred_sqr_bar),
            (G_back_bar, d_back_bar, P_back_sqr_bar),
            (mz, Pz_sqr),
            (m_new_bar, P_new_sqr_bar),
            (m_new, P_new_sqr),
        ) = ekf1_sqr_filter_step_preconditioned(
            A_bar,
            b_bar,
            Q_step_sqr_bar,
            T_h,
            m_prev_bar,
            P_prev_sqr_bar,
            measure,
            t=t_i,
        )

        ll_step = _log_likelihood_contrib(mz, Pz_sqr)
        ll = ll + ll_step

        outputs = (
            m_pred_bar,
            P_pred_sqr_bar,
            G_back_bar,
            d_back_bar,
            P_back_sqr_bar,
            mz,
            Pz_sqr,
            sigma_sqr,
            m_new_bar,
            P_new_sqr_bar,
            m_new,
            P_new_sqr,
        )
        return (m_new_bar, P_new_sqr_bar, ll), outputs

    init_carry = (m_0_bar, P_0_sqr_bar, np.array(0.0))
    (_, _, ll), outputs = jax.lax.scan(scan_body, init_carry, ts[1:])

    (
        m_pred_seq_bar,
        P_pred_seq_sqr_bar,
        G_back_seq_bar,
        d_back_seq_bar,
        P_back_seq_sqr_bar,
        mz_seq,
        Pz_seq_sqr,
        sigma_sqr_seq,
        m_updates_bar,
        P_updates_sqr_bar,
        m_updates,
        P_updates_sqr,
    ) = outputs

    m_seq = np.concatenate([mu_0[None, :], m_updates], axis=0)
    P_seq_sqr = np.concatenate([P_0_sqr[None, :, :], P_updates_sqr], axis=0)
    m_seq_bar = np.concatenate([m_0_bar[None, :], m_updates_bar], axis=0)
    P_seq_sqr_bar = np.concatenate([P_0_sqr_bar[None, :, :], P_updates_sqr_bar], axis=0)

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
        sigma_sqr_seq,
        T_h,
        ll,
    )
