"""Evidence for selecting hyperparameters from data, and information criteria.

Write ``Z`` for the ODE information at the grid nodes (observed as zero), ``Y``
for the data and ``U`` for the trajectory of a hidden input. A filter that
updates on the ODE information and then on the data at every step reports

    log_likelihood     = sum_n log p(z_n | z_<n, y_<n)
    log_likelihood_obs = sum_n log p(y_n | z_<=n, y_<n)

whose sum is ``log p(Z, Y)`` under the (linearized) model. The *conditional*
evidence ``log p(Y | Z) = log p(Z, Y) - log p(Z)`` conditions every
observation on all of the ODE information, including the later one. It
therefore differs from ``log_likelihood_obs`` whenever later ODE information is
informative about earlier data, which is the generic case. There are two
estimators of it:

- ``fenrir_loglik`` (Tronarp, Bosch & Hennig 2022): one pass on the ODE
  information alone, then a Kalman filter backwards through its backward
  conditionals, updating on the data.
- ``conditional_loglik``: a pass with data minus a pass without. On one shared
  linear-Gaussian model (one ``AffineSurrogate``) it equals Fenrir's value
  exactly. With one-shot filters each pass linearizes at its own predictions,
  and the difference is Dalton's estimator (Wu & Lysy 2024), which generally
  differs from Fenrir's.

``information_criteria`` measures how much the ODE information and the data
say about a hidden input, at a frozen linearization.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..filters.fixed_grid import (
    _concrete,
    check_grid,
    check_surrogate,
    grid_transitions,
    resolve_observations,
    timeline_transitions,
)
from ..filters.kernel import mask_rows, observe, scale_gaussian
from ..filters.result import FilterResult
from ..gaussian import AffineModel, Gaussian, marginalize
from ..gaussian.algebra import _safe_qr
from ..information import ObsModel
from ..information.observations import MEASUREMENT_TIME_ATOL, MEASUREMENT_TIME_RTOL
from ..linearization.surrogate import AffineSurrogate
from ..priors.base import Transition
from ..priors.joint import JointPrior


def fenrir_loglik(data_free: FilterResult, obs_model: ObsModel) -> Array:
    """``log p(Y | Z)`` by Fenrir: the data filtered backwards through a data-free pass.

    The pass conditioned on the ODE information alone defines a backward Markov
    chain ``x_n | x_{n+1}``. A Kalman filter runs it from the last node to the
    first and updates on the data where they are present, summing the evidence
    of each update (Tronarp, Bosch & Hennig 2022, Prop. 3.1). Observation step
    ``i`` of ``obs_model`` lives at node ``i + 1``; node 0 has none.

    With data-only nodes the pass without data must run on the timeline of
    ``obs_model``: pass ``obs_model=obs_model.without_data()`` to the solver.
    Its backward conditionals then reach the data-only nodes, while its ODE
    information is that of the grid.

    Args:
        data_free: A ``FilterResult`` conditioned on the ODE information only,
            with backward conditionals: ``gaussian_filter`` or
            ``affine_filter`` without data, or with
            ``obs_model=obs_model.without_data()``.
        obs_model: The data, on the result's nodes.

    Returns:
        The scalar ``log p(Y | Z)``.

    Raises:
        ValueError: If the result conditioned on data already (that would count
            the data twice), carries no backward conditionals, or ``obs_model``
            is on another grid.
    """
    if data_free.log_likelihood_obs is not None:
        raise ValueError(
            "fenrir_loglik needs a pass without data; this result already "
            "conditioned on data, which would count them twice."
        )
    backward = data_free.backward
    n_steps = backward.G.shape[0]
    nodes = None if obs_model.t is None else _concrete(data_free.t, float)
    timeline = None if obs_model.t is None else _concrete(obs_model.t, float)
    mismatch = obs_model.c_seq.shape[0] != n_steps
    if not mismatch and nodes is not None and timeline is not None:
        mismatch = nodes.shape != timeline.shape or not onp.allclose(
            nodes, timeline, rtol=MEASUREMENT_TIME_RTOL, atol=MEASUREMENT_TIME_ATOL
        )
    if mismatch:
        raise ValueError(
            f"obs_model has {obs_model.c_seq.shape[0]} steps but the result has "
            f"{n_steps}, or they are at other times. With data-only nodes, run "
            "the pass without data with obs_model=obs_model.without_data()."
        )

    def model(c: Array) -> AffineModel:
        return AffineModel(obs_model.H, c, obs_model.R_sqr)

    last = Gaussian(data_free.m[-1], data_free.P_sqr[-1])
    x, _, ll = observe(last, model(obs_model.c_seq[-1]), obs_model.mask[-1])
    # Step i predicts back to node i, whose data are those of step i - 1.
    c_at_node = np.concatenate(
        [np.zeros_like(obs_model.c_seq[:1]), obs_model.c_seq[:-1]]
    )
    mask_at_node = np.concatenate(
        [np.zeros_like(obs_model.mask[:1]), obs_model.mask[:-1]]
    )

    def step(carry, inputs):
        x, ll = carry
        G, d, P_sqr, c, mask = inputs
        x = marginalize(G, d, P_sqr, x)
        x, _, ll_n = observe(x, model(c), mask)
        return (x, ll + ll_n), None

    (_, ll), _ = jax.lax.scan(
        step,
        (x, ll),
        (backward.G, backward.d, backward.P_sqr, c_at_node, mask_at_node),
        reverse=True,
    )
    return ll


def conditional_loglik(data: FilterResult, data_free: FilterResult) -> Array:
    """``log p(Y | Z) = log p(Z, Y) - log p(Z)`` from a pass with and one without data.

    On one shared linear-Gaussian model, with both passes ``affine_filter``
    runs of the same ``AffineSurrogate``, this is exact and equals
    ``fenrir_loglik`` of the data-free pass. With one-shot filters
    (``gaussian_filter``) each pass linearizes at its own predictions, so the
    two terms belong to different linearizations: that difference is Dalton's
    estimator (Wu & Lysy 2024), which generally differs from Fenrir's.

    Args:
        data: A pass conditioned on the ODE information and the data.
        data_free: A pass conditioned on the ODE information only.

    Returns:
        The scalar ``data.log_likelihood + data.log_likelihood_obs -
        data_free.log_likelihood``.

    Raises:
        ValueError: If ``data`` carries no data evidence, or ``data_free`` does.
    """
    if data.log_likelihood_obs is None:
        raise ValueError("The first result must be a pass with data.")
    if data_free.log_likelihood_obs is not None:
        raise ValueError("The second result must be a pass without data.")
    return data.log_likelihood + data.log_likelihood_obs - data_free.log_likelihood


class InformationCriteria(NamedTuple):
    """Information about a hidden input ``U`` (in nats), at a frozen linearization.

    Attributes:
        I_UZ: ``I(U; Z)``, what the ODE information says about the input.
        I_UY_given_Z: ``I(U; Y | Z)``, what the data add to it.
        D: ``sum_n I(U; y_n | z_<=n, y_<n)``, the directed information of the
            data about the input, as the filter sees it.
    """

    I_UZ: Array
    I_UY_given_Z: Array
    D: Array


def _logdet(F: Array) -> Array:
    """``log|F.T @ F|`` of a triangular square-root factor ``F``."""
    return 2.0 * np.sum(np.log(np.abs(np.diag(F))))


def _predicted_factor(P_sqr: Array, transition: Transition) -> Array:
    """Square root of the predicted covariance ``A P A.T + Q``.

    A preconditioned transition acts on ``x / T_diag``: the factor moves into
    those coordinates and back, as in the filter's prediction.
    """
    x = Gaussian(np.zeros(P_sqr.shape[0], dtype=P_sqr.dtype), P_sqr)
    s = transition.T_diag
    frame = x if s is None else scale_gaussian(x, 1.0 / s)
    predicted = marginalize(transition.A, transition.b, transition.Q_sqr, frame)
    return (predicted if s is None else scale_gaussian(predicted, s)).P_sqr


def _updated_factors(P_sqr: Array, model: AffineModel) -> tuple[Array, Array]:
    """Innovation factor and updated factor of observing ``H x + v``, from one QR.

    The triangular factor of the pre-array ``[[R_sqr, 0], [P_sqr H.T, P_sqr]]``
    is ``[[S_sqr, K], [0, P_sqr_new]]``: its Gram matrix
    ``[[S, H P], [P H.T, P]]`` has the innovation covariance
    ``S = H P H.T + R`` in the first block, and ``P_sqr_new`` factors the
    Schur complement ``P - P H.T S^-1 H P``, the updated covariance.
    """
    k, n = model.H.shape
    pre = np.block(
        [
            [model.R_sqr, np.zeros((k, n), dtype=P_sqr.dtype)],
            [P_sqr @ model.H.T, P_sqr],
        ]
    )
    post = _safe_qr(pre)
    return post[:k, :k], post[k:, k:]


def _innovation_logdets(
    P_0_sqr: Array,
    transitions: tuple[Transition | None, Transition | None],
    surrogate: AffineSurrogate,
    obs_model: ObsModel | None = None,
) -> tuple[Array, Array]:
    """``sum_n log|S_n|`` of the ODE information and of the data along a pass.

    The innovation covariances do not depend on the means, so the pass carries
    the state's square-root covariance alone: one QR per prediction and one per
    update, which gives the innovation factor and the updated factor together
    (``_updated_factors``). Rows without data are masked as in the filters and
    add nothing; without ``obs_model`` the data's sum is zero.
    """
    constant, per_step = transitions
    data = None if obs_model is None else (obs_model.c_seq, obs_model.mask)

    def step(P_sqr: Array, inputs):
        transition, model, step_data = inputs
        step_transition = constant if transition is None else transition
        assert step_transition is not None
        P_sqr = _predicted_factor(P_sqr, step_transition)
        S_sqr, P_sqr = _updated_factors(P_sqr, AffineModel(*model))
        logdet_obs = np.zeros((), dtype=P_sqr.dtype)
        if step_data is not None:
            assert obs_model is not None
            c, mask = step_data
            observation = AffineModel(obs_model.H, c, obs_model.R_sqr)
            S_obs_sqr, P_sqr = _updated_factors(P_sqr, mask_rows(observation, mask))
            logdet_obs = _logdet(S_obs_sqr)
        return P_sqr, (_logdet(S_sqr), logdet_obs)

    _, (logdet_ode, logdet_obs) = jax.lax.scan(
        step, np.asarray(P_0_sqr), (per_step, surrogate, data)
    )
    return np.sum(logdet_ode), np.sum(logdet_obs)


def _on_timeline(
    surrogate: AffineSurrogate, interval: ArrayLike, grid_node: ArrayLike
) -> AffineSurrogate:
    """The surrogate per step of a timeline, switched off at the data-only nodes.

    A step takes the model of its grid interval. At a data-only node every row
    is absent (``mask_rows``): a zero row with unit noise, whose innovation
    factor is the identity, so it adds nothing to the log-determinants.
    """
    H, c, R_sqr = jax.tree.map(lambda a: a[interval], tuple(surrogate))

    def model(H: Array, c: Array, R_sqr: Array, on: Array) -> AffineModel:
        return mask_rows(AffineModel(H, c, R_sqr), np.broadcast_to(on, c.shape))

    return AffineSurrogate(*jax.vmap(model)(H, c, R_sqr, np.asarray(grid_node)))


def conditional_state_cov_sqr(P_0_sqr: ArrayLike, n_state: int) -> Array:
    """Square root of the state block's covariance given the input block.

    The Schur complement, from the factor: with the columns reordered to
    ``[input | state]``, the trailing block of the triangular factor is the
    conditional factor of the state given the input.

    Args:
        P_0_sqr: A square-root factor of the joint covariance, state block first.
        n_state: Dimension of the state block.
    """
    P_0_sqr = np.asarray(P_0_sqr)
    reordered = np.concatenate([P_0_sqr[:, n_state:], P_0_sqr[:, :n_state]], axis=1)
    R = _safe_qr(reordered)
    n_input = P_0_sqr.shape[1] - n_state
    return R[n_input:, n_input:]


def information_criteria(
    m_0: ArrayLike,
    P_0_sqr: ArrayLike,
    prior: JointPrior,
    surrogate: AffineSurrogate,
    ts: ArrayLike,
    obs_model: ObsModel,
    *,
    P_0_state_sqr: ArrayLike | None = None,
) -> InformationCriteria:
    """Mutual and directed informations of the ODE information and data about ``U``.

    At a frozen linearization everything is jointly Gaussian and the means drop
    out of the informations:

        I(U; Z)     = 1/2 [log|Cov(Z)|    - log|Cov(Z | U)|]
        I(U; Y, Z)  = 1/2 [log|Cov(Z, Y)| - log|Cov(Z, Y | U)|]
        I(U; Y | Z) = I(U; Y, Z) - I(U; Z)
        D           = 1/2 sum_n [log|S^obs_n| - log|S^obs_n given U|]

    Each log-determinant is a sum of innovation log-determinants along a filter
    pass, so four passes give them all: the joint model and the state block
    alone (given ``U``), each without and with the data. The innovation
    covariances do not depend on the means, so each pass carries only the
    square-root covariance, with one QR per prediction and per update. This
    is valid because the joint prior's block-diagonal transition makes the
    state block evolve independently given ``U``, and a frozen surrogate is
    affine in ``U``, so conditioning on ``U`` only shifts the residual means.
    The same holds for the dependence of the data on ``U``, which is why the
    data rows can be restricted to the state's columns.

    Args:
        m_0: Initial mean of the joint state.
        P_0_sqr: Square root of the joint initial covariance.
        prior: The ``JointPrior`` of state and input.
        surrogate: The frozen linearization (one affine model per step of
            ``ts``).
        ts: Time grid.
        obs_model: The data for the same grid. With data-only nodes the passes
            with the data run over its timeline, with the ODE information at
            the grid nodes only, as in ``affine_filter``.
        P_0_state_sqr: Square root of the state block's initial covariance given
            the input. Defaults to the Schur complement of ``P_0``
            (``conditional_state_cov_sqr``).

    Returns:
        The ``InformationCriteria``.

    Raises:
        TypeError: If ``prior`` is not a ``JointPrior``.
        ValueError: If the shapes of the surrogate, the data or the initial law
            do not match the prior, or ``obs_model`` was built for another grid.
    """
    if not isinstance(prior, JointPrior):
        raise TypeError(f"prior must be a JointPrior, got {type(prior).__name__}.")
    m_0 = np.asarray(m_0)
    P_0_sqr = np.asarray(P_0_sqr)
    D_x = prior.prior_x.state_dim
    if m_0.shape != (prior.state_dim,) or obs_model.H.shape[1] != prior.state_dim:
        raise ValueError(
            f"m_0 and the data must act on the joint state of dimension "
            f"{prior.state_dim}; got m_0 of shape {m_0.shape} and data on "
            f"{obs_model.H.shape[1]} columns."
        )
    ts = check_grid(ts)
    n_steps = ts.shape[0] - 1
    check_surrogate(surrogate, n_steps, m_0.shape[0])
    _, timeline = resolve_observations(obs_model, ts)
    P_0_state_sqr = (
        conditional_state_cov_sqr(P_0_sqr, D_x)
        if P_0_state_sqr is None
        else np.asarray(P_0_state_sqr)
    )
    state_surrogate = AffineSurrogate(
        surrogate.H[:, :, :D_x], surrogate.c, surrogate.R_sqr
    )
    state_obs = obs_model._replace(H=obs_model.H[:, :D_x])
    joint = grid_transitions(prior, ts)
    given_u = grid_transitions(prior.prior_x, ts)

    logdet_z, _ = _innovation_logdets(P_0_sqr, joint, surrogate)
    logdet_z_given_u, _ = _innovation_logdets(P_0_state_sqr, given_u, state_surrogate)
    if timeline is not None:
        # The passes with the data run over the timeline; its data-only steps
        # carry the surrogate rows switched off, which add nothing.
        joint = (None, timeline_transitions(prior, ts, timeline)[1])
        given_u = (None, timeline_transitions(prior.prior_x, ts, timeline)[1])
        surrogate = _on_timeline(surrogate, timeline.interval, timeline.grid_node)
        state_surrogate = _on_timeline(
            state_surrogate, timeline.interval, timeline.grid_node
        )
    # With the data: the ODE rows' innovations (now also given the earlier data)
    # and the data's; together they factorize Cov(Z, Y).
    logdet_z_data, logdet_obs = _innovation_logdets(
        P_0_sqr, joint, surrogate, obs_model
    )
    logdet_z_data_given_u, logdet_obs_given_u = _innovation_logdets(
        P_0_state_sqr, given_u, state_surrogate, state_obs
    )

    I_UZ = 0.5 * (logdet_z - logdet_z_given_u)
    I_UYZ = 0.5 * (
        (logdet_z_data + logdet_obs) - (logdet_z_data_given_u + logdet_obs_given_u)
    )
    D = 0.5 * (logdet_obs - logdet_obs_given_u)
    return InformationCriteria(I_UZ=I_UZ, I_UY_given_Z=I_UYZ - I_UZ, D=D)
