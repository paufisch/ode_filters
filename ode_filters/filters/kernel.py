"""The filter step every solver is built from.

One step of a Gaussian ODE filter moves the state ``x_n ~ N(m, P)`` to time
``t_{n+1}`` and conditions it on what is known there:

1. **predict** with the prior's transition, after **calibrating** the diffusion
   from the ODE residual at the predicted mean (see
   ``ode_filters.calibration.policies``), and keep the backward conditional
   ``p(x_n | x_{n+1})`` for smoothing;
2. **update** on the ODE information (and conservation laws), linearized
   online at the prediction or given as a frozen affine model;
3. **observe** the data, if any, with a per-row mask.

The fixed-grid, adaptive and iterated solvers differ only in how they choose
the step sizes and the information, not in the step itself.

**Preconditioning** is a change of variables inside the step. A preconditioned
prior's transition (``Transition.T_diag``) acts on ``x_bar = x / T_diag``; the
step moves into those coordinates to predict and back for the update, so
everything it returns is in the original coordinates.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple, Protocol

import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..calibration.policies import Calibration
from ..gaussian import (
    AffineModel,
    Conditional,
    Gaussian,
    invert,
    log_likelihood,
    marginalize,
    update,
)
from ..gaussian.algebra import _safe_qr
from ..information.protocol import MeasurementModel
from ..linearization.correction import Correction
from ..priors.base import Transition


class StepOutput(NamedTuple):
    """Everything one filter step produces (original coordinates).

    Attributes:
        predicted: The prediction ``p(x_{n+1} | data up to t_n)``.
        backward: The backward conditional ``p(x_n | x_{n+1})``.
        innovation: Predictive distribution of the ODE-information residual.
        filtered: The updated state at ``t_{n+1}``.
        sigma_sqr: The diffusion estimate of the step.
        log_likelihood: The ODE-information evidence of the step.
        innovation_obs: Predictive distribution of the data residual (``None``
            without data). Rows without data at this step have zero mean and
            unit variance, so they add nothing to the whitened residual.
        log_likelihood_obs: The data evidence of the step (``None`` without
            data; zero at a step without data).
    """

    predicted: Gaussian
    backward: Conditional
    innovation: Gaussian
    filtered: Gaussian
    sigma_sqr: Array
    log_likelihood: Array
    innovation_obs: Gaussian | None = None
    log_likelihood_obs: Array | None = None


class Information(Protocol):
    """What a step conditions on, and how it is linearized."""

    @property
    def ode_dim(self) -> int:
        """Number of leading rows that are ODE residual (they drive calibration)."""
        ...

    def linearize(self, m: Array, *, t: ArrayLike) -> AffineModel:
        """The affine model at the predicted mean (drives the calibration)."""
        ...

    def condition(
        self, x: Gaussian, model: AffineModel, *, t: ArrayLike
    ) -> tuple[Gaussian, Gaussian]:
        """``(posterior, innovation)`` of the update of ``x``."""
        ...

    def lhs_selector(self) -> Array:
        """The selector of the ODE's left-hand side (EK0 Jacobian of its rows)."""
        ...


class OnlineInformation(NamedTuple):
    """A measurement model linearized at each step's prediction by a correction."""

    measure: MeasurementModel
    correction: Correction

    @property
    def ode_dim(self) -> int:
        return self.measure.ode_dim

    def linearize(self, m: Array, *, t: ArrayLike) -> AffineModel:
        return self.correction.linearize(self.measure, m, t=t)

    def condition(
        self, x: Gaussian, model: AffineModel, *, t: ArrayLike
    ) -> tuple[Gaussian, Gaussian]:
        return self.correction.correct(self.measure, x, model, t=t)

    def lhs_selector(self) -> Array:
        return self.measure.E_constraint  # type: ignore[attr-defined]


class FrozenInformation(NamedTuple):
    """An affine model fixed in advance (one step of an ``AffineSurrogate``).

    All of its rows drive the diffusion estimate.
    """

    model: AffineModel

    @property
    def ode_dim(self) -> int:
        return self.model.H.shape[0]

    def linearize(self, m: Array, *, t: ArrayLike) -> AffineModel:
        return self.model

    def condition(
        self, x: Gaussian, model: AffineModel, *, t: ArrayLike
    ) -> tuple[Gaussian, Gaussian]:
        return update(model, x)

    def lhs_selector(self) -> Array:
        raise NotImplementedError(
            "A frozen affine model has no ODE left-hand-side selector; use a "
            "calibration that does not need one."
        )


def scale_gaussian(x: Gaussian, s: Array) -> Gaussian:
    """The Gaussian of ``s * x`` (elementwise ``s``): a diagonal change of variables."""
    return Gaussian(x.m * s, x.P_sqr * s[None, :])


def scale_conditional(c: Conditional, s: Array) -> Conditional:
    """``p(x | y)`` from ``p(x_bar | y_bar)`` with ``x = s * x_bar``, ``y = s * y_bar``."""
    return Conditional(s[:, None] * c.G / s[None, :], c.d * s, c.P_sqr * s[None, :])


def predict(x: Gaussian, transition: Transition) -> tuple[Gaussian, Conditional]:
    """Prediction and backward conditional of one transition.

    For a preconditioned transition both are computed in its coordinates and
    returned in the original ones.
    """
    s = transition.T_diag
    x_frame = x if s is None else scale_gaussian(x, 1.0 / s)
    predicted = marginalize(transition.A, transition.b, transition.Q_sqr, x_frame)
    backward = invert(transition.A, transition.Q_sqr, x_frame, predicted)
    if s is None:
        return predicted, backward
    return scale_gaussian(predicted, s), scale_conditional(backward, s)


def observe(
    x: Gaussian, model: AffineModel, mask: Array
) -> tuple[Gaussian, Gaussian, Array]:
    """Condition on the data rows of ``model`` that ``mask`` marks as present.

    A row without data is replaced by an exact observation of zero with unit
    noise and a zero row in ``H``: it leaves the state unchanged, its innovation
    is ``N(0, 1)`` and uncorrelated with the other rows, and its log-density
    ``-log(2 pi) / 2`` is removed from the evidence.

    Args:
        x: The state before the data.
        model: The observation model of the step (``c = -y``).
        mask: Boolean, one entry per row of ``model``.

    Returns:
        ``(posterior, innovation, log_likelihood)``.
    """
    present = mask.astype(x.m.dtype)
    R_sqr = _safe_qr(
        np.concatenate([model.R_sqr * present[None, :], np.diag(1.0 - present)], axis=0)
    )
    masked = AffineModel(model.H * present[:, None], model.c * present, R_sqr)
    posterior, innovation = update(masked, x)
    n_absent = present.shape[0] - np.sum(present)
    ll = log_likelihood(innovation) + 0.5 * n_absent * np.log(2.0 * np.pi)
    return posterior, innovation, ll


def filter_step(
    x: Gaussian,
    transition: Transition,
    information: Information,
    *,
    t: ArrayLike,
    calibration: Calibration,
    scale_noise: Callable[[Array, Array], Array],
    observation: AffineModel | None = None,
    mask: Array | None = None,
) -> StepOutput:
    """One step: calibrate, predict, update on the information, observe.

    Args:
        x: The state at the previous node.
        transition: The prior's transition over the step.
        information: The ODE information of the step.
        t: The time of the step's update (the next node).
        calibration: Diffusion calibration policy.
        scale_noise: Applies a diffusion to the prior's noise
            (``prior.apply_state_sigma_sqr``).
        observation: The step's data model, or ``None`` for none.
        mask: Which rows of ``observation`` carry data.

    Returns:
        The step's ``StepOutput``.
    """
    s = transition.T_diag
    m_prev = x.m if s is None else x.m / s
    m_pred_frame = transition.A @ m_prev + transition.b
    m_pred = m_pred_frame if s is None else m_pred_frame * s

    # The diffusion is estimated from the ODE residual at the predicted mean,
    # before the predicted covariance exists; the ODE rows alone drive it.
    model = information.linearize(m_pred, t=t)
    d = information.ode_dim
    residual = (model.H @ m_pred + model.c)[:d]

    def in_frame(H: Array) -> Array:
        return H if s is None else H * s[None, :]

    sigma_sqr, Q_sqr = calibration.calibrate(
        residual,
        in_frame(model.H[:d]),
        transition.Q_sqr,
        scale=scale_noise,
        H_ek0=lambda: in_frame(information.lhs_selector()),
    )
    predicted, backward = predict(x, transition._replace(Q_sqr=Q_sqr))
    filtered, innovation = information.condition(predicted, model, t=t)
    output = StepOutput(
        predicted=predicted,
        backward=backward,
        innovation=innovation,
        filtered=filtered,
        sigma_sqr=sigma_sqr,
        log_likelihood=log_likelihood(innovation),
    )
    if observation is None:
        return output
    assert mask is not None
    filtered, innovation_obs, ll_obs = observe(filtered, observation, mask)
    return output._replace(
        filtered=filtered, innovation_obs=innovation_obs, log_likelihood_obs=ll_obs
    )
