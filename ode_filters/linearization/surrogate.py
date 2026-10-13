"""Frozen linearizations: an information operator linearized along a trajectory.

Linearized along a fixed trajectory, an ODE filter is a linear-Gaussian
state-space model. ``AffineSurrogate`` is that model's observation part, one
affine model per filter step; ``affine_filter`` filters it, and
``iterated_smoother`` refits it at the smoothed trajectory.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
from jax import Array
from jax.typing import ArrayLike

from ..gaussian import AffineModel
from ..information.protocol import MeasurementModel
from .correction import taylor_linearize
from .statistical import QuadratureRule, slr_linearize


class AffineSurrogate(NamedTuple):
    """One affine information model per filter step, stacked.

    On a grid ``ts`` with nodes ``0..N``, step ``i`` (``0..N-1``) moves from node
    ``i`` to node ``i + 1``, and its residual is
    ``z_i = H[i] @ x_{i+1} + c[i] + v_i`` with ``v_i ~ N(0, R_sqr[i].T @ R_sqr[i])``;
    the filter observes ``z_i = 0``. A linearization trajectory is accordingly
    the states at nodes ``1..N``.

    Attributes:
        H: Observation matrices, shape ``[N, m, D]``.
        c: Offsets, shape ``[N, m]``.
        R_sqr: Upper-triangular square roots of the effective noise (including
            any linearization residual), shape ``[N, m, m]``.
    """

    H: Array
    c: Array
    R_sqr: Array

    def model(self, i: ArrayLike) -> AffineModel:
        """The affine model of step ``i``."""
        return AffineModel(self.H[i], self.c[i], self.R_sqr[i])


def taylor_surrogate(
    measure: MeasurementModel, m_seq: Array, *, ts: ArrayLike
) -> AffineSurrogate:
    """First-order Taylor (EK1) linearization of ``measure`` along a trajectory.

    Linearizing at a filter result's ``m_pred`` reproduces that EK1 filter.

    Args:
        measure: ODE-information model.
        m_seq: Linearization points, the states at nodes ``1..N``, shape
            ``[N, D]``.
        ts: The ``N`` update times, i.e. the grid's ``ts[1:]``.

    Returns:
        The ``AffineSurrogate`` with the measurement noise as ``R_sqr``.
    """
    models = jax.vmap(lambda m, t: taylor_linearize(measure, m, t=t))(m_seq, ts)
    return AffineSurrogate(models.H, models.c, models.R_sqr)


def slr_surrogate(
    measure: MeasurementModel,
    m_seq: Array,
    P_seq_sqr: Array,
    *,
    ts: ArrayLike,
    rule: QuadratureRule = "gauss_hermite",
    n_nodes: int = 4,
) -> AffineSurrogate:
    """Statistical linearization of ``measure`` over a trajectory of Gaussians.

    The linearization residual ``Omega`` of each step is part of its effective
    noise ``R_sqr`` (see ``slr_linearize``).

    Args:
        measure: ODE-information model.
        m_seq: Means at nodes ``1..N``, shape ``[N, D]``.
        P_seq_sqr: Square-root covariances at nodes ``1..N``, shape
            ``[N, D, D]``.
        ts: The ``N`` update times, i.e. the grid's ``ts[1:]``.
        rule: Quadrature rule (see ``QuadratureCorrection``).
        n_nodes: Nodes per dimension for the Gauss-Hermite rule.

    Returns:
        The ``AffineSurrogate``.
    """
    models = jax.vmap(
        lambda m, P, t: slr_linearize(measure, m, P, t=t, rule=rule, n_nodes=n_nodes)
    )(m_seq, P_seq_sqr, ts)
    return AffineSurrogate(models.H, models.c, models.R_eff_sqr)
