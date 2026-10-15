"""Named Gaussian objects shared by every layer of the library.

All covariances are stored in square-root form: ``P = P_sqr.T @ P_sqr`` with
``P_sqr`` upper triangular (see ``ode_filters.gaussian.algebra``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as np
from jax import Array


class Gaussian(NamedTuple):
    """A Gaussian ``N(m, P)`` with ``P = P_sqr.T @ P_sqr``.

    Stacked along a leading axis it is a sequence of marginals, e.g. the
    smoothed solution: ``m`` of shape ``[K, n]`` and ``P_sqr`` of shape
    ``[K, n, n]``.

    Attributes:
        m: Mean, shape ``[n]``.
        P_sqr: Square-root covariance, shape ``[n, n]``.
    """

    m: Array
    P_sqr: Array

    @property
    def cov(self) -> Array:
        """Covariance ``P = P_sqr.T @ P_sqr`` (batched over leading axes)."""
        return np.swapaxes(self.P_sqr, -1, -2) @ self.P_sqr

    @property
    def std(self) -> Array:
        """Marginal standard deviations ``sqrt(diag(P))``, shaped like ``m``.

        Computed from the factor without forming ``P``: ``P_ii`` is the squared
        norm of column ``i`` of ``P_sqr``.
        """
        return np.linalg.norm(self.P_sqr, axis=-2)


class Conditional(NamedTuple):
    """An affine Gaussian conditional ``x | y ~ N(G y + d, P)``.

    The filters store one per step, ``p(x_n | x_{n+1})``: the backward
    conditional that smoothing and posterior sampling run on.

    Attributes:
        G: Gain, shape ``[n, k]``.
        d: Offset, shape ``[n]``.
        P_sqr: Square-root covariance, shape ``[n, n]``.
    """

    G: Array
    d: Array
    P_sqr: Array


class AffineModel(NamedTuple):
    """An affine Gaussian observation model ``z | x ~ N(H x + c, R)``.

    The filters condition on ``z = 0``: an ODE-information or data model
    linearized at some state, with ``c`` holding the (negated) data.

    Attributes:
        H: Observation matrix, shape ``[k, n]``.
        c: Offset, shape ``[k]``.
        R_sqr: Square-root noise covariance, shape ``[k, k]``.
    """

    H: Array
    c: Array
    R_sqr: Array
