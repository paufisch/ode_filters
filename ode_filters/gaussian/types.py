"""Named Gaussian objects shared by every layer of the library.

All covariances are stored in square-root form: ``P = P_sqr.T @ P_sqr`` with
``P_sqr`` upper triangular (see ``ode_filters.gaussian.algebra``).
"""

from __future__ import annotations

from typing import NamedTuple

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
