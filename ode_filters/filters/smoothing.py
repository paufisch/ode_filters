"""Smoothing: ``rts_smoother``."""

from __future__ import annotations

import jax
import jax.numpy as np

from ..gaussian import Conditional, Gaussian, marginalize
from .result import FilterResult


def rts_smoother(result: FilterResult) -> Gaussian:
    """Rauch-Tung-Striebel smoothing of a filter result.

    Runs the stored backward conditionals ``p(x_n | x_{n+1})`` from the last
    filtered marginal back to the first. Works on any result that carries them:
    ``gaussian_filter``, ``affine_filter``, and ``gaussian_filter_adaptive`` with
    ``smoother=True`` (one composite conditional per save interval).

    Args:
        result: A ``FilterResult`` with backward conditionals.

    Returns:
        The smoothed marginals as a stacked ``Gaussian``: means ``[K, D]`` and
        square-root covariances ``[K, D, D]`` (unpacks as ``m, P_sqr``).

    Raises:
        ValueError: If the result carries no backward conditionals.
    """
    backward = result.backward
    last = Gaussian(result.m[-1], result.P_sqr[-1])

    def step(x_next: Gaussian, conditional: Conditional):
        x = marginalize(conditional.G, conditional.d, conditional.P_sqr, x_next)
        return x, x

    _, earlier = jax.lax.scan(step, last, backward, reverse=True)
    return Gaussian(
        np.concatenate([earlier.m, last.m[None]], axis=0),
        np.concatenate([earlier.P_sqr, last.P_sqr[None]], axis=0),
    )
