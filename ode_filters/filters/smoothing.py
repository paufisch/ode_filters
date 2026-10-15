"""Smoothing, posterior sampling and interpolation of filter results.

A filter pass stores, per step, the backward conditional ``p(x_n | x_{n+1})``
of the prediction. Run backwards from the last filtered marginal, these
conditionals give the smoothing posterior of the whole trajectory:

- ``rts_smoother`` marginalizes through them: the Rauch-Tung-Striebel
  smoothed marginals at the grid nodes;
- ``sample_posterior`` samples through them: joint sample paths;
- ``interpolate`` bridges between two nodes with the prior, for filtered or
  smoothed marginals at any time inside the grid.
"""

from __future__ import annotations

import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian import Conditional, Gaussian, invert, marginalize
from ..priors.base import BasePrior
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


def sample_posterior(key: Array, result: FilterResult) -> Array:
    """A sample path from the smoothing posterior of a filter result.

    Draws the last state from its filtered marginal, then each earlier state
    from the backward conditional ``p(x_n | x_{n+1})`` -- an exact joint sample,
    not a sample from the marginals. The key is split into one key per node.
    Vmap over keys for many paths.

    With exact ODE information (zero noise ``R``) the posterior is singular and
    the path for a given key depends on which square-root factor the QR returns:
    a batched and a looped run can differ, and the path is not smooth in the
    model's parameters. Each path is still an exact draw. A small ODE noise,
    e.g. ``measure.with_noise(1e-10)``, makes the paths reproducible.

    Args:
        key: A ``jax.random`` key.
        result: A ``FilterResult`` with backward conditionals.

    Returns:
        The path at ``result.t``, shape ``[K, D]``.

    Raises:
        ValueError: If the result carries no backward conditionals.
    """
    backward = result.backward
    keys = jax.random.split(key, result.m.shape[0])
    D = result.m.shape[1]
    last = result.m[-1] + jax.random.normal(keys[-1], (D,)) @ result.P_sqr[-1]

    def step(x_next, inputs):
        conditional, k = inputs
        x = (
            conditional.G @ x_next
            + conditional.d
            + jax.random.normal(k, (D,)) @ conditional.P_sqr
        )
        return x, x

    _, earlier = jax.lax.scan(step, last, (backward, keys[:-1]), reverse=True)
    return np.concatenate([earlier, last[None]], axis=0)


def interpolate(
    result: FilterResult,
    prior: BasePrior,
    ts: ArrayLike,
    *,
    smoothed: bool = True,
) -> Gaussian:
    """Marginals of a fixed-grid result at times between its grid nodes.

    For ``t`` in ``[t_n, t_{n+1}]`` the prior bridges the two nodes: predict from
    the filtered state at ``t_n`` to ``t``, and, for the smoothed marginal,
    condition on the smoothed state at ``t_{n+1}`` through the backward
    conditional of the remaining step. Each sub-step uses the diffusion its step
    propagated with (``result.diffusion``), so the result agrees with the
    filter and the smoother at the grid nodes.

    Args:
        result: A ``gaussian_filter`` / ``affine_filter`` result.
        prior: The prior the result was computed with.
        ts: Query times in ``[t_0, t_K]``, shape ``[M]``.
        smoothed: Return smoothed marginals (default) or filtered ones (the
            prediction from the last node before ``t``).

    Returns:
        The marginals at ``ts`` as a stacked ``Gaussian``.

    Raises:
        ValueError: If the result has no per-step diffusion (an adaptive result).
    """
    if result.diffusion is None:
        raise ValueError(
            "interpolate needs a fixed-grid result (gaussian_filter or "
            "affine_filter); an adaptive result has no per-step diffusion."
        )
    diffusion = result.diffusion
    grid = result.t
    ts = np.asarray(ts, dtype=float)
    smooth = rts_smoother(result) if smoothed else None
    last_step = grid.shape[0] - 2

    def transition(h: Array, n: Array) -> tuple[Array, Array, Array]:
        Q_sqr = prior.apply_state_sigma_sqr(prior.Q_sqr(h), diffusion[n])
        return prior.A(h), prior.b(h), Q_sqr

    def at(t: Array) -> Gaussian:
        n = np.clip(np.searchsorted(grid, t, side="right") - 1, 0, last_step)
        step = grid[n + 1] - grid[n]
        tau = t - grid[n]
        # Interior times only: at a node the bridge degenerates (a zero step),
        # so the node's own marginal is returned instead.
        inside = (tau > 0.0) & (tau < step)
        tau_safe = np.where(inside, tau, 0.5 * step)
        A1, b1, Q1 = transition(tau_safe, n)
        filtered = Gaussian(result.m[n], result.P_sqr[n])
        predicted = marginalize(A1, b1, Q1, filtered)
        if smooth is None:
            value = predicted
            node = Gaussian(
                np.where(tau >= step, result.m[n + 1], result.m[n]),
                np.where(tau >= step, result.P_sqr[n + 1], result.P_sqr[n]),
            )
        else:
            A2, b2, Q2 = transition(step - tau_safe, n)
            ahead = marginalize(A2, b2, Q2, predicted)
            back = invert(A2, Q2, predicted, ahead)
            value = marginalize(
                back.G,
                back.d,
                back.P_sqr,
                Gaussian(smooth.m[n + 1], smooth.P_sqr[n + 1]),
            )
            node = Gaussian(
                np.where(tau >= step, smooth.m[n + 1], smooth.m[n]),
                np.where(tau >= step, smooth.P_sqr[n + 1], smooth.P_sqr[n]),
            )
        return jax.tree.map(lambda a, b: np.where(inside, a, b), value, node)

    return jax.vmap(at)(ts)
