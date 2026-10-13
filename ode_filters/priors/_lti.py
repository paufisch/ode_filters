"""Discretisation helpers for linear time-invariant priors (square-root MFD)."""

from __future__ import annotations

import functools

import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.scipy.linalg import expm
from jax.typing import ArrayLike

# Number of Gauss-Legendre nodes for the Matern square-root matrix-fraction
# decomposition. 64 reconstructs Q(h) to ~1e-13 for every order up to the float64
# order ceiling (q ~ 18, where the dense Q is hopelessly ill-conditioned); more
# nodes only accumulate extra rounding without raising the ceiling.
QUAD_NODES = 64


@functools.lru_cache
def _leggauss(n_quad: int) -> tuple[onp.ndarray, onp.ndarray]:
    return onp.polynomial.legendre.leggauss(n_quad)


def _gauss_legendre(n_quad: int, q: int) -> tuple[Array, Array]:
    """Gauss-Legendre nodes and weights on ``[-1, 1]`` for the square-root MFD.

    Raises:
        ValueError: If ``n_quad <= q``. The QR in
            ``_sqr_noise_via_quadrature`` stacks one block of rows per node
            (one row per dispersion column), so fewer than ``q + 1`` nodes yields
            a factor with fewer rows than the ``q + 1`` derivative blocks of the
            state instead of a square one -- which is not a mildly less accurate
            answer but a structurally wrong shape, and it would otherwise surface
            only much later as a confusing shape error from inside a solve.
    """
    if n_quad <= q:
        raise ValueError(
            f"n_quad must exceed q to give a square noise factor, got "
            f"n_quad={n_quad!r} with q={q!r}. The Gauss-Legendre rule contributes "
            "one block of rows per node to the QR, so n_quad >= q + 1 is a shape "
            "requirement, not an accuracy preference."
        )
    gl_nodes, gl_weights = _leggauss(n_quad)
    return np.asarray(gl_nodes), np.asarray(gl_weights)


def _sqr_noise_via_quadrature(
    F: Array, G: Array, gl_nodes: Array, gl_weights: Array, h: ArrayLike
) -> Array:
    """Upper-triangular square root ``R`` of the LTI process noise
    ``Q(h) = int_0^h e^{F u} G G.T e^{F.T u} du`` (so ``R.T @ R == Q(h)``).

    Square-root matrix-fraction decomposition (Bosch, Hennig & Tronarp 2023): the
    factor is the ``R`` of a QR of the Gauss-Legendre-propagated dispersion columns
    ``sqrt(w_k) e^{F u_k} G``. ``G`` may be rank one (Matern / scalar IOUP) or rank
    ``d`` (matrix-rate IOUP). The dense, ill-conditioned ``Q(h)`` is never formed or
    factorized. jit/vmap/grad-safe (reduced-mode QR).
    """
    half_h = 0.5 * h
    u = half_h * (gl_nodes + 1.0)
    w = half_h * gl_weights
    propagated = jax.vmap(lambda u_k: expm(F * u_k) @ G)(u)  # (n_quad, dim, ncol)
    n_nodes, dim, ncol = propagated.shape
    cols = (propagated * np.sqrt(w)[:, None, None]).transpose(0, 2, 1)
    cols = cols.reshape(n_nodes * ncol, dim)
    return np.linalg.qr(cols, mode="reduced")[1]
