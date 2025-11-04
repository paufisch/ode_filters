from math import factorial
from operator import index
from typing import Optional

import numpy as np


def _make_iwp_state_matrices(q: int):
    """Return callables producing the transition A(h) and diffusion Q(h) matrices.

    Parameters
    ----------
    q : int
        Smoothness order of the integrated Wiener process.

    Returns
    -------
    tuple[callable, callable]
        Functions A(h) and Q(h) that accept a positive scalar h and return
        square numpy arrays of shape (q + 1, q + 1).
    """
    q = index(q)
    if q < 0:
        raise ValueError("q must be a non-negative integer.")

    dim = q + 1

    def A(h: float) -> np.ndarray:
        if h < 0:
            raise ValueError("h must be non-negative.")
        mat = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(i, dim):
                mat[i, j] = h ** (j - i) / factorial(j - i)
        return mat

    def Q(h: float) -> np.ndarray:
        if h < 0:
            raise ValueError("h must be non-negative.")
        mat = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(dim):
                power = 2 * q + 1 - i - j
                denom = (2 * q + 1 - i - j) * factorial(q - i) * factorial(q - j)
                mat[i, j] = (h**power) / denom
        return mat

    return A, Q


class IWP:
    """q-times integrated Wiener process prior for d-dimensional systems."""

    def __init__(self, q: int, d: int, Xi: Optional[np.ndarray] = None):
        if not isinstance(q, int):
            raise TypeError("q must be an integer.")
        if q < 0:
            raise ValueError("q must be non-negative.")

        if not isinstance(d, int):
            raise TypeError("d must be an integer.")
        if d <= 0:
            raise ValueError("d must be positive.")

        xi = np.eye(d, dtype=float) if Xi is None else np.asarray(Xi, dtype=float)
        if xi.shape != (d, d):
            raise ValueError(f"Xi must have shape ({d}, {d}), got {xi.shape}.")

        self._A, self._Q = _make_iwp_state_matrices(q)
        self.q = q
        self._dim = d
        self.xi = xi
        self._id = np.eye(d, dtype=xi.dtype)

    def A(self, h: float) -> np.ndarray:
        """State transition matrix for step size h."""
        return np.kron(self._A(self._validate_h(h)), self._id)

    def Q(self, h: float) -> np.ndarray:
        """Process noise (diffusion) matrix for step size h."""
        return np.kron(self._Q(self._validate_h(h)), self.xi)

    @staticmethod
    def _validate_h(h: float) -> float:
        if h < 0:
            raise ValueError("h must be non-negative.")
        return float(h)
