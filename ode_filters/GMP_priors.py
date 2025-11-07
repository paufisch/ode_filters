import array
from math import comb, factorial
from operator import index
from typing import Callable, Optional

import jax
import jax.experimental.jet
import jax.numpy as jnp
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


def taylor_mode_initialization(vf: Callable, inits: array, q: int) -> jnp.ndarray:
    """Return flattened Taylor‑mode coefficients produced via JAX Jet.

    Parameters
    ----------
    vf : callable
        Vector field whose Taylor coefficients are required.
    inits : array-like
        Initial value around which the expansion takes place.
    q : int
        Number of higher-order coefficients to compute.

    Returns
    -------
    jnp.ndarray
        Concatenated Taylor coefficients (including the initial value).
    """
    if not callable(vf):
        raise TypeError("vf must be callable.")
    q = index(q)
    if q < 0:
        raise ValueError("q must be a non-negative integer.")

    base_state = jnp.asarray(inits)
    coefficients: list[jnp.ndarray] = [base_state]
    series_terms: list[jnp.ndarray] = []

    for order in range(q):
        primals_out, series_out = jax.experimental.jet.jet(
            vf,
            primals=(base_state,),
            series=(tuple(series_terms),),
        )

        updated_series = [jnp.asarray(primals_out)]
        updated_series.extend(jnp.asarray(term) for term in series_out)
        series_terms = updated_series

        coefficients.append(series_terms[-1])

    leaves = jax.tree_util.tree_leaves(coefficients)
    init = jnp.concatenate([jnp.ravel(arr) for arr in leaves])
    D = init.shape[0]
    return init, np.zeros((D, D))


def _make_iwp_precond_state_matrices(q: int):
    """Return callables producing the transition, diffusion, and scaling matrices."""
    q = index(q)
    if q < 0:
        raise ValueError("q must be a non-negative integer.")

    dim = q + 1

    A_bar = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            n = q - i
            k = q - j
            if 0 <= k <= n:
                A_bar[i, j] = comb(n, k)

    Q_bar = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            Q_bar[i, j] = 1.0 / (2 * q + 1 - i - j)

    factorials = np.array([float(factorial(q - idx)) for idx in range(dim)])

    def T(h: float) -> np.ndarray:
        if h < 0:
            raise ValueError("h must be non-negative.")
        h = float(h)
        sqrt_h = np.sqrt(h)
        powers = q - np.arange(dim)
        diag_entries = sqrt_h * (h**powers) / factorials
        return np.diag(diag_entries)

    return A_bar, Q_bar, T


class IWP_precond:
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

        self._A_bar, self._Q_bar, self._T = _make_iwp_precond_state_matrices(q)
        self.q = q
        self._dim = d
        self.xi = xi
        self._id = np.eye(d, dtype=xi.dtype)

    def A(self) -> np.ndarray:
        """State transition matrix for step size h."""
        return np.kron(self._A_bar, self._id)

    def Q(self) -> np.ndarray:
        """Process noise (diffusion) matrix for step size h."""
        return np.kron(self._Q_bar, self.xi)

    def T(self, h: float) -> np.ndarray:
        """Scaling matrix for step size h."""
        return np.kron(self._T(self._validate_h(h)), self._id)

    @staticmethod
    def _validate_h(h: float) -> float:
        if h < 0:
            raise ValueError("h must be non-negative.")
        return float(h)
