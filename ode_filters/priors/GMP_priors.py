from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from math import comb, factorial
from operator import index

import jax
import jax.experimental.jet
import jax.numpy as np
from numpy.typing import ArrayLike

Array = np.ndarray
MatrixFunction = Callable[[float], Array]
VectorField = Callable[[np.ndarray], np.ndarray]


class BasePrior(ABC):
    def __init__(self, q: int, d: int, Xi: np.ndarray | None = None):
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

        self.q = q
        self._dim = d
        self.xi = xi
        self._id = np.eye(d, dtype=xi.dtype)
        self._b = np.zeros(d * (q + 1))

    @staticmethod
    def _validate_h(h: float) -> float:
        if h < 0:
            raise ValueError("h must be non-negative.")
        return float(h)

    @abstractmethod
    def A(self, h: float) -> np.ndarray:
        pass

    @abstractmethod
    def b(self, h: float) -> np.ndarray:
        pass

    @abstractmethod
    def Q(self, h: float) -> np.ndarray:
        pass


def taylor_mode_initialization(
    vf: VectorField, inits: ArrayLike, q: int, t0: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened Taylor-mode coefficients produced via JAX Jet.

    Args:
        vf: Vector field whose Taylor coefficients are required.
        inits: Initial value around which the expansion takes place.
        q: Number of higher-order coefficients to compute.
        t0: Initial time for Taylor expansion (default 0.0).

    Returns:
        Tuple of (coefficients, covariance) where:
        - coefficients are the flattened Taylor coefficients as numpy array
        - covariance is a zero covariance matrix as numpy array
    """
    if not callable(vf):
        raise TypeError("vf must be callable.")
    q = index(q)
    if q < 0:
        raise ValueError("q must be a non-negative integer.")

    def _vf(x):
        return vf(x, t=t0)

    base_state = np.asarray(inits)
    coefficients: list[np.ndarray] = [base_state]
    series_terms: list[np.ndarray] = []

    for order in range(q):
        primals_out, series_out = jax.experimental.jet.jet(
            _vf,
            primals=(base_state,),
            series=(tuple(series_terms),),
        )

        updated_series = [np.asarray(primals_out)]
        updated_series.extend(np.asarray(term) for term in series_out)
        series_terms = updated_series

        coefficients.append(series_terms[-1])

    leaves = jax.tree_util.tree_leaves(coefficients)
    init = np.concatenate([np.ravel(arr) for arr in leaves])
    D = init.shape[0]
    return init, np.zeros((D, D))


def _make_iwp_state_matrices(q: int) -> tuple[MatrixFunction, MatrixFunction]:
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

    def A(h: float) -> Array:
        if h < 0:
            raise ValueError("h must be non-negative.")
        mat = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(i, dim):
                mat = mat.at[i, j].set(h ** (j - i) / factorial(j - i))
        return mat

    def Q(h: float) -> Array:
        if h < 0:
            raise ValueError("h must be non-negative.")
        mat = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            for j in range(dim):
                power = 2 * q + 1 - i - j
                denom = (2 * q + 1 - i - j) * factorial(q - i) * factorial(q - j)
                mat = mat.at[i, j].set((h**power) / denom)
        return mat

    return A, Q


def _make_iwp_precond_state_matrices(
    q: int,
) -> tuple[Array, Array, MatrixFunction]:
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
                A_bar = A_bar.at[i, j].set(comb(n, k))

    Q_bar = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            Q_bar = Q_bar.at[i, j].set(1.0 / (2 * q + 1 - i - j))

    factorials = np.array([float(factorial(q - idx)) for idx in range(dim)])

    def T(h: float) -> Array:
        if h < 0:
            raise ValueError("h must be non-negative.")
        h = float(h)
        sqrt_h = np.sqrt(h)
        powers = q - np.arange(dim)
        diag_entries = sqrt_h * (h**powers) / factorials
        return np.diag(diag_entries)

    return A_bar, Q_bar, T


class IWP(BasePrior):
    """Integrated Wiener Process prior model."""

    def __init__(self, q: int, d: int, Xi: np.ndarray | None = None):
        super().__init__(q, d, Xi)
        self._A, self._Q = _make_iwp_state_matrices(q)

    def A(self, h: float) -> np.ndarray:
        """Return the state transition matrix for step size h.

        Args:
            h: Step size.

        Returns:
            State transition matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._A(self._validate_h(h)), self._id)

    def b(self, h: float) -> np.ndarray:
        """Return the drift vector for step size h.

        Args:
            h: Step size.

        Returns:
            Zero drift vector (shape [(q+1)*d]).
        """
        return self._b

    def Q(self, h: float) -> np.ndarray:
        """Return the diffusion matrix for step size h.

        Args:
            h: Step size.

        Returns:
            Diffusion matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._Q(self._validate_h(h)), self.xi)


class PrecondIWP(BasePrior):
    """Preconditioned Integrated Wiener Process prior.

    Uses a preconditioning transformation T(h) to make matrices stepsize-independent.
    The transformation matrices A() and Q() are constant (independent of h), while
    the stepsize dependence is absorbed into T(h).
    """

    def __init__(self, q: int, d: int, Xi: np.ndarray | None = None):
        super().__init__(q, d, Xi)
        self._A_bar, self._Q_bar, self._T = _make_iwp_precond_state_matrices(q)

    def A(self) -> np.ndarray:
        """Return the constant preconditioning transition matrix.

        Returns:
            Constant transition matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._A_bar, self._id)

    def b(self) -> np.ndarray:
        """Return the zero drift vector.

        Returns:
            Zero drift vector (shape [(q+1)*d]).
        """
        return self._b

    def Q(self) -> np.ndarray:
        """Return the constant preconditioning diffusion matrix.

        Returns:
            Constant diffusion matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._Q_bar, self.xi)

    def T(self, h: float) -> np.ndarray:
        """Return the stepsize-dependent preconditioning transformation.

        Args:
            h: Step size.

        Returns:
            Preconditioning transformation matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._T(self._validate_h(h)), self._id)
