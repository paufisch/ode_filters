from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from math import comb, factorial
from operator import index

import jax
import jax.experimental.jet
import jax.numpy as np
from jax import Array
from jax.scipy.linalg import expm
from jax.typing import ArrayLike

MatrixFunction = Callable[[float], Array]
VectorField = Callable[[ArrayLike], Array]


class BasePrior(ABC):
    def __init__(self, q: int, d: int, Xi: ArrayLike | None = None):
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
        eye_d = np.eye(d)
        basis = np.eye(q + 1)
        self._E0 = np.kron(basis[0:1], eye_d)
        self._E1 = np.kron(basis[1:2], eye_d)
        self._E2 = np.kron(basis[2:3], eye_d) if q >= 2 else None

    @property
    def E0(self) -> Array:
        """State extraction matrix (shape [d, (q+1)*d])."""
        return self._E0

    @property
    def E1(self) -> Array:
        """First derivative extraction matrix (shape [d, (q+1)*d])."""
        return self._E1

    @property
    def E2(self) -> Array | None:
        """Second derivative extraction matrix (shape [d, (q+1)*d]), or None if q < 2."""
        return self._E2

    @staticmethod
    def _validate_h(h: float) -> float:
        if h < 0:
            raise ValueError("h must be non-negative.")
        return float(h)

    @abstractmethod
    def A(self, h: float) -> Array:
        pass  # pragma: no cover

    @abstractmethod
    def b(self, h: float) -> Array:
        pass  # pragma: no cover

    @abstractmethod
    def Q(self, h: float) -> Array:
        pass  # pragma: no cover


def taylor_mode_initialization(
    vf: VectorField,
    inits: ArrayLike | tuple[ArrayLike, ...],
    q: int,
    t0: float = 0.0,
    order: int = 1,
) -> tuple[Array, Array]:
    """Return flattened Taylor-mode coefficients produced via JAX Jet.

    Args:
        vf: Vector field whose Taylor coefficients are required.
            For order=1: vf(x, *, t) -> dx/dt
            For order=2: vf(x, dx, *, t) -> d²x/dt²
        inits: Initial value(s) around which the expansion takes place.
            For order=1: x0 (initial state)
            For order=2: (x0, dx0) tuple of initial state and velocity
        q: Number of higher-order coefficients to compute.
        t0: Initial time for Taylor expansion (default 0.0).
        order: ODE order (1 or 2, default 1).

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
    order = index(order)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2.")

    # Normalize inits to a tuple of initial derivatives: (x0,) or (x0, dx0)
    if order == 1:
        init_derivs = (np.asarray(inits),)
    else:
        if not isinstance(inits, (tuple, list)) or len(inits) != order:
            raise ValueError(
                f"For order={order}, inits must be a tuple of {order} arrays."
            )
        init_derivs = tuple(np.asarray(x) for x in inits)
        if not all(x.shape == init_derivs[0].shape for x in init_derivs):
            raise ValueError("All initial derivatives must have the same shape.")

    d = init_derivs[0].shape[0]

    # Augmented state: [x0, dx0, ...] and augmented vector field
    # For order k: y = [x, v1, ..., v_{k-1}], dy/dt = [v1, ..., v_{k-1}, vf(...)]
    base_state = np.concatenate(init_derivs)

    def aug_vf(y):
        derivs = [y[i * d : (i + 1) * d] for i in range(order)]
        highest_deriv = vf(*derivs, t=t0)
        return np.concatenate([*derivs[1:], highest_deriv])

    coefficients: list[Array] = [init_derivs[0]]
    series_terms: list[Array] = []

    for _ in range(q):
        primals_out, series_out = jax.experimental.jet.jet(
            aug_vf,
            primals=(base_state,),
            series=(tuple(series_terms),),
        )

        updated_series = [np.asarray(primals_out)]
        updated_series.extend(np.asarray(term) for term in series_out)
        series_terms = updated_series

        # Extract x-part (first d components) from the augmented derivative
        coefficients.append(series_terms[-1][:d])

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

    def __init__(self, q: int, d: int, Xi: ArrayLike | None = None):
        super().__init__(q, d, Xi)
        self._A, self._Q = _make_iwp_state_matrices(q)

    def A(self, h: float) -> Array:
        """Return the state transition matrix for step size h.

        Args:
            h: Step size.

        Returns:
            State transition matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._A(self._validate_h(h)), self._id)

    def b(self, h: float) -> Array:
        """Return the drift vector for step size h.

        Args:
            h: Step size.

        Returns:
            Zero drift vector (shape [(q+1)*d]).
        """
        return self._b

    def Q(self, h: float) -> Array:
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

    def __init__(self, q: int, d: int, Xi: ArrayLike | None = None):
        super().__init__(q, d, Xi)
        self._A_bar, self._Q_bar, self._T = _make_iwp_precond_state_matrices(q)

    def A(self) -> Array:
        """Return the constant preconditioning transition matrix.

        Returns:
            Constant transition matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._A_bar, self._id)

    def b(self) -> Array:
        """Return the zero drift vector.

        Returns:
            Zero drift vector (shape [(q+1)*d]).
        """
        return self._b

    def Q(self) -> Array:
        """Return the constant preconditioning diffusion matrix.

        Returns:
            Constant diffusion matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._Q_bar, self.xi)

    def T(self, h: float) -> Array:
        """Return the stepsize-dependent preconditioning transformation.

        Args:
            h: Step size.

        Returns:
            Preconditioning transformation matrix (shape [(q+1)*d, (q+1)*d]).
        """
        return np.kron(self._T(self._validate_h(h)), self._id)


def _matern_companion_form(length_scale: float, q: int) -> tuple[Array, Array, float]:
    """Construct the companion form matrices for a Matern GP prior.

    Reference: Sarkka & Hartikainen (2010, equations 12.34-12.35)

    Parameters
    ----------
    length_scale : float
        Length scale parameter.
    q : int
        Smoothness parameter exponent (nu = q + 1/2, q in Z).

    Returns
    -------
    F : Array
        Drift matrix (shape [D, D]) - companion form matrix.
    L : Array
        Diffusion vector (shape [D]).
    q_coeff : float
        Diffusion coefficient.
    """
    if length_scale <= 0:
        raise ValueError("Length scale must be positive.")
    if not isinstance(q, int) or q < 0:
        raise ValueError("Smoothness exponent q must be a non-negative integer.")

    # Compute nu = q + 1/2 and state dimensionality D = nu + 1/2 = q + 1
    D = q + 1

    # Compute lambda = sqrt(2*nu / length_scale) = sqrt((2q + 1) / length_scale)
    lam = np.sqrt((2.0 * q + 1.0) / length_scale)

    # Construct F matrix (companion form)
    F = np.zeros((D, D), dtype=float)

    # Super-diagonal: ones
    for i in range(D - 1):
        F = F.at[i, i + 1].set(1.0)

    # Last row: [-a_j * λ^(D-j) for j = 0, ..., D-1]
    # where a_j = C(D, j) are binomial coefficients
    for j in range(D):
        a_j = comb(D, j)  # Binomial coefficient C(D, j)
        power = D - j  # Power goes from D down to 1
        F = F.at[D - 1, j].set(-a_j * (lam**power))

    # Construct L vector: [0, 0, ..., 0, 1]ᵀ
    L = np.zeros((D, 1), dtype=float).at[-1, 0].set(1.0)

    # Diffusion coefficient: q = sigma^2 [(D-1)!]^2 / (2D-2)! * (2*lambda)^(2D-1)
    # assuming sigma = 1
    numerator = float(factorial(D - 1) ** 2)
    denominator = float(factorial(2 * D - 2))
    q = (numerator / denominator) * ((2.0 * lam) ** (2 * D - 1))

    return F, L, float(q)


class MaternPrior(BasePrior):
    """Matern Gaussian process prior model using block matrix exponential."""

    def __init__(
        self,
        q: int,
        d: int,
        length_scale: float,
        Xi: ArrayLike | None = None,
    ):
        """Initialize the Matern prior.

        This creates a Matern process prior for a d-dimensional process where each
        dimension is modeled independently by a q+1 times integrated Matern process
        of the same length scale with possibly different output scale (Xi).

        Args:
            q: Smoothness order.
            d: State dimension.
            length_scale: Length scale of the process.
            Xi: Optional scaling matrix (shape [d, d]).
        """
        # TODO: check if the above assumptions permit Xi to be a non-diagonal matrix.
        super().__init__(q, d, Xi)
        self._F, self._L, self._q = _matern_companion_form(length_scale, q)
        # self._Q_param = np.asarray(self._q, dtype=float)
        self.S = self._q * self._L @ self._L.T  # Precompute S = L @ Q @ L.T
        self.n = self._F.shape[0]

        # Defensive checks - _matern_companion_form always returns valid shapes
        if self._F.shape != (self.n, self.n):  # pragma: no cover
            raise ValueError(f"F must be square, got shape {self._F.shape}")
        if self._L.shape[0] != self.n:  # pragma: no cover
            raise ValueError(
                f"L first dimension must match F: {self._L.shape[0]} != {self.n}"
            )
        if self.S.shape != (self.n, self.n):  # pragma: no cover
            raise ValueError(
                f"L @ Q @ L.T must be square with shape ({self.n}, {self.n}), "
                f"got {self.S.shape}"
            )

    def _expm_block_matrix(self, h: float) -> Array:
        """Compute exp(H*h) for Hamiltonian block matrix.

        Args:
            h: Step size.

        Returns:
            Matrix exponential of the block Hamiltonian (shape [2n, 2n]).
        """
        H = np.block(
            [
                [self._F, self.S],
                [np.zeros_like(self._F), -self._F.T],
            ]
        )
        return expm(H * h)

    def A_and_Q(self, h: float) -> tuple[Array, Array]:
        """Compute both A(h) and Q(h) efficiently in a single expm call.
        This is sometimes called matrix fraction decomposition (MFD)

        Args:
            h: Step size.

        Returns:
            Tuple of (A_h, Q_h) where:
            - A_h: State transition matrix (shape [n, n])
            - Q_h: Diffusion matrix (shape [n, n])
        """
        h = self._validate_h(h)
        expm_H = self._expm_block_matrix(h)

        A_h = expm_H[: self.n, : self.n]
        Q_h = expm_H[: self.n, self.n :] @ A_h.T

        return A_h, Q_h

    def A(self, h: float) -> Array:
        """Return the state transition matrix for step size h.

        Args:
            h: Step size.

        Returns:
            State transition matrix (shape [n, n]).
        """
        A_h, _ = self.A_and_Q(self._validate_h(h))
        return np.kron(A_h, self._id)

    def b(self, h: float) -> Array:
        """Return the drift vector for step size h.

        Args:
            h: Step size.

        Returns:
            Zero drift vector (shape [n]).
        """
        return np.zeros(self.n)

    def Q(self, h: float) -> Array:
        """Return the diffusion matrix for step size h.

        Args:
            h: Step size.

        Returns:
            Diffusion matrix (shape [n, n]).
        """
        _, Q_h = self.A_and_Q(self._validate_h(h))
        Q_h = 0.5 * (Q_h + Q_h.T)
        return np.kron(Q_h, self.xi)


class JointPrior(BasePrior):
    """Joint prior combining independent state (x) and hidden/input (u) priors.

    Creates a block-diagonal prior structure where state and hidden evolution
    are independent. The resulting matrices are block-diagonal with zeros
    in the off-diagonal blocks.

    For joint state-parameter estimation, the hidden state u can represent
    unknown parameters that appear in the ODE but evolve according to their
    own prior (e.g., IWP or Matern).

    Projection matrices:
        E0: Extracts [x, u] - zeroth derivatives of both (shape [d_x + d_u, D])
        E0_x: Extracts x only - zeroth derivative of state (shape [d_x, D])
        E0_hidden: Extracts u only - zeroth derivative of hidden (shape [d_u, D])
        E1: Extracts dx/dt - first derivative of state x (shape [d_x, D])
        E2: Extracts d^2x/dt^2 - second derivative of x (shape [d_x, D]), if q >= 2

    Args:
        prior_x: BasePrior instance for state evolution.
        prior_u: BasePrior instance for hidden/input evolution.
    """

    def __init__(self, prior_x: BasePrior, prior_u: BasePrior) -> None:
        if not isinstance(prior_x, BasePrior):
            raise TypeError(f"prior_x must be BasePrior instance, got {type(prior_x)}")
        if not isinstance(prior_u, BasePrior):
            raise TypeError(f"prior_u must be BasePrior instance, got {type(prior_u)}")

        self._prior_x = prior_x
        self._prior_u = prior_u

        # Precompute dimensions and zero blocks for efficiency
        _D_x = (prior_x.q + 1) * prior_x._dim
        _D_u = (prior_u.q + 1) * prior_u._dim
        self._zeros = np.zeros((_D_x, _D_u))
        zeros_up = np.zeros((prior_x._dim, _D_u))
        zeros_down = np.zeros((prior_u._dim, _D_x))

        # E0 extracts [x, u] - both zeroth derivatives (for measurements on both)
        self._E0 = np.block([[prior_x.E0, zeros_up], [zeros_down, prior_u.E0]])

        # E0_x extracts x only (for ODE vector field)
        self._E0_x = np.block([[prior_x.E0, zeros_up]])

        # E0_hidden extracts u only (for hidden states in vector field)
        self._E0_hidden = np.block([[zeros_down, prior_u.E0]])

        # E1 extracts dx/dt (first derivative of x, for ODE constraint)
        self._E1 = np.block([[prior_x.E1, zeros_up]])

        # E2 for second-order systems (extracts d^2x/dt^2 from state x)
        self._E2 = (
            np.block([[prior_x.E2, zeros_up]]) if prior_x.E2 is not None else None
        )

    @property
    def E0_x(self) -> Array:
        """State extraction matrix for x only (shape [d_x, D]).

        Use this as E0 in the measurement model when you have hidden states.
        """
        return self._E0_x

    @property
    def E0_hidden(self) -> Array:
        """Hidden state extraction matrix for u only (shape [d_u, D]).

        Use this as E0_hidden in the measurement model.
        """
        return self._E0_hidden

    def A(self, h: float) -> Array:
        """Return the block-diagonal state transition matrix.

        Args:
            h: Step size.

        Returns:
            Block-diagonal transition matrix with state and input blocks.
        """
        return np.block(
            [[self._prior_x.A(h), self._zeros], [self._zeros.T, self._prior_u.A(h)]]
        )

    def b(self, h: float) -> Array:
        """Return the concatenated drift vector.

        Args:
            h: Step size.

        Returns:
            Drift vector concatenating state and input drifts.
        """
        return np.concatenate([self._prior_x.b(h), self._prior_u.b(h)])

    def Q(self, h: float) -> Array:
        """Return the block-diagonal diffusion matrix.

        Args:
            h: Step size.

        Returns:
            Block-diagonal diffusion matrix with state and input blocks.
        """
        return np.block(
            [[self._prior_x.Q(h), self._zeros], [self._zeros.T, self._prior_u.Q(h)]]
        )
