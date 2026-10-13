"""Joint priors of a state and a hidden input: ``JointPrior``, ``PrecondJointPrior``."""

from __future__ import annotations

import jax.numpy as np
from jax import Array
from jax.scipy.linalg import block_diag
from jax.typing import ArrayLike

from ..gaussian.algebra import _safe_sqrt
from .base import BasePrior
from .iwp import PrecondIWP
from .matern import PrecondMaternPrior


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
        # A preconditioned sub-prior's A/Q_sqr live in preconditioned (bar)
        # coordinates; JointPrior carries no T(h), so the filter would treat them
        # as plain-space matrices and silently return garbage.
        for name, sub in (("prior_x", prior_x), ("prior_u", prior_u)):
            if isinstance(sub, (PrecondIWP, PrecondMaternPrior)):
                raise TypeError(
                    f"{name} is preconditioned ({type(sub).__name__}); JointPrior "
                    "needs plain priors. Use PrecondJointPrior to combine "
                    "preconditioned priors."
                )

        self._prior_x = prior_x
        self._prior_u = prior_u

        # State-prior order is the natural choice for adaptive step-size
        # control: convergence order in h depends on the regularity of the
        # ODE-state prior, not on the input prior.
        self.q = prior_x.q
        self._dim = prior_x._dim + prior_u._dim

        # Precompute dimensions and zero blocks for efficiency
        _D_x = (prior_x.q + 1) * prior_x._dim
        _D_u = (prior_u.q + 1) * prior_u._dim
        self._D_x = _D_x
        self._D_u = _D_u
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

    def A(self, h: ArrayLike) -> Array:
        """Return the block-diagonal state transition matrix.

        Args:
            h: Step size.

        Returns:
            Block-diagonal transition matrix with state and input blocks.
        """
        return np.block(
            [[self._prior_x.A(h), self._zeros], [self._zeros.T, self._prior_u.A(h)]]
        )

    def b(self, h: ArrayLike) -> Array:
        """Return the concatenated drift vector.

        Args:
            h: Step size.

        Returns:
            Drift vector concatenating state and input drifts.
        """
        return np.concatenate([self._prior_x.b(h), self._prior_u.b(h)])

    def Q(self, h: ArrayLike) -> Array:
        """Return the block-diagonal diffusion matrix.

        Args:
            h: Step size.

        Returns:
            Block-diagonal diffusion matrix with state and input blocks.
        """
        return np.block(
            [[self._prior_x.Q(h), self._zeros], [self._zeros.T, self._prior_u.Q(h)]]
        )

    def Q_sqr(self, h: ArrayLike) -> Array:
        """Block-diagonal upper-triangular square root of ``Q(h)``.

        ``Q`` is block-diagonal, so its square root is the block-diagonal of the
        sub-priors' square roots -- this delegates to each sub-prior's
        ``Q_sqr`` (closed-form for ``IWP``) rather than factorizing the
        dense joint ``Q(h)``.
        """
        return np.block(
            [
                [self._prior_x.Q_sqr(h), self._zeros],
                [self._zeros.T, self._prior_u.Q_sqr(h)],
            ]
        )

    @property
    def E0_state(self) -> Array:
        """State-only value projection (Bosch et al. 2022 sec. 3)."""
        return self._E0_x

    @property
    def xi(self) -> Array:
        """Component correlation of the stacked solution ``[x, u]``.

        Block-diagonal ``blkdiag(Xi_x, Xi_u)``, shape ``[d_x + d_u, d_x + d_u]``:
        the state and hidden processes are independent.
        """
        return block_diag(self._prior_x.xi, self._prior_u.xi)

    @property
    def xi_state(self) -> Array:
        """State-prior component-correlation matrix (shape ``[d_x, d_x]``).

        Diagonal-mode calibration only requires the state block to be
        per-component diagonal; the hidden block's xi can be arbitrary.
        """
        return self._prior_x.xi

    def apply_state_sigma_sqr(self, Q_sqr: Array, sigma_sqr: ArrayLike) -> Array:
        """Scale only the state block of ``Q_sqr`` by ``sqrt(sigma_sqr)``.

        Accepts a scalar (uniform state-block scaling) or a length-``d_x``
        vector (per-component state-block scaling used by
        ``calibration="diagonal"``). In both cases the input block is
        untouched. ``Q`` is block-diagonal, so its upper-triangular sqrt is
        also block-diagonal: column-scaling the state block of the sqrt by
        ``sqrt(sigma_sqr_tiled)`` yields ``blkdiag(diag(sigma) Q_x diag(sigma),
        Q_u)`` (where ``diag(sigma)`` collapses to a scalar in the scalar
        case).
        """
        sigma_sqr = np.asarray(sigma_sqr)
        if sigma_sqr.ndim == 0:
            return Q_sqr.at[: self._D_x, : self._D_x].multiply(_safe_sqrt(sigma_sqr))
        if sigma_sqr.shape != (self._prior_x._dim,):
            raise ValueError(
                f"sigma_sqr must be scalar or length-{self._prior_x._dim} vector; "
                f"got shape {sigma_sqr.shape}."
            )
        tiled = np.tile(sigma_sqr, self._prior_x.q + 1)
        return Q_sqr.at[:, : self._D_x].multiply(_safe_sqrt(tiled)[None, :])

    def apply_state_sigma_to_cov_sqr(self, P_sqr: Array, sigma_sqr: ArrayLike) -> Array:
        """Scale only the state block of a full-state covariance sqrt.

        ``P = P_sqr.T @ P_sqr``. Multiplying columns ``[:, :D_x]`` of the
        upper-triangular sqrt by ``sqrt(sigma_sqr)`` scales rows and
        columns ``:D_x`` of ``P`` by ``sigma_sqr`` and leaves the input
        block alone.
        """
        D_x = (self._prior_x.q + 1) * self._prior_x._dim
        return P_sqr.at[:, :D_x].multiply(_safe_sqrt(sigma_sqr))


class PrecondJointPrior:
    """Joint prior for preconditioned priors combining state (x) and hidden (u).

    This is the preconditioned version of JointPrior, designed specifically for
    PrecondIWP priors. The matrices A(), b(), Q() are constant (stepsize-independent),
    with the stepsize dependence absorbed into T(h).

    Note: Mixing preconditioned and non-preconditioned priors is not supported
    and would lead to inconsistent filter behavior.

    Projection matrices:
        E0: Extracts [x, u] - zeroth derivatives of both (shape [d_x + d_u, D])
        E0_x: Extracts x only - zeroth derivative of state (shape [d_x, D])
        E0_hidden: Extracts u only - zeroth derivative of hidden (shape [d_u, D])
        E1: Extracts dx/dt - first derivative of state x (shape [d_x, D])
        E2: Extracts d^2x/dt^2 - second derivative of x (shape [d_x, D]), if q >= 2

    Args:
        prior_x: PrecondIWP instance for state evolution.
        prior_u: PrecondIWP instance for hidden/input evolution.

    Example:
        >>> prior_x = PrecondIWP(q=2, d=2)
        >>> prior_u = PrecondIWP(q=1, d=1)
        >>> joint = PrecondJointPrior(prior_x, prior_u)
        >>> A = joint.A()      # Constant transition matrix
        >>> T_h = joint.T(0.1) # Stepsize-dependent transformation
    """

    def __init__(
        self,
        prior_x: PrecondIWP | PrecondMaternPrior,
        prior_u: PrecondIWP | PrecondMaternPrior,
    ) -> None:
        _allowed = (PrecondIWP, PrecondMaternPrior)
        if not isinstance(prior_x, _allowed):
            raise TypeError(
                f"prior_x must be PrecondIWP or PrecondMaternPrior, got {type(prior_x)}. "
                "Use JointPrior for non-preconditioned priors."
            )
        if not isinstance(prior_u, _allowed):
            raise TypeError(
                f"prior_u must be PrecondIWP or PrecondMaternPrior, got {type(prior_u)}. "
                "Use JointPrior for non-preconditioned priors."
            )

        self._prior_x = prior_x
        self._prior_u = prior_u

        # Store dimensions
        self.q = prior_x.q  # Use state prior's q for compatibility
        self._dim = prior_x._dim + prior_u._dim

        # Precompute dimensions and zero blocks for efficiency
        _D_x = (prior_x.q + 1) * prior_x._dim
        _D_u = (prior_u.q + 1) * prior_u._dim
        self._D_x = _D_x
        self._D_u = _D_u
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

        # Precompute drift vector (always zero for IWP)
        self._b = np.zeros(_D_x + _D_u)

    @property
    def E0(self) -> Array:
        """Combined state extraction matrix (shape [d_x + d_u, D])."""
        return self._E0

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

    @property
    def E1(self) -> Array:
        """First derivative extraction matrix for x (shape [d_x, D])."""
        return self._E1

    @property
    def E2(self) -> Array | None:
        """Second derivative extraction matrix for x (shape [d_x, D]), or None."""
        return self._E2

    def A(self, h: ArrayLike | None = None) -> Array:
        """Return the block-diagonal transition matrix.

        Args:
            h: Step size (passed through to sub-priors).

        Returns:
            Block-diagonal transition matrix with state and hidden blocks.
        """
        return np.block(
            [[self._prior_x.A(h), self._zeros], [self._zeros.T, self._prior_u.A(h)]]
        )

    def b(self, h: ArrayLike | None = None) -> Array:
        """Return the zero drift vector.

        Args:
            h: Step size (passed through to sub-priors).

        Returns:
            Zero drift vector (shape [D_x + D_u]).
        """
        return self._b

    def Q(self, h: ArrayLike | None = None) -> Array:
        """Return the block-diagonal diffusion matrix.

        Args:
            h: Step size (passed through to sub-priors).

        Returns:
            Block-diagonal diffusion matrix with state and hidden blocks.
        """
        return np.block(
            [[self._prior_x.Q(h), self._zeros], [self._zeros.T, self._prior_u.Q(h)]]
        )

    def Q_sqr(self, h: ArrayLike | None = None) -> Array:
        """Block-diagonal upper-triangular square root of ``Q``.

        Delegates to each sub-prior's ``Q_sqr`` (constant for ``PrecondIWP``)
        rather than factorizing the dense joint ``Q``.
        """
        return np.block(
            [
                [self._prior_x.Q_sqr(h), self._zeros],
                [self._zeros.T, self._prior_u.Q_sqr(h)],
            ]
        )

    def T(self, h: ArrayLike) -> Array:
        """Return the stepsize-dependent block-diagonal transformation matrix.

        Args:
            h: Step size.

        Returns:
            Block-diagonal preconditioning transformation matrix.
        """
        T_x = self._prior_x.T(h)
        T_u = self._prior_u.T(h)
        zeros_xu = np.zeros((self._D_x, self._D_u))
        return np.block([[T_x, zeros_xu], [zeros_xu.T, T_u]])

    @property
    def E0_state(self) -> Array:
        """State-only value projection (Bosch et al. 2022 sec. 3)."""
        return self._E0_x

    @property
    def xi(self) -> Array:
        """Component correlation of the stacked solution ``[x, u]``.

        Block-diagonal ``blkdiag(Xi_x, Xi_u)``, shape ``[d_x + d_u, d_x + d_u]``:
        the state and hidden processes are independent.
        """
        return block_diag(self._prior_x.xi, self._prior_u.xi)

    @property
    def xi_state(self) -> Array:
        """State-prior component-correlation matrix (shape ``[d_x, d_x]``)."""
        return self._prior_x.xi

    def apply_state_sigma_sqr(self, Q_sqr: Array, sigma_sqr: ArrayLike) -> Array:
        """Scale only the state block of ``Q_sqr`` by ``sqrt(sigma_sqr)``.

        Accepts a scalar or a length-``d_x`` vector. See
        ``JointPrior.apply_state_sigma_sqr`` for the rationale.
        """
        sigma_sqr = np.asarray(sigma_sqr)
        if sigma_sqr.ndim == 0:
            return Q_sqr.at[: self._D_x, : self._D_x].multiply(_safe_sqrt(sigma_sqr))
        if sigma_sqr.shape != (self._prior_x._dim,):
            raise ValueError(
                f"sigma_sqr must be scalar or length-{self._prior_x._dim} vector; "
                f"got shape {sigma_sqr.shape}."
            )
        tiled = np.tile(sigma_sqr, self._prior_x.q + 1)
        return Q_sqr.at[:, : self._D_x].multiply(_safe_sqrt(tiled)[None, :])

    def apply_state_sigma_to_cov_sqr(self, P_sqr: Array, sigma_sqr: ArrayLike) -> Array:
        """Scale only the state block of a full-state covariance sqrt.

        See ``JointPrior.apply_state_sigma_to_cov_sqr`` for the rationale.
        """
        return P_sqr.at[:, : self._D_x].multiply(_safe_sqrt(sigma_sqr))
