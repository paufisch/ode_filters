"""Joint prior of a state and a hidden input: ``JointPrior``."""

from __future__ import annotations

import jax.numpy as np
from jax import Array
from jax.scipy.linalg import block_diag
from jax.typing import ArrayLike

from .base import BasePrior, Transition


class JointPrior(BasePrior):
    """Joint prior of a state ``x`` and a hidden input ``u``, independent a priori.

    The state is ``[x-block, u-block]``, each derivative-major, and every matrix
    is block-diagonal. The hidden input can be a latent force or an unknown
    parameter that enters the ODE (see ``ODEInformationWithHidden``) and evolves
    by its own prior. The diffusion calibration scales the state block only
    (Schmidt et al. 2021).

    Either block may be preconditioned (``PrecondIWP``, ``PrecondMaternPrior``):
    the joint transition then carries that block's preconditioner and the
    identity for a plain one.

    Selectors:
        E0: ``[x, u]``, shape ``[d_x + d_u, D]``.
        E0_x: ``x`` alone, shape ``[d_x, D]`` (the ODE's ``E0``).
        E0_hidden: ``u`` alone, shape ``[d_u, D]``.
        E1, E2: ``x'`` and ``x''`` (``E2`` is ``None`` if ``q_x < 2``).

    Args:
        prior_x: Prior of the state.
        prior_u: Prior of the hidden input.
    """

    prior_x: BasePrior
    prior_u: BasePrior

    def __init__(self, prior_x: BasePrior, prior_u: BasePrior) -> None:
        for name, sub in (("prior_x", prior_x), ("prior_u", prior_u)):
            if not isinstance(sub, BasePrior):
                raise TypeError(f"{name} must be a BasePrior, got {type(sub)}.")
        self.prior_x = prior_x
        self.prior_u = prior_u

    @property
    def q(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
        """The state prior's order: it sets the convergence order of the solver."""
        return self.prior_x.q

    @property
    def d(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Dimension of ``[x, u]``."""
        return self.prior_x.d + self.prior_u.d

    @property
    def xi(self) -> Array:  # pyright: ignore[reportIncompatibleVariableOverride]
        """``blkdiag(Xi_x, Xi_u)``: the two processes are independent."""
        return block_diag(self.prior_x.xi, self.prior_u.xi)

    @property
    def state_dim(self) -> int:
        """Dimension of the joint state, ``D_x + D_u``."""
        return self.prior_x.state_dim + self.prior_u.state_dim

    def _pad(self, x_rows: Array | None, u_rows: Array | None) -> Array:
        """Selector rows for the joint state from rows for one block."""
        D_x, D_u = self.prior_x.state_dim, self.prior_u.state_dim
        blocks = []
        if x_rows is not None:
            blocks.append(np.concatenate([x_rows, np.zeros((x_rows.shape[0], D_u))], 1))
        if u_rows is not None:
            blocks.append(np.concatenate([np.zeros((u_rows.shape[0], D_x)), u_rows], 1))
        return np.concatenate(blocks, axis=0)

    @property
    def E0(self) -> Array:
        """Selector of ``[x, u]``, shape ``[d_x + d_u, D]``."""
        return self._pad(self.prior_x.E0, self.prior_u.E0)

    @property
    def E0_x(self) -> Array:
        """Selector of ``x`` alone: the ``E0`` of the ODE information."""
        return self._pad(self.prior_x.E0, None)

    @property
    def E0_hidden(self) -> Array:
        """Selector of the hidden input ``u``."""
        return self._pad(None, self.prior_u.E0)

    @property
    def E1(self) -> Array:
        """Selector of ``x'``, shape ``[d_x, D]``."""
        return self._pad(self.prior_x.E1, None)

    @property
    def E2(self) -> Array | None:
        """Selector of ``x''``, shape ``[d_x, D]``, or ``None`` if ``q_x < 2``."""
        E2 = self.prior_x.E2
        return None if E2 is None else self._pad(E2, None)

    def A(self, h: ArrayLike) -> Array:
        """Block-diagonal of the blocks' transition matrices."""
        return block_diag(self.prior_x.A(h), self.prior_u.A(h))

    def b(self, h: ArrayLike) -> Array:
        """The blocks' drifts, stacked."""
        return np.concatenate([self.prior_x.b(h), self.prior_u.b(h)])

    def Q_sqr(self, h: ArrayLike) -> Array:
        """Block-diagonal of the blocks' square roots (``Q`` is block-diagonal)."""
        return block_diag(self.prior_x.Q_sqr(h), self.prior_u.Q_sqr(h))

    def transition(self, h: ArrayLike) -> Transition:
        """Block-diagonal transition of the two sub-priors.

        If either sub-prior is preconditioned, the joint transition is in the
        coordinates of both blocks' preconditioners (the identity for a plain
        block).
        """
        tr_x = self.prior_x.transition(h)
        tr_u = self.prior_u.transition(h)
        T_diag = None
        if tr_x.T_diag is not None or tr_u.T_diag is not None:
            T_x = (
                np.ones(self.prior_x.state_dim) if tr_x.T_diag is None else tr_x.T_diag
            )
            T_u = (
                np.ones(self.prior_u.state_dim) if tr_u.T_diag is None else tr_u.T_diag
            )
            T_diag = np.concatenate([T_x, T_u])
        return Transition(
            block_diag(tr_x.A, tr_u.A),
            np.concatenate([tr_x.b, tr_u.b]),
            block_diag(tr_x.Q_sqr, tr_u.Q_sqr),
            T_diag,
        )

    @property
    def E0_state(self) -> Array:
        """Selector of ``x``: the adaptive error estimate ignores the input."""
        return self.E0_x

    @property
    def xi_state(self) -> Array:
        """The state prior's ``Xi``; the input's may be arbitrary."""
        return self.prior_x.xi

    @property
    def xi_state_diagonal(self) -> bool | None:
        """Whether the state prior's ``Xi`` was diagonal at construction."""
        return self.prior_x.xi_state_diagonal

    @property
    def _state_block(self) -> tuple[int, int, int]:
        return self.prior_x.state_dim, self.prior_x.d, self.prior_x.q
