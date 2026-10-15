"""ODE information operators: the residual of the ODE, plus conservation laws.

An ODE ``x^(nu) = f(a_1, ..., a_k, t)`` is information about the prior's state
``Y`` through the residual

    r(Y, t) = E_lhs @ Y - f(S_1 @ Y, ..., S_k @ Y, t),

which the filters observe as zero (Tronarp et al. 2019; Bosch et al. 2022).
``E_lhs`` selects the derivative on the left-hand side and the ``S_i`` select
the vector field's arguments. The four public classes differ only in those
selectors:

| Class | ODE | ``E_lhs`` | arguments |
| --- | --- | --- | --- |
| ``ODEInformation`` | ``x' = f(x, t)`` | ``E1`` | ``E0`` |
| ``ODEInformationWithHidden`` | ``x' = f(x, u, t)`` | ``E1`` | ``E0``, ``E0_hidden`` |
| ``SecondOrderODEInformation`` | ``x'' = f(x, x', t)`` | ``E2`` | ``E0``, ``E1`` |
| ``SecondOrderODEInformationWithHidden`` | ``x'' = f(x, x', u, t)`` | ``E2`` | ``E0``, ``E1``, ``E0_hidden`` |

``Conservation`` laws ``A x = p`` add affine rows below the ODE residual.
"""

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..gaussian.algebra import _safe_cholesky_sqr
from .noise import noise_matrix
from .observations import Measurement


def _check_selectors(**selectors: ArrayLike | None) -> None:
    """Validate the selection matrices of an ODE-information model.

    All must be 2-D with one column per state entry, and ``E1`` / ``E2`` need one
    row per solution component, like ``E0``; ``E0_hidden`` may have any number
    of rows. Catches e.g. a sliced ``E1`` that would otherwise broadcast
    silently into a wrong residual.
    """
    shapes = {}
    for name, E in selectors.items():
        if E is None:
            hint = " (a prior provides E2 only for q >= 2)" if name == "E2" else ""
            raise ValueError(f"{name} is None{hint}.")
        shape = np.shape(E)
        if len(shape) != 2:
            raise ValueError(
                f"{name} must be a 2-D selection matrix, got shape {shape}."
            )
        shapes[name] = shape
    if len({shape[1] for shape in shapes.values()}) > 1:
        listed = ", ".join(f"{name}: {shape}" for name, shape in shapes.items())
        raise ValueError(
            f"Selection matrices must all have one column per state entry; got {listed}."
        )
    d = shapes["E0"][0]
    for name in ("E1", "E2"):
        if name in shapes and shapes[name][0] != d:
            raise ValueError(
                f"{name} must have as many rows as E0 ({d}), got shape {shapes[name]}."
            )


class Conservation(eqx.Module):
    """A conservation law ``A @ x = p``, always active.

    Attributes:
        A: Constraint matrix, shape ``[k, d]`` (on ``x = E0 @ state``) or
            ``[k, state_dim]`` with ``full_state=True``.
        p: Conserved values, shape ``[k]``.
        full_state: Whether ``A`` acts on the full state instead of ``x``.
    """

    A: Array
    p: Array
    full_state: bool = eqx.field(static=True, default=False)

    def __init__(self, A: ArrayLike, p: ArrayLike, full_state: bool = False):
        self.A = np.asarray(A)
        self.p = np.asarray(p)
        self.full_state = full_state
        if self.A.ndim != 2 or self.p.ndim != 1:
            raise ValueError(
                f"Conservation needs a 2-D 'A' and a 1-D 'p', got shapes "
                f"{self.A.shape} and {self.p.shape}."
            )
        if self.A.shape[0] != self.p.shape[0]:
            raise ValueError(
                f"A.shape[0] ({self.A.shape[0]}) must match p.shape[0] ({self.p.shape[0]})."
            )

    def residual(self, x: Array) -> Array:
        """``A @ x - p``."""
        return self.A @ x - self.p

    @property
    def dim(self) -> int:
        """Number of conserved quantities."""
        return int(self.p.shape[0])


class BaseODEInformation(eqx.Module):
    """The residual ``E_lhs @ Y - f(S_1 @ Y, ..., S_k @ Y, t)`` plus conservation rows.

    Use one of the four public subclasses; they set the selectors. All of them
    implement the ``MeasurementModel`` protocol and are ``equinox`` modules:
    their arrays are pytree leaves, the vector field is static.

    Attributes:
        vf: The vector field ``f``.
        E0: Selector of the solution ``x`` (conservation laws act on ``E0 @ Y``).
        E_lhs: Selector of the ODE's left-hand side (``E1``, or ``E2`` for a
            second-order ODE): the Jacobian of the residual's affine part, which
            an EK0 linearization uses for the ODE rows.
        selectors: Selectors of the vector field's arguments, in order.
        constraints: Conservation laws.
        R: Noise covariance of the ODE rows, shape ``[d, d]`` (zero by default:
            the ODE holds exactly).
    """

    vf: Callable[..., Array] = eqx.field(static=True)
    E0: Array
    E_lhs: Array
    selectors: tuple[Array, ...]
    constraints: tuple[Conservation, ...]
    R: Array

    @property
    def ode_dim(self) -> int:
        """Number of ODE-residual rows at the top of ``g`` (they drive calibration).

        Conservation rows live on other scales, so the diffusion is estimated
        from these rows alone (Bosch, Tronarp & Hennig 2022, *Pick-and-Mix
        Information Operators*, Sec. 3).
        """
        return int(self.E_lhs.shape[0])

    @property
    def state_dim(self) -> int:
        """Dimension of the prior's state."""
        return int(self.E0.shape[1])

    @property
    def E_args(self) -> Array:
        """The vector field's argument selectors, stacked: ``[S_1; ...; S_k]``.

        The residual is affine in the state except through ``E_args @ state``.
        Statistical linearization integrates over this projected subspace only
        (see ``slr_linearize``), which keeps the quadrature dimension equal to
        the ODE dimension rather than the state dimension.
        """
        return np.concatenate(self.selectors, axis=0)

    def _arguments(self, state: Array) -> list[Array]:
        return [S @ state for S in self.selectors]

    def _conservation_rows(self, state: Array) -> tuple[list[Array], list[Array]]:
        """Residuals and (constant) Jacobians of the conservation laws."""
        x = self.E0 @ state
        residuals, jacobians = [], []
        for c in self.constraints:
            residuals.append(c.residual(state if c.full_state else x))
            jacobians.append(c.A if c.full_state else c.A @ self.E0)
        return residuals, jacobians

    def _ode_residual(self, state: Array, *, t: ArrayLike) -> Array:
        return self.E_lhs @ state - self.vf(*self._arguments(state), t=t)

    def _ode_jacobian(self, state: Array, *, t: ArrayLike) -> Array:
        args = self._arguments(state)
        jacobians = jax.jacfwd(
            lambda *a: self.vf(*a, t=t), argnums=tuple(range(len(args)))
        )(*args)
        H = self.E_lhs
        for J, S in zip(jacobians, self.selectors, strict=True):
            H = H - J @ S
        return H

    def g(self, state: Array, *, t: ArrayLike) -> Array:
        """The residual at ``state``: ODE rows, then conservation rows."""
        state = self._validate_state(state)
        conservation, _ = self._conservation_rows(state)
        return np.concatenate([self._ode_residual(state, t=t), *conservation])

    def jacobian_g(self, state: Array, *, t: ArrayLike) -> Array:
        """The Jacobian of the residual at ``state``."""
        state = self._validate_state(state)
        _, conservation = self._conservation_rows(state)
        return np.concatenate([self._ode_jacobian(state, t=t), *conservation])

    def linearize(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
        """First-order (EK1) linearization: ``H x + c`` matches ``g`` to first order.

        Args:
            state: Linearization point.
            t: Time.

        Returns:
            ``(H, c)``.
        """
        state = self._validate_state(state)
        H = self.jacobian_g(state, t=t)
        return H, self.g(state, t=t) - H @ state

    def linearize_ek0(self, state: Array, *, t: ArrayLike) -> tuple[Array, Array]:
        """Zeroth-order (EK0) linearization: the vector field held constant.

        The ODE rows of the Jacobian reduce to ``E_lhs``; the conservation rows
        are affine and exact. Only the residual is evaluated, never the
        vector-field Jacobian.

        Args:
            state: Linearization point.
            t: Time.

        Returns:
            ``(H, c)`` with ``H x + c`` equal to the residual at ``state``.
        """
        state = self._validate_state(state)
        _, conservation = self._conservation_rows(state)
        H = np.concatenate([self.E_lhs, *conservation])
        return H, self.g(state, t=t) - H @ state

    def noise_sqr(self, *, t: ArrayLike) -> Array:
        """Upper-triangular square root of the noise of all rows.

        The ODE rows carry ``R``; conservation rows are exact.
        """
        k = self.ode_dim + sum(c.dim for c in self.constraints)
        R = np.zeros((k, k)).at[: self.ode_dim, : self.ode_dim].set(self.R)
        return _safe_cholesky_sqr(R, k)

    def with_noise(self, R: ArrayLike) -> BaseODEInformation:
        """A copy whose ODE rows carry the noise ``R``.

        Args:
            R: A variance (scalar), per-row variances (length ``ode_dim``) or a
                covariance (``[ode_dim, ode_dim]``).
        """
        return eqx.tree_at(lambda m: m.R, self, noise_matrix(R, self.ode_dim, "R"))

    def _validate_state(self, state: Array) -> Array:
        return check_state(state, self.state_dim)


def check_state(state: ArrayLike, state_dim: int) -> Array:
    """``state`` as an array, checked to be one-dimensional of length ``state_dim``."""
    state = np.asarray(state)
    if state.ndim != 1:
        raise ValueError("'state' must be a one-dimensional array.")
    if state.shape[0] != state_dim:
        raise ValueError(f"'state' must have length {state_dim}, got {state.shape[0]}.")
    return state


def _init_ode_information(
    model: BaseODEInformation,
    vf: Callable[..., Array],
    E0: ArrayLike,
    E_lhs: ArrayLike,
    selectors: tuple[ArrayLike, ...],
    constraints: list[Conservation] | tuple[Conservation, ...] | None,
    R: ArrayLike | None,
) -> None:
    """Set and validate the fields of an ODE-information model."""
    model.vf = vf
    model.E0 = np.asarray(E0)
    model.E_lhs = np.asarray(E_lhs)
    model.selectors = tuple(np.asarray(S) for S in selectors)
    d = model.E_lhs.shape[0]
    model.R = np.zeros((d, d)) if R is None else noise_matrix(R, d, "R")
    model.constraints = tuple(constraints or ())
    for c in model.constraints:
        if isinstance(c, Measurement):
            raise TypeError(
                "Data are not part of the ODE information: build an obs_model with "
                "prepare_observations([Measurement(...)], E0, ts) and pass "
                "obs_model= to the solver. constraints= takes Conservation laws."
            )
        if not isinstance(c, Conservation):
            raise TypeError(
                f"constraints must be Conservation laws, got {type(c).__name__}."
            )
        expected = model.E0.shape[1] if c.full_state else model.E0.shape[0]
        label = "state" if c.full_state else "solution"
        if c.A.shape[1] != expected:
            raise ValueError(
                f"Conservation A has {c.A.shape[1]} columns but the {label} has "
                f"dimension {expected}."
            )


class ODEInformation(BaseODEInformation):
    """First-order ODE ``x' = f(x, t)``.

    Args:
        vf: Vector field ``vf(x, *, t) -> x'``.
        E0: Selector of ``x``, shape ``[d, D]``.
        E1: Selector of ``x'``, shape ``[d, D]``.
        constraints: Conservation laws.
        R: Noise of the ODE rows (default zero).
    """

    def __init__(
        self,
        vf: Callable[..., Array],
        E0: ArrayLike,
        E1: ArrayLike,
        constraints: list[Conservation] | None = None,
        *,
        R: ArrayLike | None = None,
    ):
        _check_selectors(E0=E0, E1=E1)
        _init_ode_information(self, vf, E0, E1, (E0,), constraints, R)


class ODEInformationWithHidden(BaseODEInformation):
    """First-order ODE driven by a hidden input: ``x' = f(x, u, t)``.

    The input ``u`` (a latent force, an unknown parameter) is part of the state,
    with its own prior (see ``JointPrior``).

    Args:
        vf: Vector field ``vf(x, u, *, t) -> x'``.
        E0: Selector of ``x``, shape ``[d_x, D]``.
        E1: Selector of ``x'``, shape ``[d_x, D]``.
        E0_hidden: Selector of ``u``, shape ``[d_u, D]``.
        constraints: Conservation laws (on ``x``).
        R: Noise of the ODE rows (default zero).
    """

    def __init__(
        self,
        vf: Callable[..., Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E0_hidden: ArrayLike,
        constraints: list[Conservation] | None = None,
        *,
        R: ArrayLike | None = None,
    ):
        _check_selectors(E0=E0, E1=E1, E0_hidden=E0_hidden)
        _init_ode_information(self, vf, E0, E1, (E0, E0_hidden), constraints, R)


class SecondOrderODEInformation(BaseODEInformation):
    """Second-order ODE ``x'' = f(x, x', t)``.

    Args:
        vf: Vector field ``vf(x, v, *, t) -> x''``.
        E0: Selector of ``x``, shape ``[d, D]``.
        E1: Selector of ``x'``, shape ``[d, D]``.
        E2: Selector of ``x''``, shape ``[d, D]``.
        constraints: Conservation laws.
        R: Noise of the ODE rows (default zero).
    """

    def __init__(
        self,
        vf: Callable[..., Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E2: ArrayLike,
        constraints: list[Conservation] | None = None,
        *,
        R: ArrayLike | None = None,
    ):
        _check_selectors(E0=E0, E1=E1, E2=E2)
        _init_ode_information(self, vf, E0, E2, (E0, E1), constraints, R)


class SecondOrderODEInformationWithHidden(BaseODEInformation):
    """Second-order ODE driven by a hidden input: ``x'' = f(x, x', u, t)``.

    Args:
        vf: Vector field ``vf(x, v, u, *, t) -> x''``.
        E0: Selector of ``x``, shape ``[d_x, D]``.
        E1: Selector of ``x'``, shape ``[d_x, D]``.
        E2: Selector of ``x''``, shape ``[d_x, D]``.
        E0_hidden: Selector of ``u``, shape ``[d_u, D]``.
        constraints: Conservation laws (on ``x``).
        R: Noise of the ODE rows (default zero).
    """

    def __init__(
        self,
        vf: Callable[..., Array],
        E0: ArrayLike,
        E1: ArrayLike,
        E2: ArrayLike,
        E0_hidden: ArrayLike,
        constraints: list[Conservation] | None = None,
        *,
        R: ArrayLike | None = None,
    ):
        _check_selectors(E0=E0, E1=E1, E2=E2, E0_hidden=E0_hidden)
        _init_ode_information(self, vf, E0, E2, (E0, E1, E0_hidden), constraints, R)
