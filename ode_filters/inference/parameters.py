"""Unconstrained-parameter wrappers for gradient-based inference.

Many ODE parameters are constrained: a diffusion scale, an observation noise
or a decay rate must be positive. Optimizers and samplers, however, operate on
unconstrained values. These wrappers store a parameter in unconstrained space
and expose the constrained value via ``unwrap``, following the
``equinox``/``paramax`` (and GPJax) pattern.

Wrap a parameter with its *constrained* value::

    theta = {"rate": PositiveReal(0.5)}        # 0.5 > 0

and recover the constrained value (e.g. inside a likelihood) with::

    unwrap(theta)                              # {"rate": 0.5}

``unwrap`` is the identity on plain arrays, so wrapping is opt-in and
non-breaking: ``marginal_loglik`` unwraps ``theta`` for you, so
``jax.grad`` / optimizers see the unconstrained leaves while the model sees the
constrained values.
"""

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as np
from jax import Array


def _softplus_inv(y: Array) -> Array:
    """Inverse of ``jax.nn.softplus`` (stable for large inputs)."""
    # softplus(x) ~ x for large x, where log(expm1(y)) would overflow.
    return np.where(y > 20.0, y, np.log(np.expm1(y)))


class AbstractParameter(eqx.Module):
    """A parameter stored in unconstrained space.

    Subclasses hold the raw (unconstrained) value as their array leaf and map it
    to the constrained value in ``unwrap``. Being an ``equinox.Module`` they
    are registered pytrees, so the unconstrained leaf is what ``jax.grad`` and
    Optax see.
    """

    @abc.abstractmethod
    def unwrap(self) -> Array:
        """Return the constrained value."""
        raise NotImplementedError


class Real(AbstractParameter):
    """An unconstrained real parameter (identity transform).

    Attributes:
        value: The (unconstrained) value, used as-is.
    """

    value: Array

    def __init__(self, value: Any):
        self.value = np.asarray(value, dtype=float)

    def unwrap(self) -> Array:
        """Return the value unchanged."""
        return self.value


class PositiveReal(AbstractParameter):
    """A strictly-positive parameter, stored via the softplus bijection.

    Construct it with the *constrained* (positive) value; it stores the
    corresponding unconstrained value and returns the positive value from
    ``unwrap``.

    Attributes:
        unconstrained: The stored unconstrained value (``softplus_inv(value)``).
    """

    unconstrained: Array

    def __init__(self, value: Any):
        self.unconstrained = _softplus_inv(np.asarray(value, dtype=float))

    def unwrap(self) -> Array:
        """Return ``softplus(unconstrained)``."""
        return jax.nn.softplus(self.unconstrained)


def unwrap(tree: Any) -> Any:
    """Replace every ``AbstractParameter`` in a pytree with its value.

    Identity on leaves that are not ``AbstractParameter``, so it is safe to
    call on any ``theta`` (plain arrays pass through unchanged).
    """
    return jax.tree_util.tree_map(
        lambda x: x.unwrap() if isinstance(x, AbstractParameter) else x,
        tree,
        is_leaf=lambda x: isinstance(x, AbstractParameter),
    )


class NonNegativeReal(AbstractParameter):
    """A non-negative parameter (e.g. a noise floor), via softplus.

    Like ``PositiveReal``, but construction at exactly zero is allowed: it starts
    at the smallest value softplus represents usefully (about ``2e-9``), so the
    optimizer can move the parameter away from the boundary or towards it.

    Attributes:
        unconstrained: The stored unconstrained value.
    """

    unconstrained: Array

    def __init__(self, value: Any):
        value = np.asarray(value, dtype=float)
        self.unconstrained = _softplus_inv(np.maximum(value, _NON_NEGATIVE_FLOOR))

    def unwrap(self) -> Array:
        """Return ``softplus(unconstrained)``."""
        return jax.nn.softplus(self.unconstrained)


# softplus(-20) ~ 2e-9: small enough to stand for zero next to a noise floor,
# large enough that the gradient (its sigmoid, ~2e-9) does not underflow.
_NON_NEGATIVE_FLOOR = 2.0e-9
