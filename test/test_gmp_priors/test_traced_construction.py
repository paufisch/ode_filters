"""Every prior must be constructible from *traced* hyperparameters.

Gradient-based hyperparameter inference rebuilds the prior from ``log_params``
inside the objective on every evaluation, so each constructor runs under
``jax.jit`` / ``jax.grad`` with a traced ``Xi`` and, for the Matern family, a
traced ``length_scale``. The suite had no coverage of that path -- every other
test builds its priors eagerly -- and two construction-time defects went through
it while 661 tests passed:

* the IWP ``Q_bar`` finiteness backstop tested a JAX array with a Python
  ``bool()``, raising ``TracerBoolConversionError`` for *any* traced ``Xi``; and
* :func:`_make_matern_sqr_noise` cast ``lambda`` with ``float()``, so a traced
  ``length_scale`` raised ``ConcretizationTypeError``.

Together those made the library unusable for the workflow it exists to support.
``test_grad_safety.py`` guards the same property for the solver *loops*, taking
priors as given; this file guards it for prior *construction*.

The finite-difference checks matter as much as the "does not raise" ones: a cast
that silently freezes a hyperparameter at its traced value's constant, rather
than raising, would produce a zero or wrong gradient and pass a smoke test.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import IWP, JointPrior, MaternPrior, PrecondIWP
from ode_filters.priors.gmp_priors import (
    PrecondMaternPrior,
    _check_iwp_q_bar_factor,
)

H = 0.1


def _central_difference(f, x, eps=1e-6):
    x = onp.asarray(x, dtype=float)
    out = onp.zeros_like(x)
    for i in range(x.size):
        lo, hi = x.copy(), x.copy()
        lo[i] -= eps
        hi[i] += eps
        out[i] = (float(f(np.asarray(hi))) - float(f(np.asarray(lo)))) / (2 * eps)
    return out


# `Q_sqr` is the step-size-dependent factor every filter step consumes, so it is
# the right scalarisation: a prior whose hyperparameters do not reach it is not
# actually being fitted.
def _iwp(p):
    return np.sum(IWP(3, 2, Xi=np.exp(p[0]) * onp.eye(2)).Q_sqr(H) ** 2)


def _precond_iwp(p):
    return np.sum(PrecondIWP(3, 2, Xi=np.exp(p[0]) * onp.eye(2)).Q_sqr(H) ** 2)


def _matern(p):
    pr = MaternPrior(2, 1, length_scale=np.exp(p[1]), Xi=np.exp(p[0]) * onp.eye(1))
    return np.sum(pr.Q_sqr(H) ** 2)


def _precond_matern(p):
    pr = PrecondMaternPrior(2, 1, np.exp(p[1]), Xi=np.exp(p[0]) * onp.eye(1))
    return np.sum(pr.Q_sqr(H) ** 2)


def _joint(p):
    pr = JointPrior(
        IWP(3, 2, Xi=np.exp(p[0]) * onp.eye(2)),
        MaternPrior(1, 1, length_scale=np.exp(p[2]), Xi=np.exp(p[1]) * onp.eye(1)),
    )
    return np.sum(pr.Q_sqr(H) ** 2)


CASES = [
    ("IWP", _iwp, [0.3]),
    ("PrecondIWP", _precond_iwp, [0.3]),
    ("MaternPrior", _matern, [0.3, 0.7]),
    ("PrecondMaternPrior", _precond_matern, [0.3, 0.7]),
    ("JointPrior", _joint, [0.3, -0.2, 0.7]),
]


@pytest.mark.parametrize("name, fn, x0", CASES, ids=[c[0] for c in CASES])
def test_prior_constructs_under_jit(name, fn, x0):
    """Construction from traced hyperparameters must not raise."""
    value = jax.jit(fn)(np.asarray(x0))
    assert onp.isfinite(float(value))


@pytest.mark.parametrize("name, fn, x0", CASES, ids=[c[0] for c in CASES])
def test_prior_grad_matches_finite_difference(name, fn, x0):
    """And every hyperparameter must actually receive a gradient."""
    grad = onp.asarray(jax.jit(jax.grad(fn))(np.asarray(x0)))
    expected = _central_difference(fn, x0)

    assert onp.all(onp.isfinite(grad))
    # A frozen (cast) hyperparameter shows up as an exactly-zero entry.
    assert onp.all(onp.abs(grad) > 0.0), f"{name}: some hyperparameter has no gradient"
    onp.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-8)


def test_finiteness_backstop_fires_eagerly_but_is_skipped_under_trace():
    """The IWP backstop keeps its eager behaviour and becomes a no-op under trace.

    It is a construction-time sanity check against a pathological ``q``; eager
    construction is where such a ``q`` is introduced, so that is where it must
    still fail loudly. Under trace it cannot be evaluated at all, and silently
    skipping is the only option that does not break every traced construction.
    """
    bad = np.asarray([[np.inf, 0.0], [0.0, 1.0]])
    good = np.eye(2)

    with pytest.raises(ValueError, match="non-finite square-root factor"):
        _check_iwp_q_bar_factor(bad, q=1)
    assert _check_iwp_q_bar_factor(good, q=1) is None

    # Under trace both are accepted -- no raise, no leaked tracer.
    @jax.jit
    def _traced(x):
        _check_iwp_q_bar_factor(x, q=1)
        return np.sum(x)

    assert onp.isfinite(float(_traced(good)))
