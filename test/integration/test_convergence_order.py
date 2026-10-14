"""Convergence order of the filter and the smoother, by prior, correction and q.

Modelled on ProbNumDiffEq.jl (``test/convergence.jl``) and Diffrax
(``test_integrate.py::test_ode_order``): solve a problem with an exact solution
on a sequence of uniform grids, fit the slope of ``log(error)`` against
``log(h)``, and require it to lie in a two-sided band around the expected order.

The problem is ``LINEAR_2D`` of ``test.helpers`` (a damped rotation,
``x' = L x`` on ``[0, 3]``), whose solution ``expm(L t) x0`` is exact. Two
errors per grid:

- **filter**: the filtered mean at the final time;
- **smoother**: the largest error of the smoothed mean over the interior nodes
  (the last node is the filtered one, the first the exact initial value).

The calibration is the default ``"dynamic"``. The theory proves global order ``q``
for the filter mean of an IWP(q) prior (Kersting, Sullivan & Hennig 2020) and
for the MAP trajectory (Tronarp, Saerkkae & Hennig 2021). With the exact
Taylor-mode initial state the measured order is ``q + 1`` for every prior and
both corrections, filter and smoother alike, so the bands sit around ``q + 1``.

Measured slopes (this module, float64; the range over the four priors IWP,
PrecondIWP, MaternPrior(length_scale=1) and IOUPPrior(rate=-1)):

====  =============  =============  =============  =============
 q     EK1 filter     EK1 smoother   EK0 filter     EK0 smoother
====  =============  =============  =============  =============
 1     1.99 - 2.00    1.99 - 2.00    1.97 - 2.00    1.98 - 2.01
 2     2.87 - 3.03    2.80 - 2.96    2.90 - 3.01    2.88 - 3.00
 3     3.92 - 4.08    3.76 - 4.00    3.95 - 4.11    3.96 - 4.08
 4     4.72 - 5.06    4.87 - 5.04    4.83 - 4.98    4.79 - 4.97
 5     5.90 - 6.05    5.74 - 5.96    5.90 - 6.11    5.86 - 6.10
 6     6.83 - 7.46    6.49 - 6.94    --             --
====  =============  =============  =============  =============

The band is ``q + 1 +- 0.5``, which rejects the neighbouring orders ``q`` and
``q + 2``; at q = 6, where the window between the pre-asymptotic regime and
the round-off floor is narrow, it is ``[6.25, 7.75]``.

Deviations. IWP and PrecondIWP agree to all printed digits (the same process,
in other coordinates). ``MaternPrior`` reaches the asymptotic regime later:
its mean reversion (rate ``sqrt(2q + 1)``) is a lower-order term that fades
only as ``h -> 0``, so it gives the low slopes at q = 2 and q = 4 and needs
smaller steps at q >= 5 (``MATERN_STEPS``; on the common steps it measured
6.54 / 5.49 at q = 5 and 6.27 / 5.16 at q = 6, filter / smoother). EK0 needs
smaller steps than EK1 from q = 3 on, and is not usable at q = 6 here: with
``"dynamic"`` its error grows to ~7e3 for ``T/128 <= h <= T/12`` (with
``"none"`` it stalls near 1e-5) and is back at the round-off floor by
``h = T/256``, so there is no window to fit; q = 6 is tested with EK1 only.

To compile each (prior, correction, q) once, all grids of a combination have
the same number of nodes: the grid of ``N`` steps over ``[0, T]`` is extended
with further steps of the same size past ``T``. The filter is causal, so its
value at ``t_N = T`` is exactly the ``N``-step solve; for the smoother, the
backward conditionals past ``T`` are replaced by identities and the smoothing
pass starts from the filtered state at ``T``, which is ``rts_smoother`` on the
``N``-step solve.

The full matrix is marked ``slow``; a subset that touches every prior, both
corrections and every q runs by default.
"""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    IOUPPrior,
    MaternPrior,
    ODEInformation,
    PrecondIWP,
    TaylorCorrection,
    gaussian_filter,
    rts_smoother,
    taylor_mode_initialization,
)
from test.helpers import LINEAR_2D

T_END = LINEAR_2D.tspan[1]

PRIORS = {
    "iwp": lambda q: IWP(q, LINEAR_2D.d),
    "precond_iwp": lambda q: PrecondIWP(q, LINEAR_2D.d),
    # length_scale=1: the rate is sqrt(2q + 1) under either length-scale convention.
    "matern": lambda q: MaternPrior(q, LINEAR_2D.d, length_scale=1.0),
    # A rate that is not the ODE's own dynamics (with rate=L the prior mean would
    # solve the linear ODE exactly and leave nothing to converge).
    "ioup": lambda q: IOUPPrior(q, LINEAR_2D.d, rate=-1.0),
}

# Step counts over [0, T] per (correction order, q): inside the asymptotic
# regime and above the round-off floor for every prior. Ratio-2 sequences
# where the window allows; at high q the window between the pre-asymptotic
# regime and the floor is narrow, so the steps are spaced more densely.
STEPS = {
    (1, 1): (32, 64, 128, 256),
    (1, 2): (32, 64, 128, 256),
    (1, 3): (32, 64, 128, 256),
    (1, 4): (32, 64, 128, 256),
    (1, 5): (48, 64, 96, 128),
    (1, 6): (24, 32, 48, 64),
    (0, 1): (32, 64, 128, 256),
    (0, 2): (32, 64, 128, 256),
    (0, 3): (64, 128, 256, 512),
    (0, 4): (48, 96, 192, 384),
    (0, 5): (128, 192, 256, 384),
}

# MaternPrior reaches the asymptotic regime later at high q (see the module
# docstring), so it gets smaller steps there.
MATERN_STEPS = {
    (1, 5): (96, 128, 192, 256),
    (1, 6): (96, 128, 192, 256),
}

# Errors below this are at the round-off floor (the solution is O(1)) and are
# dropped from the fit; a fit needs at least three points.
FLOOR = 1e-13

# Two-sided band around the expected order q + 1, wider at q = 6.
BAND = {q: (q + 1 - 0.5, q + 1 + 0.5) for q in range(1, 6)}
BAND[6] = (6.25, 7.75)

# The default run: every prior, both corrections and every q at least once.
DEFAULT = {
    ("iwp", 1, 1),
    ("iwp", 0, 3),
    ("precond_iwp", 1, 4),
    ("precond_iwp", 0, 2),
    ("matern", 1, 2),
    ("matern", 0, 1),
    ("ioup", 1, 5),
    ("ioup", 0, 4),
    ("iwp", 1, 6),
}


def steps(prior_name: str, order: int, q: int) -> tuple[int, ...]:
    """The step counts over ``[0, T]`` of a combination."""
    if prior_name == "matern" and (order, q) in MATERN_STEPS:
        return MATERN_STEPS[order, q]
    return STEPS[order, q]


@cache
def _error_fn(prior_name: str, order: int, q: int):
    """``(h, n) -> (filter error, smoother error)`` for ``n`` steps of size ``h``."""
    prior = PRIORS[prior_name](q)
    n_max = max(steps(prior_name, order, q))
    m_0, P_0_sqr = taylor_mode_initialization(LINEAR_2D.vf, LINEAR_2D.x0, q)
    measure = ODEInformation(LINEAR_2D.vf, prior.E0, prior.E1)
    correction = TaylorCorrection(order=order)
    E0 = prior.E0
    D = prior.state_dim
    assert LINEAR_2D.solution is not None

    @jax.jit
    def errors(h, n):
        ts = h * np.arange(n_max + 1, dtype=float)
        res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, correction=correction)
        truth = LINEAR_2D.solution(ts)
        err_filter = np.max(np.abs(E0 @ res.m[n] - truth[n]))

        # Smooth the n-step solve: start from the filtered state at t_n and pass
        # unchanged through the conditionals past it.
        past = (np.arange(n_max) >= n)[:, None]
        res_n = res._replace(
            m=res.m.at[-1].set(res.m[n]),
            P_sqr=res.P_sqr.at[-1].set(res.P_sqr[n]),
            G_back=np.where(past[..., None], np.eye(D), res.G_back),
            d_back=np.where(past, 0.0, res.d_back),
            P_back_sqr=np.where(past[..., None], 0.0, res.P_back_sqr),
        )
        m_s, _ = rts_smoother(res_n)
        nodes = np.arange(n_max + 1)
        interior = ((nodes >= 1) & (nodes < n))[:, None]
        err_smooth = np.max(np.where(interior, np.abs(m_s @ E0.T - truth), 0.0))
        return err_filter, err_smooth

    return errors


def measured_errors(prior_name: str, order: int, q: int) -> tuple[list, list]:
    """Filter and smoother errors over the step counts of ``STEPS``."""
    errors = _error_fn(prior_name, order, q)
    out = [errors(T_END / n, n) for n in steps(prior_name, order, q)]
    return [float(e[0]) for e in out], [float(e[1]) for e in out]


def fitted_order(ns, errors) -> float:
    """Slope of ``log(error)`` against ``log(h)``, without the round-off floor."""
    keep = [(n, e) for n, e in zip(ns, errors, strict=True) if e > FLOOR]
    assert len(keep) >= 3, f"fewer than three errors above the floor: {errors}"
    log_h = onp.log([T_END / n for n, _ in keep])
    log_e = onp.log([e for _, e in keep])
    return float(onp.polyfit(log_h, log_e, 1)[0])


def _cases():
    for prior_name in PRIORS:
        for order, q in STEPS:
            marks = () if (prior_name, order, q) in DEFAULT else (pytest.mark.slow,)
            yield pytest.param(
                prior_name, order, q, id=f"{prior_name}-ek{order}-q{q}", marks=marks
            )


@pytest.mark.parametrize(("prior_name", "order", "q"), list(_cases()))
def test_filter_and_smoother_converge_at_order_q_plus_one(prior_name, order, q):
    ns = steps(prior_name, order, q)
    err_filter, err_smooth = measured_errors(prior_name, order, q)
    low, high = BAND[q]
    for label, errors in (("filter", err_filter), ("smoother", err_smooth)):
        slope = fitted_order(ns, errors)
        assert low <= slope <= high, (
            f"{label} order {slope:.2f} outside [{low}, {high}] for steps {ns}; "
            f"errors {[f'{e:.2e}' for e in errors]}"
        )


def test_the_order_fit_drops_the_round_off_floor():
    ns = (10, 20, 40, 80)
    errors = [1e-6, 1e-6 / 2**4, 1e-6 / 2**8, 1e-16]
    assert fitted_order(ns, errors) == pytest.approx(4.0)
    with pytest.raises(AssertionError, match="three errors"):
        fitted_order(ns, [1e-6, 1e-15, 1e-15, 1e-16])
