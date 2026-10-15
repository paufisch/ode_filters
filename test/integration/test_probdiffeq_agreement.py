"""Fixed-grid EK1 agrees with probdiffeq 0.8 on the same grid and prior.

probdiffeq's dense EK1 (``correction_ts1`` with ``ssm_fact="dense"``) on an
integrated Wiener process prior with unit output scale is the same algorithm as
``gaussian_filter`` with ``IWP(q, d)`` and ``Xi = I``; its ``strategy_smoother``
gives the RTS smoother. On Lotka-Volterra the means and standard deviations of
every derivative agree, filter and smoother, without calibration and with the
dynamic one.

Convention differences, all of them notation:

- probdiffeq stores the *lower* Cholesky factor ``L`` of a covariance
  (``cov = L L.T``), ode_filters an upper square root (``P = P_sqr.T P_sqr``);
  the standard deviations are the row norms of ``L`` and the column norms of
  ``P_sqr``.
- probdiffeq's ``output_scale`` is a standard deviation: its square is
  ode_filters' ``sigma_sqr``. ``ivpsolvers.solver`` keeps it at one
  (``calibration="none"``, where ode_filters still reports its estimate);
  ``solver_dynamic`` sets it per step to the same quasi-MLE
  ``r.T (H Q H.T)^-1 r / d`` before the prediction (``"dynamic"``).
- The dense state is derivative-major, ``[x, x', ..., x^(q)]``, in both, and
  probdiffeq's ``tcoeffs`` are the derivatives themselves (not divided by
  ``k!``), so they equal ``taylor_mode_initialization``.
- probdiffeq discretizes the IWP in the coordinates of the preconditioner
  ``h^(q - k + 1/2) / (q - k)!``, which is ``PrecondIWP``'s; both ode_filters
  priors agree with it.

Measured largest relative differences over all derivatives and nodes, for
both priors (means relative to each entry's scale, standard deviations
entrywise): without calibration means 4.9e-14 (filter) and 4.9e-13
(smoother), standard deviations 3.8e-14; with ``"dynamic"`` means 1.8e-12 /
3.7e-12 and standard deviations 3.9e-11 / 3.3e-11. The calibrated diffusion
differs by ~1e-11 relative: the residual ``r`` is a difference of nearly
equal terms, and the two libraries form it and its whitening in different orders.
"""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    ODEInformation,
    PrecondIWP,
    gaussian_filter,
    rts_smoother,
    taylor_mode_initialization,
)
from test.helpers import LOTKA_VOLTERRA

probdiffeq = pytest.importorskip("probdiffeq")
from probdiffeq import ivpsolve, ivpsolvers, taylor  # noqa: E402

# probdiffeq's own code needs JAX's standard dtype promotion.
pytestmark = pytest.mark.standard_promotion

Q = 3
TS = np.linspace(0.0, 5.0, 101)
TOLERANCE = {"none": 1e-11, "dynamic": 1e-9}


@cache
def _probdiffeq(strategy: str, calibration: str):
    """Marginal means and standard deviations of a probdiffeq fixed-grid solve."""
    vf = LOTKA_VOLTERRA.vf
    tcoeffs = taylor.odejet_padded_scan(
        lambda y: vf(y, t=TS[0]), (LOTKA_VOLTERRA.x0,), num=Q
    )
    init, ibm, ssm = ivpsolvers.prior_wiener_integrated(tcoeffs, ssm_fact="dense")
    correction = ivpsolvers.correction_ts1(vf, ode_order=1, ssm=ssm)
    make_strategy = {
        "filter": ivpsolvers.strategy_filter,
        "smoother": ivpsolvers.strategy_smoother,
    }[strategy]
    make_solver = {"none": ivpsolvers.solver, "dynamic": ivpsolvers.solver_dynamic}[
        calibration
    ]
    solver = make_solver(
        make_strategy(ssm=ssm), prior=ibm, correction=correction, ssm=ssm
    )
    sol = jax.jit(
        lambda: ivpsolve.solve_fixed_grid(init, grid=TS, solver=solver, ssm=ssm)
    )()
    L = sol.marginals.cholesky
    return sol.marginals.mean, np.sqrt(np.einsum("kij,kij->ki", L, L))


def _relative(a, b):
    """Largest difference relative to the scale of each state entry over time."""
    return float(np.max(np.abs(a - b) / np.max(np.abs(b), axis=0, keepdims=True)))


@pytest.mark.parametrize("prior_type", [IWP, PrecondIWP])
@pytest.mark.parametrize("calibration", ["none", "dynamic"])
def test_means_and_stds_match_probdiffeq(calibration, prior_type):
    prior = prior_type(Q, LOTKA_VOLTERRA.d)
    m_0, P_0_sqr = taylor_mode_initialization(LOTKA_VOLTERRA.vf, LOTKA_VOLTERRA.x0, Q)
    measure = ODEInformation(LOTKA_VOLTERRA.vf, prior.E0, prior.E1)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration=calibration)
    smoothed = rts_smoother(res)
    ours = {"filter": (res.m, res.P_sqr), "smoother": smoothed}
    tol = TOLERANCE[calibration]
    for strategy, (m, P_sqr) in ours.items():
        mean, std = _probdiffeq(strategy, calibration)
        std_ours = np.sqrt(np.sum(P_sqr**2, axis=-2))
        assert _relative(mean, m) < tol, strategy
        # Elementwise: the standard deviations grow over orders of magnitude
        # from the exact initial state (zero variance, left out).
        rel_std = np.abs(std - std_ours)[1:] / std_ours[1:]
        assert float(np.max(rel_std)) < tol, strategy
