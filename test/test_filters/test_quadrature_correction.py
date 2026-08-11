"""``QuadratureCorrection``: statistical linearization as a Correction strategy.

Proves the three properties that make it safe to ship as an alternative to EK1:

1. it is a **no-op on affine problems** -- bit-comparable to EK1 -- so nothing
   linear changes behaviour;
2. its update is exactly the affine update built from the SLR surrogate, i.e.
   the innovation covariance carries ``Omega`` where EK1's does not;
3. it composes with everything a Correction is expected to compose with:
   ``gaussian_filter``, the sequential observation path, the preconditioned
   path, ``jit`` and reverse-mode ``grad``.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.filters import (
    QuadratureCorrection,
    TaylorCorrection,
    gaussian_filter,
)
from ode_filters.filters.ode_filter_step import sqr_filter_step
from ode_filters.filters.statistical_linearization import slr_linearize
from ode_filters.inference import sqr_inversion, sqr_marginalization
from ode_filters.measurement import (
    Measurement,
    ODEInformation,
    prepare_observations,
)
from ode_filters.priors import IWP, PrecondIWP, taylor_mode_initialization

TSPAN = (0.0, 1.5)
N_STEPS = 24


def logistic(x, *, t):
    return x * (1.0 - x)


def cubic(x, *, t):
    return -0.7 * x + 0.35 * x**3


def linear_vf(x, *, t):
    return -0.7 * x


def _measure_for(vf, q=3):
    """Just the measurement model, for tests that drive a Correction directly."""
    prior = IWP(q=q, d=1)
    return ODEInformation(vf, prior.E0, prior.E1)


def _prior_and_measure(vf, q=3):
    prior = IWP(q=q, d=1)
    return prior, ODEInformation(vf, prior.E0, prior.E1)


def _setup(vf=logistic, q=3, x0=0.1, jitter=0.0):
    prior = IWP(q=q, d=1)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    mu_0, P0_sqr = taylor_mode_initialization(vf, np.array([x0]), q=q)
    if jitter:
        P0_sqr = P0_sqr + jitter * np.eye(q + 1)
    return prior, measure, mu_0, P0_sqr


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_defaults():
    corr = QuadratureCorrection()
    assert corr.rule == "gauss_hermite"
    assert corr.n_nodes == 4
    assert corr.max_iters == 1


def test_rejects_bad_configuration_at_construction():
    """A bad rule must fail eagerly, not inside a traced scan body."""
    with pytest.raises(ValueError, match="gauss_hermite"):
        QuadratureCorrection(rule="unscented")
    with pytest.raises(ValueError, match="max_iters must be >= 1"):
        QuadratureCorrection(max_iters=0)
    with pytest.raises(ValueError, match="n_nodes must be >= 1"):
        QuadratureCorrection(n_nodes=0)


# ---------------------------------------------------------------------------
# Relation to EK1
# ---------------------------------------------------------------------------


def test_affine_field_reproduces_ek1():
    """Nothing changes on a linear problem: Omega is zero and H is the Jacobian."""
    prior, measure, mu_0, P0_sqr = _setup(vf=linear_vf)
    ek1 = gaussian_filter(
        mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, correction=TaylorCorrection(1)
    )
    for corr in (
        QuadratureCorrection(n_nodes=4),
        QuadratureCorrection(rule="cubature"),
    ):
        slr = gaussian_filter(
            mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, correction=corr
        )
        assert onp.allclose(slr.m, ek1.m, atol=1e-12)
        assert onp.allclose(slr.Pz_sqr, ek1.Pz_sqr, atol=1e-12)
        assert float(slr.log_likelihood) == pytest.approx(
            float(ek1.log_likelihood), abs=1e-9
        )


def test_nonlinear_field_inflates_the_innovation_covariance():
    """The whole point: ``S = H P H^T + Omega + R`` where EK1 has ``H P H^T + R``.

    Checked at a wide predictive spread, where the Jensen gap is visible; in a
    well-resolved pure solve the spread collapses and the two agree.
    """
    measure = _measure_for(cubic)
    m_pred = np.array([0.9, -0.3, 0.2, 0.05])
    P_pred_sqr = 0.35 * np.eye(4)

    ek1 = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=0.2)
    slr = QuadratureCorrection(n_nodes=4).correct(measure, m_pred, P_pred_sqr, t=0.2)

    S_ek1 = ek1.Pz_sqr.T @ ek1.Pz_sqr
    S_slr = slr.Pz_sqr.T @ slr.Pz_sqr
    assert float(S_slr[0, 0]) > float(S_ek1[0, 0])
    # And the means genuinely differ (spread-corrected H and c).
    assert not onp.allclose(slr.m, ek1.m, atol=1e-6)


def test_correct_equals_the_affine_update_from_the_surrogate():
    """The correction is exactly ``sqr_marginalization`` + ``sqr_inversion`` on the SLR model."""
    measure = _measure_for(cubic)
    m_pred = np.array([0.6, -0.1, 0.3, 0.02])
    P_pred_sqr = 0.2 * np.eye(4)

    model = slr_linearize(measure, m_pred, P_pred_sqr, t=0.4, n_nodes=4)
    mz, Pz_sqr = sqr_marginalization(
        model.H, model.c, model.R_eff_sqr, m_pred, P_pred_sqr
    )
    _, m_ref, P_ref_sqr = sqr_inversion(
        model.H, m_pred, P_pred_sqr, mz, Pz_sqr, model.R_eff_sqr
    )

    res = QuadratureCorrection(n_nodes=4).correct(measure, m_pred, P_pred_sqr, t=0.4)
    assert onp.allclose(res.m, m_ref, atol=1e-14)
    assert onp.allclose(res.P_sqr.T @ res.P_sqr, P_ref_sqr.T @ P_ref_sqr, atol=1e-14)
    assert onp.allclose(res.mz, mz, atol=1e-14)


def test_zero_spread_prediction_collapses_onto_ek1():
    """EK1 is the zero-spread limit, so a deterministic prediction agrees exactly."""
    measure = _measure_for(cubic)
    m_pred = np.array([0.6, -0.1, 0.3, 0.02])
    P_pred_sqr = np.zeros((4, 4))

    ek1 = TaylorCorrection(order=1).correct(measure, m_pred, P_pred_sqr, t=0.4)
    slr = QuadratureCorrection(n_nodes=4).correct(measure, m_pred, P_pred_sqr, t=0.4)
    assert onp.allclose(slr.m, ek1.m, atol=1e-14)
    assert onp.allclose(slr.mz, ek1.mz, atol=1e-14)


# ---------------------------------------------------------------------------
# IPLF (posterior linearization within a step)
# ---------------------------------------------------------------------------


def test_iplf_single_iteration_is_the_one_shot_filter():
    measure = _measure_for(cubic)
    m_pred = np.array([0.9, -0.3, 0.2, 0.05])
    P_pred_sqr = 0.35 * np.eye(4)

    one = QuadratureCorrection(n_nodes=4, max_iters=1).correct(
        measure, m_pred, P_pred_sqr, t=0.2
    )
    ref = QuadratureCorrection(n_nodes=4).correct(measure, m_pred, P_pred_sqr, t=0.2)
    assert onp.allclose(one.m, ref.m, atol=1e-14)


def test_iplf_refits_over_the_updated_density():
    """``max_iters > 1`` must relinearize over the *posterior* spread, not the prior."""
    measure = _measure_for(cubic)
    m_pred = np.array([0.9, -0.3, 0.2, 0.05])
    P_pred_sqr = 0.35 * np.eye(4)

    one = QuadratureCorrection(n_nodes=4, max_iters=1).correct(
        measure, m_pred, P_pred_sqr, t=0.2
    )
    many = QuadratureCorrection(n_nodes=4, max_iters=4).correct(
        measure, m_pred, P_pred_sqr, t=0.2
    )
    assert not onp.allclose(many.m, one.m, atol=1e-8)

    # The fixed point: one more pass from the converged surrogate reproduces it.
    res = QuadratureCorrection(n_nodes=4, max_iters=8).correct(
        measure, m_pred, P_pred_sqr, t=0.2
    )
    res2 = QuadratureCorrection(n_nodes=4, max_iters=9).correct(
        measure, m_pred, P_pred_sqr, t=0.2
    )
    assert onp.allclose(res.m, res2.m, atol=1e-8)


def test_iplf_is_a_no_op_on_affine_fields():
    measure = _measure_for(linear_vf)
    m_pred = np.array([0.9, -0.3, 0.2, 0.05])
    P_pred_sqr = 0.35 * np.eye(4)
    one = QuadratureCorrection(max_iters=1).correct(measure, m_pred, P_pred_sqr, t=0.2)
    many = QuadratureCorrection(max_iters=5).correct(measure, m_pred, P_pred_sqr, t=0.2)
    assert onp.allclose(many.m, one.m, atol=1e-13)


# ---------------------------------------------------------------------------
# Composition with the solver paths
# ---------------------------------------------------------------------------


def test_solves_a_nonlinear_ode_accurately():
    """A pure solve must still converge: the correction changes the constant, not the order."""
    prior, measure, mu_0, P0_sqr = _setup()

    def exact(t, x0=0.1):
        return x0 * np.exp(t) / (1 - x0 + x0 * np.exp(t))

    res = gaussian_filter(
        mu_0,
        P0_sqr,
        prior,
        measure,
        TSPAN,
        N_STEPS,
        correction=QuadratureCorrection(n_nodes=4),
    )
    assert float(np.max(np.abs(res.m[:, 0] - exact(res.t)))) < 1e-6


def test_reaches_through_sqr_filter_step():
    prior, measure = _prior_and_measure(cubic)
    A, b = prior.A(0.1), prior.b(0.1)
    Q_sqr = prior.Q_sqr(0.1)
    m0 = np.array([0.7, -0.2, 0.1, 0.0])
    P0 = 0.3 * np.eye(4)

    _, _, (mz, _), (m_new, _) = sqr_filter_step(
        A, b, Q_sqr, m0, P0, measure, t=0.2, correction=QuadratureCorrection(n_nodes=4)
    )
    assert bool(np.all(np.isfinite(m_new)))
    assert bool(np.all(np.isfinite(mz)))


def test_works_with_the_observation_path():
    prior, measure, mu_0, P0_sqr = _setup()
    ts = np.linspace(*TSPAN, N_STEPS + 1)
    z_t = ts[::6][1:]
    obs = Measurement(
        A=np.eye(1), z=np.array([[0.15], [0.25], [0.4], [0.6]]), z_t=z_t, noise=1e-3
    )
    obs_model = prepare_observations([obs], prior.E0, ts)

    res = gaussian_filter(
        mu_0,
        P0_sqr,
        prior,
        measure,
        TSPAN,
        N_STEPS,
        correction=QuadratureCorrection(n_nodes=4),
        obs_model=obs_model,
    )
    assert res.log_likelihood_obs is not None
    assert bool(np.all(np.isfinite(res.m)))


def test_works_on_the_preconditioned_path():
    """``E_args`` transports to bar coordinates, so the projected marginal matches."""
    q = 3
    prior = PrecondIWP(q=q, d=1)
    measure = ODEInformation(logistic, prior.E0, prior.E1)
    mu_0, P0_sqr = taylor_mode_initialization(logistic, np.array([0.1]), q=q)

    plain_prior = IWP(q=q, d=1)
    plain = gaussian_filter(
        mu_0,
        P0_sqr,
        plain_prior,
        ODEInformation(logistic, plain_prior.E0, plain_prior.E1),
        TSPAN,
        N_STEPS,
        correction=QuadratureCorrection(n_nodes=4),
    )
    precond = gaussian_filter(
        mu_0,
        P0_sqr,
        prior,
        measure,
        TSPAN,
        N_STEPS,
        correction=QuadratureCorrection(n_nodes=4),
    )
    assert onp.allclose(precond.m, plain.m, atol=1e-8)


def test_jit_and_grad_safe():
    prior, measure, mu_0, P0_sqr = _setup(vf=cubic, jitter=1e-3)

    def terminal(mu):
        return gaussian_filter(
            mu,
            P0_sqr,
            prior,
            measure,
            TSPAN,
            N_STEPS,
            correction=QuadratureCorrection(n_nodes=4),
        ).m[-1, 0]

    assert onp.isfinite(float(jax.jit(terminal)(mu_0)))
    assert bool(np.all(np.isfinite(jax.grad(terminal)(mu_0))))


def test_cubature_scales_to_a_multidimensional_ode():
    """The quadrature dimension is the ODE dimension, so ``d > 1`` stays cheap."""

    def lotka_volterra(x, *, t):
        return np.array([1.5 * x[0] - x[0] * x[1], x[0] * x[1] - 3.0 * x[1]])

    q, d = 2, 2
    prior = IWP(q=q, d=d)
    measure = ODEInformation(lotka_volterra, prior.E0, prior.E1)
    assert measure.E_args.shape == (2, prior.E0.shape[1])
    mu_0, P0_sqr = taylor_mode_initialization(lotka_volterra, np.array([1.0, 1.0]), q=q)

    for corr in (
        QuadratureCorrection(rule="cubature"),
        QuadratureCorrection(n_nodes=3),
    ):
        res = gaussian_filter(
            mu_0, P0_sqr, prior, measure, (0.0, 1.0), 30, correction=corr
        )
        assert bool(np.all(np.isfinite(res.m)))
