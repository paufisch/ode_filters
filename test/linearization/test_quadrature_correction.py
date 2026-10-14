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

from typing import NamedTuple

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    QuadratureCorrection,
    TaylorCorrection,
    gaussian_filter,
)
from ode_filters.calibration import NoCalibration
from ode_filters.filters.kernel import OnlineInformation, filter_step
from ode_filters.gaussian import AffineModel, Gaussian, update
from ode_filters.information import (
    Measurement,
    ODEInformation,
    prepare_observations,
)
from ode_filters.linearization.statistical import slr_linearize
from ode_filters.priors import IWP, PrecondIWP, taylor_mode_initialization
from test.helpers import assert_grad_matches_fd_vector, cov

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


class _Result(NamedTuple):
    m: np.ndarray
    P_sqr: np.ndarray
    mz: np.ndarray
    Pz_sqr: np.ndarray


def _correct(correction, measure, m_pred, P_pred_sqr, *, t):
    """Linearize at the prediction and update, as a filter step does."""
    model = correction.linearize(measure, m_pred, t=t)
    posterior, innovation = correction.correct(
        measure, Gaussian(m_pred, P_pred_sqr), model, t=t
    )
    return _Result(posterior.m, posterior.P_sqr, innovation.m, innovation.P_sqr)


def _prior_and_measure(vf, q=3):
    prior = IWP(q=q, d=1)
    return prior, ODEInformation(vf, prior.E0, prior.E1)


def _setup(vf=logistic, q=3, x0=0.1, jitter=0.0):
    prior = IWP(q=q, d=1)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([x0]), q=q)
    if jitter:
        P_0_sqr = P_0_sqr + jitter * np.eye(q + 1)
    return prior, measure, m_0, P_0_sqr


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_defaults():
    corr = QuadratureCorrection()
    assert corr.rule == "gauss_hermite"
    assert corr.n_nodes == 4
    assert corr.max_iters == 1


@pytest.mark.wrong_input_types
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
    prior, measure, m_0, P_0_sqr = _setup(vf=linear_vf)
    ek1 = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*TSPAN, N_STEPS + 1),
        correction=TaylorCorrection(1),
    )
    for corr in (
        QuadratureCorrection(n_nodes=4),
        QuadratureCorrection(rule="cubature"),
    ):
        slr = gaussian_filter(
            m_0,
            P_0_sqr,
            prior,
            measure,
            np.linspace(*TSPAN, N_STEPS + 1),
            correction=corr,
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

    ek1 = _correct(TaylorCorrection(order=1), measure, m_pred, P_pred_sqr, t=0.2)
    slr = _correct(QuadratureCorrection(n_nodes=4), measure, m_pred, P_pred_sqr, t=0.2)

    S_ek1 = ek1.Pz_sqr.T @ ek1.Pz_sqr
    S_slr = slr.Pz_sqr.T @ slr.Pz_sqr
    assert float(S_slr[0, 0]) > float(S_ek1[0, 0])
    # And the means genuinely differ (spread-corrected H and c).
    assert not onp.allclose(slr.m, ek1.m, atol=1e-6)


def test_correct_equals_the_affine_update_from_the_surrogate():
    """The correction is exactly the affine update with the SLR model."""
    measure = _measure_for(cubic)
    m_pred = np.array([0.6, -0.1, 0.3, 0.02])
    P_pred_sqr = 0.2 * np.eye(4)

    model = slr_linearize(measure, m_pred, P_pred_sqr, t=0.4, n_nodes=4)
    posterior, innovation = update(
        AffineModel(model.H, model.c, model.R_eff_sqr), Gaussian(m_pred, P_pred_sqr)
    )
    m_ref, P_ref_sqr, mz = posterior.m, posterior.P_sqr, innovation.m

    res = _correct(QuadratureCorrection(n_nodes=4), measure, m_pred, P_pred_sqr, t=0.4)
    assert onp.allclose(res.m, m_ref, atol=1e-14)
    assert onp.allclose(res.P_sqr.T @ res.P_sqr, P_ref_sqr.T @ P_ref_sqr, atol=1e-14)
    assert onp.allclose(res.mz, mz, atol=1e-14)


def test_zero_spread_prediction_collapses_onto_ek1():
    """EK1 is the zero-spread limit, so a deterministic prediction agrees exactly."""
    measure = _measure_for(cubic)
    m_pred = np.array([0.6, -0.1, 0.3, 0.02])
    P_pred_sqr = np.zeros((4, 4))

    ek1 = _correct(TaylorCorrection(order=1), measure, m_pred, P_pred_sqr, t=0.4)
    slr = _correct(QuadratureCorrection(n_nodes=4), measure, m_pred, P_pred_sqr, t=0.4)
    assert onp.allclose(slr.m, ek1.m, atol=1e-14)
    assert onp.allclose(slr.mz, ek1.mz, atol=1e-14)


# ---------------------------------------------------------------------------
# IPLF (posterior linearization within a step)
# ---------------------------------------------------------------------------


def test_iplf_single_iteration_is_the_one_shot_filter():
    measure = _measure_for(cubic)
    m_pred = np.array([0.9, -0.3, 0.2, 0.05])
    P_pred_sqr = 0.35 * np.eye(4)

    one = _correct(
        QuadratureCorrection(n_nodes=4, max_iters=1), measure, m_pred, P_pred_sqr, t=0.2
    )
    ref = _correct(QuadratureCorrection(n_nodes=4), measure, m_pred, P_pred_sqr, t=0.2)
    assert onp.allclose(one.m, ref.m, atol=1e-14)


def test_iplf_refits_over_the_updated_density():
    """``max_iters > 1`` must relinearize over the *posterior* spread, not the prior."""
    measure = _measure_for(cubic)
    m_pred = np.array([0.9, -0.3, 0.2, 0.05])
    P_pred_sqr = 0.35 * np.eye(4)

    one = _correct(
        QuadratureCorrection(n_nodes=4, max_iters=1), measure, m_pred, P_pred_sqr, t=0.2
    )
    many = _correct(
        QuadratureCorrection(n_nodes=4, max_iters=4), measure, m_pred, P_pred_sqr, t=0.2
    )
    assert not onp.allclose(many.m, one.m, atol=1e-8)

    # The fixed point: one more pass from the converged surrogate reproduces it.
    res = _correct(
        QuadratureCorrection(n_nodes=4, max_iters=8), measure, m_pred, P_pred_sqr, t=0.2
    )
    res2 = _correct(
        QuadratureCorrection(n_nodes=4, max_iters=9), measure, m_pred, P_pred_sqr, t=0.2
    )
    assert onp.allclose(res.m, res2.m, atol=1e-8)


def test_iplf_is_a_no_op_on_affine_fields():
    measure = _measure_for(linear_vf)
    m_pred = np.array([0.9, -0.3, 0.2, 0.05])
    P_pred_sqr = 0.35 * np.eye(4)
    one = _correct(
        QuadratureCorrection(max_iters=1), measure, m_pred, P_pred_sqr, t=0.2
    )
    many = _correct(
        QuadratureCorrection(max_iters=5), measure, m_pred, P_pred_sqr, t=0.2
    )
    assert onp.allclose(many.m, one.m, atol=1e-13)


# ---------------------------------------------------------------------------
# Composition with the solver paths
# ---------------------------------------------------------------------------


def test_solves_a_nonlinear_ode_accurately():
    """A pure solve must still converge: the correction changes the constant, not the order."""
    prior, measure, m_0, P_0_sqr = _setup()

    def exact(t, x0=0.1):
        return x0 * np.exp(t) / (1 - x0 + x0 * np.exp(t))

    res = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*TSPAN, N_STEPS + 1),
        correction=QuadratureCorrection(n_nodes=4),
    )
    assert float(np.max(np.abs(res.m[:, 0] - exact(res.t)))) < 1e-6


def test_reaches_through_the_filter_step():
    """The step predicts with the prior and then applies the correction itself."""
    prior, measure = _prior_and_measure(cubic)
    x = Gaussian(np.array([0.7, -0.2, 0.1, 0.0]), 0.3 * np.eye(4))
    out = filter_step(
        x,
        prior.transition(0.1),
        OnlineInformation(measure, QuadratureCorrection(n_nodes=4)),
        t=0.2,
        calibration=NoCalibration(),
        scale_noise=prior.apply_state_sigma_sqr,
    )
    A, Q_sqr = prior.A(0.1), prior.Q_sqr(0.1)
    assert onp.allclose(out.predicted.m, A @ x.m, rtol=1e-13)
    P_pred = A @ cov(x.P_sqr) @ A.T + cov(Q_sqr)
    assert onp.allclose(cov(out.predicted.P_sqr), P_pred, rtol=1e-12, atol=1e-15)
    expected = _correct(
        QuadratureCorrection(n_nodes=4),
        measure,
        out.predicted.m,
        out.predicted.P_sqr,
        t=0.2,
    )
    assert onp.allclose(out.filtered.m, expected.m, rtol=1e-12)
    assert onp.allclose(out.innovation.m, expected.mz, rtol=1e-12)


def test_works_with_the_observation_path():
    prior, measure, m_0, P_0_sqr = _setup()
    ts = np.linspace(*TSPAN, N_STEPS + 1)
    z_t = ts[::6][1:]
    obs = Measurement(
        A=np.eye(1), z=np.array([[0.15], [0.25], [0.4], [0.6]]), z_t=z_t, noise=1e-3
    )
    obs_model = prepare_observations([obs], prior.E0, ts)

    res = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*TSPAN, N_STEPS + 1),
        correction=QuadratureCorrection(n_nodes=4),
        obs_model=obs_model,
    )
    # The data evidence is the density of the data innovations of the steps
    # with data.
    assert obs_model is not None
    assert res.log_likelihood_obs is not None
    assert res.mz_obs is not None and res.Pz_obs_sqr is not None
    has_data = obs_model.mask[:, 0]
    variance = cov(res.Pz_obs_sqr)[:, 0, 0]
    logpdf = -0.5 * (np.log(2 * np.pi * variance) + res.mz_obs[:, 0] ** 2 / variance)
    expected = np.sum(np.where(has_data, logpdf, 0.0))
    assert float(res.log_likelihood_obs) == pytest.approx(float(expected), rel=1e-12)


def test_works_on_the_preconditioned_path():
    """``E_args`` transports to bar coordinates, so the projected marginal matches."""
    q = 3
    prior = PrecondIWP(q=q, d=1)
    measure = ODEInformation(logistic, prior.E0, prior.E1)
    m_0, P_0_sqr = taylor_mode_initialization(logistic, np.array([0.1]), q=q)

    plain_prior = IWP(q=q, d=1)
    plain = gaussian_filter(
        m_0,
        P_0_sqr,
        plain_prior,
        ODEInformation(logistic, plain_prior.E0, plain_prior.E1),
        np.linspace(*TSPAN, N_STEPS + 1),
        correction=QuadratureCorrection(n_nodes=4),
    )
    precond = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        measure,
        np.linspace(*TSPAN, N_STEPS + 1),
        correction=QuadratureCorrection(n_nodes=4),
    )
    assert onp.allclose(precond.m, plain.m, atol=1e-8)


def test_jit_and_grad_safe():
    prior, measure, m_0, P_0_sqr = _setup(vf=cubic, jitter=1e-3)

    def terminal(mu):
        return gaussian_filter(
            mu,
            P_0_sqr,
            prior,
            measure,
            np.linspace(*TSPAN, N_STEPS + 1),
            correction=QuadratureCorrection(n_nodes=4),
        ).m[-1, 0]

    assert float(jax.jit(terminal)(m_0)) == pytest.approx(
        float(terminal(m_0)), rel=1e-12
    )
    # A step's ODE residual is ~1e-7 here, and the calibrated diffusion is
    # quadratic in it, so the output curves on that scale: a small step.
    assert_grad_matches_fd_vector(terminal, m_0, eps=1e-8, rtol=1e-4)


def test_cubature_scales_to_a_multidimensional_ode():
    """The quadrature dimension is the ODE dimension, so ``d > 1`` stays cheap."""

    def lotka_volterra(x, *, t):
        return np.array([1.5 * x[0] - x[0] * x[1], x[0] * x[1] - 3.0 * x[1]])

    q, d = 2, 2
    prior = IWP(q=q, d=d)
    measure = ODEInformation(lotka_volterra, prior.E0, prior.E1)
    assert measure.E_args.shape == (2, prior.E0.shape[1])
    m_0, P_0_sqr = taylor_mode_initialization(lotka_volterra, np.array([1.0, 1.0]), q=q)

    def solve(correction):
        ts = np.linspace(0.0, 1.0, 30 + 1)
        return gaussian_filter(
            m_0, P_0_sqr, prior, measure, ts, correction=correction
        ).m

    cubature = solve(QuadratureCorrection(rule="cubature"))
    hermite = solve(QuadratureCorrection(n_nodes=3))
    # Both rules are exact for the expected Jacobian of a quadratic field, and
    # in a pure ODE solve the predictive spread (the Jensen gap to EK1) is tiny.
    assert onp.allclose(cubature, hermite, rtol=1e-10, atol=1e-12)
    assert onp.allclose(cubature, solve(TaylorCorrection()), rtol=1e-8, atol=1e-10)
