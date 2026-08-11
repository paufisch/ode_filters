"""Iterated posterior-linearization smoother.

The invariants that matter:

1. ``n_iters=0`` is *exactly* the one-shot SLR filter plus RTS -- so the
   iteration is a strict extension of a known baseline and the comparison
   "does relinearizing help?" is a clean one;
2. on an affine vector field every pass is a no-op, so IPLS reduces to the exact
   linear-Gaussian smoother for any ``n_iters``;
3. the affine filter driving the passes agrees with the ordinary filter when fed
   the same surrogate, so the outer loop is the only thing that differs.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.filters import (
    QuadratureCorrection,
    TaylorCorrection,
    affine_filter_scan,
    gaussian_filter,
    ipls_smoother,
    rts_smoother,
)
from ode_filters.filters.statistical_linearization import slr_linearize
from ode_filters.measurement import (
    Measurement,
    ODEInformation,
    prepare_observations,
)
from ode_filters.priors import IWP, PrecondIWP, taylor_mode_initialization

TSPAN = (0.0, 2.0)
N_STEPS = 32


def logistic(x, *, t):
    return x * (1.0 - x)


def linear_vf(x, *, t):
    return -0.7 * x


def _setup(vf=logistic, q=3, x0=0.1, jitter=0.0):
    prior = IWP(q=q, d=1)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    mu_0, P0_sqr = taylor_mode_initialization(vf, np.array([x0]), q=q)
    if jitter:
        P0_sqr = P0_sqr + jitter * np.eye(q + 1)
    return prior, measure, mu_0, P0_sqr


def _observations(prior, n_obs=6, noise=1e-2):
    """Sparse position observations of the logistic solution, with noise."""
    ts = np.linspace(*TSPAN, N_STEPS + 1)
    z_t = ts[1:][:: (N_STEPS // n_obs)]
    truth = 0.1 * np.exp(z_t) / (1 - 0.1 + 0.1 * np.exp(z_t))
    z = (truth + 0.02 * np.sin(37.0 * z_t))[:, None]
    obs = Measurement(A=np.eye(1), z=z, z_t=z_t, noise=noise)
    return prepare_observations([obs], prior.E0, ts)


# ---------------------------------------------------------------------------
# Baseline equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("with_obs", [False, True])
def test_zero_iterations_is_the_one_shot_baseline(with_obs):
    prior, measure, mu_0, P0_sqr = _setup()
    obs_model = _observations(prior) if with_obs else None

    base = gaussian_filter(
        mu_0,
        P0_sqr,
        prior,
        measure,
        TSPAN,
        N_STEPS,
        correction=QuadratureCorrection(n_nodes=4),
        calibration="none",
        obs_model=obs_model,
    )
    m_ref, P_ref_sqr = rts_smoother(prior, base)

    res = ipls_smoother(
        mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, n_iters=0, obs_model=obs_model
    )
    assert onp.allclose(res.m, m_ref, atol=1e-14)
    assert onp.allclose(res.P_sqr, P_ref_sqr, atol=1e-14)
    assert onp.allclose(res.m_filt, base.m, atol=1e-14)
    assert float(res.log_likelihood) == pytest.approx(float(base.log_likelihood))
    assert res.n_iters == 0
    assert onp.allclose(res.t, np.linspace(*TSPAN, N_STEPS + 1))


def test_affine_field_makes_every_pass_a_no_op():
    """Omega vanishes and H is constant, so IPLS is the exact linear smoother."""
    prior, measure, mu_0, P0_sqr = _setup(vf=linear_vf)
    obs_model = _observations(prior)

    base = gaussian_filter(
        mu_0,
        P0_sqr,
        prior,
        measure,
        TSPAN,
        N_STEPS,
        correction=TaylorCorrection(1),
        calibration="none",
        obs_model=obs_model,
    )
    m_ref, _ = rts_smoother(prior, base)

    for n_iters in (0, 1, 4):
        res = ipls_smoother(
            mu_0,
            P0_sqr,
            prior,
            measure,
            TSPAN,
            N_STEPS,
            n_iters=n_iters,
            obs_model=obs_model,
        )
        assert onp.allclose(res.m, m_ref, atol=1e-10)


def test_iterations_move_the_estimate_on_a_nonlinear_problem_with_data():
    """With observations keeping the spread up, relinearizing changes the answer."""
    prior, measure, mu_0, P0_sqr = _setup(q=2, jitter=1e-2)
    obs_model = _observations(prior, n_obs=8, noise=1e-2)

    runs = [
        ipls_smoother(
            mu_0,
            P0_sqr,
            prior,
            measure,
            TSPAN,
            N_STEPS,
            n_iters=k,
            obs_model=obs_model,
        )
        for k in (0, 1, 6)
    ]
    assert all(bool(np.all(np.isfinite(r.m))) for r in runs)
    assert not onp.allclose(runs[1].m, runs[0].m, atol=1e-12)
    # And the iteration converges: further passes stop moving the estimate.
    extra = ipls_smoother(
        mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, n_iters=7, obs_model=obs_model
    )
    assert onp.allclose(extra.m, runs[2].m, atol=1e-7)


def test_solution_stays_accurate():
    prior, measure, mu_0, P0_sqr = _setup()

    def exact(t, x0=0.1):
        return x0 * np.exp(t) / (1 - x0 + x0 * np.exp(t))

    res = ipls_smoother(mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, n_iters=3)
    assert float(np.max(np.abs(res.m[:, 0] - exact(res.t)))) < 1e-5


# ---------------------------------------------------------------------------
# The affine driver
# ---------------------------------------------------------------------------


def test_affine_filter_scan_matches_the_ordinary_filter_on_the_same_surrogate():
    """Feeding the EK1 surrogate reproduces an EK1 filter step-for-step.

    Linearizing at the *filtered* means of an EK1 run and replaying those fixed
    surrogates must reproduce that run, which is what makes the outer loop the
    only moving part.
    """
    prior, measure, mu_0, P0_sqr = _setup()
    ts, h = np.linspace(*TSPAN, N_STEPS + 1, retstep=True)

    ek1 = gaussian_filter(
        mu_0,
        P0_sqr,
        prior,
        measure,
        TSPAN,
        N_STEPS,
        correction=TaylorCorrection(1),
        calibration="none",
    )
    # EK1 linearizes at the predicted mean of each step.
    H_seq, c_seq = jax.vmap(lambda m, t: measure.linearize(m, t=t))(ek1.m_pred, ts[1:])
    R_sqr = measure.get_noise(t=0.0)
    R_seq_sqr = np.broadcast_to(R_sqr, (N_STEPS, *R_sqr.shape))

    m_seq, P_seq_sqr, *_ = affine_filter_scan(
        mu_0, P0_sqr, prior.A(h), prior.b(h), prior.Q_sqr(h), H_seq, c_seq, R_seq_sqr
    )
    assert onp.allclose(m_seq, ek1.m, atol=1e-12)
    assert onp.allclose(P_seq_sqr, ek1.P_sqr, atol=1e-12)


def test_affine_filter_scan_applies_observations():
    prior, measure, mu_0, P0_sqr = _setup()
    ts, h = np.linspace(*TSPAN, N_STEPS + 1, retstep=True)
    obs_model = _observations(prior)

    models = jax.vmap(
        lambda t: slr_linearize(measure, mu_0, P0_sqr + 1e-2 * np.eye(4), t=t)
    )(ts[1:])
    args = (
        mu_0,
        P0_sqr,
        prior.A(h),
        prior.b(h),
        prior.Q_sqr(h),
        models.H,
        models.c,
        models.R_eff_sqr,
    )
    without = affine_filter_scan(*args)
    with_obs = affine_filter_scan(*args, obs_model=obs_model)

    assert not onp.allclose(with_obs[0], without[0], atol=1e-8)
    assert float(without[6]) == 0.0  # no observation likelihood without data
    assert float(with_obs[6]) != 0.0


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_obs_likelihood_is_reported_only_with_data():
    prior, measure, mu_0, P0_sqr = _setup()
    plain = ipls_smoother(mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, n_iters=1)
    assert plain.log_likelihood_obs is None

    with_obs = ipls_smoother(
        mu_0,
        P0_sqr,
        prior,
        measure,
        TSPAN,
        N_STEPS,
        n_iters=1,
        obs_model=_observations(prior),
    )
    assert with_obs.log_likelihood_obs is not None
    assert onp.isfinite(float(with_obs.log_likelihood_obs))


def test_cubature_rule_is_selectable():
    prior, measure, mu_0, P0_sqr = _setup()
    res = ipls_smoother(
        mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, n_iters=2, rule="cubature"
    )
    assert bool(np.all(np.isfinite(res.m)))


def test_rejects_negative_iterations():
    prior, measure, mu_0, P0_sqr = _setup()
    with pytest.raises(ValueError, match="n_iters must be >= 0"):
        ipls_smoother(mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS, n_iters=-1)


def test_rejects_preconditioned_priors():
    q = 3
    prior = PrecondIWP(q=q, d=1)
    measure = ODEInformation(logistic, prior.E0, prior.E1)
    mu_0, P0_sqr = taylor_mode_initialization(logistic, np.array([0.1]), q=q)
    with pytest.raises(NotImplementedError, match="preconditioned"):
        ipls_smoother(mu_0, P0_sqr, prior, measure, TSPAN, N_STEPS)


def test_jit_and_grad_safe_with_a_positive_definite_initial_covariance():
    """Reverse mode needs a PD ``P_0_sqr`` -- see the ``ipls_smoother`` note."""
    prior, measure, mu_0, P0_sqr = _setup(jitter=1e-3)

    def terminal(mu):
        return ipls_smoother(mu, P0_sqr, prior, measure, TSPAN, N_STEPS, n_iters=2).m[
            -1, 0
        ]

    assert onp.isfinite(float(jax.jit(terminal)(mu_0)))
    assert bool(np.all(np.isfinite(jax.grad(terminal)(mu_0))))
