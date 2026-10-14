"""Online diffusion calibration in the fixed-grid filter."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    Conservation,
    JointPrior,
    ODEInformation,
    ODEInformationWithHidden,
    PrecondIWP,
    gaussian_filter,
    rts_smoother,
    taylor_mode_initialization,
)
from ode_filters.calibration.sigma import (
    posthoc_mle_sigma_sqr,
    quasi_mle_sigma_sqr_from_Q,
)
from test.helpers import (
    assert_close,
    assert_grad_matches_fd,
    assert_grad_matches_fd_vector,
)

CALIBRATIONS = ["dynamic", "diagonal", "diagonal_ekf0", "none"]


def logistic_vf(x, *, t):
    return x * (1 - x)


def _logistic(prior_cls=IWP, q=2):
    prior = prior_cls(q=q, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(logistic_vf, np.array([0.1]), q=q)
    return prior, m_0, P_0_sqr, ODEInformation(logistic_vf, prior.E0, prior.E1)


TS = np.linspace(0.0, 5.0, 51)


def test_each_steps_diffusion_is_the_quasi_mle_at_its_prediction():
    prior, m_0, P_0_sqr, measure = _logistic()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    h = float(TS[1] - TS[0])
    for i in range(TS.shape[0] - 1):
        m_pred = prior.A(h) @ res.m[i] + prior.b(h)
        H, c = measure.linearize(m_pred, t=TS[i + 1])
        expected = quasi_mle_sigma_sqr_from_Q(H @ m_pred + c, H, prior.Q_sqr(h))
        assert float(res.sigma_sqr[i]) == pytest.approx(float(expected), rel=1e-10)


def test_calibrated_solution_is_accurate_and_smoothable():
    prior, m_0, P_0_sqr, measure = _logistic()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    assert abs(float(res.m[-1, 0]) - float(1 / (1 + 9 * np.exp(-5.0)))) < 1e-3
    assert np.allclose(rts_smoother(res).m[-1], res.m[-1])


def test_diagonal_calibration_returns_one_diffusion_per_component():
    prior, m_0, P_0_sqr, measure = _logistic()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="diagonal")
    assert res.sigma_sqr.shape == (TS.shape[0] - 1, 1)
    # With one component, the per-component estimate is the scalar one.
    scalar = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="dynamic")
    assert_close(res.sigma_sqr[:, 0], scalar.sigma_sqr, rtol=1e-10)
    assert_close(res.m, scalar.m, rtol=1e-10)


def test_posthoc_calibration_of_an_uncalibrated_run():
    prior, m_0, P_0_sqr, measure = _logistic()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration="none")
    sigma_sqr = float(posthoc_mle_sigma_sqr(res.mz, res.Pz_sqr))
    assert 0.0 < sigma_sqr < 1e3


def _joint(prior_cls):
    joint = JointPrior(prior_cls(q=2, d=2), prior_cls(q=0, d=1, Xi=1e-3 * np.eye(1)))

    def vf(x, u, *, t):
        return -u[0] * x

    measure = ODEInformationWithHidden(vf, joint.E0_x, joint.E1, joint.E0_hidden)
    D = joint.E0.shape[1]
    m_0 = np.zeros(D).at[0].set(1.0).at[1].set(0.5).at[6].set(0.7)
    return joint, m_0, 1e-6 * np.eye(D), measure


@pytest.mark.parametrize("kind", ["single", "joint"])
@pytest.mark.parametrize("calibration", CALIBRATIONS)
def test_preconditioning_leaves_calibrated_runs_unchanged(kind, calibration):
    """Preconditioning is a change of variables: trajectory and diffusion agree."""
    setups = (
        [_logistic(IWP), _logistic(PrecondIWP)]
        if kind == "single"
        else [_joint(IWP), _joint(PrecondIWP)]
    )
    ts = np.linspace(0.0, 2.0, 31)
    plain, precond = (
        gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration=calibration)
        for prior, m_0, P_0_sqr, measure in setups
    )
    assert np.allclose(plain.m, precond.m, atol=1e-8)
    assert np.allclose(plain.sigma_sqr, precond.sigma_sqr, atol=1e-8)


@pytest.mark.parametrize("calibration", ["diagonal", "diagonal_ekf0"])
def test_diagonal_calibration_resolves_a_multiscale_problem(calibration):
    def vf(x, *, t):
        return 2.0 * x * (1 - x)

    m_0, P_0_sqr = taylor_mode_initialization(vf, np.asarray([1e-5, 1e-10]), q=3)
    ts = np.linspace(0.0, 15.0, 501)
    results = []
    for prior in (IWP(q=3, d=2), PrecondIWP(q=3, d=2)):
        measure = ODEInformation(vf, prior.E0, prior.E1)
        results.append(
            gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration=calibration)
        )
    plain, precond = results
    assert np.allclose(plain.m[-1], precond.m[-1], atol=1e-8)
    assert abs(float(plain.m[-1, 1]) - 0.99907) < 1e-3  # the true x_2(15)


@pytest.mark.parametrize("calibration", CALIBRATIONS)
def test_differentiable_in_the_initial_state_and_in_xi(calibration):
    _, m_0, P_0_sqr, _ = _logistic()
    ts = np.linspace(0.0, 1.0, 6)

    def by_mean(m):
        prior = IWP(q=2, d=1)
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        res = gaussian_filter(m, P_0_sqr, prior, measure, ts, calibration=calibration)
        return res.log_likelihood

    def by_xi(xi_diag):
        prior = IWP(q=2, d=1, Xi=np.diag(xi_diag))
        measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
        res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration=calibration)
        return res.log_likelihood

    assert_grad_matches_fd_vector(by_mean, m_0)
    assert_grad_matches_fd(lambda xi: by_xi(xi[None]), np.array(1.0))


def test_a_concrete_non_diagonal_xi_is_still_rejected():
    prior = IWP(q=2, d=2, Xi=np.array([[1.0, 0.3], [0.3, 1.0]]))
    measure = ODEInformation(lambda x, *, t: -x, prior.E0, prior.E1)
    D = prior.E0.shape[1]
    with pytest.raises(ValueError, match="Xi to be diagonal"):
        gaussian_filter(
            np.zeros(D), np.eye(D), prior, measure, TS[:6], calibration="diagonal"
        )


def test_conservation_rows_do_not_drive_the_diffusion():
    def vf_sir(x, *, t, beta=0.5, gamma=0.1):
        return np.array(
            [-beta * x[0] * x[1], beta * x[0] * x[1] - gamma * x[1], gamma * x[1]]
        )

    prior = IWP(q=2, d=3)
    m_0, P_0_sqr = taylor_mode_initialization(vf_sir, np.array([0.99, 0.01, 0.0]), q=2)
    measure = ODEInformation(
        vf_sir,
        prior.E0,
        prior.E1,
        constraints=[Conservation(np.array([[1.0, 1.0, 1.0]]), np.array([1.0]))],
    )
    ts = np.linspace(0.0, 2.0, 31)
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts)
    Q_sqr = prior.Q_sqr(float(ts[1] - ts[0]))
    d = measure.ode_dim
    for n in (0, 15, 29):
        H, c = measure.linearize(res.m_pred[n], t=ts[n + 1])
        residual = (H @ res.m_pred[n] + c)[:d]
        expected = quasi_mle_sigma_sqr_from_Q(residual, H[:d], Q_sqr)
        assert np.allclose(res.sigma_sqr[n], expected, rtol=1e-8)
