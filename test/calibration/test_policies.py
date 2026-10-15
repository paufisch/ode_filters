"""The online calibration policies and their floors."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    DiagonalCalibration,
    DynamicCalibration,
    JointPrior,
    NoCalibration,
    ODEInformation,
    gaussian_filter,
    gaussian_filter_adaptive,
    taylor_mode_initialization,
)
from ode_filters.calibration import as_calibration
from test.helpers import assert_close

TS = np.linspace(0.0, 1.0, 11)


def _at_rest():
    """``x' = -x`` started at its equilibrium: the residual is identically zero."""

    def vf(x, *, t):
        return -x

    prior = IWP(q=2, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.0]), q=2)
    return prior, ODEInformation(vf, prior.E0, prior.E1), m_0, P_0_sqr


def test_without_a_floor_the_diffusion_collapses_at_an_equilibrium():
    prior, measure, m_0, P_0_sqr = _at_rest()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    assert onp.all(onp.asarray(res.sigma_sqr) == 0.0)


@pytest.mark.parametrize(
    "calibration",
    [DynamicCalibration(min_sigma_sqr=1e-30), DiagonalCalibration(min_sigma_sqr=1e-30)],
    ids=["dynamic", "diagonal"],
)
def test_a_floor_keeps_the_diffusion_positive(calibration):
    prior, measure, m_0, P_0_sqr = _at_rest()
    res = gaussian_filter(m_0, P_0_sqr, prior, measure, TS, calibration=calibration)
    assert onp.all(onp.asarray(res.sigma_sqr) >= 1e-30)
    assert bool(np.all(np.isfinite(res.m))) and bool(np.all(np.isfinite(res.P_sqr)))


def test_a_floor_on_the_adaptive_path():
    prior, measure, m_0, P_0_sqr = _at_rest()
    res = gaussian_filter_adaptive(
        m_0,
        P_0_sqr,
        prior,
        measure,
        TS[::5],
        calibration=DynamicCalibration(min_sigma_sqr=1e-30),
        record_steps=50,
    )
    n = int(res.steps.count)
    assert onp.all(onp.asarray(res.steps.sigma_sqr[:n]) >= 1e-30)
    assert bool(res.success)


def test_the_floor_is_opt_in():
    def vf(x, *, t):
        return x * (1 - x)

    prior = IWP(q=2, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.1]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    default = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    zero = gaussian_filter(
        m_0,
        P_0_sqr,
        prior,
        measure,
        TS,
        calibration=DynamicCalibration(min_sigma_sqr=0.0),
    )
    assert_close(zero.m, default.m, rtol=0.0)


def test_no_calibration_reports_but_does_not_apply_the_estimate():
    def vf(x, *, t):
        return x * (1 - x)

    prior = IWP(q=2, d=1)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.1]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    none = gaussian_filter(
        m_0, P_0_sqr, prior, measure, TS, calibration=NoCalibration()
    )
    dynamic = gaussian_filter(m_0, P_0_sqr, prior, measure, TS)
    assert bool(np.all(none.sigma_sqr > 0.0))
    assert not np.allclose(none.P_sqr, dynamic.P_sqr)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("none", NoCalibration()),
        ("dynamic", DynamicCalibration()),
        ("diagonal", DiagonalCalibration()),
        ("diagonal_ekf0", DiagonalCalibration(jacobian="ek0")),
    ],
)
def test_string_shorthands(name, expected):
    assert as_calibration(name) == expected
    assert as_calibration(expected) is expected


@pytest.mark.wrong_input_types
def test_invalid_specifications_raise():
    with pytest.raises(ValueError, match="calibration must be"):
        as_calibration("dynamc")
    with pytest.raises(TypeError, match="calibration must be"):
        as_calibration(1.0)
    with pytest.raises(ValueError, match="jacobian"):
        DiagonalCalibration(jacobian="ek2")


def test_diagonal_policy_requires_a_diagonal_state_xi():
    coupled = IWP(q=2, d=2, Xi=np.array([[1.0, 0.5], [0.5, 1.0]]))
    with pytest.raises(ValueError, match="diagonal"):
        DiagonalCalibration().validate(coupled)
    # Only the state block counts for a joint prior.
    joint = JointPrior(
        IWP(q=2, d=1), IWP(q=1, d=2, Xi=np.array([[1.0, 0.5], [0.5, 1.0]]))
    )
    DiagonalCalibration().validate(joint)


def test_diagonal_validation_holds_under_jit():
    """Under jit Xi is traced; the flag recorded at construction still counts."""
    coupled = IWP(q=1, d=2, Xi=np.array([[1.0, 0.5], [0.5, 1.0]]))
    assert coupled.xi_diagonal is False
    with pytest.raises(ValueError, match="diagonal"):
        eqx.filter_jit(DiagonalCalibration().validate)(coupled)
    eqx.filter_jit(DiagonalCalibration().validate)(IWP(q=1, d=2))  # diagonal: fine
    # For a joint prior only the state block counts, also under jit.
    joint = JointPrior(
        IWP(q=2, d=1), IWP(q=1, d=2, Xi=np.array([[1.0, 0.5], [0.5, 1.0]]))
    )
    assert joint.xi_state_diagonal is True
    eqx.filter_jit(DiagonalCalibration().validate)(joint)


def test_diagonal_validation_trusts_a_prior_built_from_a_traced_xi():
    def check(scale):
        prior = IWP(q=1, d=2, Xi=scale * np.ones((2, 2)))
        assert prior.xi_diagonal is None  # unknown
        DiagonalCalibration().validate(prior)
        return scale

    jax.jit(check)(1.0)  # does not raise: the traced Xi cannot be inspected
