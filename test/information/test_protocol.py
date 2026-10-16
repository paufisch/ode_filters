"""The measurement-model protocols: the required and the optional members.

A custom model needs the members of ``MeasurementModel`` only. Each optional
member is checked where its feature calls it: a model that declares it works
with that feature, and a model that lacks it fails with an error naming it.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import (
    IWP,
    BlackBoxMeasurement,
    DiagonalCalibration,
    MeasurementModel,
    ODEInformation,
    SecondOrderODEInformation,
    TaylorCorrection,
    TransformedMeasurement,
    gaussian_filter,
    taylor_mode_initialization,
)
from ode_filters.information import (
    EK0Linearizable,
    HasLHSSelector,
    HasNoiseCovariance,
    NoiseSettable,
)
from test.helpers import assert_close

OPTIONAL = (HasLHSSelector, EK0Linearizable, HasNoiseCovariance, NoiseSettable)
PRIOR = IWP(q=2, d=1)


def logistic(x, *, t):
    return x * (1.0 - x)


REFERENCE = ODEInformation(logistic, PRIOR.E0, PRIOR.E1)


class MinimalModel:
    """The logistic residual ``E1 Y - f(E0 Y)`` with the required members only."""

    ode_dim = 1

    def g(self, state, *, t):
        return PRIOR.E1 @ state - logistic(PRIOR.E0 @ state, t=t)

    def jacobian_g(self, state, *, t):
        return jax.jacfwd(lambda s: self.g(s, t=t))(state)

    def linearize(self, state, *, t):
        H = self.jacobian_g(state, t=t)
        return H, self.g(state, t=t) - H @ state

    def noise_sqr(self, *, t):
        return np.zeros((1, 1))


class WithLHSSelector(MinimalModel):
    E_lhs = PRIOR.E1


class WithEK0(MinimalModel):
    def linearize_ek0(self, state, *, t):
        return PRIOR.E1, self.g(state, t=t) - PRIOR.E1 @ state


class WithNoiseCovariance(MinimalModel):
    R = 0.5 * np.eye(1)


def solve(measure, **kwargs):
    m_0, P_0_sqr = taylor_mode_initialization(logistic, np.array([0.1]), q=2)
    ts = np.linspace(0.0, 2.0, 21)
    return gaussian_filter(m_0, P_0_sqr, PRIOR, measure, ts, **kwargs)


@pytest.mark.parametrize(
    "model",
    [
        REFERENCE,
        SecondOrderODEInformation(lambda x, v, *, t: -x, PRIOR.E0, PRIOR.E1, PRIOR.E2),
    ],
    ids=["first_order", "second_order"],
)
def test_ode_information_implements_every_protocol(model):
    assert isinstance(model, MeasurementModel)
    for protocol in OPTIONAL:
        assert isinstance(model, protocol), protocol.__name__


@pytest.mark.parametrize(
    "model",
    [
        BlackBoxMeasurement(lambda s, *, t: s[:1], state_dim=3, obs_dim=1),
        TransformedMeasurement(REFERENCE, lambda s: s),
    ],
    ids=["black_box", "transformed"],
)
def test_black_box_and_transformed_models_declare_their_noise_only(model):
    assert isinstance(model, MeasurementModel)
    assert isinstance(model, HasNoiseCovariance)
    assert isinstance(model, NoiseSettable)
    assert not isinstance(model, HasLHSSelector)
    assert not isinstance(model, EK0Linearizable)


def test_a_model_with_the_required_members_only_runs_like_ode_information():
    model = MinimalModel()
    assert isinstance(model, MeasurementModel)
    assert not any(isinstance(model, protocol) for protocol in OPTIONAL)
    got, want = solve(model), solve(REFERENCE)
    assert_close(got.m, want.m, rtol=1e-12)
    assert_close(got.log_likelihood, want.log_likelihood, rtol=1e-12)


def test_a_missing_optional_member_raises_an_error_that_names_it():
    model = MinimalModel()
    with pytest.raises(TypeError, match="has no linearize_ek0"):
        solve(model, correction=TaylorCorrection(order=0))
    with pytest.raises(AttributeError, match="has no 'E_lhs'"):
        solve(model, calibration=DiagonalCalibration(jacobian="ek0"))
    transformed = TransformedMeasurement(model, lambda s: s)
    with pytest.raises(AttributeError, match="base has no 'R'"):
        _ = transformed.R
    with pytest.raises(AttributeError, match="base has no 'with_noise'"):
        transformed.with_noise(0.1)


def test_one_optional_member_is_enough_for_the_feature_that_calls_it():
    """Each feature checks only the member that it calls."""
    ek0 = TaylorCorrection(order=0)
    assert isinstance(WithEK0(), EK0Linearizable)
    assert not isinstance(WithEK0(), HasLHSSelector)
    assert_close(solve(WithEK0(), correction=ek0).m, solve(REFERENCE, correction=ek0).m)

    diagonal = DiagonalCalibration(jacobian="ek0")
    assert isinstance(WithLHSSelector(), HasLHSSelector)
    assert not isinstance(WithLHSSelector(), EK0Linearizable)
    assert_close(
        solve(WithLHSSelector(), calibration=diagonal).sigma_sqr,
        solve(REFERENCE, calibration=diagonal).sigma_sqr,
    )

    transformed = TransformedMeasurement(WithNoiseCovariance(), lambda s: s)
    assert not isinstance(WithNoiseCovariance(), NoiseSettable)
    assert_close(transformed.R, WithNoiseCovariance.R, rtol=0.0)
    with pytest.raises(AttributeError, match="base has no 'with_noise'"):
        transformed.with_noise(0.1)
