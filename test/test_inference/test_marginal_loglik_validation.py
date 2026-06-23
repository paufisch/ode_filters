"""Error-path coverage for marginal_loglik (observations required)."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters import InferenceProblem, marginal_loglik
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP


def test_marginal_loglik_requires_data():
    prior = IWP(q=2, d=1)

    def build(theta):
        measure = ODEInformation(lambda x, *, t: -x, prior.E0, prior.E1)
        return np.array([1.0, 0.0, 0.0]), np.eye(3), measure

    problem = InferenceProblem(build=build, prior=prior, tspan=(0.0, 1.0), N=10)
    with pytest.raises(ValueError, match="requires observations"):
        marginal_loglik({}, None, model=problem)
