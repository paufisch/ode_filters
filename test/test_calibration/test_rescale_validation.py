"""Error-path coverage for ode_filters.calibration.rescale (prior-aware path)."""

from __future__ import annotations

import jax.numpy as np
import pytest

from ode_filters.calibration.rescale import rescale_sqr_seq
from ode_filters.priors.gmp_priors import IWP


def test_prior_aware_rescale_seq_bad_shape_raises():
    """A prior-aware sequence rescale with a mismatched sigma vector must raise."""
    prior = IWP(q=2, d=1)
    P_seq = np.zeros((3, 3, 3))  # 3 steps
    # 1D sigma whose length != number of steps falls through to the validation.
    with pytest.raises(ValueError, match="sigma_sqr must be scalar"):
        rescale_sqr_seq(P_seq, np.zeros(2), prior=prior)
