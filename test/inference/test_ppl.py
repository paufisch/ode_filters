"""Smoke tests for PPL integration (slice 2).

Exercises the example scripts directly: a short NUTS run via BlackJAX and NumPyro
over the decay rate, checking the chain is finite and the posterior mean lands
near the truth. Counts are kept small; tolerances are loose to avoid flakiness.
"""

from __future__ import annotations

import pathlib
import sys

import jax.numpy as np
import pytest

# The example scripts live in <repo>/examples (not an installed package).
_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

LAM_TRUE = 0.8


def test_blackjax_samples_recover_rate():
    pytest.importorskip("blackjax")
    import blackjax_inference

    samples = blackjax_inference.run(num_warmup=80, num_samples=150, seed=0)
    assert samples.shape == (150,)
    assert np.all(np.isfinite(samples))
    assert np.all(samples > 0.0)  # softplus parameterization
    assert abs(float(np.mean(samples)) - LAM_TRUE) < 0.3


def test_numpyro_samples_recover_rate():
    pytest.importorskip("numpyro")
    import numpyro_inference

    samples = numpyro_inference.run(num_warmup=80, num_samples=150, seed=0)
    assert samples.shape == (150,)
    assert np.all(np.isfinite(samples))
    assert np.all(samples > 0.0)  # LogNormal prior
    assert abs(float(np.mean(samples)) - LAM_TRUE) < 0.3
