"""The example scripts that sample with NumPyro and BlackJAX.

A short NUTS run of each script over the decay rate: the chain is finite and the
posterior mean lands near the truth. Counts are small and tolerances loose to
avoid flakiness; the runs are marked ``slow`` (deselect with ``-m "not slow"``).
"""

from __future__ import annotations

import importlib.util
import pathlib
from types import ModuleType

import jax.numpy as np
import pytest

# The example scripts live in <repo>/examples (not an installed package).
_EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"

LAM_TRUE = 0.8


def _load_example(name: str) -> ModuleType:
    """Import ``examples/<name>.py`` without putting the directory on sys.path."""
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.slow
@pytest.mark.standard_promotion
def test_blackjax_samples_recover_rate():
    pytest.importorskip("blackjax")
    samples = _load_example("blackjax_inference").run(
        num_warmup=80, num_samples=150, seed=0
    )
    assert samples.shape == (150,)
    assert np.all(np.isfinite(samples))
    assert np.all(samples > 0.0)  # softplus parameterization
    assert abs(float(np.mean(samples)) - LAM_TRUE) < 0.3


@pytest.mark.slow
@pytest.mark.standard_promotion
def test_numpyro_samples_recover_rate():
    pytest.importorskip("numpyro")
    samples = _load_example("numpyro_inference").run(
        num_warmup=80, num_samples=150, seed=0
    )
    assert samples.shape == (150,)
    assert np.all(np.isfinite(samples))
    assert np.all(samples > 0.0)  # LogNormal prior
    assert abs(float(np.mean(samples)) - LAM_TRUE) < 0.3
