"""Solver outputs must match the recorded reference values.

See ``reference/scenarios.py`` for what is recorded and ``reference/README.md``
for the changes made to the recorded values since they were first taken.
"""

from __future__ import annotations

from pathlib import Path

import numpy as onp
import pytest

from test.helpers import assert_close
from test.integration.reference.scenarios import SCENARIOS

REFERENCE = onp.load(Path(__file__).parent / "reference" / "reference_outputs.npz")

# Relative to the largest entry of each recorded array (see ``assert_close``).
RTOL = 1e-9


@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_matches_reference(name):
    outputs = SCENARIOS[name]()
    recorded = {
        key.split("/", 1)[1]: REFERENCE[key]
        for key in REFERENCE.files
        if key.split("/", 1)[0] == name
    }
    assert set(outputs) == set(recorded), (sorted(outputs), sorted(recorded))
    for field, expected in recorded.items():
        try:
            assert_close(outputs[field], expected, rtol=RTOL)
        except AssertionError as err:
            raise AssertionError(f"{name}/{field}: {err}") from None
