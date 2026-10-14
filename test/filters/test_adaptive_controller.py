"""Tests for the step-size controllers."""

from __future__ import annotations

import jax
import jax.numpy as np
import pytest

from ode_filters import PController, PIController, StepSizeController

# ---------------------------------------------------------------------------
# PI controller
# ---------------------------------------------------------------------------


class TestPIController:
    def test_unit_error_keeps_step_near_safety(self):
        ctrl = PIController(order=2)
        h_new = ctrl.propose(h=0.1, err=1.0, err_prev=None)
        assert h_new == pytest.approx(0.1 * 0.9, rel=1e-12)

    def test_shrinks_on_large_error(self):
        ctrl = PIController(order=2)
        h_new = ctrl.propose(h=0.1, err=10.0, err_prev=None)
        assert h_new < 0.1

    def test_grows_on_small_error(self):
        ctrl = PIController(order=2)
        h_new = ctrl.propose(h=0.1, err=0.01, err_prev=None)
        assert h_new > 0.1

    def test_clips_at_max_factor(self):
        ctrl = PIController(order=2, max_factor=2.0)
        # Tiny error would otherwise blow up the step.
        h_new = ctrl.propose(h=0.1, err=1e-12, err_prev=None)
        assert h_new == pytest.approx(0.2, rel=1e-12)

    def test_clips_at_min_factor(self):
        ctrl = PIController(order=2, min_factor=0.1)
        h_new = ctrl.propose(h=0.1, err=1e12, err_prev=None)
        assert h_new == pytest.approx(0.01, rel=1e-12)


# ---------------------------------------------------------------------------
# P controller
# ---------------------------------------------------------------------------


class TestPController:
    def test_unit_error_keeps_step_near_safety(self):
        ctrl = PController(order=2)
        h_new = ctrl.propose(h=0.1, err=1.0)
        assert h_new == pytest.approx(0.1 * 0.9, rel=1e-12)

    def test_shrinks_on_large_error(self):
        ctrl = PController(order=2)
        h_new = ctrl.propose(h=0.1, err=10.0)
        assert h_new < 0.1

    def test_grows_on_small_error(self):
        ctrl = PController(order=2)
        h_new = ctrl.propose(h=0.1, err=0.01)
        assert h_new > 0.1

    def test_ignores_err_prev(self):
        ctrl = PController(order=2)
        h_a = ctrl.propose(h=0.1, err=0.5, err_prev=None)
        h_b = ctrl.propose(h=0.1, err=0.5, err_prev=999.0)
        assert h_a == pytest.approx(h_b, rel=1e-12)

    def test_default_alpha_is_one_over_order(self):
        ctrl = PController(order=4)
        # h_new / (h * safety) = err^(-alpha) = 16^(-1/4) = 0.5
        h_new = ctrl.propose(h=1.0, err=16.0)
        assert h_new == pytest.approx(0.9 * 0.5, rel=1e-12)

    def test_explicit_alpha_override(self):
        ctrl = PController(order=2, alpha=0.5, safety=1.0)
        h_new = ctrl.propose(h=1.0, err=4.0)
        # err^(-0.5) = 0.5
        assert h_new == pytest.approx(0.5, rel=1e-12)

    def test_satisfies_protocol(self):
        assert isinstance(PController(order=2), StepSizeController)
        assert isinstance(PIController(order=2), StepSizeController)

    def test_clips_at_min_and_max_factor(self):
        ctrl = PController(order=2, min_factor=0.25, max_factor=3.0)
        assert ctrl.propose(h=1.0, err=1e12) == pytest.approx(0.25, rel=1e-12)
        assert ctrl.propose(h=1.0, err=1e-12) == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize(
    "ctrl", [PController(order=2), PIController(order=2)], ids=["P", "PI"]
)
def test_controller_propose_is_traceable(ctrl):
    """The jittable adaptive solver calls ``propose`` inside its traced loop."""
    eager = ctrl.propose(0.1, 0.5, 0.8)
    jitted = jax.jit(ctrl.propose)(0.1, 0.5, 0.8)
    assert np.allclose(eager, jitted, rtol=1e-12)


def test_pi_controller_non_positive_err_prev_means_no_memory():
    """Inside traced code "no memory" is a non-positive err_prev, not None."""
    ctrl = PIController(order=2)
    no_memory = ctrl.propose(0.1, 0.5, None)
    assert np.allclose(ctrl.propose(0.1, 0.5, -1.0), no_memory, rtol=1e-12)
    assert not np.allclose(ctrl.propose(0.1, 0.5, 0.8), no_memory)


@pytest.mark.parametrize(
    "ctrl", [PController(order=2), PIController(order=2)], ids=["P", "PI"]
)
def test_controller_shrinks_to_min_factor_on_non_finite_error(ctrl):
    for err in (np.nan, np.inf):
        assert np.allclose(ctrl.propose(1.0, err, None), ctrl.min_factor)
