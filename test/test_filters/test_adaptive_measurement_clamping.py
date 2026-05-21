"""Adaptive loop must land on fixed Measurement times.

The adaptive step controller decides h purely from the local ODE-defect
error and has no view of where Measurement constraints fire. Without an
extra clamp, large adaptive steps step *past* measurement times and
``Measurement.find_index(t_next)`` returns ``None`` -- silently dropping
the observation. The clamp in ``ekf1_sqr_adaptive_loop`` should land on
each measurement time exactly, like the existing ``t_end`` clamp does.
"""

from __future__ import annotations

import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.filters import ekf1_sqr_adaptive_loop
from ode_filters.measurement.measurement_models import (
    MEASUREMENT_TIME_ATOL,
    MEASUREMENT_TIME_RTOL,
    Conservation,
    Measurement,
    ODEInformation,
    ODEInformationWithHidden,
)
from ode_filters.priors.gmp_priors import (
    IWP,
    JointPrior,
    taylor_mode_initialization,
)


def logistic_vf(x, *, t):
    return x * (1 - x)


def _setup_logistic(constraints=None):
    """Build a small logistic-ODE filter problem with optional constraints."""
    prior = IWP(q=2, d=1)
    x0 = np.array([0.1])
    mu_0, Sigma_0_sqr = taylor_mode_initialization(logistic_vf, x0, q=2)
    measure = ODEInformation(logistic_vf, prior.E0, prior.E1, constraints=constraints)
    return prior, mu_0, Sigma_0_sqr, measure


def _t_in_seq(t_seq, t_target):
    """True if t_target appears in t_seq within find_index tolerance."""
    diffs = onp.abs(onp.asarray(t_seq) - t_target)
    return bool(
        onp.any(diffs <= MEASUREMENT_TIME_ATOL + MEASUREMENT_TIME_RTOL * abs(t_target))
    )


# ---------------------------------------------------------------------------
# 1. Clamping fires: adaptive step lands on the measurement time.
# ---------------------------------------------------------------------------


class TestClampFires:
    def test_single_measurement_lands_in_t_seq(self):
        """A measurement at a mid-interval irrational t must appear in t_seq."""
        t_m = 0.37  # chosen so that a coarse initial step would step over it
        meas = Measurement(
            A=np.array([[1.0]]),
            z=np.array([[0.6]]),
            z_t=np.array([t_m]),
        )
        prior, mu_0, S0, measure = _setup_logistic(constraints=[meas])

        result = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            (0.0, 1.0),
            h_init=0.5,  # would naturally step from 0.0 to 0.5 past t_m
            atol=1e-4,
            rtol=1e-2,
        )

        assert _t_in_seq(result.t_seq, t_m), (
            f"measurement time {t_m} not found in t_seq: {onp.asarray(result.t_seq)}"
        )

    def test_multiple_measurements_all_land_in_order(self):
        """All measurement times should appear, and stay in order."""
        z_t = onp.array([0.13, 0.41, 0.79])
        meas = Measurement(
            A=np.array([[1.0]]),
            z=np.array([[0.2], [0.55], [0.85]]),
            z_t=np.asarray(z_t),
        )
        prior, mu_0, S0, measure = _setup_logistic(constraints=[meas])

        result = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            prior,
            measure,
            (0.0, 1.0),
            h_init=0.5,
            atol=1e-4,
            rtol=1e-2,
        )

        t_seq = onp.asarray(result.t_seq)
        for t_m in z_t:
            assert _t_in_seq(t_seq, t_m), f"missed measurement at {t_m}"
        # Monotone, no duplicates.
        assert onp.all(onp.diff(t_seq) > 0)


# ---------------------------------------------------------------------------
# 2. The measurement actually shapes the posterior.
# ---------------------------------------------------------------------------


class TestMeasurementApplied:
    def test_posterior_differs_from_predicted_at_measurement_step(self):
        """At a measurement step the posterior must differ from the prior.

        Directly checks that the EKF update applied the measurement at the
        clamped step: ``m_seq[i]`` (posterior) is meaningfully different from
        ``m_pred_seq[i-1]`` (prior pre-update). Avoids comparing two
        adaptive runs (each with its own grid) which is noisier and depends
        on interpolation.
        """
        t_m = 0.37
        # Off-trajectory observation: logistic(0.37, 0.1) is ~0.137, so an
        # observation of 0.6 with tight noise must pull the posterior up.
        meas = Measurement(
            A=np.array([[1.0]]),
            z=np.array([[0.6]]),
            z_t=np.array([t_m]),
            noise=1e-4,
        )
        prior, mu_0, S0, measure = _setup_logistic(constraints=[meas])

        # Loosen the initial prior so the measurement has visible weight.
        S0_loose = 1e-1 * np.eye(S0.shape[0])

        r = ekf1_sqr_adaptive_loop(
            mu_0,
            S0_loose,
            prior,
            measure,
            (0.0, 1.0),
            h_init=0.5,
            atol=1e-4,
            rtol=1e-2,
        )

        t_seq = onp.asarray(r.t_seq)
        # Index of the t_seq entry that landed on t_m (clamp guarantees it).
        i_post = int(onp.argmin(onp.abs(t_seq - t_m)))
        assert _t_in_seq([t_seq[i_post]], t_m)

        # m_pred_seq is indexed by step (length N); m_seq has length N+1, so
        # the prior for the i_post-th t_seq entry is m_pred_seq[i_post - 1].
        x_post = float((prior.E0 @ r.m_seq[i_post])[0])
        x_pre = float((prior.E0 @ r.m_pred_seq[i_post - 1])[0])

        # The update should pull the posterior substantially toward 0.6.
        assert x_post > x_pre + 0.05, (
            f"measurement did not shift posterior: pre={x_pre:.4f} post={x_post:.4f}"
        )


# ---------------------------------------------------------------------------
# 3. No-op when there are no Measurement constraints.
# ---------------------------------------------------------------------------


class TestNoOpWhenEmpty:
    def test_t_seq_when_no_constraints(self):
        """With no Measurement constraints the loop runs unchanged."""
        prior, mu_0, S0, measure = _setup_logistic(constraints=None)
        assert len(measure.measurement_times()) == 0
        result = ekf1_sqr_adaptive_loop(
            mu_0, S0, prior, measure, (0.0, 1.0), atol=1e-4, rtol=1e-2
        )
        assert float(result.t_seq[-1]) == pytest.approx(1.0, rel=1e-12)

    def test_conservation_only_does_not_clamp(self):
        """Conservation constraints are always-on; they expose no fixed times."""
        cons = Conservation(A=np.array([[1.0]]), p=np.array([0.5]))
        _, _, _, measure = _setup_logistic(constraints=[cons])
        assert len(measure.measurement_times()) == 0


# ---------------------------------------------------------------------------
# 4. The cursor stays consistent across rejected steps and tight grids.
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_meas_at_t_start_or_t_end_is_dropped(self):
        """Endpoint measurements are already covered by the endpoint clamp."""
        meas = Measurement(
            A=np.array([[1.0]]),
            z=np.array([[0.1], [0.6]]),
            z_t=np.array([0.0, 1.0]),
        )
        prior, mu_0, S0, measure = _setup_logistic(constraints=[meas])
        result = ekf1_sqr_adaptive_loop(
            mu_0, S0, prior, measure, (0.0, 1.0), atol=1e-4, rtol=1e-2
        )
        # Loop completed; endpoint shows up via the t_end clamp.
        assert float(result.t_seq[-1]) == pytest.approx(1.0, rel=1e-12)

    def test_meas_below_h_min_raises_with_clear_message(self):
        """When the measurement clamp drives h below h_min, error explains why."""
        # Place a measurement 0.005 from t_start, h_min=0.1.
        meas = Measurement(
            A=np.array([[1.0]]),
            z=np.array([[0.5]]),
            z_t=np.array([0.005]),
        )
        prior, mu_0, S0, measure = _setup_logistic(constraints=[meas])
        with pytest.raises(RuntimeError, match="measurement time"):
            ekf1_sqr_adaptive_loop(
                mu_0,
                S0,
                prior,
                measure,
                (0.0, 1.0),
                h_init=0.5,
                h_min=0.1,
                atol=1e-4,
                rtol=1e-2,
            )


# ---------------------------------------------------------------------------
# 5. Compose with JointPrior (the user's active work area).
# ---------------------------------------------------------------------------


class TestJointPriorComposition:
    def test_measurement_lands_with_joint_prior(self):
        """Clamping must work inside the joint-prior adaptive path too."""
        prior_x = IWP(q=2, d=2)
        prior_u = IWP(q=0, d=1, Xi=1e-3 * np.eye(1))
        joint = JointPrior(prior_x, prior_u)

        def vf_hidden(x, u, *, t):
            return -u[0] * x

        # Measurement on the (length-2) ODE state at a mid-interval time.
        t_m = 0.41
        meas = Measurement(
            A=np.array([[1.0, 0.0]]),  # observe first component
            z=np.array([[0.5]]),
            z_t=np.array([t_m]),
            noise=1e-4,
        )

        measure = ODEInformationWithHidden(
            vf_hidden,
            E0=joint.E0_x,
            E1=joint.E1,
            E0_hidden=joint.E0_hidden,
            constraints=[meas],
        )
        D = joint.E0.shape[1]
        mu_0 = np.zeros(D).at[0].set(1.0).at[1].set(0.5).at[6].set(0.7)
        S0 = 1e-6 * np.eye(D)

        result = ekf1_sqr_adaptive_loop(
            mu_0,
            S0,
            joint,
            measure,
            (0.0, 1.0),
            h_init=0.5,
            atol=1e-4,
            rtol=1e-2,
        )
        assert _t_in_seq(result.t_seq, t_m)
