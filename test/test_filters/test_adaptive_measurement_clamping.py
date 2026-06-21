"""Observations assimilated at their save-at times by the adaptive filter.

The adaptive solver lands exactly on every ``save_at`` time, and external
observations (built with :func:`prepare_observations`, aligned to ``save_at``)
are assimilated at those times. These end-to-end tests check that a precise
observation at its time visibly shapes the filtered posterior.

The mechanics of the (removed) embedded-``Measurement``-in-measure clamping --
``measure.measurement_times()`` snapping, ``find_index`` tolerance, the Python
``sqr_adaptive_loop`` landing on embedded-measurement times -- have no
analog in the public save-at API and are covered for the obs-model path by
``test/test_filters/test_adaptive_solve.py`` (the ``test_obs_model_*`` tests).
"""

from __future__ import annotations

import jax.numpy as np

from ode_filters import (
    IWP,
    JointPrior,
    ODEInformation,
    gaussian_filter_adaptive,
    prepare_observations,
    taylor_mode_initialization,
)
from ode_filters.measurement.measurement_models import (
    Measurement,
    ODEInformationWithHidden,
)


def logistic_vf(x, *, t):
    return x * (1 - x)


def _setup_logistic():
    """Build a small logistic-ODE filter problem."""
    prior = IWP(q=2, d=1)
    x0 = np.array([0.1])
    mu_0, Sigma_0_sqr = taylor_mode_initialization(logistic_vf, x0, q=2)
    measure = ODEInformation(logistic_vf, prior.E0, prior.E1)
    return prior, mu_0, Sigma_0_sqr, measure


class TestMeasurementApplied:
    def test_observation_shifts_posterior_at_its_time(self):
        """A precise observation at a save time pulls the filtered state to it.

        Off-trajectory observation: logistic(t=0.4, x0=0.1) is ~0.142, so an
        observation of 0.6 with tight noise must pull the posterior up at that
        time relative to the unobserved (free) solve on the same grid.
        """
        prior, mu_0, S0, measure = _setup_logistic()
        save_at = np.linspace(0.0, 1.0, 6)  # save_at[2] == 0.4
        t_idx = 2

        free = gaussian_filter_adaptive(
            mu_0, S0, prior, measure, save_at, atol=1e-2, rtol=1e-2, calibration="none"
        )

        meas = Measurement(
            A=np.array([[1.0]]),
            z=np.array([[0.6]]),
            z_t=np.array([float(save_at[t_idx])]),
            noise=1e-8,
        )
        obs = prepare_observations([meas], prior.E0, save_at)
        pulled = gaussian_filter_adaptive(
            mu_0,
            S0,
            prior,
            measure,
            save_at,
            obs_model=obs,
            atol=1e-2,
            rtol=1e-2,
            calibration="none",
        )

        x_free = float((prior.E0 @ free.m[t_idx])[0])
        x_pulled = float((prior.E0 @ pulled.m[t_idx])[0])
        # The free trajectory is well below 0.6; the observation pulls it up.
        assert x_pulled > x_free + 0.05, (
            f"observation did not shift posterior: free={x_free:.4f} "
            f"pulled={x_pulled:.4f}"
        )
        assert abs(x_pulled - 0.6) < 0.05, (
            f"posterior not assimilated toward 0.6: pulled={x_pulled:.4f}"
        )


class TestJointPriorComposition:
    def test_observation_assimilated_with_joint_prior(self):
        """Assimilation must work inside the joint-prior adaptive path too."""
        prior_x = IWP(q=2, d=2)
        prior_u = IWP(q=0, d=1, Xi=1e-3 * np.eye(1))
        joint = JointPrior(prior_x, prior_u)

        def vf_hidden(x, u, *, t):
            return -u[0] * x

        measure = ODEInformationWithHidden(
            vf_hidden,
            E0=joint.E0_x,
            E1=joint.E1,
            E0_hidden=joint.E0_hidden,
        )
        D = joint.E0.shape[1]
        mu_0 = np.zeros(D).at[0].set(1.0).at[1].set(0.5).at[6].set(0.7)
        S0 = 1e-6 * np.eye(D)

        save_at = np.linspace(0.0, 1.0, 6)  # save_at[2] == 0.4
        t_idx = 2

        free = gaussian_filter_adaptive(
            mu_0, S0, joint, measure, save_at, atol=1e-4, rtol=1e-2, calibration="none"
        )

        # Observe the first ODE component; E0_x extracts the length-2 ODE state.
        meas = Measurement(
            A=np.array([[1.0, 0.0]]),  # observe first component of ODE state
            z=np.array([[0.5]]),
            z_t=np.array([float(save_at[t_idx])]),
            noise=1e-6,
        )
        obs = prepare_observations([meas], joint.E0_x, save_at)
        pulled = gaussian_filter_adaptive(
            mu_0,
            S0,
            joint,
            measure,
            save_at,
            obs_model=obs,
            atol=1e-4,
            rtol=1e-2,
            calibration="none",
        )

        x_free = float((joint.E0_x @ free.m[t_idx])[0])
        x_pulled = float((joint.E0_x @ pulled.m[t_idx])[0])
        # Free value is ~0.97; the observation of 0.5 pulls the component down.
        assert abs(x_pulled - 0.5) < abs(x_free - 0.5) - 0.1, (
            f"observation not assimilated under joint prior: "
            f"free={x_free:.4f} pulled={x_pulled:.4f}"
        )
