"""Statistical linear regression of the ODE residual, against analytic oracles.

The load-bearing tests are the *exactness ladder*: for a cubic vector field
every SLR quantity has a closed form, and a Gauss-Hermite rule of the matching
degree must reproduce it to machine precision. Because the closed form is
derived independently of the quadrature code (see ``_cubic_slr_reference``),
this pins the implementation rather than merely checking self-consistency.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters.filters.statistical_linearization import (
    MAX_NODES,
    check_arg_projection,
    cubature_rule,
    gauss_hermite_rule,
    quadrature_nodes,
    slr_linearize,
)
from ode_filters.measurement import ODEInformation
from ode_filters.measurement.measurement_models import (
    BlackBoxMeasurement,
    Conservation,
    ODEInformationWithHidden,
    SecondOrderODEInformation,
    SecondOrderODEInformationWithHidden,
)
from ode_filters.priors import IWP

A_LIN, EPS = -0.7, 0.35


def cubic_vf(x, *, t):
    """Scalar cubic field: the case where every Gaussian moment is closed-form."""
    return A_LIN * x + EPS * x**3


def affine_vf(x, *, t):
    return A_LIN * x


def _random_gaussian(state_dim: int, seed: int = 0):
    m = jax.random.normal(jax.random.PRNGKey(seed), (state_dim,)) * 0.8
    L = jax.random.normal(jax.random.PRNGKey(seed + 1), (state_dim, state_dim)) * 0.4
    P_sqr = np.linalg.qr(L)[1]
    return m, P_sqr


def _cubic_slr_reference(prior, m, P_sqr):
    """Closed-form SLR of ``E1 X - (a E0 X + eps (E0 X)^3)`` under ``N(m, P)``.

    Derived from the Gaussian moments of a cubic, independently of the
    quadrature implementation. ``Omega`` reproduces the published
    ``6 eps^2 P^2 (3 m^2 + P)``.
    """
    P = P_sqr.T @ P_sqr
    m_x = (prior.E0 @ m)[0]
    P_xx = (prior.E0 @ P @ prior.E0.T)[0, 0]

    E_jac = A_LIN + 3 * EPS * (m_x**2 + P_xx)
    H = prior.E1 - E_jac * prior.E0
    E_g = prior.E1 @ m - (A_LIN * m_x + EPS * (m_x**3 + 3 * m_x * P_xx))
    c = E_g - H @ m
    Omega = 6 * EPS**2 * P_xx**2 * (3 * m_x**2 + P_xx)
    return H, c, np.array([[Omega]])


# ---------------------------------------------------------------------------
# Quadrature rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("n_nodes", [2, 3, 4])
def test_gauss_hermite_moments(dim, n_nodes):
    """The rule must reproduce the standard-normal moments it claims to."""
    nodes, weights = gauss_hermite_rule(dim, n_nodes)
    assert nodes.shape == (n_nodes**dim, dim)
    assert bool(np.all(weights > 0))
    assert float(np.sum(weights)) == pytest.approx(1.0)
    assert onp.allclose(weights @ nodes, 0.0, atol=1e-12)
    cov = np.einsum("k,ki,kj->ij", weights, nodes, nodes)
    assert onp.allclose(cov, np.eye(dim), atol=1e-12)


@pytest.mark.parametrize("dim", [1, 2, 4])
def test_cubature_moments(dim):
    nodes, weights = cubature_rule(dim)
    assert nodes.shape == (2 * dim, dim)
    assert bool(np.all(weights > 0))
    assert float(np.sum(weights)) == pytest.approx(1.0)
    cov = np.einsum("k,ki,kj->ij", weights, nodes, nodes)
    assert onp.allclose(cov, np.eye(dim), atol=1e-12)


def test_gauss_hermite_exactness_degree():
    """``n`` nodes integrate degree ``2n - 1`` exactly and degree ``2n`` not."""
    nodes, weights = gauss_hermite_rule(1, 2)
    x = nodes[:, 0]
    assert float(weights @ x**3) == pytest.approx(0.0, abs=1e-12)  # degree 3: exact
    assert float(weights @ x**4) != pytest.approx(3.0, abs=1e-6)  # degree 4: not
    nodes, weights = gauss_hermite_rule(1, 4)
    x = nodes[:, 0]
    assert float(weights @ x**6) == pytest.approx(15.0)  # degree 6: exact


def test_quadrature_nodes_dispatch():
    gh = quadrature_nodes("gauss_hermite", 2, 3)
    assert gh[0].shape == (9, 2)
    cub = quadrature_nodes("cubature", 2, 3)  # n_nodes ignored
    assert cub[0].shape == (4, 2)


def test_quadrature_rule_errors():
    with pytest.raises(ValueError, match="gauss_hermite"):
        quadrature_nodes("unscented", 2, 3)
    with pytest.raises(ValueError, match="n_nodes must be >= 1"):
        gauss_hermite_rule(2, 0)
    with pytest.raises(ValueError, match=f"MAX_NODES={MAX_NODES}"):
        gauss_hermite_rule(8, 5)


# ---------------------------------------------------------------------------
# The exactness ladder against the analytic cubic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_nodes", [2, 3, 4, 5])
def test_cubic_H_and_c_exact_from_two_nodes(n_nodes):
    """``H`` and ``c`` need degree 3, so two Gauss-Hermite nodes already suffice."""
    prior = IWP(q=3, d=1)
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1)
    m, P_sqr = _random_gaussian(prior.E0.shape[1])
    H_ref, c_ref, _ = _cubic_slr_reference(prior, m, P_sqr)

    model = slr_linearize(measure, m, P_sqr, t=0.3, n_nodes=n_nodes)
    assert onp.allclose(model.H, H_ref, atol=1e-12)
    assert onp.allclose(model.c, c_ref, atol=1e-12)


def test_cubic_omega_needs_four_nodes():
    """``Omega`` involves the sixth moment, so it needs ``2n - 1 >= 6``."""
    prior = IWP(q=3, d=1)
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1)
    m, P_sqr = _random_gaussian(prior.E0.shape[1])
    _, _, Omega_ref = _cubic_slr_reference(prior, m, P_sqr)

    def omega(n_nodes):
        model = slr_linearize(measure, m, P_sqr, t=0.3, n_nodes=n_nodes)
        return model.Omega_sqr.T @ model.Omega_sqr

    # Degree-3 rules get the mean and gain right but not the residual variance.
    assert not onp.allclose(omega(2), Omega_ref, atol=1e-8)
    assert not onp.allclose(omega(3), Omega_ref, atol=1e-8)
    # Degree 7 >= 6: exact.
    assert onp.allclose(omega(4), Omega_ref, atol=1e-12)
    assert onp.allclose(omega(5), Omega_ref, atol=1e-12)


def test_cubature_matches_degree_three_rule():
    """Spherical cubature is third-degree: exact ``H``/``c``, inexact ``Omega``."""
    prior = IWP(q=3, d=1)
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1)
    m, P_sqr = _random_gaussian(prior.E0.shape[1])
    H_ref, c_ref, Omega_ref = _cubic_slr_reference(prior, m, P_sqr)

    model = slr_linearize(measure, m, P_sqr, t=0.3, rule="cubature")
    assert onp.allclose(model.H, H_ref, atol=1e-12)
    assert onp.allclose(model.c, c_ref, atol=1e-12)
    Omega = model.Omega_sqr.T @ model.Omega_sqr
    assert float(Omega[0, 0]) > 0.0
    assert not onp.allclose(Omega, Omega_ref, atol=1e-8)


def test_R_eff_is_R_plus_omega():
    """The effective noise square root must factor ``R + Omega`` exactly."""
    prior = IWP(q=3, d=1)
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1)
    measure.R = 1e-4
    m, P_sqr = _random_gaussian(prior.E0.shape[1])

    model = slr_linearize(measure, m, P_sqr, t=0.3, n_nodes=4)
    Omega = model.Omega_sqr.T @ model.Omega_sqr
    R_sqr = measure.get_noise(t=0.3)
    R_eff = model.R_eff_sqr.T @ model.R_eff_sqr
    assert onp.allclose(R_eff, R_sqr.T @ R_sqr + Omega, atol=1e-14)
    assert model.R_eff_sqr.shape == (1, 1)


def test_omega_is_psd_and_residuals_are_centred():
    prior = IWP(q=3, d=1)
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1)
    m, P_sqr = _random_gaussian(prior.E0.shape[1], seed=7)
    model = slr_linearize(measure, m, P_sqr, t=0.1, n_nodes=4)

    Omega = model.Omega_sqr.T @ model.Omega_sqr
    assert bool(np.all(np.linalg.eigvalsh(Omega) >= -1e-14))
    # The weighted fit residuals sum to zero, which is what makes Omega a
    # covariance rather than a second moment.
    _, weights = gauss_hermite_rule(1, 4)
    resid = model.Omega_sqr / np.sqrt(weights)[:, None]
    assert onp.allclose(weights @ resid, 0.0, atol=1e-12)


def test_affine_field_is_a_no_op():
    """Omega vanishes and H is the exact Jacobian when the field is affine."""
    prior = IWP(q=3, d=1)
    measure = ODEInformation(affine_vf, prior.E0, prior.E1)
    m, P_sqr = _random_gaussian(prior.E0.shape[1])
    H_ek1, c_ek1 = measure.linearize(m, t=0.3)

    for kwargs in ({"n_nodes": 4}, {"rule": "cubature"}):
        model = slr_linearize(measure, m, P_sqr, t=0.3, **kwargs)
        assert onp.allclose(model.Omega_sqr, 0.0, atol=1e-14)
        assert onp.allclose(model.H, H_ek1, atol=1e-14)
        assert onp.allclose(model.c, c_ek1, atol=1e-14)


def test_zero_spread_recovers_ek1():
    """EK1 is the zero-spread limit of statistical linearization."""
    prior = IWP(q=3, d=1)
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1)
    m, _ = _random_gaussian(prior.E0.shape[1])
    H_ek1, c_ek1 = measure.linearize(m, t=0.3)

    model = slr_linearize(measure, m, np.zeros((4, 4)), t=0.3, n_nodes=4)
    assert onp.allclose(model.H, H_ek1, atol=1e-14)
    assert onp.allclose(model.c, c_ek1, atol=1e-14)
    assert onp.allclose(model.Omega_sqr, 0.0, atol=1e-14)


def test_lift_is_irrelevant_to_the_surrogate():
    """The surrogate must not depend on how nodes are lifted off the subspace.

    The residual is affine in the complement of ``E_args``, so shifting the
    linearization mean along a null-space direction of ``E_args`` may change
    ``c`` (the residual really is different there) but must leave ``H`` and
    ``Omega`` untouched.
    """
    prior = IWP(q=3, d=1)
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1)
    m, P_sqr = _random_gaussian(prior.E0.shape[1])
    shift = np.array([0.0, 0.0, 2.5, -1.0])  # orthogonal to E0

    a = slr_linearize(measure, m, P_sqr, t=0.3, n_nodes=4)
    b = slr_linearize(measure, m + shift, P_sqr, t=0.3, n_nodes=4)
    assert onp.allclose(a.H, b.H, atol=1e-12)
    assert onp.allclose(
        a.Omega_sqr.T @ a.Omega_sqr, b.Omega_sqr.T @ b.Omega_sqr, atol=1e-12
    )


def test_conservation_rows_are_affine_and_carry_no_omega():
    """Conservation constraints are affine, so they contribute nothing to Omega."""
    prior = IWP(q=3, d=1)
    cons = Conservation(A=np.array([[1.0]]), p=np.array([0.5]))
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1, constraints=[cons])
    m, P_sqr = _random_gaussian(prior.E0.shape[1])

    model = slr_linearize(measure, m, P_sqr, t=0.3, n_nodes=4)
    assert model.H.shape == (2, 4)
    Omega = model.Omega_sqr.T @ model.Omega_sqr
    assert onp.allclose(Omega[1, :], 0.0, atol=1e-14)
    assert onp.allclose(Omega[:, 1], 0.0, atol=1e-14)
    assert float(Omega[0, 0]) > 0.0


def test_jit_and_grad_safe():
    prior = IWP(q=3, d=1)
    measure = ODEInformation(cubic_vf, prior.E0, prior.E1)
    m, P_sqr = _random_gaussian(prior.E0.shape[1])

    def total(mu):
        model = slr_linearize(measure, mu, P_sqr, t=0.3, n_nodes=4)
        return np.sum(model.H) + np.sum(model.c) + np.sum(model.Omega_sqr**2)

    assert onp.isfinite(float(jax.jit(total)(m)))
    assert bool(np.all(np.isfinite(jax.grad(total)(m))))


# ---------------------------------------------------------------------------
# E_args declarations
# ---------------------------------------------------------------------------


def test_E_args_declarations_are_correct():
    """Every shipped measurement model must declare a valid argument projection."""
    prior = IWP(q=3, d=1)
    E0, E1 = prior.E0, prior.E1
    E2 = np.eye(4)[2:3]
    Eh = np.eye(4)[3:4]

    models = [
        (ODEInformation(cubic_vf, E0, E1), 1),
        (ODEInformationWithHidden(lambda x, u, *, t: x**3 + u * x**2, E0, E1, Eh), 2),
        (SecondOrderODEInformation(lambda x, v, *, t: x**3 - v**3, E0, E1, E2), 2),
        (
            SecondOrderODEInformationWithHidden(
                lambda x, v, u, *, t: x**3 - v**3 + u**2, E0, E1, E2, Eh
            ),
            3,
        ),
    ]
    state = np.array([0.4, -0.2, 0.9, 0.3])
    for measure, expected_p in models:
        assert measure.E_args.shape == (expected_p, 4)
        # The defining property: the Jacobian depends on the state only through
        # E_args @ state. A too-small E_args would silently break SLR.
        check_arg_projection(measure, state, t=0.2)


def test_check_arg_projection_catches_a_bad_declaration():
    """A projection that misses a nonlinear coordinate must be rejected."""
    prior = IWP(q=3, d=1)

    class BadE0Args(SecondOrderODEInformation):
        @property
        def E_args(self):  # claims x only, but the field also reads v
            return self._E0

    measure = BadE0Args(
        lambda x, v, *, t: x**3 - v**3, prior.E0, prior.E1, np.eye(4)[2:3]
    )
    with pytest.raises(ValueError, match="does not span"):
        check_arg_projection(measure, np.array([0.4, -0.2, 0.9, 0.3]), t=0.0)


def test_check_arg_projection_accepts_full_rank_projection():
    """Nothing to check when the projection already spans the state."""
    measure = BlackBoxMeasurement(
        lambda state, *, t: np.array([state[0] ** 3]), state_dim=2, obs_dim=1
    )
    check_arg_projection(measure, np.array([0.3, -0.4]), t=0.0)  # returns, no raise


def test_falls_back_to_full_state_without_E_args():
    """A duck-typed model without ``E_args`` integrates over the whole state."""
    measure = BlackBoxMeasurement(
        lambda state, *, t: np.array([state[0] ** 3]), state_dim=2, obs_dim=1
    )
    m = np.array([0.3, -0.4])
    P_sqr = np.array([[0.2, 0.05], [0.0, 0.3]])
    model = slr_linearize(measure, m, P_sqr, t=0.0, n_nodes=4)

    P_xx = (P_sqr.T @ P_sqr)[0, 0]
    assert float(model.H[0, 0]) == pytest.approx(3 * (m[0] ** 2 + P_xx))
    assert float(model.H[0, 1]) == pytest.approx(0.0, abs=1e-14)
