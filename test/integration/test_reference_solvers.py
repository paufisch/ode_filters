"""Harder problems, checked against outside oracles.

- **Stiff van der Pol** (``eps = 1e-6``) with EK1 and the adaptive solver,
  against SciPy's implicit ``Radau`` at ``rtol = atol = 1e-10``.
- **A DETEST subset** (Hull, Enright, Fellen & Sedgwick 1972): the scalar
  problems A2 and A4 and the orbit problems D1 and D3 against their closed
  forms, the systems B1 and B5 against SciPy's ``DOP853`` at ``1e-13``; with
  the adaptive EK1 at two tolerances. A3, the non-autonomous one, documents
  that the Taylor-mode initialization holds ``t`` fixed.
- **A 20-dimensional linear system** against ``expm``.

All measured numbers quoted below are float64 on this suite's setup.
"""

from __future__ import annotations

from functools import cache

import jax
import jax.numpy as np
import numpy as onp
import pytest
from scipy.integrate import solve_ivp

from ode_filters import (
    IWP,
    ODEInformation,
    PrecondIWP,
    gaussian_filter,
    gaussian_filter_adaptive,
    rts_smoother,
    taylor_mode_initialization,
)

# ---------------------------------------------------------------------------
# Stiff van der Pol against Radau
# ---------------------------------------------------------------------------

VDP_EPS = 1e-6
VDP_SAVE_AT = np.linspace(0.0, 2.0, 41)


def _vdp(y, *, t):
    """Van der Pol in Hairer's scaling: stiffness ratio ~1/eps on the slow arcs."""
    return np.array([y[1], ((1.0 - y[0] ** 2) * y[1] - y[0]) / VDP_EPS])


def _vdp_radau():
    def f(t, y):
        return [y[1], ((1.0 - y[0] ** 2) * y[1] - y[0]) / VDP_EPS]

    def jac(t, y):
        return [
            [0.0, 1.0],
            [(-2.0 * y[0] * y[1] - 1.0) / VDP_EPS, (1.0 - y[0] ** 2) / VDP_EPS],
        ]

    sol = solve_ivp(
        f,
        (0.0, 2.0),
        [2.0, 0.0],
        method="Radau",
        jac=jac,
        t_eval=onp.asarray(VDP_SAVE_AT),
        rtol=1e-10,
        atol=1e-10,
    )
    assert sol.success
    return sol.y.T


def test_stiff_van_der_pol_matches_radau():
    """EK1 (PrecondIWP, q = 5) at ``atol = rtol = 1e-6``.

    Measured: 4898 accepted and 2463 rejected steps; largest error 1.4e-7
    (``y1``) and 1.4e-6 (``y2``) of the component's largest value.
    """
    prior = PrecondIWP(q=5, d=2)
    y0 = np.array([2.0, 0.0])
    m_0, P_0_sqr = taylor_mode_initialization(_vdp, y0, prior.q)
    measure = ODEInformation(_vdp, prior.E0, prior.E1)
    res = jax.jit(
        lambda: gaussian_filter_adaptive(
            m_0,
            P_0_sqr,
            prior,
            measure,
            VDP_SAVE_AT,
            atol=1e-6,
            rtol=1e-6,
            max_steps=20_000,
        )
    )()
    assert bool(res.success)
    ref = _vdp_radau()
    err = onp.abs(onp.asarray(res.m @ prior.E0.T) - ref)
    scaled = err.max(axis=0) / onp.abs(ref).max(axis=0)
    assert scaled.max() < 2e-5, scaled
    assert int(np.sum(res.n_accepted)) < 10_000


# ---------------------------------------------------------------------------
# DETEST non-stiff subset
# ---------------------------------------------------------------------------

DETEST_T = 20.0
DETEST_SAVE_AT = np.linspace(0.0, DETEST_T, 21)


def _a2(y, *, t):
    return -0.5 * y**3


def _a3(y, *, t):
    return y * np.cos(t)


def _a4(y, *, t):
    return 0.25 * y * (1.0 - y / 20.0)


def _b1(y, *, t):
    return np.array([2.0 * (y[0] - y[0] * y[1]), -(y[1] - y[0] * y[1])])


def _b5(y, *, t):
    return np.array([y[1] * y[2], -y[0] * y[2], -0.51 * y[0] * y[1]])


def _orbit(y, *, t):
    r_cubed = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return np.array([y[2], y[3], -y[0] / r_cubed, -y[1] / r_cubed])


def _orbit_y0(e):
    return np.array([1.0 - e, 0.0, 0.0, np.sqrt((1.0 + e) / (1.0 - e))])


def _kepler(e, ts):
    """Closed-form orbit of eccentricity ``e`` (mean anomaly ``t``, period 2 pi)."""
    out = []
    for t in onp.asarray(ts):
        E = t
        for _ in range(50):
            E = E - (E - e * onp.sin(E) - t) / (1.0 - e * onp.cos(E))
        dE = 1.0 / (1.0 - e * onp.cos(E))
        b = onp.sqrt(1.0 - e**2)
        out.append(
            [onp.cos(E) - e, b * onp.sin(E), -onp.sin(E) * dE, b * onp.cos(E) * dE]
        )
    return onp.array(out)


def _dop853(vf, y0):
    sol = solve_ivp(
        lambda t, y: onp.asarray(vf(np.asarray(y), t=t)),
        (0.0, DETEST_T),
        onp.asarray(y0),
        method="DOP853",
        t_eval=onp.asarray(DETEST_SAVE_AT),
        rtol=1e-13,
        atol=1e-13,
    )
    return sol.y.T


DETEST_TS = onp.asarray(DETEST_SAVE_AT)  # the save times, on the host
DETEST = {
    "A2": (_a2, np.array([1.0]), lambda: (1.0 / onp.sqrt(DETEST_TS + 1.0))[:, None]),
    "A4": (
        _a4,
        np.array([1.0]),
        lambda: (20.0 / (1.0 + 19.0 * onp.exp(-DETEST_TS / 4.0)))[:, None],
    ),
    "B1": (_b1, np.array([1.0, 3.0]), lambda: _dop853(_b1, np.array([1.0, 3.0]))),
    "B5": (
        _b5,
        np.array([0.0, 1.0, 1.0]),
        lambda: _dop853(_b5, np.array([0.0, 1.0, 1.0])),
    ),
    "D1": (_orbit, _orbit_y0(0.1), lambda: _kepler(0.1, DETEST_TS)),
    "D3": (_orbit, _orbit_y0(0.5), lambda: _kepler(0.5, DETEST_TS)),
}
DETEST_TOLS = (1e-6, 1e-9)


@cache
def _detest_errors(name: str) -> list[float]:
    """DETEST's mixed error ``|y - y_ref| / (1 + |y_ref|)``, largest over the run.

    One per tolerance of ``DETEST_TOLS``; the tolerance is a traced argument,
    so the solver compiles once per problem.
    """
    vf, y0, reference = DETEST[name]
    prior = IWP(q=4, d=y0.shape[0])
    m_0, P_0_sqr = taylor_mode_initialization(vf, y0, prior.q)
    measure = ODEInformation(vf, prior.E0, prior.E1)

    @jax.jit
    def solve(tol):
        return gaussian_filter_adaptive(
            m_0,
            P_0_sqr,
            prior,
            measure,
            DETEST_SAVE_AT,
            atol=tol,
            rtol=tol,
            max_steps=20_000,
        )

    ref = reference()
    errors = []
    for tol in DETEST_TOLS:
        res = solve(tol)
        assert bool(res.success)
        x = onp.asarray(res.m @ prior.E0.T)
        errors.append(float((onp.abs(x - ref) / (1.0 + onp.abs(ref))).max()))
    return errors


@pytest.mark.parametrize("name", sorted(DETEST))
def test_detest_error_follows_the_tolerance(name):
    """Error within a constant of the tolerance, and shrinking with it.

    The controller bounds the local error; the global error is a problem-
    dependent multiple of it. Measured error / tolerance at 1e-6 and 1e-9:
    A2 0.04 / 0.12, A4 0.05 / 0.07, B1 5.2 / 4.5, B5 0.16 / 0.17,
    D1 0.67 / 7.0, D3 6.8 / 4.0 (orbit phase errors accumulate).
    """
    errors = _detest_errors(name)
    for err, tol in zip(errors, DETEST_TOLS, strict=True):
        assert err < 100.0 * tol, f"error {err:.2e} at tolerance {tol:.0e}"
    # Three decades of tolerance buy at least one and a half decades of
    # accuracy (measured: 2.0 -- 3.2 decades).
    assert errors[1] < 10**-1.5 * errors[0], errors


def _exact_derivatives_of_exp_sin(order: int) -> list[float]:
    """``d^k/dt^k exp(sin t)`` at ``t = 0``, the solution of DETEST A3."""
    f = lambda t: np.exp(np.sin(t))  # noqa: E731
    out = []
    for _ in range(order + 1):
        out.append(float(f(0.0)))
        f = jax.grad(f)
    return out


@pytest.mark.xfail(
    strict=True,
    reason="taylor_mode_initialization evaluates the vector field at the fixed "
    "time t0, so for a non-autonomous field the derivatives from the third on "
    "miss the explicit t-dependence (here 1, 1, 1, 1, 1 instead of 1, 1, 1, 0, "
    "-3)",
)
def test_taylor_initialization_of_a_non_autonomous_field_is_exact():
    """DETEST A3, ``y' = y cos t``: the initial state should be exact."""
    q = 4
    m_0, _ = taylor_mode_initialization(_a3, np.array([1.0]), q)
    expected = _exact_derivatives_of_exp_sin(q)
    onp.testing.assert_allclose(onp.asarray(m_0), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# A 20-dimensional linear system against expm
# ---------------------------------------------------------------------------


def _linear_system(d: int = 20):
    """``A = V B V.T``: ten damped rotations in a random orthonormal basis.

    Real parts of the eigenvalues in ``[-0.93, -0.12]``, imaginary parts up to
    2.7, so the components mix and oscillate on different time scales.
    """
    key_basis, key_re, key_im, key_x0 = jax.random.split(jax.random.PRNGKey(0), 4)
    V, _ = np.linalg.qr(jax.random.normal(key_basis, (d, d)))
    re = -jax.random.uniform(key_re, (d // 2,), minval=0.05, maxval=1.0)
    im = jax.random.uniform(key_im, (d // 2,), minval=0.0, maxval=3.0)
    B = jax.scipy.linalg.block_diag(
        *[np.array([[a, b], [-b, a]]) for a, b in zip(re, im, strict=True)]
    )
    return V @ B @ V.T, jax.random.normal(key_x0, (d,))


@cache
def _linear_system_solve(N: int, calibration: str):
    A, x0 = _linear_system()

    def vf(x, *, t):
        return A @ x

    prior = IWP(q=3, d=x0.shape[0])
    m_0, P_0_sqr = taylor_mode_initialization(vf, x0, prior.q)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    ts = np.linspace(0.0, 5.0, N + 1)

    @jax.jit
    def run():
        res = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration=calibration)
        m_s, P_s_sqr = rts_smoother(res)
        truth = jax.vmap(lambda t: jax.scipy.linalg.expm(A * t) @ x0)(ts)
        scale = np.max(np.abs(truth))
        err_final = np.max(np.abs(prior.E0 @ res.m[-1] - truth[-1])) / scale
        err_path = np.max(np.abs(m_s @ prior.E0.T - truth)) / scale
        std = np.sqrt(np.sum((P_s_sqr @ prior.E0.T) ** 2, axis=-2))
        z = (m_s @ prior.E0.T - truth)[1:] / std[1:]
        return err_final, err_path, np.sqrt(np.mean(z**2))

    return tuple(float(v) for v in run())


@pytest.mark.parametrize("calibration", ["dynamic", "diagonal"])
def test_twenty_dimensional_linear_system_matches_expm(calibration):
    """EK1, IWP(q = 3) on ``[0, 5]`` with 200 and 400 steps.

    Measured (relative to the largest solution entry), N = 200 / 400: final
    error 2.8e-7 / 1.8e-8 (dynamic) and 1.6e-7 / 1.4e-8 (diagonal), smoother
    path error 2.8e-7 / 1.9e-8 and 3.0e-7 / 1.9e-8, so order 3.9 and 4.0;
    smoother RMS z at N = 400: 0.37 and 0.27.
    """
    coarse = _linear_system_solve(200, calibration)
    fine = _linear_system_solve(400, calibration)
    assert fine[0] < 3e-7 and fine[1] < 3e-7, fine
    order = onp.log2(coarse[1] / fine[1])
    assert 3.5 <= order <= 4.5, order
    assert 10**-1.5 <= fine[2] <= 10**1.5, fine[2]
