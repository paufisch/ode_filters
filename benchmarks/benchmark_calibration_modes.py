"""Benchmark calibration modes on a moderately stiff problem (van der Pol, mu=10).

Compares three diffusion-calibration recipes available in ``ode_filters``:

1. **Fixed-step + post-hoc MLE** -- run ``ekf1_sqr_loop`` with ``sigma=1``,
   then call :func:`posthoc_mle_sigma_sqr` to get one global ``sigma_hat^2``
   and :func:`rescale_sqr_seq` to multiply every stored covariance. This is
   probdiffeq's ``calibrate='fixed'`` mode.

2. **Fixed-step + online dynamic** -- run :func:`ekf1_sqr_loop_dynamic`,
   which estimates ``sigma_hat^2_n`` per step from the process-noise
   residual ``H Q(h) H.T`` and bakes it into ``Q_h`` *before* propagation.
   This is probdiffeq's ``MLEDiffusion`` ("dynamic") convention.

3. **Adaptive step + online dynamic** -- :func:`ekf1_sqr_adaptive_loop` with
   PI step controller and the same per-step dynamic calibration.

On a stiff problem, the true local diffusion varies by orders of magnitude
across the trajectory -- the relaxation oscillation has fast inner layers
separated by slow outer arcs. The single-scalar post-hoc mode necessarily
overestimates uncertainty on the slow arcs and underestimates on the layers;
the dynamic mode tracks the variation; the adaptive mode additionally lets
the step size respond.

We use mu=10 over [0, 30] (≈ two relaxation cycles). Larger mu makes the
problem genuinely stiff and the fixed-step EKF1 modes break down -- pick
``ekf1_sqr_adaptive_loop`` for those.

Reference: :func:`scipy.integrate.solve_ivp` with method='LSODA' and tight
tolerances.

Run with: ``uv run python benchmarks/benchmark_calibration_modes.py``
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from ode_filters.calibration import posthoc_mle_sigma_sqr, rescale_sqr_seq
from ode_filters.filters import (
    PIController,
    ekf1_sqr_adaptive_loop,
    ekf1_sqr_loop,
    ekf1_sqr_loop_dynamic,
)
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Stiff test problem: van der Pol with mu = 1000.
# x'' - mu (1 - x^2) x' + x = 0
# State: (x, dx/dt). At mu=1000 this is a textbook stiff relaxation oscillator
# with three orders of magnitude between the slow arcs and the fast jumps.
# ---------------------------------------------------------------------------
MU = 10.0


def vdp(state, *, t):
    x, v = state[0], state[1]
    return jnp.array([v, MU * (1.0 - x * x) * v - x])


def vdp_scipy(t, y):
    x, v = y
    return [v, MU * (1.0 - x * x) * v - x]


X0 = jnp.array([2.0, 0.0])
TSPAN = (0.0, 30.0)  # ~ two relaxation oscillations at mu=10
Q_PRIOR = 4  # IWP smoothness -- higher to handle the fast transition layers

# Reference solution from a stiff classical solver.
print("Computing reference (LSODA)...")
ref = solve_ivp(
    vdp_scipy,
    TSPAN,
    np.array(X0),
    method="LSODA",
    rtol=1e-10,
    atol=1e-12,
    dense_output=True,
)
print(f"  reference took {len(ref.t)} internal steps")


def _setup(q=Q_PRIOR):
    prior = IWP(q=q, d=2)
    mu0, S0 = taylor_mode_initialization(vdp, X0, q=q)
    measure = ODEInformation(vdp, prior.E0, prior.E1)
    return prior, mu0, S0, measure


def _l2_error_at_endpoint(m_final, prior):
    x_pred = np.asarray(prior.E0 @ m_final)
    x_ref = ref.sol(TSPAN[1])
    return float(np.sqrt(np.mean((x_pred - x_ref) ** 2)))


# ---------------------------------------------------------------------------
# Mode 1: fixed-step + post-hoc global MLE.
# ---------------------------------------------------------------------------
print("\nMode 1: fixed-step + post-hoc global MLE")
N_FIXED = 5000
prior, mu0, S0, measure = _setup()
r_fixed = ekf1_sqr_loop(mu0, S0, prior, measure, TSPAN, N_FIXED)
m_seq_fixed = jnp.stack(list(r_fixed[0]))
P_sqr_fixed = jnp.stack(list(r_fixed[1]))
mz_fixed = jnp.stack(list(r_fixed[-3]))
Pz_sqr_fixed = jnp.stack(list(r_fixed[-2]))

sigma_global = float(posthoc_mle_sigma_sqr(mz_fixed, Pz_sqr_fixed))
P_sqr_fixed_calibrated = rescale_sqr_seq(P_sqr_fixed, sigma_global)
err_fixed = _l2_error_at_endpoint(m_seq_fixed[-1], prior)
print(f"  steps={N_FIXED}, global sigma^2 = {sigma_global:.3e}")
print(f"  endpoint L2 error = {err_fixed:.3e}")


# ---------------------------------------------------------------------------
# Mode 2: fixed-step + online dynamic per-step.
# ---------------------------------------------------------------------------
print("\nMode 2: fixed-step + online dynamic per-step")
prior, mu0, S0, measure = _setup()
r_dyn = ekf1_sqr_loop_dynamic(mu0, S0, prior, measure, TSPAN, N_FIXED)
m_seq_dyn = jnp.stack(list(r_dyn[0]))
P_sqr_dyn = jnp.stack(list(r_dyn[1]))
sigma_dyn = np.asarray(r_dyn[9])
err_dyn = _l2_error_at_endpoint(m_seq_dyn[-1], prior)
print(
    f"  steps={N_FIXED}, sigma^2: "
    f"min={sigma_dyn.min():.2e}, max={sigma_dyn.max():.2e}, "
    f"median={np.median(sigma_dyn):.2e}, "
    f"max/min = {sigma_dyn.max() / sigma_dyn.min():.1e}"
)
print(f"  endpoint L2 error = {err_dyn:.3e}")


# ---------------------------------------------------------------------------
# Mode 3: adaptive step + online dynamic per-step.
# ---------------------------------------------------------------------------
print("\nMode 3: adaptive step + online dynamic per-step")
prior, mu0, S0, measure = _setup()
r_adapt = ekf1_sqr_adaptive_loop(
    mu0,
    S0,
    prior,
    measure,
    TSPAN,
    atol=1e-5,
    rtol=1e-3,
    controller=PIController(order=prior.q),
)
ts_adapt = np.asarray(r_adapt.t_seq)
m_adapt = np.stack([np.asarray(m) for m in r_adapt.m_seq])
sigma_adapt = np.asarray(r_adapt.sigma_sqr_seq)
h_adapt = np.asarray(r_adapt.h_seq)
err_adapt = _l2_error_at_endpoint(r_adapt.m_seq[-1], prior)
print(
    f"  accepted={len(r_adapt.h_seq)}, rejected={r_adapt.n_rejected}, "
    f"h: min={h_adapt.min():.2e}, max={h_adapt.max():.2e}"
)
print(
    f"  sigma^2: min={sigma_adapt.min():.2e}, max={sigma_adapt.max():.2e}, "
    f"median={np.median(sigma_adapt):.2e}, "
    f"max/min = {sigma_adapt.max() / sigma_adapt.min():.1e}"
)
print(f"  endpoint L2 error = {err_adapt:.3e}")


# ---------------------------------------------------------------------------
# Plot: x(t) trajectories, sigma^2 traces, calibrated 2-sigma bands.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# Reference + means.
t_grid = np.linspace(TSPAN[0], TSPAN[1], 4000)
x_ref_grid = ref.sol(t_grid)[0]
ts_fixed = np.linspace(TSPAN[0], TSPAN[1], N_FIXED + 1)
ax = axes[0]
ax.plot(t_grid, x_ref_grid, "k-", lw=0.6, label="LSODA reference")
ax.plot(
    ts_fixed, np.asarray(m_seq_fixed[:, 0]), "C0--", lw=0.8, label="fixed + post-hoc"
)
ax.plot(ts_fixed, np.asarray(m_seq_dyn[:, 0]), "C1:", lw=0.8, label="fixed + dynamic")
ax.plot(ts_adapt, m_adapt[:, 0], "C2-", lw=0.8, label="adaptive + dynamic")
ax.set_ylabel("x(t)")
ax.set_title(f"van der Pol, mu={MU:g}  --  three calibration modes")
ax.legend(loc="upper right", fontsize=8)

# 2-sigma bands on x(t).
ax = axes[1]
std_fixed = np.sqrt(
    np.einsum(
        "nij,nij->n",
        np.asarray(P_sqr_fixed_calibrated[:, :, :1]),
        np.asarray(P_sqr_fixed_calibrated[:, :, :1]),
    )
    + 1e-300
)
P_fixed_full = np.einsum(
    "nij,nik->njk",
    np.asarray(P_sqr_fixed_calibrated),
    np.asarray(P_sqr_fixed_calibrated),
)
P_dyn_full = np.einsum("nij,nik->njk", np.asarray(P_sqr_dyn), np.asarray(P_sqr_dyn))
std_x_fixed = np.sqrt(np.maximum(P_fixed_full[:, 0, 0], 0.0))
std_x_dyn = np.sqrt(np.maximum(P_dyn_full[:, 0, 0], 0.0))
ax.semilogy(ts_fixed, std_x_fixed, "C0--", lw=0.8, label="fixed + post-hoc")
ax.semilogy(ts_fixed, std_x_dyn, "C1:", lw=0.8, label="fixed + dynamic")
ax.set_ylabel(r"$\sigma_x(t)$")
ax.set_title("Posterior std on x(t)  --  dynamic adapts to local stiffness")
ax.legend(loc="upper right", fontsize=8)

# Per-step sigma^2 traces.
ax = axes[2]
ax.semilogy(ts_fixed[1:], sigma_dyn, "C1.-", ms=2, lw=0.5, label="fixed + dynamic")
ax.semilogy(ts_adapt[1:], sigma_adapt, "C2.-", ms=2, lw=0.5, label="adaptive + dynamic")
ax.axhline(
    sigma_global,
    color="C0",
    ls="--",
    lw=1.0,
    label=f"fixed + post-hoc global = {sigma_global:.1e}",
)
ax.set_xlabel("t")
ax.set_ylabel(r"$\widehat{\sigma}^2$")
ax.set_title("Per-step diffusion estimate")
ax.legend(loc="lower right", fontsize=8)

plt.tight_layout()
out_path = "benchmarks/benchmark_calibration_modes.png"
plt.savefig(out_path, dpi=130)
print(f"\nSaved figure to {out_path}")


# ---------------------------------------------------------------------------
# Summary table.
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"{'mode':<32} {'work':<22} {'endpoint L2':<14} {'sigma^2':<10}")
print("-" * 78)
print(
    f"{'fixed + post-hoc global MLE':<32} {f'{N_FIXED} steps':<22} "
    f"{err_fixed:<14.3e} {sigma_global:<10.3e}"
)
print(
    f"{'fixed + online dynamic':<32} {f'{N_FIXED} steps':<22} "
    f"{err_dyn:<14.3e} {f'~{np.median(sigma_dyn):.2e}':<10}"
)
print(
    f"{'adaptive + online dynamic':<32} "
    f"{f'{len(h_adapt)} acc / {r_adapt.n_rejected} rej':<22} "
    f"{err_adapt:<14.3e} {f'~{np.median(sigma_adapt):.2e}':<10}"
)
print("=" * 78)
