# Notation & conventions

The guides and API assume a small, consistent vocabulary. This page is the single
place that maps every code symbol to the maths, and states the conventions the
library follows.

## Symbols

| Symbol (code) | Maths | Meaning |
| --- | --- | --- |
| `d` | $d$ | dimension of the ODE (number of state components) |
| `q` | $q$ | smoothness order: number of derivatives modelled (an IWP tracks $x, \dot x, \ldots, x^{(q)}$) |
| state dim | $D = d\,(q+1)$ | length of the stacked state vector $Y$ |
| `Y` / `m` | $Y(t)$, $m$ | the stacked state (solution + derivatives) and its posterior mean |
| `E0` | $E_0$ | selection matrix extracting the 0th derivative $x$ from $Y$ |
| `E1` | $E_1$ | selection matrix extracting the 1st derivative $\dot x$ |
| `E2` | $E_2$ | 2nd derivative $\ddot x$ (second-order ODEs only) |
| `Xi` | $\Xi$ | structural scale matrix of the prior diffusion (shape $d\times d$) |
| `sigma^2` | $\sigma^2$ | scalar (or per-component) diffusion scale, set by calibration |
| `A`, `b`, `Q` | $A(h), b(h), Q(h)$ | prior transition matrix, drift, and process-noise covariance over a step `h` |
| `H`, `c`, `R` | $H, c, R$ | linearized measurement: Jacobian, offset, noise covariance |
| `m_pred`, `P_pred` | $m^-, P^-$ | predicted (prior) mean and covariance before an update |
| `m_z`, `P_z` | $m_z, P_z$ | predicted-observation marginal (the ODE-defect innovation) |
| `mu_0`, `Sigma_0` | $m_0, P_0$ | initial mean and covariance |

!!! note "Prefer `E0`/`E1` over `q`/`d` in the API"
    Functions take the selection matrices `E0`, `E1` (`E2`) rather than raw `q`/`d`
    integers. This keeps the measurement model agnostic to how the prior lays out
    its state, and lets `JointPrior` expose `E0_x` / `E0_hidden` for joint
    state–parameter models. Construct them from a prior (`prior.E0`, `prior.E1`).

## The square-root convention

Every covariance is stored and propagated as a **square-root factor**, never as a
dense matrix. A name ending in `_sqr` is an upper-triangular factor `A_sqr` with

$$ A = A_{\text{sqr}}^{\top}\, A_{\text{sqr}}. $$

To recover a covariance, multiply back: `P = P_sqr.T @ P_sqr`. To extract a standard
deviation, take `np.sqrt(np.diag(P_sqr.T @ P_sqr))`. The filter never forms `P`
explicitly — predictions and updates are QR-based operations on the factors
(Krämer & Hennig 2024), which keeps covariances positive-definite and well
conditioned even at high order `q`.

## Other conventions

- **Keyword-only time.** Vector fields and measurement methods take `t` as a
  keyword-only argument: `def vf(x, *, t): ...`. This makes the time dependence
  explicit and prevents positional mistakes.
- **`tspan` is a tuple.** Pass `tspan=(t0, t1)`, not a list — it must be hashable to
  serve as a `jax.jit` static argument. See [Sharp bits](sharp-bits.md).
- **64-bit precision.** Probabilistic ODE solvers are sensitive to round-off; enable
  `jax.config.update("jax_enable_x64", True)` (the test suite does this in
  `conftest.py`).
- **Pure JAX.** Everything is `jit`/`grad`/`vmap`-compatible on the scan-based code
  paths; `import jax.numpy as np` is used throughout.

## See also

- [What is a probabilistic ODE solver?](probabilistic-ode-solvers.md) — the concepts behind these symbols.
- [Sharp bits](sharp-bits.md) — the conventions that bite if ignored.
