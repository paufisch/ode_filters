# Notation & conventions

The guides and the API reference use one consistent set of symbols. This page
maps every code symbol to its mathematical counterpart and states the
conventions the library follows.

## Symbols

| Symbol (code) | Math | Meaning |
| --- | --- | --- |
| `d` | $d$ | dimension of the ODE (number of state components) |
| `q` | $q$ | smoothness order: number of derivatives modeled (an IWP tracks $x, \dot x, \ldots, x^{(q)}$) |
| state dim | $D = d\,(q+1)$ | length of the stacked state vector $Y$ |
| `Y` / `m` | $Y(t)$, $m$ | the stacked state (solution + derivatives) and its posterior mean |
| `E0` | $E_0$ | selection matrix extracting the 0th derivative $x$ from $Y$ |
| `E1` | $E_1$ | selection matrix extracting the 1st derivative $\dot x$ |
| `E2` | $E_2$ | 2nd derivative $\ddot x$ (second-order ODEs only) |
| `Xi` | $\Xi$ | structural scale matrix of the prior diffusion (shape $d\times d$) |
| `sigma^2` | $\sigma^2$ | scalar (or per-component) diffusion scale, set by calibration |
| `diffusion` | $\sigma^2_n$ | the diffusion step $n$ propagated with (`sigma_sqr` under a calibrating policy, one under `"none"`) |
| `A`, `b`, `Q` | $A(h), b(h), Q(h)$ | prior transition matrix, drift, and process-noise covariance over a step `h` |
| `H`, `c`, `R` | $H, c, R$ | linearized measurement: Jacobian, offset, noise covariance |
| `m_pred`, `P_pred` | $m^-, P^-$ | predicted (prior) mean and covariance before an update |
| `mz`, `Pz_sqr` | $m_z, P_z$ | innovation: the predicted residual and its square-root covariance |
| `m_0`, `P_0_sqr` | $m_0, P_0$ | initial mean and square-root covariance |
| `ts` | $t_0, \dots, t_N$ | the time grid: filter step $n$ updates at `ts[n + 1]` |

!!! note "Prefer `E0`/`E1` over `q`/`d` in the API"
    Functions take the selection matrices `E0`, `E1` (`E2`) rather than raw `q`/`d`
    integers. The measurement model then does not depend on how the prior lays
    out its state, and `JointPrior` can expose `E0_x` / `E0_hidden` for joint
    state–parameter models. Construct them from a prior (`prior.E0`, `prior.E1`).

## The square-root convention

Every covariance is stored and propagated as a **square-root factor**, never as a
dense matrix. A name ending in `_sqr` is an upper-triangular factor `A_sqr` with

$$ A = A_{\text{sqr}}^{\top}\, A_{\text{sqr}}. $$

To recover a covariance, multiply back: `P = P_sqr.T @ P_sqr`, or use the
`Gaussian` properties: `rts_smoother(result).cov`, and `.std` for the marginal
standard deviations (computed from the factor's column norms, without forming
`P`). Two kinds of names break the suffix rule:

- **`sigma_sqr` is not a factor.** It is $\sigma^2$ itself, and so are the
  names built on it: `min_sigma_sqr`, `quasi_mle_sigma_sqr`,
  `apply_state_sigma_sqr`.
- **Model inputs are covariances, not factors.** `Xi`, `R=` and `noise=` take
  the covariance (or a variance); the models factor them internally
  (`measure.noise_sqr(t=t)`, `ObsModel.R_sqr`).

The filter does not form `P` explicitly. Predictions and updates are QR-based
operations on the factors (Krämer & Hennig 2024). The represented covariance
`P = P_sqr.T @ P_sqr` is a Gram matrix and therefore positive semidefinite by
construction. The filter also never forms the ill-conditioned dense `Q`, so the
factors stay well conditioned at high order `q`.

## Other conventions

- **Keyword-only time.** Vector fields and measurement methods take `t` as a
  keyword-only argument: `def vf(x, *, t): ...`. This makes the time dependence
  explicit and prevents passing `t` in the position of another argument.
- **Time grids are arrays.** The solvers take the grid `ts` itself (any strictly
  increasing 1-D array); a uniform one is `np.linspace(t0, t1, N + 1)`.
- **64-bit precision.** Probabilistic ODE solvers are sensitive to round-off.
  Enable double precision with `jax.config.update("jax_enable_x64", True)` (the
  test suite does this in `conftest.py`).
- **Pure JAX.** Every scan-based code path is compatible with `jit`, `grad` and
  `vmap`, and the code imports `jax.numpy` as `np` throughout.

## See also

- [What is a probabilistic ODE solver?](probabilistic-ode-solvers.md): the concepts behind these symbols.
- [Sharp bits](sharp-bits.md): the errors that follow when these conventions are ignored.
