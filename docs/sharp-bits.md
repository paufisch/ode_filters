# Sharp bits & FAQ

The footguns that catch people, each with the fix, followed by a short FAQ. (Named
after JAX's own [Sharp Bits](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html).)

## Sharp bits

### Enable 64-bit precision

Probabilistic ODE solvers propagate covariances and are sensitive to round-off. In
float32 you may see covariances lose positive-definiteness or the log-likelihood go
`NaN`. Turn on double precision **before** any array is created:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### `tspan` must be a tuple

`tspan` is a `jax.jit` *static* argument, so it must be hashable:

```python
tspan = (0.0, 10.0)   # ✅ tuple
tspan = [0.0, 10.0]   # ❌ list -> unhashable, breaks under jit
```

### Covariances are square-root factors

Anything named `*_sqr` is an upper-triangular factor, not a covariance. Reconstruct
with `P = P_sqr.T @ P_sqr`; for standard deviations use
`np.sqrt(np.diag(P_sqr.T @ P_sqr))`. See [Notation](notation.md).

### Build models outside `jit`; pass `prior`/`measure`/`N`/`tspan` as static

The prior, measurement model, integer `N`, and `tspan` are static configuration —
construct them in Python and mark them static (the loops already do this for you via
`static_argnums`). Only arrays (`mu_0`, `Sigma_0_sqr`, parameters) are traced.

### Gradient-based inference runs on the fixed-grid scan path only

`marginal_loglik` / `fit` and any `jax.grad` over a solve must use the **fixed-grid
`*_scan`** loops. The Python-`for` loops build lists and the adaptive driver uses a
Python `while` loop — neither is reverse-differentiable. (See the *Fixed-grid only*
note in [Parameter estimation](parameter-estimation.md).)

### Calibration is opt-in, and should be off during parameter inference

Diffusion calibration sizes the uncertainty, but a *dynamic* calibration absorbs
model–data misfit into `sigma^2` and confounds the likelihood over parameters. Use
`calibration="none"` (the default in the inference layer) when fitting parameters,
and set the prior scale `Xi` deliberately. See [Diffusion calibration](calibration.md).

## FAQ

### My covariance blows up / the log-likelihood is `NaN`

In order: (1) enable float64; (2) check `q` is not too high for the step size (very
high order on a coarse grid is ill-conditioned); (3) make sure the initial
covariance is sensible (`taylor_mode_initialization` gives a good one); (4) for
stiff problems, take smaller steps or use adaptive stepping.

### The uncertainty band looks far too tight (or too wide)

That is a calibration symptom, not a bug. Plot the whitened residuals (see
[Diffusion calibration](calibration.md) → *What calibration cannot fix*): a
well-calibrated filter has `||z_n||^2 / d ≈ 1`. Too tight ≫ 1, too wide ≪ 1.

### How does this compare to `scipy` / Diffrax / probdiffeq / ProbNum?

`scipy`/Diffrax are classical solvers — fast point trajectories, no uncertainty.
`ode_filters`, [probdiffeq](https://github.com/pnkraemer/probdiffeq), and
[ProbNum](https://probnum.readthedocs.io/) are *probabilistic* solvers (Gaussian
posterior + likelihood). `ode_filters` focuses on a transparent square-root
EKF/RTS substrate with first-class custom measurement models and a parameter-/
latent-force-inference layer. See [What is a probabilistic ODE solver?](probabilistic-ode-solvers.md).

### Which loop do I call?

See [How to choose](how-to-choose.md). Short version: `ekf1_sqr_loop` for a simple
fixed-step solve, the `*_scan` variants for jit/grad/vmap, and
`ekf1_sqr_adaptive_loop` for tolerance-based stepping.

### What do the ten values returned by the filter mean?

See the filter [API reference](api/filters/index.md) and the annotated
[Quickstart](examples/quickstart.ipynb) step 4.
