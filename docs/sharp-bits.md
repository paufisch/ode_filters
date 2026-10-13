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

### Covariances are square-root factors

Anything named `*_sqr` is an upper-triangular factor, not a covariance. Reconstruct
with `P = P_sqr.T @ P_sqr`; for standard deviations use
`np.sqrt(np.diag(P_sqr.T @ P_sqr))`. The one exception is the diffusion:
`sigma_sqr` (and `min_sigma_sqr`) is sigma *squared*, a variance. See
[Notation](notation.md).

### The solvers are not pre-jitted: wrap them yourself

`gaussian_filter` and friends are plain functions. The simplest way to compile one
is `equinox.filter_jit`: it traces the arrays and treats everything else (prior,
measurement model, `tspan`, `N`, `calibration`, `correction`, ...) as static:

```python
import equinox as eqx

solve = eqx.filter_jit(gaussian_filter)
result = solve(mu_0, P0_sqr, prior, measure, (0.0, 10.0), 100, calibration="none")
```

With `jax.jit`, every non-array argument has to be marked static, keyword
arguments included, and static arguments must be hashable, so `tspan` has to be a
tuple rather than a list:

```python
solve = jax.jit(
    gaussian_filter,
    static_argnums=(2, 3, 4, 5),  # prior, measure, tspan, N
    static_argnames=("calibration", "correction"),
)
```

### Build models once and don't mutate them under `jit`

Priors and measurement models are static arguments, compared by identity. Two
consequences:

- **A new model object recompiles**, even if it is equal to the old one. Build
  the prior and measurement model once and reuse them across calls.
- **A mutated model is not noticed.** After `measure.R = 0.5`, a jitted solve
  that has already seen `measure` keeps using the old noise. Build a new model
  instead (which recompiles), or call the solver without `jit`.

Only arrays (`mu_0`, `P0_sqr`, the data in an `obs_model`, parameters) are traced,
so differentiate with respect to those.

### Preconditioned priors: fixed grid only, some fields in bar coordinates

`PrecondIWP` / `PrecondMaternPrior` / `PrecondJointPrior` run the recursion in
rescaled coordinates `x = T @ x_bar`. They work with `gaussian_filter` (without an
`obs_model`), but `gaussian_filter_adaptive` and `JointPrior` reject them. In their
results, `m_pred`, `P_pred_sqr` and the backward-pass fields are in bar
coordinates; see the [filters API](api/filters/index.md).

### A solve is differentiable on both stepping modes

Both `gaussian_filter` (fixed grid) and `gaussian_filter_adaptive` (save-at-grid)
are `jit`/`grad`/`vmap`-compatible, so any `jax.grad` over a solve works either way.

### Dynamic calibration absorbs model–data misfit

Diffusion calibration sizes the uncertainty, but a *dynamic* calibration absorbs
model–data misfit into `sigma^2`. When you want the misfit to stay visible (e.g. in
a likelihood), use `calibration="none"` and set the prior scale matrix `Xi` deliberately.
See [Diffusion calibration](calibration.md).

## FAQ

### My covariance blows up / the log-likelihood is `NaN`

In order: (1) enable float64; (2) check `q` is not too high for the step size (very
high order on a coarse grid is ill-conditioned); (3) make sure the initial
covariance is sensible (`taylor_mode_initialization` pairs the mean with a
zero/Dirac covariance); (4) for
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
EKF/RTS substrate with first-class custom measurement models. See
[What is a probabilistic ODE solver?](probabilistic-ode-solvers.md).

### Which function do I call?

See [How to choose](how-to-choose.md). Short version: `gaussian_filter` for a
fixed-step solve and `gaussian_filter_adaptive` for tolerance-based stepping; both
support jit/grad/vmap.

### What do the fields of the filter result mean?

See the filter [API reference](api/filters/index.md) and the annotated
[Quickstart](examples/quickstart.ipynb) step 4.
