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
with `P = P_sqr.T @ P_sqr`, or use the `Gaussian` properties: for smoothed
standard deviations, `rts_smoother(result).std`; for filtered ones,
`result.filtered.std`. The one exception is the diffusion:
`sigma_sqr` (and a policy's `min_sigma_sqr`) is sigma *squared*, a variance. See
[Notation](notation.md).

### The solvers are not pre-jitted: wrap them yourself

`gaussian_filter` and friends are plain functions. The simplest way to compile one
is `equinox.filter_jit`: it traces the arrays (including the time grid `ts`) and
treats everything else (prior, measurement model, `calibration`, `correction`,
...) as static:

```python
import equinox as eqx

solve = eqx.filter_jit(gaussian_filter)
result = solve(m_0, P_0_sqr, prior, measure, np.linspace(0.0, 10.0, 101))
```

A traced grid -- an argument of the compiled function, as above -- is handled
step by step (one transition per step). A concrete uniform grid computes the
prior's transition once: a NumPy or JAX array outside `jit`, or one the
compiled function closes over. For a fixed grid and a prior whose transition is
costly (a `MaternPrior` takes a matrix exponential and a quadrature per
transition), close over it:

```python
ts = np.linspace(0.0, 10.0, 101)
solve = eqx.filter_jit(lambda m_0, prior: gaussian_filter(m_0, P_0_sqr, prior, measure, ts))
```

### Models are pytrees: use `equinox.filter_jit`, not `jax.jit` with static arguments

Priors and measurement models are `equinox` modules. Their arrays (`Xi`, a length
scale, a rate, the selection matrices, the noise `R`) are pytree leaves; their
orders, dimensions and vector fields are static. So:

- `equinox.filter_jit` traces a model's arrays like any other argument: a new
  model with the same structure does not recompile, and one with a different
  hyperparameter is not silently stale. `jax.jit` with `static_argnums` does not
  work, because a module holding arrays is not hashable.
- Models are immutable. `measure.with_noise(R)` returns a copy with new noise;
  there is no setter.
- A model can be differentiated (`equinox.filter_grad`) or batched
  (`equinox.filter_vmap`) directly, e.g. over a length scale.

### Data times must be grid nodes

`prepare_observations` matches each measurement time `t` to a node of the grid
(within `1e-7 + 1e-5 |t|`) and raises for a time that is not one. For data off
the grid, add their times to it with
`ts = merge_measurement_times(ts, measurements)`, which adds no node for a time
within this tolerance of another. Every solver accepts non-uniform grids.
Measurements may have different times: each step conditions on the rows that
have data. Data at the initial time `ts[0]` belong in the initial state, so
`merge_measurement_times` raises for them, and also for times outside the grid.

The matching runs on the host. Inside `jax.jit`, build the `Measurement` from
concrete times: a NumPy array, Python numbers, or an array that the jitted
function closes over. Its data `z`, its matrix `A` and its noise may be traced,
so the evidence can be differentiated and batched over the data. Times that are
arguments of the jitted function are traced, and `prepare_observations` raises a
`TypeError` for them.

### A solve is differentiable on both stepping modes

Both `gaussian_filter` (fixed grid) and `gaussian_filter_adaptive` (save-at-grid)
are `jit`/`grad`/`vmap`-compatible, so any `jax.grad` over a solve works either way.

Reverse-mode differentiation of a fixed-grid solve stores the intermediates of
every step, so its memory grows with the number of steps. Pass
`checkpoint=True` to `gaussian_filter`, `affine_filter` or `iterated_smoother`
to wrap each step in `jax.checkpoint`. The backward pass then stores only the
state carried between steps and recomputes each step's intermediates, at the
cost of one more forward pass. For a Lorenz system with `q = 3` on 4000 steps,
this reduces the temporary memory of the compiled gradient of the
log-likelihood from 52 MB to 5 MB on a CPU. The values are identical and the
gradients agree to round-off. The adaptive solver has no such option, because
its loop is checkpointed already.

### A system at rest: exact values, but gradients need a floor

Dynamic calibration estimates the diffusion from each step's ODE residual. A
solve that starts exactly at an equilibrium -- or whose solution the prior
reproduces exactly, like `x' = 1` -- has residuals that vanish identically, so
the estimate is exactly zero. From an exact initial value
(`taylor_mode_initialization`) the posterior is then the exact, certain path,
and the filter, smoother, samples and dense output return it. Diagonal
calibration does the same per component, e.g. for a pendulum released at rest.

Derivatives at such a point are not right: with a zero diffusion the ODE
information has no weight, so `jax.grad` differentiates the Taylor
extrapolation of the initial value instead of the solution. An optimizer can
land there, e.g. a fit of a decay started from the initial guess `x0 = 0`.
Set a floor, `calibration=DynamicCalibration(min_sigma_sqr=1e-8)`, or use
`calibration="none"` with a deliberately chosen `Xi` (the usual choice for
inference).

### An uncertain initial value without data shrinks towards zero

The ODE information alone pulls an uncertain initial value towards the zero
solution. We solve `x' = -x` on `np.linspace(0.0, 5.0, 51)` with `IWP(q=2)`
and the initial value `x(0) ~ N(1.5, 0.5^2)`, placed along the direction that
moves all derivatives consistently:

```python
J = jax.jacfwd(lambda x0: taylor_mode_initialization(vf, x0, q=2)[0])(x0)
P_0_sqr = np.zeros((3, 3)).at[0].set(0.5 * J[:, 0])
```

After `rts_smoother`, the smoothed `x(0)` has mean `2e-7` and standard
deviation `2e-4` under the default dynamic calibration, and mean `1.33` and
standard deviation `0.47` under `calibration="none"`.

The posterior is correct under the model. Every `x = A exp(-t)` solves the
ODE, but zero residuals are most likely for the trivial solution `x = 0`,
which an `IWP` prior predicts with no discretization error. The ODE
information therefore favors small amplitudes `A`. Under `"none"` the size of
the pull follows from the prior: an `IWP` of order `q` drives `x^(q+1)` with
white noise of intensity `Xi`, so for `x' = -k x` on a long horizon the ODE
information weighs `A` by `exp(-A^2 k^(2q+1) / (4 Xi))`. This factor turns
`N(1.5, 0.5^2)` into the `N(1.33, 0.47^2)` above. Dynamic calibration
strengthens the pull: it estimates each step's diffusion from that step's
residual, which shrinks with the mean, and a smaller diffusion gives the
prior's preference for `x = 0` more weight.

Give an uncertain initial value data to pin it, or keep the initial value
exact (`taylor_mode_initialization`). One measurement of `x(1)` with noise
standard deviation 0.01 brings the smoothed `x(0)` back to within one
standard deviation of the true value, under either calibration.
`calibration="none"` weakens the effect but does not remove it.

### Sample paths depend on the factorization when the ODE information is exact

With exact ODE information (zero noise `R`, the default) the smoothing posterior
is singular, and a singular covariance has many square-root factors. Which one
the QR returns depends on round-off, so `sample_posterior` can return a
different path for the same key in a batched (`vmap`) and a looped run, or on
other hardware, and a fixed-key path is not smooth in the model's parameters
(finite differences through it do not converge). Each path is still an exact
draw from the posterior, so means, covariances and Monte Carlo estimates are
unaffected. For reproducible paths, give the ODE information a small noise, a
nugget: `measure.with_noise(1e-10)`. Keep it well below the variance the filter
predicts for the residual, since it widens the posterior a little.

### Dynamic calibration absorbs model–data misfit

Diffusion calibration sizes the uncertainty, but a *dynamic* calibration absorbs
model–data misfit into `sigma^2`. When you want the misfit to stay visible (e.g. in
a likelihood), use `calibration="none"` and set the prior scale matrix `Xi` deliberately.
See [Diffusion calibration](calibration.md).

### The smoother's band is calibrated for the filter

Every calibration policy fits the diffusion to the filter's innovations, and
the smoother inherits it. Under a constant diffusion the smoothed mean is 8.5
to 3400 times more accurate than the filtered one in our tests (`q = 2` to
`5`), while its standard deviation is at most 6.4 times smaller, so the
smoother's band is too wide. Dynamic calibration overstates the diffusion for
`q >= 2`, and its changes from step to step cost the smoother part or all of
its accuracy. See [The smoother's uncertainty](calibration.md#the-smoothers-uncertainty)
for the measurements and two recipes.

## FAQ

### My covariance blows up / the log-likelihood is `NaN`

In order: (1) enable float64; (2) check `q` is not too high for the step size (very
high order on a coarse grid is ill-conditioned); (3) make sure the initial
covariance is sensible (`taylor_mode_initialization` pairs the mean with a
zero/Dirac covariance); (4) for
stiff problems, take smaller steps or use adaptive stepping.

### The uncertainty band looks far too tight (or too wide)

That is a calibration symptom, not a bug. With data, check the whitened data
innovations: under a consistent model `nis(result.mz_obs, result.Pz_obs_sqr)`
averages the number of observed rows per step — much larger means too tight,
much smaller too wide. Without data, see [Diffusion calibration](calibration.md)
-> *What calibration cannot fix*.

### How does this compare to `scipy` / Diffrax / probdiffeq / ProbNumDiffEq.jl / rodeo?

`scipy` and Diffrax are classical solvers: fast point trajectories, no
uncertainty. The others are probabilistic solvers. [How it compares](comparison.md)
sets out what each offers and when to use which: probdiffeq is the fastest, and
`ode_filters` focuses on models that combine the ODE with data, conservation laws
and hidden inputs.

### Which function do I call?

See [How to choose](how-to-choose.md). Short version: `gaussian_filter` for a
fixed-step solve and `gaussian_filter_adaptive` for tolerance-based stepping; both
support jit/grad/vmap.

### What do the fields of the filter result mean?

See the filter [API reference](api/filters/index.md) and the annotated
[Quickstart](examples/quickstart.ipynb) step 4.
