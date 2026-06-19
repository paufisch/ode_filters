# What is a probabilistic ODE solver?

A classical ODE solver (`scipy.integrate.solve_ivp`, a Runge–Kutta method, Diffrax)
returns a **single trajectory**: its best guess at the solution. But every solver
makes a discretization error, and that error is invisible in the output — you get
a number, not a sense of how much to trust it.

A **probabilistic** ODE solver returns a **Gaussian posterior over the solution**:
a mean trajectory *and* a calibrated covariance. The mean is comparable to a
classical solver's estimate; the covariance quantifies the numerical uncertainty
introduced by taking finite steps. `ode_filters` computes this posterior with
Kalman filtering and smoothing.

!!! tip "One-sentence intuition"
    Solving an ODE is recast as **tracking a moving target**: just like a Kalman
    filter tracks an object from noisy sensor readings, a probabilistic ODE solver
    tracks the solution curve, using *the ODE itself* as the measurement.

## The four ingredients

Every solve in `ode_filters` is built from the same four pieces (you will meet
them in this order in the [Quickstart](examples/quickstart.ipynb)).

### 1. A prior over solutions

Before looking at the ODE, we say what we believe a "reasonable" solution looks
like: a **smooth function**. We encode this with an **integrated Wiener process**
(IWP) — a Gauss–Markov process whose state stacks the solution and its first `q`
derivatives,

$$ Y(t) = \big[\,x(t),\ \dot x(t),\ \ldots,\ x^{(q)}(t)\,\big]. $$

`q` is the smoothness order (how many derivatives we track); higher `q` assumes a
smoother solution and gives higher-order accuracy. The process moves between grid
points by a known linear transition, so inference stays a *linear-Gaussian*
state-space problem. See [Gauss–Markov priors](how-to-choose.md) for choosing
`q` and the prior family (IWP / Matern / Joint).

### 2. The ODE as data: the information operator

The ODE $\dot x = f(x, t)$ tells us that the **residual**

$$ g(Y) = \dot x(t) - f\big(x(t),\, t\big) $$

must be **zero** at the true solution. A probabilistic solver treats "the residual
is zero" as a **noiseless observation** of the state — an *information operator*.
At each grid point we condition the prior on this pseudo-observation: we pull the
belief toward states whose derivative actually matches the vector field. Because
`f` is usually nonlinear, the residual is linearized (this is the *extended*
Kalman filter):

- **EK0** treats `f` as locally constant (no Jacobian) — cheap and robust.
- **EK1** uses the Jacobian of `f` — more accurate, the historical default.

The choice is a pluggable [`Correction`](corrections.md); the measurement model
(`ODEInformation` and friends) supplies the residual.

### 3. The filter: a forward pass

An **extended Kalman filter** sweeps left to right across the grid. At each step it
**predicts** with the prior transition, then **updates** by conditioning on the
ODE residual. The output is the *filtering* posterior — the belief about the
solution at time $t_n$ given the ODE information up to $t_n$.

### 4. The smoother: a backward pass

A **Rauch–Tung–Striebel (RTS) smoother** then sweeps right to left, propagating
late information backwards so every time point is informed by the *whole*
trajectory. The output is the *smoothing* posterior — the quantity you usually
want to plot.

## Calibrating the uncertainty

The covariance is only honest if it is the right *size*. Its scale is set by a
diffusion parameter $\sigma^2$ on top of the prior. `ode_filters` estimates
$\sigma^2$ from the observed residuals (a quasi-MLE), so the error bars match the
actual discretization error rather than being arbitrary. This is what makes the
shaded band in the Quickstart meaningful. See [Diffusion calibration](calibration.md).

## Why bother (vs `scipy` / Diffrax)?

- **Uncertainty quantification** — a calibrated error band, for free, in one pass.
- **A differentiable likelihood** — the solver returns a marginal log-likelihood of
  the solution, so you can *fit ODE parameters to data* by gradient descent or MCMC.
  See [Parameter estimation](parameter-estimation.md).
- **Structured information** — conservation laws, partial/noisy observations, and
  hidden parameters all enter as additional measurements in the same machinery.

If you only need a fast point trajectory for a well-behaved ODE, a classical solver
is the right tool. Reach for a probabilistic solver when you need the *uncertainty*
or the *likelihood*.

## From concept to code

| Concept | In `ode_filters` |
| --- | --- |
| Prior over solutions (smoothness `q`, dimension `d`) | `IWP(q, d)` / `MaternPrior` / `JointPrior` |
| Initialize the state from `x0` and its derivatives | `taylor_mode_initialization(vf, x0, q)` |
| "Derivative must equal `f`" (the information operator) | `ODEInformation(vf, prior.E0, prior.E1)` |
| How to linearize `f` (EK0 / EK1) | `TaylorCorrection(order=0 | 1)` |
| Forward filtering pass | `gaussian_filter(...)` (or `gaussian_filter_adaptive(...)`) |
| Backward smoothing pass | `rts_smoother(prior, result)` |
| Size the error bars | `calibration=` / `ode_filters.calibration` |
| Fit parameters to data | `marginal_loglik` / `ODEFilter` + `fit` |

The symbols (`q`, `d`, `E0`/`E1`, `Xi`, `sigma^2`, the square-root convention) are
defined in [Notation & conventions](notation.md).

## Going deeper

The view of ODE solving as Bayesian filtering is due to a line of work worth
reading:

- Schober, Särkkä & Hennig (2019), *A probabilistic model for numerical solutions
  of initial value problems*.
- Tronarp, Kersting, Särkkä & Hennig (2019), *Probabilistic solutions to ODEs as
  nonlinear Bayesian filtering: a new perspective*.
- Bosch, Tronarp & Hennig (2021), *Calibrated adaptive probabilistic ODE solvers*.
- Krämer & Hennig (2024), *Stable implementation of probabilistic ODE solvers*
  (the square-root formulation used here).
- Hennig, Osborne & Kersting (2022), *Probabilistic Numerics* (textbook).

## See also

- [Quickstart](examples/quickstart.ipynb) — the four ingredients in code.
- [Notation & conventions](notation.md) — what every symbol means.
- [How to choose](how-to-choose.md) — picking the prior, order, and correction.
