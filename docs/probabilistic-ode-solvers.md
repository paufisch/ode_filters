# What is a probabilistic ODE solver?

A classical ODE solver (`scipy.integrate.solve_ivp`, a Runge–Kutta method, Diffrax)
returns a single trajectory, its estimate of the solution. Every solver makes a
discretization error, and the returned trajectory carries no information about
its size.

A **probabilistic ODE solver** returns a Gaussian posterior over the solution,
which consists of a mean trajectory and a calibrated covariance. The mean plays
the role of a classical solver's estimate. The covariance quantifies the
numerical uncertainty that the finite step size introduces. `ode_filters`
computes this posterior with Kalman filtering and smoothing.

## The four ingredients

Every solve in `ode_filters` combines the same four components. The
[Quickstart](examples/quickstart.ipynb) introduces them in the same order.

### 1. A prior over solutions

Before the ODE enters, we assume that the solution is a smooth function. We
encode this with an integrated Wiener process (IWP), a Gauss–Markov process
whose state stacks the solution and its first `q` derivatives,

$$ Y(t) = \big[\,x(t),\ \dot x(t),\ \ldots,\ x^{(q)}(t)\,\big], $$

where `q` is the smoothness order, the number of derivatives in the state. A
larger `q` assumes a smoother solution and gives a higher order of convergence.
The process moves between grid points by a known linear transition with
Gaussian noise, so the prior is a linear-Gaussian state-space model. See
[Gauss–Markov priors](how-to-choose.md) for choosing `q` and the prior family
(IWP / Matern / Joint).

### 2. The ODE as data: the information operator

The ODE $\dot x = f(x, t)$ implies that the **residual**

$$ g(Y) = \dot x(t) - f\big(x(t),\, t\big) $$

is zero at the true solution. A probabilistic solver treats the statement "the
residual is zero" as a noiseless observation of the state, and the map from the
state to this residual is the **information operator**. At each grid point we
condition the prior on this pseudo-observation, which moves the belief towards
states whose derivative matches the vector field. For a nonlinear `f` the
update linearizes the residual, as the extended Kalman filter (EKF) does:

- **EK0** treats `f` as locally constant, so a step evaluates no Jacobian.
- **EK1** linearizes `f` with its Jacobian. It is the default and is generally
  more accurate per step.

The linearization is a pluggable [`Correction`](corrections.md), and the
measurement model (`ODEInformation` and its variants) supplies the residual.

### 3. The filter: a forward pass

An extended Kalman filter traverses the grid forward in time. At each step it
predicts with the prior transition and then updates by conditioning on the ODE
residual. The output is the **filtering posterior**, the distribution of the
solution at time $t_n$ given the ODE information up to $t_n$.

### 4. The smoother: a backward pass

A Rauch–Tung–Striebel (RTS) smoother then traverses the grid backward in time.
It propagates the information of later steps to earlier ones, so the estimate at
every time point conditions on the ODE information over the whole grid. The
output is the **smoothing posterior**, the posterior to report and plot for a
completed solve.

## Calibrating the uncertainty

The covariance quantifies the numerical error only if its scale matches the
size of that error. The scale is set by a diffusion parameter $\sigma^2$ that
multiplies the process noise of the prior. By default, `ode_filters` estimates
$\sigma^2$ at every step from the observed ODE residual with an approximate
quasi-maximum-likelihood estimate (quasi-MLE; Bosch et al. 2021). The error
bars then scale with the observed residual magnitude instead of the scale of
the prior, which nothing in the problem fixes. The estimate is approximate,
because it conditions on the local linearization and assumes that the residual
covariance scales with $\sigma^2$. See [Diffusion calibration](calibration.md).

## Compared with a classical solver

- **Uncertainty quantification.** The forward pass that computes the mean also
  computes a calibrated error band, and the backward pass turns it into the
  smoothing band.
- **Structured information.** Conservation laws, partial or noisy observations
  and hidden parameters enter as additional measurements of the same filter.
- **Inference from data.** The filter also returns the likelihood of the data,
  which accounts for the numerical error. The likelihood is differentiable in
  every parameter, so the same solve serves to fit parameters, to recover an
  unknown input or to provide the log-density for a sampler.

When only a point trajectory is needed, a classical solver is the appropriate
tool, since it computes no covariance and is faster for the same accuracy. A
probabilistic solver is the choice when the numerical uncertainty is needed,
or when the ODE is one part of a model that is fitted to data.

## From data to parameters

Data enter the filter as one more measurement, so fitting an ODE to data uses
the same filter. Each data update evaluates the log-density of the observation
under its prediction, and the sum of these terms is the log-likelihood of the
data (`result.log_likelihood_obs`). This likelihood accounts for the numerical
error. On a coarse grid the solver's uncertainty widens the predictions, and
the likelihood reflects this instead of treating an inaccurate trajectory as
exact. Maximize it over the parameters with `fit_scipy`, or sample from the
posterior with NumPyro, as in
[Parameter estimation](examples/parameter-estimation.ipynb). An unknown input
`u(t)` that drives the ODE is one more block of the state with its own prior,
and the same smoother recovers it, as in
[Recovering a hidden input](examples/hidden-input.ipynb).

## From concept to code

| Concept | In `ode_filters` |
| --- | --- |
| Prior over solutions (smoothness `q`, dimension `d`) | `IWP(q, d)` / `MaternPrior` / `JointPrior` |
| Initialize the state from `x0` and its derivatives | `taylor_mode_initialization(vf, x0, q)` |
| "Derivative must equal `f`" (the information operator) | `ODEInformation(vf, prior.E0, prior.E1)` |
| How to linearize `f` (EK0 / EK1) | `TaylorCorrection(order=0 | 1)` |
| Forward filtering pass | `gaussian_filter(...)` (or `gaussian_filter_adaptive(...)`) |
| Backward smoothing pass | `rts_smoother(result)` |
| Size the error bars | `calibration=` / `ode_filters.calibration` |
| Data | `measurements=[Measurement(...)]`, or `prepare_observations(...)` passed as `obs_model=` |
| Fit parameters to data | `marginal_loglik` with `fit_scipy` / `fit` |

The symbols (`q`, `d`, `E0`/`E1`, `Xi`, `sigma^2`, the square-root convention) are
defined in [Notation & conventions](notation.md).

## Going deeper

The view of ODE solving as Bayesian filtering is due to the following line of
work:

- Schober, Särkkä & Hennig (2019), *A probabilistic model for numerical solutions
  of initial value problems*.
- Tronarp, Kersting, Särkkä & Hennig (2019), *Probabilistic solutions to ODEs as
  nonlinear Bayesian filtering: a new perspective*.
- Bosch, Tronarp & Hennig (2021), *Calibrated adaptive probabilistic ODE solvers*.
- Krämer & Hennig (2024), *Stable implementation of probabilistic ODE solvers*
  (the square-root formulation used here).
- Hennig, Osborne & Kersting (2022), *Probabilistic Numerics* (textbook).

## See also

- [Quickstart](examples/quickstart.ipynb): the four ingredients in code.
- [Notation & conventions](notation.md): the meaning of every symbol.
- [How to choose](how-to-choose.md): choosing the prior, the order and the
  correction.
