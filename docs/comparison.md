# How it compares

`ode_filters` is one of several libraries for probabilistic ODE solvers. All of
them use Gaussian filtering and smoothing with the ODE as a measurement, and
much of what `ode_filters` implements was developed by the authors of the
others: preconditioning and Taylor-mode initialization by Krämer and Hennig,
dynamic calibration and Fenrir by Bosch, Tronarp and Hennig, and Dalton by Wu
and Lysy. This page lists what each library offers. The facts about the other
libraries were checked against their source in October 2026 (probdiffeq 0.9.2,
ProbNumDiffEq.jl 0.18.0, rodeo 1.1.3). If something has changed since, please
open an issue.

## The libraries

**[probdiffeq](https://github.com/pnkraemer/probdiffeq)** (JAX) is the most
complete JAX library for probabilistic ODE solvers, and it is faster than
`ode_filters` (see [Speed](#speed)). Its dense, isotropic, block-diagonal and
matrix-free covariance factorizations let it scale to large systems. It has
adaptive steps with a filter, a fixed-interval and a fixed-point smoother; IWP,
IOUP and exponential priors; higher-order ODEs, as well as DAEs and custom
residuals (marked experimental); and parameter estimation through the
log-marginal likelihood of time series (Fenrir) or terminal values, also with
adaptive steps.

**[ProbNumDiffEq.jl](https://github.com/nathanaelbosch/ProbNumDiffEq.jl)**
(Julia) brings probabilistic solvers to DifferentialEquations.jl. It has EK0,
EK1 and diagonal EK1; IWP, IOUP and Matérn priors, with probabilistic
exponential integrators built on the IOUP prior; global, dynamic and
per-component calibration; dense, isotropic and block-diagonal covariances;
second-order ODEs and mass-matrix DAEs; manifold updates for nonlinear
invariants; and parameter inference with the filtering, Fenrir and Dalton data
likelihoods.

**[rodeo](https://github.com/mlysy/rodeo)** (JAX) is a toolbox for parameter
inference. It offers four linearizations, which it calls interrogations (EK0, a
diagonal EK1, Chkrebtii's and rodeo's own), and these ways to approximate the
likelihood of the data: a plug-in of the solution without data, Fenrir, Dalton
for Gaussian and non-Gaussian data, MAGI, and pseudo-marginal MCMC. It steps on
a uniform grid with an IWP prior, block-diagonal per variable or as one dense
block, solves higher-order ODEs directly, and leaves the diffusion to the user.

**`ode_filters`** (JAX) is built for models in which the ODE is one source of
information among others, next to conservation laws, data at any grid node,
and hidden inputs that drive the dynamics. Every solver runs the same
square-root filter step, so each of these combines with every prior,
linearization and stepping mode.

## Side by side

| | `ode_filters` | probdiffeq | ProbNumDiffEq.jl | rodeo |
| --- | --- | --- | --- | --- |
| Language | JAX | JAX | Julia | JAX |
| Priors | IWP, IOUP, Matérn, joint state–input | IWP, IOUP, exponential | IWP, IOUP, Matérn | IWP |
| Linearization | EK0, EK1, IEKF, SLR; IEKS and IPLS | EK0, EK1, diagonal EK1, IEKF | EK0, EK1, diagonal EK1 | EK0, diagonal EK1, Chkrebtii, rodeo |
| Covariance | dense | dense, isotropic, block-diagonal, matrix-free | dense, isotropic, block-diagonal | dense, block-diagonal |
| Steps | fixed (any grid), adaptive | fixed, adaptive | fixed, adaptive | fixed (uniform) |
| Calibration | none, global MLE, dynamic, per-component | none, global MLE, dynamic, per-component | none, global MLE, dynamic, per-component | set by the user |
| Additional information | second-order ODEs, linear invariants, custom residuals | higher-order ODEs; DAEs and custom residuals (experimental) | second-order ODEs, mass-matrix DAEs, nonlinear invariants | higher-order ODEs |
| Data | linear Gaussian, at any grid node, per-row masks | Gaussian, of one derivative, at the grid or save points | at data times added to the steps | linear Gaussian or any log-likelihood, on the grid |
| Data likelihoods | filtering, Fenrir, Dalton | Fenrir, terminal values | filtering, Fenrir, Dalton | plug-in, Fenrir, Dalton (also non-Gaussian), MAGI |

The table uses one name per method, whatever a library calls it. *IWP* is the
integrated Wiener process (rodeo's IBM). *EK0* and *EK1* linearize the ODE to
zeroth and first order (probdiffeq's ts0 and ts1), and *diagonal EK1* keeps
only the diagonal blocks of the Jacobian. *IEKF* re-linearizes within a step,
*IEKS* and *IPLS* re-linearize over the whole trajectory, and *SLR* is
statistical linearization. *Isotropic* covariances are the same for every ODE
dimension (ProbNumDiffEq.jl's isometric Kronecker covariance). Calibration
estimates the diffusion once for the whole solve (*global MLE*), at every step
(*dynamic*), or for each ODE dimension (*per-component*). The *filtering*
likelihood conditions on the data during the solve (`ode_filters`' sequential
evidence). *Fenrir* filters the data through the posterior of a solve without
data. *Dalton* divides the evidence of the ODE and the data together by the
evidence of the ODE alone. The *plug-in* likelihood evaluates the data at the
mean of a solve without data.

## Choosing a library

All four libraries estimate parameters from data. The choice depends on the
model and on the other features it needs.

- **Speed or large systems.** probdiffeq. It is faster (see [Speed](#speed)),
  and its covariance factorizations let it scale. `ode_filters` stores dense
  covariances, so a step costs `O(D^3)` operations in the state dimension
  `D = d (q + 1)`. Block-diagonal factorizations are planned.
- **The Julia ecosystem**, e.g. DifferentialEquations.jl problems with mass
  matrices. ProbNumDiffEq.jl.
- **Non-Gaussian data, MAGI, or several likelihood approximations to compare.**
  rodeo.
- **`ode_filters`** when the model combines the ODE with other information
  (data, conservation laws or an unknown input in the same filter), when
  hyperparameters are selected by the conditional evidence, when the
  linearization error should enter the uncertainty (statistical linearization,
  the IPLS), or when a fit is checked with innovation diagnostics.

## Speed

The [work-precision benchmark](https://github.com/paufisch/ode_filters/tree/main/benchmarks)
times `ode_filters`, probdiffeq and Diffrax on three problems, with priors of
order `q = 3`, in float64 on a CPU. Both probabilistic libraries calibrate the
diffusion at every step (`calibration="dynamic"` and probdiffeq's
`solver_dynamic`). On a fixed grid they then compute the same solution, so the
figures compare runtime alone.

probdiffeq is faster. On a fixed grid it computes the same solution 1.1 to 1.6
times as fast as `ode_filters`. With adaptive steps it reaches the same
accuracy up to three times as fast on Lotka–Volterra and van der Pol, and on
the logistic problem the two take about the same time. Diffrax's Tsit5, which
computes no covariance, is one to two orders of magnitude faster than either.

<figure markdown="span">
  ![Work-precision diagram of the three libraries with adaptive steps](assets/work-precision.svg)
  <figcaption><strong>Adaptive steps.</strong> Relative trajectory RMSE against
  the median runtime of one solve, sweeping the tolerance (atol = rtol, 1e-3 to
  1e-9). Color: library. Circles with solid lines: first-order linearization
  (EK1, probdiffeq's ts1). Triangles with dashed lines: zeroth order (EK0, ts0).
  Squares: Diffrax's Tsit5. Lower left is better. Points with an error above 1
  are left out.</figcaption>
</figure>

<figure markdown="span">
  ![Work-precision diagram of the three libraries on a fixed grid](assets/work-precision-fixed.svg)
  <figcaption><strong>Fixed grid.</strong> As above, but every solver runs the
  same uniform grid of 80 to 1280 steps, Diffrax with a constant step
  size.</figcaption>
</figure>

Measured on 2026-10-16 on an Apple M4 Pro with JAX 0.8.0, probdiffeq 0.8.0,
Diffrax 0.7.2 and `ode_filters` 0.8.0. Each runtime is the median of 20 timing
rounds after compilation. The interquartile range is at most 9 % of it and
smaller than the markers. Timings depend on the machine.

## Classical solvers

[Diffrax](https://docs.kidger.site/diffrax/) and SciPy's
[`solve_ivp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
return a point estimate of the trajectory. They compute no covariance and are
faster for the same accuracy (see [Speed](#speed)), so they are the better
choice when only the solution is needed.
