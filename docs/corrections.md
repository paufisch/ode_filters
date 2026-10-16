# Linearization schemes

A step of a probabilistic ODE filter involves two independent choices: what to
observe, and how to linearize it for the Gaussian update. The measurement model
defines what is observed, namely the ODE residual and conservation laws.
External data form a separate observation model, passed to the solvers as
`measurements=`. `ode_filters` separates the two choices. The
measurement model owns the residual `g` and its Jacobian, and a
**`Correction`** owns the transition from the predicted mean to the updated
posterior. Any measurement model composes with any correction.

## Available corrections

`TaylorCorrection(order=1)` is the first-order extended Kalman linearization
(EK1), which uses the full Jacobian of the vector field. It is the default of
every function that takes `correction=`, and it reproduces the historical
behavior exactly.

`TaylorCorrection(order=0)` is the zeroth-order linearization (EK0). It treats
the vector field as locally constant, so the ODE-defect rows of the Jacobian
reduce to the selection matrix `E_lhs` (`E1` for first-order systems, `E2` for
second-order ones), and no Jacobian of the vector field is evaluated. A step of
EK0 therefore costs less than a step of EK1. EK0 is the classic `EK0` / `ts0`
solver.

`IteratedTaylorCorrection(max_iters=k)` is the iterated EKF (IEKF). It runs a
single forward pass in which the update of each step relinearizes at the
updated mean in `k` Gauss-Newton passes, a number fixed in advance
(`max_iters=1` reproduces EK1). This reduces the local linearization error on
nonlinear problems. The fixed iteration count keeps it reverse-mode
differentiable. (The iterated smoother over the whole trajectory, the IEKS, is
a separate construct;
see [Iterated smoothing](#iterated-smoothing-iekf-ieks-and-ipls).)

`QuadratureCorrection(rule=..., n_nodes=..., max_iters=...)` is statistical
linear regression (SLR), the linearization of the statistically linearized
filter (SLF) and the sigma-point filters. Instead of the tangent at the
predicted mean, it fits the affine surrogate over the predictive distribution
and evaluates the Gaussian expectations by quadrature. Two things change
relative to EK1. `H` becomes the expected Jacobian, and the covariance `Omega`
of the linearization residual is added to the measurement noise, so the
innovation covariance is `H P H^T + Omega + R`. For an affine vector field both
changes vanish, so on linear problems this correction coincides with EK1. With
`max_iters > 1` it becomes the iterated posterior-linearization filter (IPLF).
See [Statistical linearization](#statistical-linearization) below.

## Choosing a correction

Pass `correction=` to `gaussian_filter`. The default is EK1:

```python
import jax.numpy as np
from ode_filters import gaussian_filter, TaylorCorrection
from ode_filters.information import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization


def vf(x, *, t):
    return x * (1 - x)


prior = IWP(q=2, d=1)
m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.1]), q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)
ts = np.linspace(0.0, 5.0, 51)

# Zeroth-order (EK0) solve:
result = gaussian_filter(
    m_0, P_0_sqr, prior, measure, ts, correction=TaylorCorrection(order=0)
)
```

EK0 and EK1 agree exactly when the vector field is constant, since its Jacobian
is then zero. For nonlinear dynamics they differ, and EK1 is generally more
accurate per step.

## Calibration uses the same linearization

Each step linearizes once, at the predicted mean, and both the diffusion
calibration and the update use that linearization: an EK0 filter estimates
`sigma^2` with the EK0 Jacobian, an EK1 filter with the EK1 one. (A statistical
linearization needs the predicted covariance it is calibrating, so a
`QuadratureCorrection` calibrates with the EK1 linearization at the mean.) The
per-component policy `DiagonalCalibration(jacobian="ek0")` (shorthand
`"diagonal_ekf0"`) always uses the ODE's left-hand-side selector (`E1` for a
first-order ODE, `E2` for a second-order one), whatever the correction. See
[Diffusion Calibration](calibration.md).

## Statistical linearization

The residual `g(X) = E_lhs X - f(E_args X)` is affine in the state except
through the arguments of the vector field, so by Stein's identity the SLR
quantities reduce to expectations under the projected marginal
`N(E_args m, E_args P E_args^T)`:

```
H     = E[jacobian_g]                        (= E_lhs - E[J_f] E_args)
c     = E[g] - H m
Omega = Var[g] - H P H^T = Var[g(X) - H X - c]
```

The quadrature therefore runs in the `p = E_args.shape[0]` dimensions of the
vector field's arguments (`p = d` for a first-order ODE) instead of the
`d (q+1)` dimensions of the state. The tensor-product Gauss-Hermite rule with
`m` nodes per dimension has `m^p` nodes, a count set by the arguments and not by
the state, which keeps this spectrally accurate rule affordable. For the same
reason, the sigma-point instability at `q >= 3` reported in the literature does
not apply here. The filter takes no square root of the ill-conditioned full
`P`, only of the `p x p` projected block.

**Choosing the rule.** `"gauss_hermite"` (default) is exact for polynomials of
degree `2 * n_nodes - 1`. A vector field of polynomial degree `k` needs
`n_nodes >= ceil((k+1)/2)` for `H` and `c`, and `n_nodes >= ceil((2k+1)/2)` for
`Omega` as well. A cubic field (Duffing, van der Pol) is therefore reproduced
exactly at `n_nodes=4`, and a quadratic one (Lotka-Volterra, SIRD) at
`n_nodes=3`. `"cubature"` is the third-degree spherical rule. It has `2p`
nodes, so its cost is linear in the dimension, and it is the general fallback
for non-polynomial or high-dimensional fields. The classic scaled unscented
transform is not offered, because its `beta` term puts a negative weight on the
center point, which breaks the square-root construction of `Omega`.

**Custom measurement models should declare `E_args`.** The ODE-information
models stack their argument selectors (`E0`; `E0, E1` for a second-order ODE;
plus `E0_hidden` with a hidden input). A model without `E_args` (a
`BlackBoxMeasurement`) is integrated over the whole state, which is correct but
costs `n_nodes ** state_dim` nodes. If the `E_args` of a custom model omits a
coordinate in which the residual is nonlinear, the linearization is wrong and no
error is raised. Validate such a model with
`check_arg_projection(measure, state, t=...)`, which perturbs the state along
the null space of `E_args` and checks that the Jacobian does not change.

**When it matters.** The closure error scales with the predictive spread of
the vector field's arguments. In a pure ODE solve this spread vanishes as
`h -> 0`, so the statistical linearization converges to EK1 and changes
nothing. It changes the result when the spread is bounded below, as with a
latent force, sparse or noisy data, or large steps, because the Jensen gap then
does not vanish under grid refinement.

## Frozen linearizations: `affine_filter`

Linearized along a fixed trajectory, the ODE filter is a linear-Gaussian
state-space model. `taylor_surrogate` (EK1 at given points) and `slr_surrogate`
(statistical linearization over given Gaussians) build the observation part of
that model, one affine model per step, as an `AffineSurrogate`. `affine_filter`
filters it and returns the same `FilterResult` as `gaussian_filter`, so
`rts_smoother` and every other function that takes a `FilterResult` work on it
unchanged.

```python
from ode_filters import affine_filter, gaussian_filter, taylor_surrogate

ek1 = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration="none")
surrogate = taylor_surrogate(measure, ek1.m_pred, ts=ts[1:])
frozen = affine_filter(m_0, P_0_sqr, prior, surrogate, ts)  # reproduces ek1
```

Step `i` of the surrogate updates at `ts[i + 1]`, so a linearization trajectory
consists of the states at nodes `1..N`. With the linearization frozen, the
evidence is a smooth function of the hyperparameters. The iterated smoother
below iterates this frozen linearization.

## Iterated smoothing: IEKF, IEKS and IPLS

Three schemes iterate a linearization. They differ in what they iterate and
over which part of the trajectory:

| Scheme | Iterates | Over | API |
| --- | --- | --- | --- |
| IEKF | the linearization point (Gauss-Newton on the per-step MAP) | one step, within a forward pass | `IteratedTaylorCorrection` |
| IEKS | the linearization point, at the smoothed means | the whole trajectory | `iterated_smoother(linearization="taylor")` |
| IPLS | the spread of a statistical linearization, the smoothed marginals | the whole trajectory | `iterated_smoother(linearization="slr")` |

The IEKS is Gauss-Newton on the MAP objective of the whole trajectory (Bell,
1994; for ODE filters, Tronarp, Saerkkae & Hennig, 2021). Each pass linearizes
at the smoothed means of the previous pass, and filters and smooths the
resulting linear-Gaussian model. The iteration converges to the MAP trajectory.
The IPLS (Garcia-Fernandez, Svensson & Saerkkae, 2017) refits a statistical
linearization over the smoothed marginals instead and converges to a
moment-matched Gaussian.

```python
from ode_filters import iterated_smoother

# measurements: a list of Measurement objects, or None for the ODE alone
result = iterated_smoother(
    m_0, P_0_sqr, prior, measure, ts,
    linearization="taylor", n_iters=5, measurements=measurements,
)
result.m, result.P_sqr   # smoothed trajectory of the final pass
result.filter            # its FilterResult (evidence, innovations)
result.surrogate         # the linearization it used
result.step              # RMS change of the means per pass
```

Without `init`, the first linearization point is a single filter and smoother
pass with the matching correction, and `n_iters=0` returns exactly this pass. A
comparison of `n_iters=0` with `n_iters > 0` therefore isolates the effect of
iterating. `init=` warm-starts from a given trajectory (e.g. a converged run at
nearby hyperparameters), and `diagnostics=` records any function of each pass's
`IterationState`.

**Which one when.** The IEKF improves the result when the linearization error
of a single step is large, as for large steps on a stiff or strongly nonlinear
field. It cannot use information from later steps. The trajectory-level
schemes condition on all of the information. When data or a latent force keep
the spread of the vector field's arguments from vanishing, this is what reduces
the spread, because a single ODE-residual update removes variance mainly in the
direction of the highest derivative. Use the IEKS for a MAP estimate, and the
IPLS where the linearization error itself should enter the uncertainty.

The diffusion is fixed across passes (a per-pass `sigma^2` would change with
the linearization it is conditioned on), so the prior carries its own scale, as
with `calibration="none"`. Reverse-mode differentiation works from a singular
`P_0_sqr`, including the Dirac that `taylor_mode_initialization` returns.

## Writing a custom correction

`Correction` is an `equinox.Module`, and a filter step calls it twice. First,
`linearize(measure, m, *, t)` returns the `AffineModel` at the predicted mean.
This linearization drives the diffusion calibration, which runs before the
predicted covariance exists, and its default is EK1. Second,
`correct(measure, x, model, *, t)` returns `(posterior, innovation)` as
`Gaussian`s. Because the correction owns the update, it can iterate it, and the
IEKF and the statistical linearization use this to iterate their updates.

## See also

- [How to choose](how-to-choose.md): choosing the prior, the order and the
  correction.
- API reference for [`ode_filters.linearization`](api/linearization/index.md) (`Correction`, `TaylorCorrection`, the surrogates) and [`ode_filters.filters`](api/filters/index.md) (`affine_filter`, `iterated_smoother`).
