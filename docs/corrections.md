# Linearization Schemes (Corrections)

A probabilistic ODE filter step has two independent concerns: *what* to observe
(the measurement model -- the ODE residual and conservation laws; external data
observations are a separate `obs_model` path built with `prepare_observations`)
and *how* to linearize it for the Gaussian update. `ode_filters` separates these. The
measurement model owns the residual `g` and its Jacobian; a **`Correction`**
strategy owns the predicted-mean -> updated-posterior transition. Any measurement
model composes with any correction.

## Available corrections

`TaylorCorrection(order=1)` -- the first-order extended Kalman linearization
(EK1), using the full vector-field Jacobian. This is the default everywhere and
reproduces the historical behavior exactly.

`TaylorCorrection(order=0)` -- the zeroth-order linearization (EK0): the vector
field is treated as locally constant, so the ODE-defect rows of the Jacobian
reduce to the selection matrix `E_constraint` (`E1` for first-order systems, `E2`
for second-order; no vector-field Jacobian). EK0 is cheaper --
no Jacobian -- and is the classic `EK0` / `ts0` solver.

`IteratedTaylorCorrection(max_iters=k)` -- the iterated EKF (IEKF): a single
forward pass in which *each step's update* relinearizes at the updated mean for
`k` fixed Gauss-Newton passes (`max_iters=1` reproduces EK1). This reduces local
linearization error on nonlinear problems. The fixed iteration count keeps it
reverse-mode differentiable. (The whole-trajectory iterated *smoother*, IEKS, is a
separate construct and is not yet implemented.)

`QuadratureCorrection(rule=..., n_nodes=..., max_iters=...)` -- statistical
linear regression (the SLF / sigma-point family). Instead of a tangent at the
predicted mean, it fits the affine surrogate *over the predictive spread*, with
the Gaussian expectations evaluated by quadrature. Two things change relative to
EK1: `H` becomes the *expected* Jacobian, and the linearization-residual
covariance `Omega` is added to the measurement noise, so the innovation
covariance is `H P H^T + Omega + R`. Both vanish for an affine vector field, so
this correction is a no-op on linear problems. With `max_iters > 1` it becomes
the iterated posterior-linearization filter (IPLF). See
[Statistical linearization](#statistical-linearization) below.

## Choosing a correction

Pass `correction=` to `gaussian_filter`; it defaults to EK1:

```python
import jax.numpy as np
from ode_filters import gaussian_filter, TaylorCorrection
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization


def vf(x, *, t):
    return x * (1 - x)


prior = IWP(q=2, d=1)
mu_0, S0 = taylor_mode_initialization(vf, np.array([0.1]), q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)

# Zeroth-order (EK0) solve:
result = gaussian_filter(
    mu_0, S0, prior, measure, (0.0, 5.0), N=50,
    correction=TaylorCorrection(order=0),
)
```

EK0 and EK1 agree exactly when the vector field is constant (zero Jacobian); they
differ for nonlinear dynamics, with EK1 generally more accurate per step.

## Calibration uses its own linearization

`correction=` controls only the *update*. Diffusion calibration (the
`calibration` argument) has its own linearization choice, because the ODE-defect
residual that drives `sigma^2` is identical for EK0 and EK1 -- only the
denominator's Jacobian differs. To pair an EK0 update with an EK0-style
(`E1`-based) sigma estimate, use `calibration="diagonal_ekf0"`. See
[Diffusion Calibration](calibration.md).

## Statistical linearization

`g(X) = E_c X - f(E_args X)` is affine in the state *except* through the vector
field's arguments, so by Stein's identity the SLR quantities reduce to
expectations under the **projected** marginal `N(E_args m, E_args P E_args^T)`:

```
H     = E[jacobian_g]                        (= E_c - E[J_f] E_args)
c     = E[g] - H m
Omega = Var[g] - H P H^T = Var[g(X) - H X - c]
```

The quadrature therefore runs in `p = E_args.shape[0]` dimensions -- the ODE
dimension -- not in the `d (q+1)` state dimension. That is what makes the
spectrally accurate rule affordable (`m^p` is indexed by the ODE's arity), and
it is why the sigma-point instability at `q >= 3` reported in the literature does
not apply here: no square root of the ill-conditioned full `P` is ever taken,
only of the `p x p` projected block.

**Choosing the rule.** `"gauss_hermite"` (default) is exact for polynomials of
degree `2 * n_nodes - 1`. A vector field of polynomial degree `k` needs
`n_nodes >= ceil((k+1)/2)` for `H` and `c`, and `n_nodes >= ceil((2k+1)/2)` for
`Omega` as well -- so a cubic (Duffing, van der Pol) is *exactly* reproduced at
`n_nodes=4` and a quadratic (Lotka-Volterra, SIRD) at `n_nodes=3`.
`"cubature"` is the third-degree spherical rule: `2p` nodes, dimension-linear,
the general fallback for non-polynomial or high-dimensional fields. The classic
scaled unscented transform is deliberately not offered -- its `beta` term puts a
negative weight on the centre point, which breaks the square-root construction
of `Omega`.

**Custom measurement models must declare `E_args`.** The base implementation
returns `E0`, correct for first-order models; the second-order and hidden-state
models override it. A subclass whose vector field reads more than `E0 @ state`
must override it too, or the quadrature silently misses the spread in the
omitted directions. Validate with `check_arg_projection(measure, state, t=...)`,
which perturbs along the null space of `E_args` and checks the Jacobian is
unchanged.

**When it matters.** The closure error scales with the predictive spread of the
vector field's arguments. In a pure ODE solve that spread vanishes as `h -> 0`,
so this collapses back onto EK1 and buys nothing. It becomes load-bearing when
the spread has a floor -- a latent force, sparse or noisy data, large steps --
because then the Jensen gap does not vanish under grid refinement.

## Iterated posterior linearization (IPLS)

`ipls_smoother` is the whole-trajectory version: it refits the surrogate at every
step over the **smoothed** marginal and re-runs filter and smoother, for a fixed
number of passes. `n_iters=0` reproduces the one-shot SLR filter plus RTS
exactly, so the "does relinearizing help?" comparison is clean.

```python
from ode_filters import ipls_smoother

result = ipls_smoother(
    mu_0, P0_sqr, prior, measure, (0.0, 5.0), N=200,
    n_iters=3, n_nodes=4, obs_model=obs_model,
)
result.m, result.P_sqr        # smoothed trajectory
result.log_likelihood_obs     # observation-channel evidence
```

Two distinct axes are separated here: `IteratedTaylorCorrection` iterates the
linearization *point* at zero spread (Gauss-Newton, converging to the MAP), while
IPLS iterates the *spread* the fit is taken over (moment matching, converging to
a moment-matched Gaussian). The trajectory-level version is the one with room to
work: a single ODE-residual update removes variance mainly in the
highest-derivative direction, so the per-step posterior spread of the vector
field's arguments is close to the predictive one -- pooling observations across
the trajectory is what actually shrinks it.

Scope: fixed grid, plain (non-preconditioned) priors, fixed process noise (the
diffusion is not recalibrated between passes, since a per-pass `sigma^2` would
chase the linearization it is conditioned on). Reverse-mode differentiation
requires a positive-definite `P_0_sqr`.

## Writing a custom correction

`Correction` is an `equinox.Module`; subclass it and implement
`correct(measure, m_pred, P_pred_sqr, *, t) -> CorrectionResult`. Returning a
finished `CorrectionResult` (rather than a one-shot `(H, c)`) lets a correction
own its update loop -- the seam that the iterated (IEKF) and sigma-point (SLR)
schemes use.

## Status

EK0, EK1, IEKF, statistical linearization (`QuadratureCorrection`, including
IPLF) and the trajectory-level `ipls_smoother` ship today. `QuadratureCorrection`
works on the plain, sequential-observation, and preconditioned paths. The
whole-trajectory IEKS, and `ipls_smoother` on preconditioned priors, are still
open.

## See also

- [How to choose](how-to-choose.md) — picking the prior, order, and correction.
- API reference for `ode_filters.filters` (`Correction`, `TaylorCorrection`).
