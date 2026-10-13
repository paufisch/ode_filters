# Linearization Schemes (Corrections)

A probabilistic ODE filter step has two independent concerns: *what* to observe
(the measurement model — the ODE residual and conservation laws; external data
observations are a separate `obs_model` path built with `prepare_observations`)
and *how* to linearize it for the Gaussian update. `ode_filters` separates these. The
measurement model owns the residual `g` and its Jacobian; a **`Correction`**
strategy owns the predicted-mean -> updated-posterior transition. Any measurement
model composes with any correction.

## Available corrections

`TaylorCorrection(order=1)` — the first-order extended Kalman linearization
(EK1), using the full vector-field Jacobian. This is the default everywhere and
reproduces the historical behavior exactly.

`TaylorCorrection(order=0)` — the zeroth-order linearization (EK0): the vector
field is treated as locally constant, so the ODE-defect rows of the Jacobian
reduce to the selection matrix `E_lhs` (`E1` for first-order systems, `E2`
for second-order; no vector-field Jacobian). EK0 is cheaper —
no Jacobian — and is the classic `EK0` / `ts0` solver.

`IteratedTaylorCorrection(max_iters=k)` — the iterated EKF (IEKF): a single
forward pass in which *each step's update* relinearizes at the updated mean for
`k` fixed Gauss-Newton passes (`max_iters=1` reproduces EK1). This reduces local
linearization error on nonlinear problems. The fixed iteration count keeps it
reverse-mode differentiable. (The whole-trajectory iterated *smoother*, IEKS, is a
separate construct and is not yet implemented.)

`QuadratureCorrection(rule=..., n_nodes=..., max_iters=...)` — statistical
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

EK0 and EK1 agree exactly when the vector field is constant (zero Jacobian); they
differ for nonlinear dynamics, with EK1 generally more accurate per step.

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

`g(X) = E_lhs X - f(E_args X)` is affine in the state *except* through the vector
field's arguments, so by Stein's identity the SLR quantities reduce to
expectations under the **projected** marginal `N(E_args m, E_args P E_args^T)`:

```
H     = E[jacobian_g]                        (= E_lhs - E[J_f] E_args)
c     = E[g] - H m
Omega = Var[g] - H P H^T = Var[g(X) - H X - c]
```

The quadrature therefore runs in `p = E_args.shape[0]` dimensions — the ODE
dimension — not in the `d (q+1)` state dimension. That is what makes the
spectrally accurate rule affordable (`m^p` is indexed by the ODE's arity), and
it is why the sigma-point instability at `q >= 3` reported in the literature does
not apply here: no square root of the ill-conditioned full `P` is ever taken,
only of the `p x p` projected block.

**Choosing the rule.** `"gauss_hermite"` (default) is exact for polynomials of
degree `2 * n_nodes - 1`. A vector field of polynomial degree `k` needs
`n_nodes >= ceil((k+1)/2)` for `H` and `c`, and `n_nodes >= ceil((2k+1)/2)` for
`Omega` as well — so a cubic (Duffing, van der Pol) is *exactly* reproduced at
`n_nodes=4` and a quadratic (Lotka-Volterra, SIRD) at `n_nodes=3`.
`"cubature"` is the third-degree spherical rule: `2p` nodes, dimension-linear,
the general fallback for non-polynomial or high-dimensional fields. The classic
scaled unscented transform is deliberately not offered — its `beta` term puts a
negative weight on the centre point, which breaks the square-root construction
of `Omega`.

**Custom measurement models should declare `E_args`.** The ODE-information
models stack their argument selectors (`E0`; `E0, E1` for a second-order ODE;
plus `E0_hidden` with a hidden input). A model without `E_args` (a
`BlackBoxMeasurement`) is integrated over the whole state, which is correct but
costs `n_nodes ** state_dim` nodes. A custom model whose `E_args` misses a
coordinate the residual is nonlinear in is silently wrong; validate it with
`check_arg_projection(measure, state, t=...)`, which perturbs along the null
space of `E_args` and checks that the Jacobian is unchanged.

**When it matters.** The closure error scales with the predictive spread of the
vector field's arguments. In a pure ODE solve that spread vanishes as `h -> 0`,
so this collapses back onto EK1 and buys nothing. It becomes load-bearing when
the spread has a floor — a latent force, sparse or noisy data, large steps —
because then the Jensen gap does not vanish under grid refinement.

## Frozen linearizations: `affine_filter`

Linearized along a fixed trajectory, the ODE filter is a linear-Gaussian
state-space model. `taylor_surrogate` (EK1 at given points) and `slr_surrogate`
(statistical linearization over given Gaussians) build that model's observation
part, one affine model per step, as an `AffineSurrogate`; `affine_filter` filters
it and returns the same `FilterResult` as `gaussian_filter`, so `rts_smoother`
and every other tool work on it unchanged.

```python
from ode_filters import affine_filter, gaussian_filter, taylor_surrogate

ek1 = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration="none")
surrogate = taylor_surrogate(measure, ek1.m_pred, ts=ts[1:])
frozen = affine_filter(m_0, P_0_sqr, prior, surrogate, ts)  # reproduces ek1
```

Step `i` of the surrogate updates at `ts[i + 1]`, so a linearization trajectory
is the states at nodes `1..N`. Freezing the linearization is what makes the
evidence a smooth function of the hyperparameters at a fixed linearization, and
what the iterated smoother below iterates.

## Iterated smoothing: IEKF, IEKS and IPLS

Three schemes iterate a linearization. They differ in *what* they iterate and
*over how much* of the trajectory:

| Scheme | Iterates | Over | API |
| --- | --- | --- | --- |
| IEKF | the linearization point (Gauss-Newton on the per-step MAP) | one step, within a forward pass | `IteratedTaylorCorrection` |
| IEKS | the linearization point, at the smoothed means | the whole trajectory | `iterated_smoother(linearization="taylor")` |
| IPLS | the spread of a statistical linearization, the smoothed marginals | the whole trajectory | `iterated_smoother(linearization="slr")` |

The IEKS is Gauss-Newton on the MAP objective of the whole trajectory (Bell,
1994; for ODE filters, Tronarp, Saerkkae & Hennig, 2021): each pass linearizes at
the previous pass's smoothed means, filters and smooths the resulting
linear-Gaussian model, and converges to the MAP trajectory. The IPLS
(Garcia-Fernandez, Svensson & Saerkkae, 2017) refits a statistical linearization
over the smoothed marginals instead and converges to a moment-matched Gaussian.

```python
from ode_filters import iterated_smoother

# obs_model: data from prepare_observations, or None for the ODE alone
result = iterated_smoother(
    m_0, P_0_sqr, prior, measure, ts,
    linearization="taylor", n_iters=5, obs_model=obs_model,
)
result.m, result.P_sqr   # smoothed trajectory of the final pass
result.filter            # its FilterResult (evidence, innovations)
result.surrogate         # the linearization it used
result.step              # RMS change of the means per pass
```

Without `init`, the first linearization point is a one-shot filter and smoother
with the matching correction, and `n_iters=0` returns exactly that, so "does
iterating help?" is a clean comparison. `init=` warm-starts from a given
trajectory (e.g. a converged run at nearby hyperparameters), and `diagnostics=`
records any function of each pass's `IterationState`.

**Which one when.** The IEKF helps when a single step is badly linearized (large
steps on a stiff or strongly nonlinear field); it cannot use later information.
The trajectory-level schemes pool all of the information, which is what shrinks
the spread of the vector field's arguments when data or a latent force keep it
from vanishing: a single ODE-residual update removes variance mainly in the
highest-derivative direction. Use the IEKS for a MAP estimate, the IPLS where
the linearization error itself should enter the uncertainty.

The diffusion is fixed across passes (a per-pass `sigma^2` would chase the
linearization it is conditioned on), so the prior carries its own scale, as with
`calibration="none"`. Reverse-mode differentiation works from a singular
`P_0_sqr`, including the Dirac that `taylor_mode_initialization` returns.

## Writing a custom correction

`Correction` is an `equinox.Module`. A filter step calls it twice: `linearize(measure, m, *, t)`
returns the `AffineModel` at the predicted mean (it drives the diffusion
calibration, before the predicted covariance exists; the default is EK1), and
`correct(measure, x, model, *, t)` returns `(posterior, innovation)` as
`Gaussian`s. Owning the update lets a correction iterate it — the seam the IEKF
and the statistical linearization use.

## See also

- [How to choose](how-to-choose.md) — picking the prior, order, and correction.
- API reference for [`ode_filters.linearization`](api/linearization/index.md) (`Correction`, `TaylorCorrection`, the surrogates) and [`ode_filters.filters`](api/filters/index.md) (`affine_filter`, `iterated_smoother`).
