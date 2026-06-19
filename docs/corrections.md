# Linearization Schemes (Corrections)

A probabilistic ODE filter step has two independent concerns: *what* to observe
(the measurement model -- the ODE residual, conservation laws, data) and *how* to
linearize it for the Gaussian update. `ode_filters` separates these. The
measurement model owns the residual `g` and its Jacobian; a **`Correction`**
strategy owns the predicted-mean -> updated-posterior transition. Any measurement
model composes with any correction.

## Available corrections

`TaylorCorrection(order=1)` -- the first-order extended Kalman linearization
(EK1), using the full vector-field Jacobian. This is the default everywhere and
reproduces the historical behavior exactly.

`TaylorCorrection(order=0)` -- the zeroth-order linearization (EK0): the vector
field is treated as locally constant, so the ODE-defect rows of the Jacobian
reduce to the selection matrix `E1` (no vector-field Jacobian). EK0 is cheaper --
no Jacobian -- and is the classic `EK0` / `ts0` solver.

`IteratedTaylorCorrection(max_iters=k)` -- the iterated EKF (IEKF): a single
forward pass in which *each step's update* relinearizes at the updated mean for
`k` fixed Gauss-Newton passes (`max_iters=1` reproduces EK1). This reduces local
linearization error on nonlinear problems. The fixed iteration count keeps it
reverse-mode differentiable, so it works inside [parameter estimation](parameter-estimation.md).
(The whole-trajectory iterated *smoother*, IEKS, is a separate construct and is
not yet implemented.)

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

## Writing a custom correction

`Correction` is an `equinox.Module`; subclass it and implement
`correct(measure, m_pred, P_pred_sqr, *, t) -> CorrectionResult`. Returning a
finished `CorrectionResult` (rather than a one-shot `(H, c)`) lets a correction
own its update loop -- the seam that iterated (IEKF) and sigma-point (UKF/SLR)
schemes will use.

## Status

EK0, EK1, and IEKF ship today, selectable on `gaussian_filter` and the inference
API (`ODEFilter` / `marginal_loglik`). Sigma-point (UKF/SLR) corrections, the
whole-trajectory IEKS, and `correction=` on the *preconditioned* path are
planned.

## See also

- [Parameter Estimation](parameter-estimation.md)
- API reference for `ode_filters.filters` (`Correction`, `TaylorCorrection`).
