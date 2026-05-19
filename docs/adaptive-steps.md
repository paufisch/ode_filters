# Adaptive Step-Size Control

`ekf1_sqr_adaptive_loop` chooses its own step size during integration. Every
proposed step is evaluated against a tolerance on the local error; rejected
steps are retried at a smaller `h`, accepted steps update a Gustafsson-style
PI controller. The local error estimate is computed from quantities the EKF
step already produces, so adaptivity adds essentially no per-step cost beyond
the controller arithmetic.

## Basic usage

```python
import jax.numpy as np
from ode_filters.filters import ekf1_sqr_adaptive_loop, rts_sqr_smoother_loop
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization


def vf(x, *, t):
    return x * (1 - x)


prior = IWP(q=2, d=1)
mu_0, S0 = taylor_mode_initialization(vf, np.array([0.1]), q=2)
measure = ODEInformation(vf, prior.E0, prior.E1)

result = ekf1_sqr_adaptive_loop(
    mu_0, S0, prior, measure, (0.0, 5.0),
    atol=1e-5, rtol=1e-3,
)

print(f"{len(result.h_seq)} accepted, {result.n_rejected} rejected")
print(f"step range: {min(result.h_seq):.3g} ... {max(result.h_seq):.3g}")
```

`result` is an `AdaptiveLoopResult` named tuple whose accepted-step sequences
are laid out so the smoother can consume them directly:

```python
m_smooth, P_smooth_sqr = rts_sqr_smoother_loop(
    result.m_seq[-1], result.P_seq_sqr[-1],
    result.G_back_seq, result.d_back_seq, result.P_back_seq_sqr,
    N=len(result.h_seq),
)
```

## Tolerances

`atol` and `rtol` follow the classical adaptive-RK convention. The
tolerance-weighted error norm is

```
err = sqrt(mean((D / (atol + rtol * |x|))^2))
```

where `D` is the per-step local error estimate (per-component standard
deviation of the calibrated process noise on the function-value axis) and `x`
is the predicted function value. A step is accepted when `err <= 1`.

Reasonable defaults: `atol=1e-4`, `rtol=1e-2`. Tighten by an order of
magnitude when you need higher accuracy; loosen on smooth problems where the
prior is the binding constraint anyway.

## Controllers

Two controllers ship with the library; both implement the
`StepSizeController` protocol (a single `propose(h, err, err_prev)` method)
and are interchangeable via the `controller=` argument.

### `PController` -- proportional (memoryless)

```
h_new = h * safety * err^(-alpha)
```

The simplest controller: it sees only the current step's error. Default
`alpha = 1 / order`. Sometimes called the "I-controller" or "elementary
controller" in the adaptive-ODE-solver literature
(Hairer-Wanner-Norsett, Soederlind) because `h` itself is the integral
of the per-step log-step adjustment. It is the controller that scipy's
`solve_ivp` uses for the elementary Runge-Kutta methods.

```python
from ode_filters.filters import PController

result = ekf1_sqr_adaptive_loop(
    mu_0, S0, prior, measure, (0.0, 5.0),
    atol=1e-5, rtol=1e-3,
    controller=PController(order=2),
)
```

Use when you want predictable, memoryless behaviour or when debugging --
the `PController` cannot oscillate around the tolerance.

### `PIController` -- proportional + integral (the default)

```
h_new = h * safety * err^(-alpha) * (err_prev / err)^(beta)
```

Adds a Gustafsson PI term using the previously accepted step's error.
This damps oscillations around `err = 1` that a pure proportional
controller can exhibit when the residual is noisy. Defaults follow
Gustafsson (1991): `alpha = 0.7 / order`, `beta = 0.4 / order`,
`safety = 0.9`. On the first step or right after a reject the I-term is
dropped, and the controller falls back to the proportional form.

```python
from ode_filters.filters import PIController

result = ekf1_sqr_adaptive_loop(
    mu_0, S0, prior, measure, (0.0, 5.0),
    atol=1e-5, rtol=1e-3,
    controller=PIController(order=2, alpha=0.4, beta=0.1, safety=0.8),
)
```

Setting `beta=0` on `PIController` is *almost* equivalent to
`PController`, except `PIController` defaults `alpha` to `0.7 / order`
(tuned to pair with the I-term) while `PController` defaults it to
`1 / order` (the standard solo-controller value). For clarity, prefer
the dedicated `PController` class when you want proportional-only.

### Common knobs

Both controllers expose `safety`, `min_factor`, and `max_factor`. The
step-ratio is clipped to `[min_factor, max_factor]` last, after all gain
terms.

## Step bounds

- `h_init`: first step. Defaults to one hundredth of the span.
- `h_min`: integrating below this raises `RuntimeError`. Triggered when the
  prior order is too low for the requested tolerance.
- `h_max`: cap on any proposed step. Defaults to the full span.

## Diagnostics

Every accepted step records its per-step quasi-MLE in `result.sigma_sqr_seq`.
A well-specified problem produces $\widehat{\sigma}^2_n$ values that stay
within an order of magnitude; large spikes indicate the prior is too smooth
for that regime (often during a stiff transient -- this is fine; the
controller responds by shrinking `h`).

`result.h_seq` plotted against `result.t_seq[:-1]` shows how the controller
adapted -- big steps on smooth stretches, small steps near features.

## Calibration off

Pass `calibrate=False` to skip the rescaling of stored covariances. The
per-step quasi-MLE is still computed and returned for diagnostics, but the
posterior covariances are reported uncalibrated. Useful for testing.

## See also

- [Diffusion Calibration](calibration.md) -- the per-step quasi-MLE that
  feeds the local error estimate.
- Wiki notes *local-error-and-adaptive-stepping*,
  *calibration-pipeline*, *whitened-residual-diagnostics* for derivations
  and failure modes.
