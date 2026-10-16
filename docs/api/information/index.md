# Information

The information module defines the ODE constraints and the observation models
of a probabilistic ODE solver. Its classes form four groups:

- **ODE classes** define the constraint that the dynamical system imposes.
- **Constraint and data classes** add conservation laws (`Conservation`) and
  noisy observations (`Measurement`, passed to a solver as `measurements=`).
- **Black-box models** accept an arbitrary user-defined measurement function.
- **Transformed models** apply a nonlinear state transformation and compose its
  Jacobian by the chain rule.

## Class Hierarchy

### ODE Information Classes

The module provides four ODE information classes, organized by ODE order and whether
hidden states are present:

| Class                                 | ODE Order | Hidden States | Vector Field Signature           |
| ------------------------------------- | --------- | ------------- | -------------------------------- |
| `ODEInformation`                      | 1st       | No            | `vf(x, *, t) -> dx/dt`           |
| `ODEInformationWithHidden`            | 1st       | Yes           | `vf(x, u, *, t) -> dx/dt`        |
| `SecondOrderODEInformation`           | 2nd       | No            | `vf(x, v, *, t) -> d^2x/dt^2`    |
| `SecondOrderODEInformationWithHidden` | 2nd       | Yes           | `vf(x, v, u, *, t) -> d^2x/dt^2` |

### Flexible Measurement Classes

| Class                    | Description                                         |
| ------------------------ | --------------------------------------------------- |
| `BlackBoxMeasurement`    | User-defined `g(state, t)` with autodiff Jacobian   |
| `TransformedMeasurement` | Wraps any model with nonlinear state transformation |

The separation into classes has three consequences:

- **No runtime conditionals.** Each class has a fixed code path, so a compiled
  solve contains no branch over the type of the model.
- **Explicit signatures.** The class determines the signature of the vector
  field.
- **Single responsibility.** Each class handles exactly one case.

## Composable Constraints

Further constraints are added with frozen dataclasses:

- **`Conservation`**: Time-invariant linear constraints `A @ x = p`, passed via
  the ODE class's `constraints=` argument.
- **`Measurement`**: Time-varying observations `A @ x = z[t]` at specified times.
  They are not constraints of the measurement model. Pass them to a solver as
  `measurements=[Measurement(...)]`, and to `marginal_loglik`, `fit` or
  `information_criteria` as the data. These functions convert them into an
  `ObsModel` with `prepare_observations([Measurement(...)], prior.E0_state, ts)`.
  Call `prepare_observations` where code needs the `ObsModel` itself
  (`fenrir_loglik` and `conditional_loglik` called directly, or one conversion
  reused in several solves), and pass the result as `obs_model=`. `A` acts on
  the solution `x = prior.E0_state @ state` (for a `JointPrior`, the solution
  without the input), or on the full state with `full_state=True`. A time
  between two grid nodes becomes a data-only node, where the filter conditions
  on the data without an update on the ODE
  information (see [sharp bits](../../sharp-bits.md#data-between-grid-nodes-add-nodes-without-ode-information)). Measurements may have
  different times, and each step conditions on the rows that have data. Built
  inside `jax.jit` from concrete times, a `Measurement` may carry traced data.

## Usage Examples

### First-Order ODE

```python
from ode_filters.priors import IWP
from ode_filters.information import ODEInformation

prior = IWP(q=2, d=1)

def vf(x, *, t):
    return -x  # exponential decay

model = ODEInformation(vf, prior.E0, prior.E1)
```

### Second-Order ODE (e.g., Harmonic Oscillator)

```python
from ode_filters.priors import IWP
from ode_filters.information import SecondOrderODEInformation

prior = IWP(q=3, d=1)
omega = 2.0

def vf(x, v, *, t):
    return -(omega**2) * x  # harmonic oscillator

model = SecondOrderODEInformation(vf, prior.E0, prior.E1, prior.E2)
```

### Joint State-Parameter Estimation

To estimate unknown parameters together with the state, use `JointPrior` with
the hidden-state classes:

```python
import jax
import jax.numpy as np

from ode_filters import gaussian_filter
from ode_filters.information import Measurement, SecondOrderODEInformationWithHidden
from ode_filters.priors import IWP, JointPrior

jax.config.update("jax_enable_x64", True)

# Prior for the state x (q >= 2, so E2 exists) and the unknown damping u, a
# slowly drifting constant (a random walk with a small diffusion)
prior_joint = JointPrior(IWP(q=3, d=1), IWP(q=0, d=1, Xi=1e-4 * np.eye(1)))


def vf(x, v, u, *, t):  # damped oscillator x'' = -x - u x'
    return -x - u * v


model = SecondOrderODEInformationWithHidden(
    vf,
    E0=prior_joint.E0_x,  # extracts x
    E1=prior_joint.E1,  # extracts dx/dt
    E2=prior_joint.E2,  # extracts d^2x/dt^2
    E0_hidden=prior_joint.E0_hidden,  # extracts u
)

# Synthetic position data from the true damping u = 0.3
ts = np.linspace(0.0, 10.0, 201)
gamma = 0.15
omega_d = np.sqrt(1.0 - gamma**2)
obs_times = ts[10::10]
x_true = np.exp(-gamma * obs_times) * (
    np.cos(omega_d * obs_times) + gamma / omega_d * np.sin(omega_d * obs_times)
)

# Data go to the solver as measurements=, not into the measure. Measurement.A
# acts on prior_joint.E0_state @ state = E0_x @ state (the position x), so it is
# [n_obs, d_x]; observation times lie on the grid.
meas = Measurement(A=np.eye(1), z=x_true[:, None], z_t=obs_times, noise=1e-4)

# State layout [x, x', x'', x''', u]: start at rest at x = 1, u unknown
m_0 = np.array([1.0, 0.0, -1.0, 0.0, 0.0])
P_0_sqr = np.diag(np.array([0.0, 0.0, 0.0, 1.0, 1.0]))

result = gaussian_filter(m_0, P_0_sqr, prior_joint, model, ts, measurements=[meas])
u_hat = prior_joint.E0_hidden @ result.m[-1]  # estimated damping: ~0.297
```

### Adding Conservation Laws

```python
from ode_filters.information import ODEInformation, Conservation
import jax.numpy as np

# Energy conservation: x1 + x2 = 1
cons = Conservation(A=np.array([[1.0, 1.0]]), p=np.array([1.0]))

model = ODEInformation(vf, E0, E1, constraints=[cons])
```

### Black-Box Measurement Models

When the residual does not have the structure of the ODE information classes,
`BlackBoxMeasurement` defines an arbitrary differentiable measurement function.
JAX computes its Jacobian by automatic differentiation.

**Example:**

```python
from ode_filters.information import BlackBoxMeasurement
import jax.numpy as np

# Custom nonlinear observation: observe squared position and velocity
def custom_g(state, *, t):
    x, v = state[0], state[1]
    return np.array([x**2, v])

model = BlackBoxMeasurement(
    g_func=custom_g,
    state_dim=6,      # full state dimension (e.g., q=2, d=2 -> D=6)
    obs_dim=2,        # observation dimension
    noise=0.01        # measurement noise variance
)

# Use like any other measurement model
H, c = model.linearize(state, t=0.0)
R = model.noise_sqr(t=0.0)
```

As the `measure` of a solver, `g` plays the role of the ODE residual: the filter
conditions on `g(state) = 0`, and all rows drive the diffusion calibration. If
`g` stacks further rows after the ODE residual (e.g. conservation constraints),
pass the number of leading rows that form the ODE defect as `ode_dim=`.

### Transformed Measurement Models

`TransformedMeasurement` wraps a measurement model with a nonlinear state
transformation `sigma(state)`. The Jacobian of the mean follows from the chain
rule, `J_total = J_g(sigma(state)) @ J_sigma(state)`. Only the mean and the
linearization are transformed. The measurement-noise covariance `R` is taken
unchanged from the base model (see `noise_sqr`) and is not propagated through
`sigma`, so `R` must be expressed in the coordinates of the base model after
the transformation.

**Use cases:**

- Nonlinear coordinate transformations (e.g., polar to Cartesian)
- Applying constraints like softmax normalization
- Feature extraction before measurement

**Example with autodiff Jacobian:**

```python
from ode_filters.information import ODEInformation, TransformedMeasurement
from ode_filters.priors import IWP
import jax
import jax.numpy as np

# Base ODE model
def vf(x, *, t):
    return -x

prior = IWP(q=2, d=2)
base_model = ODEInformation(vf, prior.E0, prior.E1)

# Apply softmax to ensure state components sum to 1
def softmax_transform(state):
    x = state[:2]  # extract position components
    x_normalized = jax.nn.softmax(x)
    return state.at[:2].set(x_normalized)

model = TransformedMeasurement(base_model, softmax_transform)

# Jacobian includes chain rule automatically
H, c = model.linearize(state, t=0.0)
```

**Example with explicit Jacobian:**

Pass `sigma_jacobian=` to replace the automatic Jacobian of `sigma`, e.g. by a
closed form:

```python
def sigma_jacobian(state):
    # Custom Jacobian implementation
    return jax.jacfwd(softmax_transform)(state)

model = TransformedMeasurement(base_model, softmax_transform, sigma_jacobian=sigma_jacobian)
```

## ODE information

::: ode_filters.information.ODEInformation

::: ode_filters.information.ODEInformationWithHidden

::: ode_filters.information.SecondOrderODEInformation

::: ode_filters.information.SecondOrderODEInformationWithHidden

## Constraints and observations

::: ode_filters.information.Conservation

::: ode_filters.information.Measurement

::: ode_filters.information.prepare_observations

::: ode_filters.information.ObsModel

## Custom measurement models

Any object with the `MeasurementModel` interface works as a solver's `measure`.
Four further members are optional. One feature needs each of them, and each has
its own protocol, which the feature checks with `isinstance` before the call:

| Protocol             | Member          | Needed by                             |
| -------------------- | --------------- | ------------------------------------- |
| `HasLHSSelector`     | `E_lhs`         | `DiagonalCalibration(jacobian="ek0")` |
| `EK0Linearizable`    | `linearize_ek0` | `TaylorCorrection(order=0)`           |
| `HasNoiseCovariance` | `R`             | `TransformedMeasurement.R`            |
| `NoiseSettable`      | `with_noise`    | `TransformedMeasurement.with_noise`   |

A model without an optional member works with every feature that does not need
it. A feature whose member is missing raises an error that names the member
(`TypeError` for `TaylorCorrection(order=0)`, `AttributeError` for the others).
The ODE-information models implement all four protocols. `BlackBoxMeasurement`
implements `HasNoiseCovariance` and `NoiseSettable`, and
`TransformedMeasurement` forwards both to its base model.

::: ode_filters.information.MeasurementModel

::: ode_filters.information.HasLHSSelector

::: ode_filters.information.EK0Linearizable

::: ode_filters.information.HasNoiseCovariance

::: ode_filters.information.NoiseSettable

::: ode_filters.information.BaseODEInformation

::: ode_filters.information.BlackBoxMeasurement

::: ode_filters.information.TransformedMeasurement
