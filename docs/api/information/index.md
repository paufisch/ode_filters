# Information

The information module provides classes for defining ODE constraints and observation
models used in probabilistic ODE solvers. The design separates concerns cleanly:

- **ODE classes** define the dynamical system constraint
- **Constraint and data classes** add conservation laws (`Conservation`) and noisy observations (`Measurement`, via an `obs_model`)
- **Black-box models** allow arbitrary user-defined measurement functions
- **Transformed models** apply nonlinear state transformations with proper Jacobians

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

This separation ensures:

- **No runtime conditionals** - Each class has a fixed code path, optimal for JAX JIT
- **Explicit signatures** - The vector field type is clear from the class choice
- **Single responsibility** - Each class handles exactly one case

## Composable Constraints

Additional constraints are added via frozen dataclasses:

- **`Conservation`**: Time-invariant linear constraints `A @ x = p`, passed via
  the ODE class's `constraints=` argument.
- **`Measurement`**: Time-varying observations `A @ x = z[t]` at specified times.
  These are *not* measure constraints; build an observation model with
  `prepare_observations([Measurement(...)], E0, ts)` and pass it as
  `obs_model=` to the solver. Every measurement time must be a node of the grid
  (add off-grid times to it, e.g. `ts = np.union1d(ts, z_t)`); measurements may
  have different times, and each step conditions on the rows that have data.

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

For estimating unknown parameters alongside the state, use `JointPrior` with
the hidden state classes:

```python
import jax
import jax.numpy as np

from ode_filters import gaussian_filter, prepare_observations
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

# Data go through obs_model, not the measure. Measurement.A acts on E0_x @ state
# (the position x), so it is [n_obs, d_x]; observation times lie on the grid.
meas = Measurement(A=np.eye(1), z=x_true[:, None], z_t=obs_times, noise=1e-4)
obs_model = prepare_observations([meas], prior_joint.E0_x, ts)

# State layout [x, x', x'', x''', u]: start at rest at x = 1, u unknown
m_0 = np.array([1.0, 0.0, -1.0, 0.0, 0.0])
P_0_sqr = np.diag(np.array([0.0, 0.0, 0.0, 1.0, 1.0]))

result = gaussian_filter(m_0, P_0_sqr, prior_joint, model, ts, obs_model=obs_model)
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

For cases where the standard ODE structure doesn't fit, use `BlackBoxMeasurement` to define an arbitrary differentiable measurement function. The Jacobian is computed automatically via JAX autodiff.

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
pass `ode_dim=` to say how many leading rows are ODE defect.

### Transformed Measurement Models

`TransformedMeasurement` wraps any existing measurement model with a nonlinear state transformation `sigma(state)`. The Jacobian of the mean is composed correctly via the chain rule: `J_total = J_g(sigma(state)) @ J_sigma(state)`. Note, however, that only the mean/linearization is transformed: the measurement-noise covariance `R` is delegated to and taken unchanged from the base model (see `noise_sqr`), and is *not* propagated through `sigma`. `R` must therefore already be expressed in the base model's post-transform coordinates.

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

For performance-critical applications, you can provide a custom Jacobian:

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
