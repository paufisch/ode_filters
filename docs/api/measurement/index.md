# Measurement Models

The measurement module provides classes for defining ODE constraints and observation
models used in probabilistic ODE solvers. The design separates concerns cleanly:

- **ODE classes** define the dynamical system constraint
- **Constraint dataclasses** define additional constraints (conservation laws, measurements)
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

- **`Conservation`**: Time-invariant linear constraints `A @ x = p`
- **`Measurement`**: Time-varying observations `A @ x = z[t]` at specified times

These can be combined freely with any ODE class.

## Usage Examples

### First-Order ODE

```python
from ode_filters.priors import IWP
from ode_filters.measurement import ODEInformation

prior = IWP(q=2, d=1)

def vf(x, *, t):
    return -x  # exponential decay

model = ODEInformation(vf, prior.E0, prior.E1)
```

### Second-Order ODE (e.g., Harmonic Oscillator)

```python
from ode_filters.priors import IWP
from ode_filters.measurement import SecondOrderODEInformation

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
from ode_filters.priors import IWP, JointPrior
from ode_filters.measurement import SecondOrderODEInformationWithHidden, Measurement

# Prior for state x and unknown damping parameter u
prior_x = IWP(q=2, d=1)
prior_u = IWP(q=2, d=1)
prior_joint = JointPrior(prior_x, prior_u)

# Damped oscillator with unknown damping
def vf(x, v, u, *, t):
    omega = 1.0
    return -(omega**2) * x - u * v

# Observations of position
A_obs = prior_joint.E0[:1, :]  # observe x only
obs = Measurement(A=A_obs, z=observations, z_t=obs_times, noise=0.01)

model = SecondOrderODEInformationWithHidden(
    vf,
    E0=prior_joint.E0_x,        # extracts x
    E1=prior_joint.E1,          # extracts dx/dt
    E2=prior_joint.E2,          # extracts d^2x/dt^2
    E0_hidden=prior_joint.E0_hidden,  # extracts u
    constraints=[obs]
)
```

### Adding Conservation Laws

```python
from ode_filters.measurement import ODEInformation, Conservation
import jax.numpy as np

# Energy conservation: x1 + x2 = 1
cons = Conservation(A=np.array([[1.0, 1.0]]), p=np.array([1.0]))

model = ODEInformation(vf, E0, E1, constraints=[cons])
```

### Black-Box Measurement Models

For cases where the standard ODE structure doesn't fit, use `BlackBoxMeasurement` to define an arbitrary differentiable measurement function. The Jacobian is computed automatically via JAX autodiff.

**Example:**

```python
from ode_filters.measurement import BlackBoxMeasurement
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
R = model.get_noise(t=0.0)
```

### Transformed Measurement Models

`TransformedMeasurement` wraps any existing measurement model with a nonlinear state transformation `sigma(state)`. The Jacobian is computed correctly via the chain rule: `J_total = J_g(sigma(state)) @ J_sigma(state)`.

**Use cases:**

- Nonlinear coordinate transformations (e.g., polar to Cartesian)
- Applying constraints like softmax normalization
- Feature extraction before measurement

**Example with autodiff Jacobian:**

```python
from ode_filters.measurement import ODEInformation, TransformedMeasurement
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

model = TransformedMeasurement(
    base_model,
    softmax_transform,
    use_autodiff_jacobian=False,
    sigma_jacobian=sigma_jacobian
)
```

---

## API Reference

::: ode_filters.measurement
handler: python
options:
show_object_full_path: true
show_source: false
members_order: source
show_signature_annotations: true
