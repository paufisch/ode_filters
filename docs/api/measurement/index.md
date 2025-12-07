# Measurement Models

The measurement module provides classes for defining ODE constraints and observation
models used in probabilistic ODE solvers. The design separates concerns cleanly:

- **ODE classes** define the dynamical system constraint
- **Constraint dataclasses** define additional constraints (conservation laws, measurements)

## Class Hierarchy

The module provides four ODE information classes, organized by ODE order and whether
hidden states are present:

| Class                                 | ODE Order | Hidden States | Vector Field Signature           |
| ------------------------------------- | --------- | ------------- | -------------------------------- |
| `ODEInformation`                      | 1st       | No            | `vf(x, *, t) -> dx/dt`           |
| `ODEInformationWithHidden`            | 1st       | Yes           | `vf(x, u, *, t) -> dx/dt`        |
| `SecondOrderODEInformation`           | 2nd       | No            | `vf(x, v, *, t) -> d^2x/dt^2`    |
| `SecondOrderODEInformationWithHidden` | 2nd       | Yes           | `vf(x, v, u, *, t) -> d^2x/dt^2` |

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

---

## API Reference

::: ode_filters.measurement
handler: python
options:
show_object_full_path: true
show_source: false
members_order: source
show_signature_annotations: true
