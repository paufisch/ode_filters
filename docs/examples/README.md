# ODE Filters Examples

This directory contains comprehensive tutorials and examples for the `ode_filters` package.

## Tutorial Structure

### 1. **quickstart.ipynb** - Getting Started
- Basic first-order ODE (Logistic equation)
- Introduction to the filtering and smoothing workflow
- Recommended starting point for new users

### 2. **examples.ipynb** - Basic First-Order ODEs
- **Logistic equation**: Population growth model
- **Lotka-Volterra**: Predator-prey dynamics
- **SIR model**: Epidemiological model

### 3. **second-order-systems.ipynb** - Second-Order ODEs
- Damped harmonic oscillator
- Joint state-parameter estimation (basic example)
- Working with second-order systems

### 4. **advanced-features.ipynb** - Advanced Capabilities ⭐ NEW
This comprehensive tutorial covers all advanced features:

#### 4.1 First-Order ODEs with Hidden States
- Joint state-parameter estimation
- Exponential decay with unknown rate parameter
- Using `JointPrior` and `ODEInformationWithHidden`

#### 4.2 Second-Order ODEs with Hidden States
- Damped harmonic oscillator with unknown damping
- Inferring physical parameters from observations
- Using `SecondOrderODEInformationWithHidden`

#### 4.3 Conservation Constraints
- SIR model with population conservation
- Enforcing algebraic constraints: `A·x = p`
- Using the `Conservation` class

#### 4.4 Linear Measurements
- Time-varying observations at discrete points
- Lotka-Volterra with sparse predator observations
- Using the `Measurement` class

### 5. **additional_info.ipynb** - Conservation and Measurements
- SIR model with conservation constraints
- Combining conservation and measurements
- Real-world data integration examples

## Feature Coverage Matrix

| Feature | Quickstart | Examples | Second-Order | Advanced | Additional |
|---------|-----------|----------|--------------|----------|------------|
| First-order ODEs | ✅ | ✅ | - | ✅ | ✅ |
| Second-order ODEs | - | - | ✅ | ✅ | - |
| Hidden states (1st order) | - | - | - | ✅ | - |
| Hidden states (2nd order) | - | - | - | ✅ | - |
| Conservation constraints | - | - | - | ✅ | ✅ |
| Time-varying measurements | - | - | - | ✅ | ✅ |
| Multi-dimensional systems | - | ✅ | - | ✅ | ✅ |
| Parameter estimation | - | - | ✅ | ✅ | - |

## Key Classes by Use Case

### Basic ODE Solving
```python
from ode_filters.filters import ekf1_sqr_loop, rts_sqr_smoother_loop
from ode_filters.measurement import ODEInformation
from ode_filters.priors import IWP, taylor_mode_initialization
```

### Second-Order Systems
```python
from ode_filters.measurement import SecondOrderODEInformation
# Use taylor_mode_initialization with order=2
```

### Hidden States (Parameter Estimation)
```python
from ode_filters.measurement import (
    ODEInformationWithHidden,
    SecondOrderODEInformationWithHidden
)
from ode_filters.priors import JointPrior
```

### Constraints
```python
from ode_filters.measurement import Conservation, Measurement
```

### Advanced Measurement Models
```python
from ode_filters.measurement import BlackBoxMeasurement, TransformedMeasurement
```

## Recommended Learning Path

1. **Start here**: `quickstart.ipynb` - Learn the basic workflow
2. **Expand**: `examples.ipynb` - See different first-order ODEs
3. **Second-order**: `second-order-systems.ipynb` - Understand higher-order systems
4. **Advanced**: `advanced-features.ipynb` - Master all advanced capabilities
5. **Real-world**: `additional_info.ipynb` - See practical applications

## Running the Examples

### Using Jupyter
```bash
jupyter notebook examples/
```

### Building Documentation
```bash
uv run mkdocs serve
```
Then visit http://localhost:8000

## Next Steps

After completing these tutorials, explore:
- **API Reference**: Detailed documentation for all functions and classes
- **Source Code**: `ode_filters/` directory for implementation details
- **Tests**: `test/` directory for additional examples and edge cases

## Contributing Examples

If you develop interesting use cases, consider contributing them! See `CONTRIBUTING.md` for guidelines.
