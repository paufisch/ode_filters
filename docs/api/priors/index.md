# Priors

A prior is a Gauss–Markov process over the solution and its first `q` derivatives;
it encodes the smoothness assumed before seeing the ODE. See
[How to choose](../../how-to-choose.md) for picking one.

| Prior | Process | Key parameters |
| --- | --- | --- |
| `IWP` | `q`-times integrated Wiener process (the default) | `q` (smoothness order), `d` (dimension), `Xi` (structural scale) |
| `MaternPrior` | Matern process of smoothness `nu = q + 1/2`, unit variance | `q` (smoothness order), `d` (dimension), `length_scale` (the textbook `l` of Rasmussen & Williams 2006, eq. 4.14; SDE rate `sqrt(2q+1) / l`), `Xi` (structural scale) |
| `IOUPPrior` | integrated Ornstein–Uhlenbeck process — the highest derivative follows `dY^(q) = R Y^(q) dt + dW` | `q`, `d`, `rate` (scalar, length-`d` vector, or `d × d` matrix), `Xi` |
| `JointPrior` | a state block stacked with a hidden block | exposes `E0_x`, `E0_hidden` for joint state–parameter models |
| `PrecondIWP`, `PrecondMaternPrior` | the same processes, computed in preconditioned coordinates | better conditioning at high `q` / small `h` |

`q` sets the method's order of accuracy; `Xi` (together with the calibrated
`sigma^2`) sets the size of the uncertainty. Initialize the state from `x0` with
`taylor_mode_initialization(vf, x0, q)` (which returns the Taylor-mode mean and a
zero/Dirac initial covariance). Symbols are defined in
[Notation](../../notation.md).

Every prior exposes the discrete-time transition `A(h)`, drift `b(h)`, process
noise `Q(h)` and its upper-triangular square root `Q_sqr(h)`
(`Q = Q_sqr.T @ Q_sqr`), all in the original coordinates. A filter step consumes
`transition(h)`: the same matrices, or for a preconditioned prior their
better-conditioned form in rescaled coordinates together with the rescaling. For `IWP` /
`PrecondIWP` this is a closed form that avoids ever factorizing the dense,
ill-conditioned `Q(h)`; for `MaternPrior` / `PrecondMaternPrior` / `IOUPPrior` it
is a square-root matrix-fraction decomposition, which likewise never forms the
dense `Q(h)`. Prefer `Q_sqr(h)` over `Q(h)` downstream: at high `q` the dense
`Q(h)` loses positive-definiteness while the square-root factor stays accurate.

## Priors

::: ode_filters.priors.IWP

::: ode_filters.priors.MaternPrior

::: ode_filters.priors.IOUPPrior

::: ode_filters.priors.JointPrior

## Preconditioned priors

The same processes, computed in coordinates rescaled by the step size,
`x = T(h) @ x_bar`: a change of variables inside each filter step, so results
are identical up to round-off and every solver accepts them. A `JointPrior` may
combine plain and preconditioned blocks.

::: ode_filters.priors.PrecondIWP

::: ode_filters.priors.PrecondMaternPrior

## The transition and the base class

::: ode_filters.priors.Transition

::: ode_filters.priors.BasePrior

## Initialization

::: ode_filters.priors.taylor_mode_initialization

::: ode_filters.priors.joint_taylor_initialization
