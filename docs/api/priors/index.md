# Priors

A prior is a Gauss–Markov process over the solution and its first `q` derivatives;
it encodes the smoothness assumed before seeing the ODE. See
[How to choose](../../how-to-choose.md) for picking one.

| Prior | Process | Key parameters |
| --- | --- | --- |
| `IWP` | `q`-times integrated Wiener process (the default) | `q` (smoothness order), `d` (dimension), `Xi` (structural scale) |
| `MaternPrior` | Matern process | smoothness + length scale |
| `JointPrior` | a state block stacked with a hidden block | exposes `E0_x`, `E0_hidden` for joint state–parameter models |
| `PrecondIWP`, `PrecondMaternPrior` | preconditioned variants | better conditioning at high `q` / small `h` |

`q` sets the method's order of accuracy; `Xi` (together with the calibrated
`sigma^2`) sets the size of the uncertainty. Initialize the state from `x0` with
`taylor_mode_initialization(vf, x0, q)`. Symbols are defined in
[Notation](../../notation.md).

::: ode_filters.priors
