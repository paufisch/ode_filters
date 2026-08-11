# Priors

A prior is a Gauss–Markov process over the solution and its first `q` derivatives;
it encodes the smoothness assumed before seeing the ODE. See
[How to choose](../../how-to-choose.md) for picking one.

| Prior | Process | Key parameters |
| --- | --- | --- |
| `IWP` | `q`-times integrated Wiener process (the default) | `q` (smoothness order), `d` (dimension), `Xi` (structural scale) |
| `MaternPrior` | Matern process | `q` (smoothness order), `d` (dimension), `length_scale`, `Xi` (structural scale) |
| `IOUPPrior` | integrated Ornstein–Uhlenbeck process — the highest derivative follows `dY^(q) = R Y^(q) dt + dW` | `q`, `d`, `rate` (scalar, length-`d` vector, or `d × d` matrix), `Xi` |
| `JointPrior` | a state block stacked with a hidden block | exposes `E0_x`, `E0_hidden` for joint state–parameter models |
| `PrecondIWP`, `PrecondMaternPrior` | preconditioned variants | better conditioning at high `q` / small `h` |

`q` sets the method's order of accuracy; `Xi` (together with the calibrated
`sigma^2`) sets the size of the uncertainty. Initialize the state from `x0` with
`taylor_mode_initialization(vf, x0, q)` (which returns the Taylor-mode mean and a
zero/Dirac initial covariance). Symbols are defined in
[Notation](../../notation.md).

Every prior exposes the discrete-time transition `A(h)`, drift `b(h)`, process
noise `Q(h)`, and — what the square-root filter actually consumes — its
upper-triangular square root `Q_sqr(h)` (`Q = Q_sqr.T @ Q_sqr`). For `IWP` /
`PrecondIWP` this is a closed form that avoids ever factorizing the dense,
ill-conditioned `Q(h)`; for `MaternPrior` / `PrecondMaternPrior` / `IOUPPrior` it
is a square-root matrix-fraction decomposition, which likewise never forms the
dense `Q(h)`. Prefer `Q_sqr(h)` over `Q(h)` downstream: at high `q` the dense
`Q(h)` loses positive-definiteness while the square-root factor stays accurate.

::: ode_filters.priors
