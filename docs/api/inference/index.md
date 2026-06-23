# Inference

The **square-root Gaussian algebra** (`sqr_marginalization`, `sqr_inversion`) is the
QR-based numerical core every filter step is built on: it propagates covariances as
upper-triangular factors `A_sqr` (with `A = A_sqr.T @ A_sqr`), never forming a dense
covariance and keeping it symmetric and positive-*semi*-definite (no indefiniteness
from round-off; see [notation](../../notation.md)).

Built on top of these primitives, this module also exposes a parameter-inference
layer — `ODEFilter`, `fit`, `marginal_loglik`, and `InferenceProblem`, plus the
`Real` / `PositiveReal` parameter wrappers — for fitting ODE parameters by
maximizing the marginal log-likelihood of external observations under the
ODE-constrained Gauss-Markov model. (`fit` minimizes the negative log-likelihood
via a caller-supplied Optax optimizer.)

::: ode_filters.inference
