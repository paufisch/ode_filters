"""Posterior sampling of an ODE parameter with BlackJAX NUTS.

Bayesian inference of the decay rate ``lam`` in ``dx/dt = -lam x`` from noisy
observations, using the differentiable :func:`ode_filters.marginal_loglik` as the
likelihood. BlackJAX samples in *unconstrained* space, so we parameterize
``lam = softplus(log_lam)`` with a standard-normal prior on ``log_lam``.

Requires the inference extra: ``pip install ode-filters[inference]`` (or
``uv sync --group dev`` in a source checkout).

Run with: ``uv run python examples/blackjax_inference.py``
"""

from __future__ import annotations

import blackjax
import jax
import jax.numpy as np
import jax.random as jr

from ode_filters import InferenceProblem, marginal_loglik
from ode_filters.measurement import Measurement, ODEInformation, prepare_observations
from ode_filters.priors import IWP

jax.config.update("jax_enable_x64", True)

X0, LAM_TRUE, TSPAN, N, NOISE = 2.0, 0.8, (0.0, 5.0), 100, 0.02


def _problem_and_data():
    ts = np.linspace(TSPAN[0], TSPAN[1], N + 1)
    x_true = X0 * np.exp(-LAM_TRUE * ts[1:])
    z = (x_true + NOISE * jr.normal(jr.PRNGKey(0), (N,))).reshape(-1, 1)
    prior = IWP(q=2, d=1, Xi=np.eye(1))
    data = prepare_observations(
        [Measurement(np.eye(1), z, ts[1:], noise=NOISE**2)], prior.E0, ts
    )

    def build(theta):
        lam = theta["lam"]
        measure = ODEInformation(lambda x, *, t: -lam * x, prior.E0, prior.E1)
        return np.array([X0, 0.0, 0.0]), np.diag(np.array([1e-2, 1.0, 1.0])), measure

    return InferenceProblem(build=build, prior=prior, tspan=TSPAN, N=N), data


def run(num_warmup: int = 300, num_samples: int = 300, seed: int = 0):
    """Return posterior samples of ``lam`` (shape ``[num_samples]``)."""
    problem, data = _problem_and_data()

    def logdensity(position):
        log_lam = position["log_lam"]
        lam = jax.nn.softplus(log_lam)
        log_prior = -0.5 * log_lam**2  # standard normal on log_lam
        return marginal_loglik({"lam": lam}, data, model=problem) + log_prior

    warmup_key, sample_key = jr.split(jr.PRNGKey(seed))
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, params), _ = warmup.run(
        warmup_key, {"log_lam": np.array(0.0)}, num_steps=num_warmup
    )
    kernel = blackjax.nuts(logdensity, **params)

    def step(carry, key):
        new_state, _ = kernel.step(key, carry)
        return new_state, new_state.position["log_lam"]

    _, log_lam = jax.lax.scan(step, state, jr.split(sample_key, num_samples))
    return jax.nn.softplus(log_lam)


def main() -> None:
    samples = run()
    print(f"true lam      = {LAM_TRUE}")
    print(
        f"posterior lam = {float(np.mean(samples)):.4f} +/- {float(np.std(samples)):.4f}"
    )


if __name__ == "__main__":
    main()
