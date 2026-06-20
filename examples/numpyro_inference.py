"""Posterior sampling of an ODE parameter with NumPyro NUTS.

Bayesian inference of the decay rate ``lam`` in ``dx/dt = -lam x`` from noisy
observations. The differentiable :func:`ode_filters.marginal_loglik` enters the
NumPyro model via ``numpyro.factor``; NumPyro handles the constrained ``LogNormal``
prior and its reparameterization automatically.

Run with: ``uv run python examples/numpyro_inference.py``
"""

from __future__ import annotations

import jax.numpy as np
import jax.random as jr
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from ode_filters import InferenceProblem, marginal_loglik
from ode_filters.measurement import Measurement, ODEInformation, prepare_observations
from ode_filters.priors import IWP

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

    def model():
        lam = numpyro.sample("lam", dist.LogNormal(0.0, 1.0))
        numpyro.factor("loglik", marginal_loglik({"lam": lam}, data, model=problem))

    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=False,
    )
    mcmc.run(jr.PRNGKey(seed))
    return mcmc.get_samples()["lam"]


def main() -> None:
    samples = run()
    print(f"true lam      = {LAM_TRUE}")
    print(
        f"posterior lam = {float(np.mean(samples)):.4f} +/- {float(np.std(samples)):.4f}"
    )


if __name__ == "__main__":
    main()
