"""One consistency battery over the solvers: jit, vmap and grad, in float64 and float32.

A small factory (``Case``) builds a solver configuration from defaults -- the
IWP prior, EK1, dynamic calibration, the fixed-grid ``gaussian_filter``, no
data, the filter result as output -- and each case changes one axis (the
pattern of probdiffeq's test ``Factory``). For every case, three
transformations must leave the numbers alone:

- ``equinox.filter_jit`` reproduces the eager run;
- ``vmap`` over a batch of initial values reproduces a Python loop;
- the reverse-mode gradient of a scalar summary of the output with respect to
  a parameter ``s`` matches central finite differences. ``s`` enters both the
  vector field and the prior's ``Xi``, so the gradient flows through the
  ODE information and the process noise alike.

A float32 tier reruns a subset of the cases in single precision and compares
them with the float64 run, at a tolerance scaled by float32's epsilon.

Square-root factors are compared through the covariances they represent: a
factor is unique only up to an orthogonal transformation, which batched and
unbatched QR choose differently.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, fields

import equinox as eqx
import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    FilterResult,
    InferenceProblem,
    IOUPPrior,
    IteratedSmootherResult,
    IteratedTaylorCorrection,
    JointPrior,
    MaternPrior,
    Measurement,
    ODEInformation,
    ODEInformationWithHidden,
    PrecondIWP,
    QuadratureCorrection,
    TaylorCorrection,
    affine_filter,
    gaussian_filter,
    gaussian_filter_adaptive,
    interpolate,
    iterated_smoother,
    joint_taylor_initialization,
    marginal_loglik,
    prepare_observations,
    rts_smoother,
    sample_posterior,
    taylor_mode_initialization,
    taylor_surrogate,
)
from test.helpers import assert_close, assert_grad_matches_fd, cov

# Grids and initial values are NumPy arrays, so they take the precision of the
# run they enter (float64 by default, float32 in the float32 tier).
Q = 2  # order of the state prior
TS = onp.linspace(0.0, 2.0, 11)
SAVE_AT = onp.linspace(0.0, 2.0, 5)
TS_QUERY = onp.array([0.1, 0.55, 1.3, 1.95])
X0 = onp.array([1.2, 0.5])
X0_BATCH = onp.array([[1.1, 0.5], [1.2, 0.6], [1.3, 0.4]])
S0 = 0.3
KEY = jax.random.PRNGKey(3)
KEYS = jax.random.split(jax.random.PRNGKey(4), 3)
# With exact ODE information the smoothing posterior is singular, and a sample
# path through a singular square-root factor depends on which of its rotations
# the QR picked -- batched and unbatched QR pick different ones, and the path
# is not smooth in the parameters. A tiny ODE noise makes the factors unique.
SAMPLE_ODE_NOISE = 1e-10

CORRECTIONS = {
    "ek0": TaylorCorrection(order=0),
    "ek1": TaylorCorrection(order=1),
    "iekf": IteratedTaylorCorrection(max_iters=3),
    "slr": QuadratureCorrection(n_nodes=3),
}


def _vf(theta):
    """Lotka-Volterra dynamics with the predator rate ``theta``."""

    def vf(x, *, t):
        return np.array([x[0] * (1.0 - x[1]), theta * x[1] * (x[0] - 1.0)])

    return vf


def _vf_hidden(theta):
    """The Lotka-Volterra dynamics driven by a hidden input ``u`` on the predator."""

    def vf(x, u, *, t):
        return np.array([x[0] * (1.0 - x[1]), theta * x[1] * (x[0] - 1.0) + u[0]])

    return vf


@dataclass(frozen=True)
class Case:
    """A solver configuration: defaults everywhere but the axis under test.

    ``output="loglik"`` is the data evidence of ``marginal_loglik``, so it runs
    with data whatever ``data`` says; ``output="sample"`` adds the tiny ODE
    noise ``SAMPLE_ODE_NOISE`` and is batched over keys rather than initial
    values (``sample_posterior``'s documented batching).
    """

    prior: str = "iwp"
    correction: str = "ek1"
    calibration: str = "dynamic"
    driver: str = "filter"
    data: bool = False
    output: str = "filter"

    def __str__(self) -> str:
        default = Case()
        changed = [
            f"{f.name}={getattr(self, f.name)}"
            for f in fields(self)
            if getattr(self, f.name) != getattr(default, f.name)
        ]
        return ",".join(changed) or "default"

    def make_prior(self, s):
        xi = np.exp(s) * np.eye(2)
        if self.prior == "iwp":
            return IWP(q=Q, d=2, Xi=xi)
        if self.prior == "precond_iwp":
            return PrecondIWP(q=Q, d=2, Xi=xi)
        if self.prior == "matern":
            return MaternPrior(q=Q, d=2, length_scale=1.0, Xi=xi)
        if self.prior == "ioup":
            return IOUPPrior(q=Q, d=2, rate=-0.5, Xi=xi)
        assert self.prior == "joint"
        return JointPrior(IWP(q=Q, d=2, Xi=xi), MaternPrior(q=1, d=1, length_scale=1.0))

    def inputs(self, x0, s):
        """``(prior, measure, m_0, P_0_sqr)`` at the initial value and parameter."""
        prior = self.make_prior(s)
        theta = 1.0 + s
        if isinstance(prior, JointPrior):
            vf = _vf_hidden(theta)
            m_0, P_0_sqr = joint_taylor_initialization(vf, x0, prior=prior)
            measure = ODEInformationWithHidden(
                vf, prior.E0_x, prior.E1, prior.E0_hidden
            )
        else:
            vf = _vf(theta)
            m_0, P_0_sqr = taylor_mode_initialization(vf, x0, q=Q)
            measure = ODEInformation(vf, prior.E0, prior.E1)
        if self.output == "sample":
            measure = measure.with_noise(SAMPLE_ODE_NOISE)
        return prior, measure, m_0, P_0_sqr

    def build(self, x0, s):
        """``(m_0, P_0_sqr, measure)``: the ``build`` of an ``InferenceProblem``."""
        _, measure, m_0, P_0_sqr = self.inputs(x0, s)
        return m_0, P_0_sqr, measure

    @property
    def grid(self):
        return SAVE_AT if self.driver == "adaptive" else TS

    def observations(self):
        """Data of the prey at every other node of the grid, or ``None``."""
        if not (self.data or self.output == "loglik"):
            return None
        prior = self.make_prior(S0)
        E0 = prior.E0_x if isinstance(prior, JointPrior) else prior.E0
        t_obs = self.grid[2::2]
        z = (1.2 - 0.2 * np.sin(1.5 * t_obs))[:, None]
        measurement = Measurement(np.array([[1.0, 0.0]]), z, t_obs, noise=1e-3)
        return prepare_observations([measurement], E0, self.grid)

    def linearization_points(self):
        """The frozen trajectory of the ``affine`` driver: a default filter's means."""
        if self.driver != "affine":
            return None
        prior, measure, m_0, P_0_sqr = self.inputs(X0, S0)
        return gaussian_filter(m_0, P_0_sqr, prior, measure, TS).m[1:]

    def solver(self):
        """``solve(x0, s, key) -> output``, with the data and trajectory bound."""
        obs = self.observations()
        m_ref = self.linearization_points()
        correction = CORRECTIONS[self.correction]

        def solve(x0, s, key):
            prior, measure, m_0, P_0_sqr = self.inputs(x0, s)
            if self.output == "loglik":
                assert obs is not None
                problem = InferenceProblem(
                    build=lambda theta: self.build(x0, theta["s"]),
                    prior=prior,
                    ts=TS,
                    calibration=self.calibration,
                    correction=correction,
                    prior_fn=lambda theta: self.make_prior(theta["s"]),
                )
                return marginal_loglik({"s": s}, obs, model=problem, channel="both")
            if self.driver == "filter":
                result = gaussian_filter(
                    m_0,
                    P_0_sqr,
                    prior,
                    measure,
                    TS,
                    correction=correction,
                    calibration=self.calibration,
                    obs_model=obs,
                )
            elif self.driver == "adaptive":
                result = gaussian_filter_adaptive(
                    m_0,
                    P_0_sqr,
                    prior,
                    measure,
                    SAVE_AT,
                    correction=correction,
                    calibration=self.calibration,
                    obs_model=obs,
                    atol=1e-6,
                    rtol=1e-6,
                    smoother=True,
                )
            elif self.driver == "affine":
                assert m_ref is not None
                surrogate = taylor_surrogate(measure, m_ref, ts=TS[1:])
                result = affine_filter(
                    m_0, P_0_sqr, prior, surrogate, TS, obs_model=obs
                )
            else:
                assert self.driver == "iterated"
                return iterated_smoother(
                    m_0, P_0_sqr, prior, measure, TS, n_iters=2, obs_model=obs
                )
            if self.output == "filter":
                return result
            if self.output == "smoother":
                return rts_smoother(result)
            if self.output == "interpolate":
                return interpolate(result, prior, TS_QUERY)
            assert self.output == "sample"
            return sample_posterior(key, result)

        return solve


AXES = {
    "prior": ("precond_iwp", "matern", "ioup", "joint"),
    "correction": ("ek0", "iekf", "slr"),
    "calibration": ("none", "diagonal", "diagonal_ekf0"),
    "driver": ("adaptive", "affine", "iterated"),
    "data": (True,),
    "output": ("smoother", "interpolate", "sample", "loglik"),
}
CASES = [Case()] + [Case(**{axis: v}) for axis, values in AXES.items() for v in values]
# The heaviest drivers, whose transformations also have dedicated tests
# (test_adaptive_solve, test_grad_safety, test_iterated), run as slow tests.
SLOW_DRIVERS = ("adaptive", "iterated")


def params(cases):
    """``pytest.param`` per case, named by its changed axis, heavy ones slow."""
    return [
        pytest.param(
            case,
            id=str(case),
            marks=[pytest.mark.slow] if case.driver in SLOW_DRIVERS else [],
        )
        for case in cases
    ]


def summary(out) -> jax.Array:
    """A scalar about equally sensitive to the means, covariances and evidence."""
    if isinstance(out, tuple) and not hasattr(out, "m"):  # (ll_ode, ll_obs)
        return (out[0] + out[1]) / (TS.shape[0] - 1)
    if not hasattr(out, "m"):  # a sample path
        return np.mean(out[:, 0] + 0.5 * out[:, 1])
    value = np.mean(out.m[:, 0] + 0.5 * out.m[:, 1])
    value = value + np.log(np.sum(np.trace(cov(out.P_sqr), axis1=1, axis2=2)))
    if isinstance(out, IteratedSmootherResult):
        out = out.filter
    if isinstance(out, FilterResult):
        value = value + out.log_likelihood / (out.t.shape[0] - 1)
    return value


def _comparable(tree) -> dict[str, jax.Array]:
    """The array leaves by path, square-root factors replaced by covariances."""
    leaves, _ = jax.tree_util.tree_flatten_with_path(eqx.filter(tree, eqx.is_array))
    out = {}
    for path, leaf in leaves:
        name = jax.tree_util.keystr(path)
        factor = name.endswith("_sqr") and not name.endswith("sigma_sqr")
        out[name] = cov(leaf) if factor else leaf
    return out


def assert_trees_close(actual, expected, *, rtol: float) -> None:
    """Every array leaf close on the scale of the leaf (see ``assert_close``).

    The iterated smoother's ``step`` is a difference of nearly equal means, so
    its error is measured on the scale of the means (one) instead of its own.
    """
    actual, expected = _comparable(actual), _comparable(expected)
    assert actual.keys() == expected.keys()
    for name, leaf in expected.items():
        atol = rtol if name == ".step" else 0.0
        try:
            assert_close(actual[name], leaf, rtol=rtol, atol=atol)
        except AssertionError as error:
            raise AssertionError(f"{name or 'output'}: {error}") from None


@functools.cache
def _jitted(case: Case):
    return eqx.filter_jit(case.solver())


@pytest.mark.parametrize("case", params(CASES))
def test_jit_matches_eager(case):
    eager = case.solver()(X0, S0, KEY)
    assert_trees_close(_jitted(case)(X0, S0, KEY), eager, rtol=1e-11)


@pytest.mark.parametrize("case", params(CASES))
def test_vmap_matches_a_loop(case):
    solve = _jitted(case)
    if case.output == "sample":
        batch, in_axes = KEYS, (None, None, 0)
        looped = [solve(X0, S0, key) for key in KEYS]
    else:
        batch, in_axes = X0_BATCH, (0, None, None)
        looped = [solve(x0, S0, KEY) for x0 in X0_BATCH]
    stacked = jax.tree.map(
        lambda *leaves: np.stack(leaves) if eqx.is_array(leaves[0]) else leaves[0],
        *looped,
    )
    batched = eqx.filter_jit(eqx.filter_vmap(case.solver(), in_axes=in_axes))
    args = [X0, S0, KEY]
    args[in_axes.index(0)] = batch
    # The step-size control of the adaptive solve amplifies the round-off of
    # the batched linear algebra.
    rtol = 1e-8 if case.driver == "adaptive" else 1e-10
    assert_trees_close(batched(*args), stacked, rtol=rtol)


@pytest.mark.parametrize("case", params(CASES))
def test_grad_matches_finite_differences(case):
    solve = case.solver()
    assert_grad_matches_fd(lambda s: summary(solve(X0, s, KEY)), np.asarray(S0))


# ---------------------------------------------------------------------------
# float32 tier
# ---------------------------------------------------------------------------

# A representative subset. IEKF and diagonal_ekf0 repeat the arithmetic of EK1
# and diagonal, the affine and iterated drivers that of the fixed-grid filter.
# The sample path is left out: in float32 the tiny ODE noise is below the
# resolution, so the posterior is singular again and the path not unique.
FLOAT32_CASES = [
    case
    for case in CASES
    if case.output != "sample"
    and case.correction != "iekf"
    and case.calibration != "diagonal_ekf0"
    and case.driver not in ("affine", "iterated")
]
RTOL32 = float(onp.sqrt(onp.finfo(onp.float32).eps))
# The answer of a solve: the posterior moments, the evidence and the diffusion.
# The backward gains are left out -- they can be ill-conditioned (the joint
# prior's are) -- and the smoother case checks what they produce.
ANSWER = (".m", ".P_sqr", ".log_likelihood", ".log_likelihood_obs", ".sigma_sqr")


@pytest.mark.parametrize("case", params(FLOAT32_CASES))
def test_float32_matches_float64(case):
    """A single-precision solve agrees with the double-precision one.

    Means, evidence and diffusion agree to ``sqrt(eps)`` of float32, relative to
    their scale, and so do the fixed-grid covariances. The adaptive solve's
    small steps make the posterior covariance much smaller than the predicted
    one, so the update cancels digits; its covariances get ``30 sqrt(eps)``.
    """
    reference = _comparable(_jitted(case)(X0, S0, KEY))
    with jax.enable_x64(False):
        single = _comparable(eqx.filter_jit(case.solver())(X0, S0, KEY))
    dtypes = {leaf.dtype for leaf in single.values()}
    assert dtypes <= {np.dtype("float32"), np.dtype("int32"), np.dtype("bool")}
    for name, leaf in reference.items():
        if case.output == "loglik" or name in ANSWER:
            factor = 30.0 if case.driver == "adaptive" and name == ".P_sqr" else 1.0
            try:
                assert_close(single[name], leaf, rtol=factor * RTOL32)
            except AssertionError as error:
                raise AssertionError(f"{name or 'output'}: {error}") from None
