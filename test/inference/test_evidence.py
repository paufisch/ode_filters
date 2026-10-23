"""Conditional evidence and information criteria against a dense oracle.

On a frozen linearization the model is linear-Gaussian, so the joint law of the
states, the ODE residuals ``Z`` and the data ``Y`` is one Gaussian. The oracle
unrolls it into dense matrices and computes every quantity from
log-determinants and Schur complements directly.
"""

from __future__ import annotations

import jax
import jax.numpy as np
import numpy as onp
import pytest

from ode_filters import (
    IWP,
    AffineSurrogate,
    InferenceProblem,
    InformationCriteria,
    JointPrior,
    MaternPrior,
    Measurement,
    ODEInformation,
    PrecondIWP,
    affine_filter,
    conditional_loglik,
    fenrir_loglik,
    gaussian_filter,
    gaussian_filter_adaptive,
    information_criteria,
    marginal_loglik,
    prepare_observations,
    taylor_mode_initialization,
)
from ode_filters.inference.evidence import conditional_state_cov_sqr
from test.helpers import (
    assert_close,
    assert_grad_matches_fd,
    cov,
    surrogate_on_merged_grid,
)

N = 6
TS = np.linspace(0.0, 1.2, N + 1)
K_ROWS = 1


def _prior(log_xi=0.0):
    return JointPrior(
        IWP(q=1, d=1, Xi=np.exp(log_xi) * np.eye(1)),
        MaternPrior(q=1, d=1, length_scale=0.8),
    )


def _setup(log_xi=0.0, seed=0):
    prior = _prior(log_xi)
    D = prior.state_dim
    keys = jax.random.split(jax.random.PRNGKey(seed), 4)
    surrogate = AffineSurrogate(
        jax.random.normal(keys[0], (N, K_ROWS, D)),
        jax.random.normal(keys[1], (N, K_ROWS)),
        0.2 * np.broadcast_to(np.eye(K_ROWS), (N, K_ROWS, K_ROWS)),
    )
    m_0 = jax.random.normal(keys[2], (D,))
    P_0_sqr = np.block(
        [
            [0.3 * np.eye(2), np.zeros((2, 2))],
            [np.zeros((2, 2)), prior.prior_u.stationary_cov_sqr()],
        ]
    )
    t_obs = TS[2::2]
    y = jax.random.normal(keys[3], (t_obs.shape[0], 1))
    obs = prepare_observations(
        [Measurement(np.eye(1), y, t_obs, noise=0.05)], prior.E0_x, TS
    )
    return prior, surrogate, m_0, P_0_sqr, obs


def _logpdf_zero(mean, S):
    k = mean.shape[0]
    _, logdet = np.linalg.slogdet(S)
    return -0.5 * (k * np.log(2 * np.pi) + logdet + mean @ np.linalg.solve(S, mean))


def _conditional_cov(S, keep, given):
    """``Cov(keep | given)`` of a Gaussian with covariance ``S``."""
    S_kk = S[np.ix_(keep, keep)]
    if len(given) == 0:
        return S_kk
    S_kg = S[np.ix_(keep, given)]
    return S_kk - S_kg @ np.linalg.solve(S[np.ix_(given, given)], S_kg.T)


def _logdet(S):
    return np.linalg.slogdet(S)[1]


def _dense(prior, surrogate, m_0, P_0_sqr, obs):
    """The joint Gaussian of (states, residuals) in time order, and index sets."""
    D = prior.state_dim
    h = TS[1] - TS[0]
    A, Q = prior.A(h), prior.Q(h)
    # States X_0..X_N: means and the full covariance.
    means, covs = [m_0], [cov(P_0_sqr)]
    for _ in range(N):
        means.append(A @ means[-1])
        covs.append(A @ covs[-1] @ A.T + Q)
    X_cov = np.zeros(((N + 1) * D, (N + 1) * D))
    for i in range(N + 1):
        for j in range(i, N + 1):
            block = covs[i] @ np.linalg.matrix_power(A, j - i).T
            X_cov = X_cov.at[i * D : (i + 1) * D, j * D : (j + 1) * D].set(block)
            X_cov = X_cov.at[j * D : (j + 1) * D, i * D : (i + 1) * D].set(block.T)
    X_mean = np.concatenate(means)

    # Residual rows in time order: z_n, then y_n where present.
    rows, offsets, noises, kinds = [], [], [], []
    for n in range(N):
        L = (
            np.zeros((K_ROWS, (N + 1) * D))
            .at[:, (n + 1) * D : (n + 2) * D]
            .set(surrogate.H[n])
        )
        rows.append(L)
        offsets.append(surrogate.c[n])
        noises.append(cov(surrogate.R_sqr[n]))
        kinds += [("z", n)] * K_ROWS
        if bool(obs.mask[n].any()):
            L = np.zeros((1, (N + 1) * D)).at[:, (n + 1) * D : (n + 2) * D].set(obs.H)
            rows.append(L)
            offsets.append(obs.c_seq[n])
            noises.append(cov(obs.R_sqr))
            kinds.append(("y", n))
    L = np.concatenate(rows)
    V_mean = L @ X_mean + np.concatenate(offsets)
    E = jax.scipy.linalg.block_diag(*noises)
    V_cov = L @ X_cov @ L.T + E
    # The joint covariance of (V, U) with U the input block at every node.
    u_idx = np.array([n * D + j for n in range(N + 1) for j in range(2, D)])
    U_sel = (
        np.zeros((u_idx.shape[0], (N + 1) * D))
        .at[np.arange(u_idx.shape[0]), u_idx]
        .set(1.0)
    )
    VU = np.concatenate([L, U_sel])
    VU_cov = VU @ X_cov @ VU.T
    VU_cov = VU_cov.at[: L.shape[0], : L.shape[0]].add(E)
    return V_mean, V_cov, VU_cov, kinds


def _oracle(prior, surrogate, m_0, P_0_sqr, obs):
    """``log p(Y | Z)`` and the three informations from the dense joint Gaussian."""
    V_mean, V_cov, VU_cov, kinds = _dense(prior, surrogate, m_0, P_0_sqr, obs)
    z = np.array([i for i, (k, _) in enumerate(kinds) if k == "z"])
    y = np.array([i for i, (k, _) in enumerate(kinds) if k == "y"])
    n_v = len(kinds)
    u = np.arange(n_v, VU_cov.shape[0])

    log_p_zy = _logpdf_zero(V_mean, V_cov)
    log_p_z = _logpdf_zero(V_mean[z], V_cov[np.ix_(z, z)])
    all_v = np.arange(n_v)
    I_UZ = 0.5 * (
        _logdet(VU_cov[np.ix_(z, z)]) - _logdet(_conditional_cov(VU_cov, z, u))
    )
    I_UYZ = 0.5 * (
        _logdet(VU_cov[np.ix_(all_v, all_v)])
        - _logdet(_conditional_cov(VU_cov, all_v, u))
    )
    D = 0.0
    for i in y:
        past = np.arange(i)
        D += 0.5 * (
            _logdet(_conditional_cov(VU_cov, np.array([i]), past))
            - _logdet(
                _conditional_cov(VU_cov, np.array([i]), np.concatenate([past, u]))
            )
        )
    return log_p_zy - log_p_z, InformationCriteria(I_UZ, I_UYZ - I_UZ, D)


def _assert_matches_oracle(log_xi, *, atol):
    prior, surrogate, m_0, P_0_sqr, obs = _setup(log_xi=log_xi)
    log_p_y_given_z, expected = _oracle(prior, surrogate, m_0, P_0_sqr, obs)
    data_free = affine_filter(m_0, P_0_sqr, prior, surrogate, TS)
    data = affine_filter(m_0, P_0_sqr, prior, surrogate, TS, obs_model=obs)
    assert_close(fenrir_loglik(data_free, obs), log_p_y_given_z, rtol=1e-9)
    assert_close(conditional_loglik(data, data_free), log_p_y_given_z, rtol=1e-9)
    criteria = information_criteria(m_0, P_0_sqr, prior, surrogate, TS, obs)
    for name in InformationCriteria._fields:
        try:
            assert_close(
                getattr(criteria, name), getattr(expected, name), rtol=1e-9, atol=atol
            )
        except AssertionError as err:
            raise AssertionError(f"{name}: {err}") from None


def test_evidence_and_informations_match_the_dense_oracle():
    _assert_matches_oracle(0.0, atol=0.0)


@pytest.mark.parametrize("log_xi", [np.log(1e-10), np.log(1e10)], ids=["1e-10", "1e10"])
def test_extreme_state_diffusions_match_the_dense_oracle(log_xi):
    """The evidence and the informations at a state diffusion ``Xi`` of ``1e+-10``.

    At ``Xi = 1e10`` the ODE information and the data say almost nothing about
    the input: the informations are about ``1e-6`` nats, each a difference of
    log-determinants of size about ``60``, which neither side computes to better
    than about ``1e-13``. They are compared to an absolute ``1e-11`` nats.
    """
    _assert_matches_oracle(log_xi, atol=1e-11)


def test_fenrir_equals_the_conditional_evidence_on_a_shared_surrogate():
    prior, surrogate, m_0, P_0_sqr, obs = _setup(seed=3)
    data_free = affine_filter(m_0, P_0_sqr, prior, surrogate, TS)
    data = affine_filter(m_0, P_0_sqr, prior, surrogate, TS, obs_model=obs)
    assert_close(
        fenrir_loglik(data_free, obs), conditional_loglik(data, data_free), rtol=1e-10
    )


def test_marginal_loglik_channels_agree_for_a_linear_ode():
    """For an affine vector field the one-shot passes are exact, so Fenrir and
    Dalton estimate the same conditional evidence."""
    prior = IWP(q=2, d=1)
    ts = np.linspace(0.0, 2.0, 21)
    t_obs = ts[4::4]
    obs = prepare_observations(
        [
            Measurement(
                np.eye(1), np.exp(-1.3 * t_obs)[:, None] + 0.01, t_obs, noise=1e-3
            )
        ],
        prior.E0,
        ts,
    )

    def build(theta):
        def vf(x, *, t):
            return -theta * x

        m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([1.0]), q=2)
        return m_0, P_0_sqr, ODEInformation(vf, prior.E0, prior.E1)

    problem = InferenceProblem(build=build, prior=prior, ts=ts)
    theta = np.array(1.1)
    fenrir = marginal_loglik(theta, obs, model=problem, channel="fenrir")
    dalton = marginal_loglik(theta, obs, model=problem, channel="dalton")
    assert_close(fenrir, dalton, rtol=1e-9)
    _, ll_obs = marginal_loglik(theta, obs, model=problem, channel="both")
    # The sequential data evidence conditions on less ODE information.
    assert not np.allclose(fenrir, ll_obs)
    assert_grad_matches_fd(
        lambda th: marginal_loglik(th, obs, model=problem, channel="fenrir"), theta
    )


def test_informations_are_differentiable_in_the_state_diffusion():
    _, surrogate, m_0, P_0_sqr, obs = _setup()

    def criterion(name):
        def f(log_xi):
            prior = _prior(log_xi)
            return getattr(
                information_criteria(m_0, P_0_sqr, prior, surrogate, TS, obs), name
            )

        return f

    assert_grad_matches_fd(criterion("I_UY_given_Z"), np.array(0.2))
    assert_grad_matches_fd(criterion("D"), np.array(0.2))


def test_informations_do_not_depend_on_preconditioning():
    """A preconditioned state prior is a change of variables inside each step."""
    _, surrogate, m_0, P_0_sqr, obs = _setup()
    plain = information_criteria(m_0, P_0_sqr, _prior(), surrogate, TS, obs)
    precond = JointPrior(
        PrecondIWP(q=1, d=1, Xi=np.eye(1)), MaternPrior(q=1, d=1, length_scale=0.8)
    )
    other = information_criteria(m_0, P_0_sqr, precond, surrogate, TS, obs)
    for name in InformationCriteria._fields:
        assert_close(getattr(other, name), getattr(plain, name), rtol=1e-9)


def test_default_conditional_state_covariance_is_the_schur_complement():
    """For the factor ``[[R_x, 0], [R_u J.T, R_u]]`` the state given the input has ``R_x``."""
    keys = jax.random.split(jax.random.PRNGKey(5), 3)
    R_x = np.triu(jax.random.normal(keys[0], (3, 3))) + 2.0 * np.eye(3)
    R_u = np.triu(jax.random.normal(keys[1], (2, 2))) + 2.0 * np.eye(2)
    J = jax.random.normal(keys[2], (3, 2))
    factor = np.block([[R_x, np.zeros((3, 2))], [R_u @ J.T, R_u]])
    assert_close(cov(conditional_state_cov_sqr(factor, 3)), cov(R_x), rtol=1e-12)
    prior, surrogate, m_0, P_0_sqr, obs = _setup()
    default = information_criteria(m_0, P_0_sqr, prior, surrogate, TS, obs)
    explicit = information_criteria(
        m_0, P_0_sqr, prior, surrogate, TS, obs, P_0_state_sqr=0.3 * np.eye(2)
    )
    assert_close(default.D, explicit.D, rtol=1e-12)


def test_noise_free_ode_rows_match_the_dense_oracle():
    """Without noise on the ODE rows every update leaves a singular covariance, and
    with the state a function of the input so is its covariance given the input."""
    prior, surrogate, m_0, _, obs = _setup()
    surrogate = surrogate._replace(R_sqr=np.zeros_like(surrogate.R_sqr))
    R_u = prior.prior_u.stationary_cov_sqr()
    J = np.array([[0.7, -0.4], [0.2, 0.5]])
    P_0_sqr = np.block([[np.zeros((2, 2)), np.zeros((2, 2))], [R_u @ J.T, R_u]])
    _, expected = _oracle(prior, surrogate, m_0, P_0_sqr, obs)
    criteria = information_criteria(m_0, P_0_sqr, prior, surrogate, TS, obs)
    for name in InformationCriteria._fields:
        assert_close(getattr(criteria, name), getattr(expected, name), rtol=1e-9)


def _noise_free_prior(log_xi, log_length_scale):
    return JointPrior(
        IWP(q=2, d=8, Xi=np.exp(log_xi) * np.eye(8)),
        MaternPrior(q=2, d=1, length_scale=np.exp(log_length_scale)),
    )


def test_informations_are_differentiable_with_noise_free_ode_rows():
    """Eight ODE rows without noise, each tying a derivative to the values and the
    input, leave a singular covariance after every update; the state starts exact,
    so its covariance given the input is singular as well. The gradients in the
    state diffusion and in the input's length scale match finite differences."""
    n_steps, d = 10, 8
    ts = np.linspace(0.0, 2.0, n_steps + 1)
    prior = _noise_free_prior(0.0, 0.0)
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    J = jax.random.normal(keys[0], (n_steps, d, d))
    b = jax.random.normal(keys[1], (n_steps, d, 1))
    H = np.concatenate(
        [prior.prior_x.E1 - J @ prior.prior_x.E0, -b @ prior.prior_u.E0], axis=2
    )
    c = jax.random.normal(keys[2], (n_steps, d))
    surrogate = AffineSurrogate(H, c, np.zeros((n_steps, d, d)))
    y = jax.random.normal(keys[3], (n_steps, d))
    obs = prepare_observations(
        [Measurement(np.eye(d), y, ts[1:], noise=0.05)], prior.E0_x, ts
    )
    m_0 = np.zeros(prior.state_dim)

    def criterion(name, wrt):
        def f(theta):
            log_xi, log_length_scale = (
                (theta, np.log(0.5)) if wrt == "xi" else (2.0, theta)
            )
            prior = _noise_free_prior(log_xi, log_length_scale)
            D_x = prior.prior_x.state_dim
            P_0_sqr = (
                np.zeros((prior.state_dim, prior.state_dim))
                .at[D_x:, D_x:]
                .set(prior.prior_u.stationary_cov_sqr())
            )
            return getattr(
                information_criteria(m_0, P_0_sqr, prior, surrogate, ts, obs), name
            )

        return f

    for name in ("I_UY_given_Z", "D"):
        assert_grad_matches_fd(criterion(name, "xi"), np.array(2.0))
        assert_grad_matches_fd(criterion(name, "length_scale"), np.array(np.log(0.5)))


def test_invalid_arguments_raise():
    prior, surrogate, m_0, P_0_sqr, obs = _setup()
    data = affine_filter(m_0, P_0_sqr, prior, surrogate, TS, obs_model=obs)
    data_free = affine_filter(m_0, P_0_sqr, prior, surrogate, TS)
    with pytest.raises(ValueError, match="twice"):
        fenrir_loglik(data, obs)
    with pytest.raises(ValueError, match="steps"):
        fenrir_loglik(data_free, obs._replace(c_seq=obs.c_seq[:-1], mask=obs.mask[:-1]))
    with pytest.raises(ValueError, match="with data"):
        conditional_loglik(data_free, data_free)
    with pytest.raises(ValueError, match="without data"):
        conditional_loglik(data, data)
    with pytest.raises(TypeError, match="JointPrior"):
        information_criteria(m_0, P_0_sqr, prior.prior_x, surrogate, TS, obs)
    with pytest.raises(ValueError, match="joint state"):
        information_criteria(m_0[:2], P_0_sqr, prior, surrogate, TS, obs)
    with pytest.raises(ValueError, match="requires data"):
        information_criteria(m_0, P_0_sqr, prior, surrogate, TS, [])

    def vf(x, *, t):
        return -x

    iwp = IWP(q=1, d=1)
    m, P = taylor_mode_initialization(vf, np.array([1.0]), q=1)
    adaptive = gaussian_filter_adaptive(
        m, P, iwp, ODEInformation(vf, iwp.E0, iwp.E1), TS
    )
    with pytest.raises(ValueError, match="backward conditionals"):
        fenrir_loglik(
            adaptive,
            prepare_observations(
                [Measurement(np.eye(1), np.ones((1, 1)), TS[1:2])], iwp.E0, TS
            ),
        )


def test_one_shot_filters_give_daltons_estimator():
    """With nonlinear dynamics the two estimators differ."""

    def vf(x, *, t):
        return x * (1.0 - x)

    prior = IWP(q=2, d=1)
    ts = np.linspace(0.0, 2.0, 11)
    m_0, P_0_sqr = taylor_mode_initialization(vf, np.array([0.3]), q=2)
    measure = ODEInformation(vf, prior.E0, prior.E1)
    obs = prepare_observations(
        [
            Measurement(
                np.eye(1), np.array([[0.5], [0.9]]), ts[np.array([4, 8])], noise=1e-3
            )
        ],
        prior.E0,
        ts,
    )
    data_free = gaussian_filter(m_0, P_0_sqr, prior, measure, ts, calibration="none")
    data = gaussian_filter(
        m_0, P_0_sqr, prior, measure, ts, calibration="none", obs_model=obs
    )
    fenrir = fenrir_loglik(data_free, obs)
    dalton = conditional_loglik(data, data_free)
    assert abs(float(fenrir - dalton)) > 1e-6


# --------------------------------------------------------------------------
# Data between the grid nodes
# --------------------------------------------------------------------------

T_OFF = onp.array([0.3, 0.4, 0.9, 1.1])  # 0.4 is a grid node


def _off_grid_setup(log_xi=0.0):
    prior, surrogate, m_0, P_0_sqr, _ = _setup(log_xi=log_xi)
    y = jax.random.normal(jax.random.PRNGKey(7), (T_OFF.shape[0], 1))
    meas = [Measurement(np.eye(1), y, T_OFF, noise=0.05)]
    obs = prepare_observations(meas, prior.E0_x, TS)
    assert obs is not None and obs.t is not None and obs.grid_node is not None
    merged = prepare_observations(meas, prior.E0_x, obs.t)
    assert merged is not None and merged.t is None
    merged_surrogate = surrogate_on_merged_grid(surrogate, obs.grid_node)
    return prior, surrogate, m_0, P_0_sqr, obs, merged_surrogate, merged


def test_off_grid_evidence_equals_the_merged_grid_without_ode_rows():
    prior, surrogate, m_0, P_0_sqr, obs, merged_surrogate, merged = _off_grid_setup()
    assert obs.t is not None
    data_free = affine_filter(
        m_0, P_0_sqr, prior, surrogate, TS, obs_model=obs.without_data()
    )
    data = affine_filter(m_0, P_0_sqr, prior, surrogate, TS, obs_model=obs)
    fenrir = fenrir_loglik(data_free, obs)
    # One frozen linearization: Fenrir and Dalton agree exactly.
    assert_close(conditional_loglik(data, data_free), fenrir, rtol=1e-10)
    merged_free = affine_filter(m_0, P_0_sqr, prior, merged_surrogate, obs.t)
    assert_close(fenrir_loglik(merged_free, merged), fenrir, rtol=1e-10)


@pytest.mark.parametrize("log_xi", [0.0, np.log(1e-4)])
def test_off_grid_informations_equal_the_merged_grid_without_ode_rows(log_xi):
    prior, surrogate, m_0, P_0_sqr, obs, merged_surrogate, merged = _off_grid_setup(
        log_xi
    )
    assert obs.t is not None
    criteria = information_criteria(m_0, P_0_sqr, prior, surrogate, TS, obs)
    expected = information_criteria(
        m_0, P_0_sqr, prior, merged_surrogate, obs.t, merged
    )
    for name in InformationCriteria._fields:
        assert_close(getattr(criteria, name), getattr(expected, name), rtol=1e-10)
    assert float(criteria.I_UY_given_Z) > 0.0


@pytest.mark.parametrize("t_obs", [TS[2::2], T_OFF], ids=["on-grid", "off-grid"])
def test_informations_take_a_list_of_measurements(t_obs):
    """A list of ``Measurement`` objects gives the values of its ``ObsModel``."""
    prior, surrogate, m_0, P_0_sqr, _ = _setup()
    y = jax.random.normal(jax.random.PRNGKey(4), (t_obs.shape[0], 1))
    meas = [Measurement(np.eye(1), y, t_obs, noise=0.05)]
    obs = prepare_observations(meas, prior.E0_state, TS)
    expected = information_criteria(m_0, P_0_sqr, prior, surrogate, TS, obs)
    criteria = information_criteria(m_0, P_0_sqr, prior, surrogate, TS, meas)
    for name in InformationCriteria._fields:
        assert_close(getattr(criteria, name), getattr(expected, name), rtol=0.0)


def test_fenrir_needs_the_pass_without_data_on_the_timeline():
    prior, surrogate, m_0, P_0_sqr, obs, merged_surrogate, _ = _off_grid_setup()
    on_grid = affine_filter(m_0, P_0_sqr, prior, surrogate, TS)
    with pytest.raises(ValueError, match="without_data"):
        fenrir_loglik(on_grid, obs)
    # As many steps, at other times.
    assert obs.t is not None
    shifted = affine_filter(
        m_0, P_0_sqr, prior, merged_surrogate, onp.asarray(obs.t) * 1.01
    )
    with pytest.raises(ValueError, match="other times"):
        fenrir_loglik(shifted, obs)
