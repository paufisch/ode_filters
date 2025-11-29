from math import factorial

import jax.numpy as np
import pytest

from ode_filters.priors.gmp_priors import (
    IWP,
    PrecondIWP,
    _make_iwp_precond_state_matrices,
    _make_iwp_state_matrices,
)


@pytest.mark.parametrize("q,h", [(0, 0.7), (1, 0.3)])
def test_make_iwp_state_matrices_matches_closed_form(q, h):
    A_fn, Q_fn = _make_iwp_state_matrices(q)
    dim = q + 1

    # Build expected matrices explicitly.
    expected_A = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(i, dim):
            expected_A = expected_A.at[i, j].set(h ** (j - i) / factorial(j - i))

    expected_Q = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            power = 2 * q + 1 - i - j
            denom = (2 * q + 1 - i - j) * factorial(q - i) * factorial(q - j)
            expected_Q = expected_Q.at[i, j].set(h**power / denom)

    assert A_fn(h) == pytest.approx(expected_A)
    assert Q_fn(h) == pytest.approx(expected_Q)


def test_make_iwp_state_matrices_invalid_inputs():
    with pytest.raises(ValueError):
        _make_iwp_state_matrices(-1)

    A_fn, Q_fn = _make_iwp_state_matrices(1)
    with pytest.raises(ValueError):
        A_fn(-0.1)
    with pytest.raises(ValueError):
        Q_fn(-0.1)


def test_iwp_identity_sigma_matches_kron():
    q, d, h = 1, 2, 0.5
    iwp = IWP(q=q, d=d)
    base_A, base_Q = _make_iwp_state_matrices(q)
    expected_A = np.kron(base_A(h), np.eye(d))
    expected_Q = np.kron(base_Q(h), np.eye(d))

    assert iwp.A(h) == pytest.approx(expected_A)
    assert iwp.Q(h) == pytest.approx(expected_Q)


def test_iwp_custom_sigma_and_validation():
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    iwp = IWP(q=1, d=2, Xi=sigma)
    h = 0.25
    base_Q = _make_iwp_state_matrices(1)[1]
    expected_Q = np.kron(base_Q(h), sigma)

    assert iwp.Q(h) == pytest.approx(expected_Q)

    with pytest.raises(ValueError):
        IWP(q=-1, d=2)

    with pytest.raises(ValueError):
        IWP(q=1, d=0)

    with pytest.raises(ValueError):
        IWP(q=1, d=2, Xi=np.eye(3))

    with pytest.raises(ValueError):
        iwp.A(-0.2)


@pytest.mark.parametrize("q_float", [0.5, 1.5])
def test_make_iwp_state_matrices_rejects_float_q(q_float):
    with pytest.raises(TypeError):
        _make_iwp_state_matrices(q_float)


@pytest.mark.parametrize("q_float", [0.5, 1.5])
def test_iwp_rejects_float_q(q_float):
    with pytest.raises(TypeError):
        IWP(q=q_float, d=2)


@pytest.mark.parametrize("d_float", [1.5, 2.0])
def test_iwp_rejects_float_d(d_float):
    with pytest.raises(TypeError):
        IWP(q=1, d=d_float)


def test_make_iwp_precond_state_matrices_rejects_negative_q():
    with pytest.raises(ValueError, match="q must be a non-negative integer"):
        _make_iwp_precond_state_matrices(-1)


def test_preconditioner_T_rejects_negative_step():
    _, _, T = _make_iwp_precond_state_matrices(1)
    with pytest.raises(ValueError, match="h must be non-negative"):
        T(-0.5)


@pytest.mark.parametrize(
    "args,expected_exception,match",
    [
        ((1.2, 1), TypeError, "q must be an integer"),
        ((-1, 1), ValueError, "q must be non-negative"),
        ((0, "1"), TypeError, "d must be an integer"),
        ((0, 0), ValueError, "d must be positive"),
    ],
)
def test_iwp_precond_constructor_rejects_invalid_q_and_d(
    args, expected_exception, match
):
    with pytest.raises(expected_exception, match=match):
        PrecondIWP(*args)


def test_iwp_precond_constructor_rejects_bad_xi_shape():
    with pytest.raises(ValueError, match=r"Xi must have shape"):
        PrecondIWP(q=0, d=1, Xi=np.eye(2))


def test_iwp_precond_validate_h_returns_float():
    assert PrecondIWP._validate_h(2) == 2.0


def test_iwp_precond_validate_h_rejects_negative():
    with pytest.raises(ValueError, match="h must be non-negative"):
        PrecondIWP._validate_h(-1.0)
