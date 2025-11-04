from math import factorial

import numpy as np
import pytest

from ode_filters.GMP_priors import IWP, _make_iwp_state_matrices


@pytest.mark.parametrize("q,h", [(0, 0.7), (1, 0.3)])
def test_make_iwp_state_matrices_matches_closed_form(q, h):
    A_fn, Q_fn = _make_iwp_state_matrices(q)
    dim = q + 1

    # Build expected matrices explicitly.
    expected_A = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(i, dim):
            expected_A[i, j] = h ** (j - i) / factorial(j - i)

    expected_Q = np.zeros((dim, dim), dtype=float)
    for i in range(dim):
        for j in range(dim):
            power = 2 * q + 1 - i - j
            denom = (2 * q + 1 - i - j) * factorial(q - i) * factorial(q - j)
            expected_Q[i, j] = h**power / denom

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
