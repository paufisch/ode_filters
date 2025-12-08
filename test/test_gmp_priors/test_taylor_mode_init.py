import jax.numpy as np
import pytest

from ode_filters.priors.gmp_priors import taylor_mode_initialization


def test_taylor_mode_initialization_q0_returns_state_flattened():
    def vf(y, *, t):
        return y

    x0 = np.array([1.0, 2.0])
    result, _ = taylor_mode_initialization(vf, x0, q=0)

    assert result.ndim == 1
    assert np.array_equal(np.asarray(result), np.asarray(x0).ravel())


def test_taylor_mode_initialization_scalar_linear_field_matches_closed_form():
    def vf(y, *, t):
        return -y  # u(t) = e^{-t} → u^{(k)}(0) = (-1)^k

    x0 = np.array([1.0])
    expected = np.array([(-1.0) ** k for k in range(4)])
    result, _ = taylor_mode_initialization(vf, x0, q=3)

    assert result.shape == expected.shape
    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_taylor_mode_initialization_vector_field_runs_and_flattens():
    def vf(y, *, t):
        a, b = 0.5, -0.3
        return np.array([a * y[0] - y[0] * y[1], b * y[1] + y[0] * y[1]])

    x0 = np.array([1.0, 2.0])
    result, _ = taylor_mode_initialization(vf, x0, q=3)

    assert result.ndim == 1
    assert result.shape[0] == len(x0) * (3 + 1)


def test_taylor_mode_initialization_rejects_invalid_inputs():
    def vf(y, *, t):
        return y

    with pytest.raises(TypeError):
        taylor_mode_initialization(123, np.array([0.0]), q=1)

    with pytest.raises(ValueError):
        taylor_mode_initialization(vf, np.array([0.0]), q=-1)


def test_taylor_mode_initialization_q1_linear_vector_field_matches_matrix_product():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])

    def vf(y, *, t):
        return A @ y

    x0 = np.array([1.0, 2.0])

    expected = np.concatenate((x0, A @ x0))
    result, _ = taylor_mode_initialization(vf, x0, q=1)

    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_taylor_mode_initialization_linear_vector_field_matches_matrix_powers():
    A = np.array([[0.0, 1.0], [-2.0, -3.0]])

    def vf(y, *, t):
        return A @ y

    x0 = np.array([1.0, 2.0])

    Ax = A @ x0
    A2x = A @ (A @ x0)
    expected = np.concatenate((x0, Ax, A2x))
    result, _ = taylor_mode_initialization(vf, x0, q=2)

    assert np.allclose(np.asarray(result), np.asarray(expected))


# =============================================================================
# Second-order ODE initialization tests
# =============================================================================


def test_taylor_mode_initialization_second_order_basic():
    """Test second-order ODE initialization for harmonic oscillator."""
    omega = 1.0

    def vf(x, v, *, t):
        return -(omega**2) * x  # d^2x/dt^2 = -omega^2 * x

    x0 = np.array([1.0])
    v0 = np.array([0.0])

    result, cov = taylor_mode_initialization(vf, (x0, v0), q=2, order=2)

    # State should be [x0, v0, a0] where a0 = -omega^2 * x0 = -1
    assert result.shape == (3,)
    assert result[0] == pytest.approx(1.0)  # x0
    assert result[1] == pytest.approx(0.0)  # v0
    assert result[2] == pytest.approx(-1.0)  # a0 = -omega^2 * x0

    # Covariance should be zero
    assert cov.shape == (3, 3)
    assert np.allclose(cov, np.zeros((3, 3)))


def test_taylor_mode_initialization_second_order_damped():
    """Test second-order ODE with damping."""
    omega, gamma = 1.0, 0.5

    def vf(x, v, *, t):
        return -(omega**2) * x - gamma * v

    x0 = np.array([1.0])
    v0 = np.array([0.5])

    result, _ = taylor_mode_initialization(vf, (x0, v0), q=1, order=2)

    # State should be [x0, v0] (q=1 gives 2 coefficients)
    assert result.shape == (2,)
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.5)


def test_taylor_mode_initialization_second_order_multidim():
    """Test second-order ODE with multi-dimensional state."""

    def vf(x, v, *, t):
        return -x - 0.1 * v

    x0 = np.array([1.0, 2.0])
    v0 = np.array([0.0, 0.0])

    result, _ = taylor_mode_initialization(vf, (x0, v0), q=2, order=2)

    # State dimension: d * (q+1) = 2 * 3 = 6
    assert result.shape == (6,)


def test_taylor_mode_initialization_rejects_invalid_order():
    """Test that invalid order raises error."""

    def vf(y, *, t):
        return y

    with pytest.raises(ValueError, match="order must be 1 or 2"):
        taylor_mode_initialization(vf, np.array([1.0]), q=1, order=3)

    with pytest.raises(ValueError, match="order must be 1 or 2"):
        taylor_mode_initialization(vf, np.array([1.0]), q=1, order=0)


def test_taylor_mode_initialization_second_order_rejects_non_tuple():
    """Test that second-order requires tuple of inits."""

    def vf(x, v, *, t):
        return -x

    with pytest.raises(ValueError, match="must be a tuple of 2 arrays"):
        taylor_mode_initialization(vf, np.array([1.0]), q=1, order=2)


def test_taylor_mode_initialization_second_order_rejects_wrong_tuple_length():
    """Test that second-order rejects wrong tuple length."""

    def vf(x, v, *, t):
        return -x

    with pytest.raises(ValueError, match="must be a tuple of 2 arrays"):
        taylor_mode_initialization(vf, (np.array([1.0]),), q=1, order=2)

    with pytest.raises(ValueError, match="must be a tuple of 2 arrays"):
        taylor_mode_initialization(
            vf, (np.array([1.0]), np.array([0.0]), np.array([0.0])), q=1, order=2
        )


def test_taylor_mode_initialization_second_order_rejects_mismatched_shapes():
    """Test that second-order rejects mismatched array shapes."""

    def vf(x, v, *, t):
        return -x

    with pytest.raises(ValueError, match="must have the same shape"):
        taylor_mode_initialization(
            vf, (np.array([1.0, 2.0]), np.array([0.0])), q=1, order=2
        )
