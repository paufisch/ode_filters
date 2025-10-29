import pytest

from ode_filters.gaussian_inference import marginalization


def test_marginalization_example1():
    """Test marginalization with simple known values."""
    import numpy as np

    # Simple 2D -> 1D case
    A = np.array([[1.0, 0.0]])  # shape [1, 2]
    b = np.array([0.0])  # shape [1]
    Q = np.array([[0.1]])  # shape [1, 1]
    mu = np.array([1.0, 2.0])  # shape [2]
    Sigma = np.eye(2)  # shape [2, 2]

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Expected: mu_z = A @ mu + b = [1, 0] @ [1, 2] + 0 = [1]
    # Expected: Sigma_z = A @ Sigma @ A.T + Q = [1, 0] @ I @ [1; 0] + 0.1 = [[1.1]]
    assert mu_z == pytest.approx([1.0])
    assert Sigma_z == pytest.approx(np.array([[1.1]]))
    assert Sigma_z.shape == (1, 1)


def test_marginalization_example2():
    """Test marginalization with simple known values.

    The 1d covariance Q is passed as an array instead of a matrix.
    """
    import numpy as np

    # Simple 2D -> 1D case
    A = np.array([[1.0, 0.0]])  # shape [1, 2]
    b = np.array([0.0])  # shape [1]
    Q = np.array([0.1])  # shape [1, 1]
    mu = np.array([1.0, 2.0])  # shape [2]
    Sigma = np.eye(2)  # shape [2, 2]

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)

    # Expected: mu_z = A @ mu + b = [1, 0] @ [1, 2] + 0 = [1]
    # Expected: Sigma_z = A @ Sigma @ A.T + Q = [1, 0] @ I @ [1; 0] + 0.1 = [[1.1]]
    assert mu_z == pytest.approx([1.0])
    assert Sigma_z == pytest.approx(np.array([[1.1]]))
