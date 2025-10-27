import numpy as np
from scipy.linalg import cho_factor, cho_solve


def marginalization(A, b, Q, mu, Sigma):
    """Marginalize out the linear transformation in a Gaussian model.

    Computes the marginal distribution of z = Ax + b given p(x) ~ N(mu, Sigma)
    and p(z|x) ~ N(Ax + b, Q).

    Args:
        A: Linear transformation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q: Observation noise covariance (shape [n_obs, n_obs]).
        mu: Prior mean (shape [n_state]).
        Sigma: Prior covariance (shape [n_state, n_state]).

    Returns:
        mu_z: Marginal mean of z (shape [n_obs]).
        Sigma_z: Marginal covariance of z (shape [n_obs, n_obs]).
    """

    mu_z = A @ mu + b
    Sigma_z = A @ Sigma @ A.T + Q

    return mu_z, Sigma_z


def inversion(A, b, Q, mu, Sigma, z):
    """Compute the posterior of x given observation z using Bayesian inversion.

    Given p(x) ~ N(mu, Sigma), p(z|x) ~ N(Ax + b, Q), and an observation z,
    computes p(x|z) ~ N(G@z + d, Lambda) efficiently using Cholesky decomposition.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q: Observation noise covariance (shape [n_obs, n_obs]).
        mu: Prior mean (shape [n_state]).
        Sigma: Prior covariance (shape [n_state, n_state]).
        z: Observation value (shape [n_obs]).

    Returns:
        posterior_mean: Posterior mean N(G@z + d, Lambda) (shape [n_state]).
        Lambda: Posterior covariance (shape [n_state, n_state]).
    """

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)
    Sigma_z_2d = np.atleast_2d(Sigma_z)
    # Efficient inverse via Cholesky decomposition instead of np.linalg.inv
    L, lower = cho_factor(Sigma_z_2d, lower=True, check_finite=False)
    Y = cho_solve((L, lower), (Sigma @ A.T).T, check_finite=False)
    G = Y.T
    
    
    d = mu - G @ mu_z
    Lambda = Sigma - G @ Sigma_z @ G.T
    return G@z+d, Lambda


def inversion2(A, mu, Sigma, mu_z, Sigma_z):
    """Compute posterior parameters without explicit observation value.

    Variant of inversion that returns the gain matrix G, offset d, and posterior
    covariance Lambda separately, given pre-computed marginal parameters.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        mu: Prior mean (shape [n_state]).
        Sigma: Prior covariance (shape [n_state, n_state]).
        mu_z: Marginal observation mean (shape [n_obs]).
        Sigma_z: Marginal observation covariance (shape [n_obs, n_obs]).

    Returns:
        G: Kalman gain matrix (shape [n_state, n_obs]).
        d: Posterior offset (shape [n_state]).
        Lambda: Posterior covariance (shape [n_state, n_state]).
    """

    Sigma_z_2d = np.atleast_2d(Sigma_z)
    # Efficient inverse via Cholesky decomposition
    L, lower = cho_factor(Sigma_z_2d, lower=True, check_finite=False)
    Y = cho_solve((L, lower), (Sigma @ A.T).T, check_finite=False)
    G = Y.T
    
    
    d = mu - G @ mu_z
    Lambda = Sigma - G @ Sigma_z @ G.T
    return G, d, Lambda

