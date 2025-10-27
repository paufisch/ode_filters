import numpy as np

def sqr_marginalization(A, b, Q_sqr, mu, Sigma_sqr):
    """Marginalize using square-root (Cholesky) covariance representation.

    Numerically stable version of marginalization that works with Cholesky
    factors (square roots) of covariance matrices. Computes the marginal of
    z = Ax + b using QR decomposition for numerical stability.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q_sqr: Cholesky factor of observation noise (shape [n_obs, n_obs]).
        mu: Prior mean (shape [n_state]).
        Sigma_sqr: Cholesky factor of prior covariance (shape [n_state, n_state]).

    Returns:
        mu_z: Marginal mean of z (shape [n_obs]).
        Sigma_z_sqr: Cholesky factor of marginal covariance (shape [n_obs + n_state, n_obs]).
    """

    mu_z = A @ mu + b
    C = np.concatenate([Sigma_sqr @ A.T, Q_sqr], axis=0)
    _, Sigma_z_sqr = np.linalg.qr(C)

    return mu_z, Sigma_z_sqr


def sqr_inversion(A, b, Q_sqr, mu, Sigma_sqr, z):
    """Inversion using square-root covariance representation for stability.

    Numerically stable Bayesian inversion using Cholesky factors and QR
    decomposition. Returns the posterior mean and Cholesky factor.

    Args:
        A: Observation matrix (shape [n_obs, n_state]).
        b: Observation offset (shape [n_obs]).
        Q_sqr: Cholesky factor of observation noise (shape [n_obs, n_obs]).
        mu: Prior mean (shape [n_state]).
        Sigma_sqr: Cholesky factor of prior covariance (shape [n_state, n_state]).
        z: Observation value (shape [n_obs]).

    Returns:
        posterior_mean: Posterior mean (shape [n_state]).
        Lambda_sqr: Cholesky factor of posterior covariance (shape [n_state, n_state]).
    """

    mu_z, Sigma_z_sqr = sqr_marginalization(A, b, Q_sqr, mu, Sigma_sqr)
    Sigma = Sigma_sqr.T @ Sigma_sqr
    Sigma_z = Sigma_z_sqr.T @ Sigma_z_sqr
    Sigma_z_2d = np.atleast_2d(Sigma_z)
    G = Sigma @ A.T @ np.linalg.inv(Sigma_z_2d)
    d = mu - G @ mu_z
    
    B = np.eye((G@A).shape[0]) - G@A
    C = np.concatenate([Sigma_sqr @ B.T, Q_sqr @ G.T], axis=0)
    _, Lambda_sqr = np.linalg.qr(C)
    
    return G@z+d, Lambda_sqr

