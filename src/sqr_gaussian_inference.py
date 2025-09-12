import numpy as np

from scipy.linalg import cho_factor, cho_solve

def conditioning(A, x, b, Q):

    return A @ x + b, Q


def sqr_marginalization(A, b, Q_sqr, mu, Sigma_sqr):

    mu_z = A @ mu + b
    C = np.concatenate([Sigma_sqr @ A.T, Q_sqr], axis=0)
    _, Sigma_z_sqr = np.linalg.qr(C)

    return mu_z, Sigma_z_sqr


def sqr_inversion(A, b, Q_sqr, mu, Sigma_sqr, z):

    mu_z, Sigma_z_sqr = sqr_marginalization(A, b, Q_sqr, mu, Sigma_sqr)
    Sigma = Sigma_sqr.T @ Sigma_sqr
    Sigma_z = Sigma_z_sqr.T @ Sigma_z_sqr
    Sigma_z_2d = np.atleast_2d(Sigma_z)
    G = Sigma @ A.T @ np.linalg.inv(Sigma_z_2d)
    d = mu - G @ mu_z
    
    B = np.eye((G@A).shape[0]) - G@A
    C = np.concatenate([Sigma_sqr @ B.T, Q_sqr @ G.T], axis=0)
    _, Lambda_sqr = np.linalg.qr(C)
    
    return conditioning(G, z, d, Lambda_sqr)

