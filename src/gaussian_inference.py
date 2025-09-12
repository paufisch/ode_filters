import numpy as np
from scipy.linalg import cho_factor, cho_solve

def conditioning(A, x, b, Q):

    return A @ x + b, Q


def marginalization(A, b, Q, mu, Sigma):

    mu_z = A @ mu + b
    Sigma_z = A @ Sigma @ A.T + Q

    return mu_z, Sigma_z


def inversion(A, b, Q, mu, Sigma, z):

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)
    Sigma_z_2d = np.atleast_2d(Sigma_z)
    #G = Sigma @ A.T @ np.linalg.inv(Sigma_z_2d) 
    #instead of the naive inverse computation more efficiently:
    L, lower = cho_factor(Sigma_z_2d, lower=True, check_finite=False)
    Y = cho_solve((L, lower), (Sigma @ A.T).T, check_finite=False)
    G = Y.T
    
    
    d = mu - G @ mu_z
    #Lambda = Sigma - G @ Sigma_z @ G.T
    B = np.eye((G@A).shape[0]) - G@A
    Lambda = B @ Sigma @ B.T + G @ Q @ G.T #Josephson Form
    return conditioning(G, z, d, Lambda)

