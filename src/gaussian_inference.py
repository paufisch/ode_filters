import numpy as np
from scipy.linalg import cho_factor, cho_solve



def marginalization(A, b, Q, mu, Sigma):
    """
    p(x) = N(mu|Sigma)
    z = Ax + b
    p(z|x) = N(A@x+b, Q)
    p(z) = N(A@mu+b, A@Sigma@A.T + Q)
    """

    mu_z = A @ mu + b
    Sigma_z = A @ Sigma @ A.T + Q

    return mu_z, Sigma_z


def inversion(A, b, Q, mu, Sigma, z):
    """
    p(x) = N(mu|Sigma)
    z = Ax + b
    p(z|x) = N(A@x+b, Q)
    p(z) = N(mu_z, Sigma_z)
    p(x|z) = N(G@z+d, Lambda) = N(A@x+b, Q) * N(mu|Sigma) / N(mu_z, Sigma_z)
    """

    mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)
    Sigma_z_2d = np.atleast_2d(Sigma_z)
    #G = Sigma @ A.T @ np.linalg.inv(Sigma_z_2d) 
    #instead of the naive inverse computation more efficiently:
    L, lower = cho_factor(Sigma_z_2d, lower=True, check_finite=False)
    Y = cho_solve((L, lower), (Sigma @ A.T).T, check_finite=False)
    G = Y.T
    
    
    d = mu - G @ mu_z
    Lambda = Sigma - G @ Sigma_z @ G.T
    #B = np.eye((G@A).shape[0]) - G@A
    #Lambda = B @ Sigma @ B.T + G @ Q @ G.T #Josephson Form
    return G@z+d, Lambda


def inversion2(A, mu, Sigma, mu_z, Sigma_z):
    """
    p(x) = N(mu|Sigma)
    z = Ax + b
    p(z|x) = N(A@x+b, Q)
    p(z) = N(mu_z, Sigma_z)
    p(x|z) = N(G@z+d, Lambda) = N(A@x+b, Q) * N(mu|Sigma) / N(mu_z, Sigma_z)
    """

    #mu_z, Sigma_z = marginalization(A, b, Q, mu, Sigma)
    Sigma_z_2d = np.atleast_2d(Sigma_z)
    #G = Sigma @ A.T @ np.linalg.inv(Sigma_z_2d) 
    #instead of the naive inverse computation more efficiently:
    L, lower = cho_factor(Sigma_z_2d, lower=True, check_finite=False)
    Y = cho_solve((L, lower), (Sigma @ A.T).T, check_finite=False)
    G = Y.T
    
    
    d = mu - G @ mu_z
    Lambda = Sigma - G @ Sigma_z @ G.T
    #B = np.eye((G@A).shape[0]) - G@A
    #Lambda = B @ Sigma @ B.T + G @ Q @ G.T #Josephson Form
    return G, d, Lambda

