from typing import Callable

import jax
import numpy as np


class ODE_information:
    def __init__(self, vf: Callable, d: int = 1, q: int = 1):
        """Initialize the ODE_information class.

        Args:
            d (int): Dimension of the state space, default is 1.
            q (int): Order of the differential equation, default is 1.
            vf (callable): The vector field function of the ODE.
        """

        # define projection matrices
        self._E0 = np.zeros(q + 1)
        self._E1 = np.zeros(q + 1)
        self._E0[0] = 1.0
        self._E1[1] = 1.0
        self._E0 = np.kron(self._E0, np.eye(d))
        self._E1 = np.kron(self._E1, np.eye(d))
        self._vf = vf

    # observation model
    def g(self, X):
        g_ODE = self._E1 @ X - self._vf(self._E0 @ X)
        return g_ODE

    def jacobian_g(self, X):
        return jax.jacfwd(self.g)(X)
