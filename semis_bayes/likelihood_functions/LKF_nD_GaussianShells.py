import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist


class LKF_nD_GaussianShells:
    """Two Gaussian shells.

    Log-evidence: -1.75 (Ndim = 2), -5.67 (5), -14.59 (10), -36.09 (20), -60.13 (30),
    -112.42 (50), -168.16 (70), -255.62 (100)

    Reference: Feroz F, Hobson M P. Multimodal nested sampling: an efficient and
    robust alternative to Markov Chain Monte Carlo methods for astronomical data
    analyses. MNRAS, 2008, 384(2): 449-463.
    """

    Nfun = 1
    LogL = True
    _LOGZ = {2: -1.75, 5: -5.67, 10: -14.59, 20: -36.09, 30: -60.13, 50: -112.42, 70: -168.16, 100: -255.62}

    def __init__(self, Ndim):
        self.Ndim = Ndim
        self.Pdis = Nataf([makedist("Uniform", lower=-6, upper=6)] * Ndim, np.eye(Ndim))
        # reference log-evidence (NaN for dimensions not listed above)
        self.logz_ref = self._LOGZ.get(Ndim, np.nan)

    def EvlLKF(self, theta):
        theta = np.asarray(theta, dtype=float)
        c = np.zeros((self.Ndim, 2))
        c[0, 0], c[0, 1] = 3.5, -3.5
        r, omega = 2.0, 0.1
        X = -1 * (np.sqrt(np.sum((theta - c[:, [0]]) ** 2, 0)) - r) ** 2 / 2 / omega ** 2
        Y = -1 * (np.sqrt(np.sum((theta - c[:, [1]]) ** 2, 0)) - r) ** 2 / 2 / omega ** 2
        XY = np.maximum(X, Y)
        return np.log(1 / np.sqrt(2 * np.pi * omega ** 2)) + XY + np.log(np.exp(X - XY) + np.exp(Y - XY))
