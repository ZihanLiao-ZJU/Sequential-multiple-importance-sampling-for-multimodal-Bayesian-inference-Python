import copy

import numpy as np
from scipy.optimize import minimize_scalar

from ._utils import flat
from .log_operations import logmean
from .support_functions.ResRat import ResRat
from .support_functions.logGauss import logGauss


def fminbnd(fun, a, b):
    """MATLAB fminbnd (golden section + parabolic interpolation, TolX = 1e-4).

    scipy's bounded method implements the same algorithm with the same
    stopping rule. Returns (x, fval).
    """
    res = minimize_scalar(fun, bounds=(a, b), method="bounded", options={"xatol": 1e-4, "maxiter": 500})
    return res.x, res.fun


class SeMIS_Bay_Nataf:
    """SeMIS Bayesian updating in Nataf-transformed (standard normal) space.

    Paper-aligned naming:
        r_i   (implementation) <-> lambda_i (theory)
        r_max (implementation) <-> r^{max}_{0:i-1} (running max of log-likelihood)

    EvlLKF() computes the SOFT TRUNCATION term
        logLi(theta) = log(min(L(theta) / exp(r_max + r_i), 1))
                     = min(logL(theta) - r_max - r_i, 0)
    The ratio beta_i(theta) in Eq. (24) is built in ResRat from two logLi terms.
    """

    def __init__(self, LKF):
        self.LKF = LKF
        self.Nfun = 4
        self.Ndim = LKF.Ndim
        self.Ncal = None
        self.X = None
        self.Y = None
        # intermediate parameters G = [r_i, r_max]
        self.G = np.array([-np.inf, -np.inf])

    def EvlLKF(self):
        """Soft truncation term logLi = min(logL - r_max - r_i, 0)."""
        logL = self.Y[0, :]
        r_i, r_max = self.G[0], self.G[1]
        if np.isinf(r_i):
            return np.zeros_like(logL)
        return np.fmin(logL - r_max - r_i, 0)

    def EvlPDF(self):
        """Prior log-density logPi."""
        return self.Y[1, :]

    def UpdObj(self, y, p):
        """Update G = [r_i, r_max].

        r_i solves Eq. (25), logmean(beta_i) = log(p), by a 1D bounded search
        over 100 sub-intervals.
        """
        obj = copy.copy(self)
        logL = flat(y)[2, :]
        r_i = obj.G[0]
        r_max = max(np.nanmax(logL), obj.G[1])
        dr_max = r_max - obj.G[1]

        maxi = 100
        err_min = np.inf

        def fun(ri):
            return abs(logmean(ResRat(obj, y, np.array([ri, r_max]))[0], None) - np.log(p))

        rmin = max(-1e12, r_i - dr_max)
        rmax = 0.0
        rbnd = (rmax - rmin) / maxi
        for i in range(1, maxi + 1):
            r_tmp, err = fminbnd(fun, rmin + (i - 1) * rbnd, rmin + i * rbnd)
            if err < err_min:
                err_min = err
                r_i = r_tmp

        G = np.array([r_i, r_max])
        obj.G = G
        return G, obj

    @property
    def FlgCvg(self):
        return bool(self.G[0] >= -1e-4)

    def EvlY(self, u):
        """Evaluate and cache the base terms obj.Y = [logL; logPi] at u (U-space)."""
        obj = copy.copy(self)
        x = obj.U2X(u)
        logL = np.asarray(obj.LKF.EvlLKF(x), dtype=float).ravel()
        obj.X = u
        obj.Y = np.vstack([logL, logGauss(u)])
        obj.Ncal = u.shape[1]
        return obj

    def UpdY(self, y):
        """Refresh and prepend the intermediate terms: y := [logLi; logPi; base terms]."""
        obj = copy.copy(self)
        y = flat(y)[2:, :]
        obj.Y = y
        return np.vstack([obj.EvlLKF(), obj.EvlPDF(), y])

    def U2X(self, u):
        """Map U-space samples to X-space via the Nataf transformation."""
        return self.LKF.Pdis.U2X(u)
