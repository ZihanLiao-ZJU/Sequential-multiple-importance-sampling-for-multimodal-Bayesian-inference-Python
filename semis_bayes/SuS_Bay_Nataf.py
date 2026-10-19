import copy

import numpy as np

from ._utils import flat, mround, randi, randperm, randsample
from .log_operations import logmean, logminus, logsum
from .support_functions.ResRat import ResRat
from .support_functions.VabCmb import VabCmb
from .support_functions.logGauss import logGauss


def _mmin(a, b):
    """MATLAB min of two arrays (complex values are compared by magnitude)."""
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        return np.where(np.abs(b) < np.abs(a), b, a)
    return np.fmin(a, b)


class SuS_Bay_Nataf:
    """Subset simulation (SuS) with Nataf transformation for Bayesian updating.

    Naming convention (consistent with SeMIS_Bay_Nataf):
        logLi : log-domain HARD truncation term (indicator)
        logPi : log prior density
        G = [r_i, r_max]
            r_i   : adaptive threshold (level parameter)
            r_max : running maximum of the log-likelihood
    """

    def __init__(self, LKF):
        self.LKF = LKF
        self.Nfun = 4
        self.Ndim = LKF.Ndim
        self.Ncal = None
        self.X = None
        self.Y = None
        self.G = np.array([-np.inf, -np.inf])

    def EvlLKF(self):
        """HARD truncation term: logLi = log(I(logL > r_max + r_i))."""
        logL = self.Y[0, :]
        r_i, r_max = self.G[0], self.G[1]
        if np.isinf(r_i):
            return np.zeros_like(logL)
        with np.errstate(divide="ignore"):
            return np.log((logL > r_max + r_i).astype(float))

    def EvlPDF(self):
        """Prior log-density logPi."""
        return self.Y[1, :]

    def UpdObj(self, y, p):
        """Update G = [r_i, r_max] with the standard SuS quantile rule."""
        obj = copy.copy(self)
        logL = flat(y)[2, :]
        Nsam = logL.size
        l_sorted = np.sort(logL)[::-1]
        Nsed = int(mround(Nsam * p))
        r_max = max(np.nanmax(logL), obj.G[1])
        fth = (l_sorted[Nsed - 1] + l_sorted[Nsed]) / 2 if Nsed > 0 else l_sorted[0]
        G = np.array([fth - r_max, r_max])
        obj.G = G
        return G, obj

    def SltSed(self, u, y, Nsam):
        """Select seeds satisfying the hard truncation condition."""
        u = flat(u)[:self.Ndim, :]
        y = flat(y)[:self.Nfun, :]
        logL = y[2, :]
        Lth = self.G[1] + self.G[0]
        Ns = mround(Nsam / np.arange(1, Nsam + 1))
        Nsed = mround(Nsam / Ns)
        ind_sed = logL > Lth
        u_can = u[:, ind_sed]
        y_can = y[:, ind_sed]
        Ncan = int(np.sum(ind_sed))
        ok = np.nonzero(Ncan - Nsed >= 0)[0]
        Nsed = int(Nsed[ok[-1]]) if ok.size else 0
        if Nsed > 0 and Ncan >= Nsed:
            randind = randperm(Ncan, Nsed)
        elif Nsed > 0 and Ncan > 0:
            randind = randi(Ncan, Nsed)
        else:
            randind = np.array([], dtype=int)
        u_sed = u_can[:, randind]
        y_sed = y_can[:, randind]
        if y_sed.size:
            y_sed = self.UpdY(y_sed)
        return u_sed, y_sed

    @property
    def FlgCvg(self):
        return bool(self.G[0] >= -1e-1)

    def BayInf(self, inp):
        """Evidence estimation and posterior resampling from the SuS levels."""
        x = inp["x"]
        y = [np.array(v, copy=True) for v in inp["y"]]
        g = inp["g"]
        Nite = inp["Nite"]

        # ---- normalize weights across levels ----
        w_res = [None] * Nite
        c = np.zeros(Nite)
        for ite in range(Nite - 1):
            w_res[ite], _ = ResRat(self, y[ite], g[ite + 1])
            c[ite + 1] = c[ite] + logmean(w_res[ite], None)
            y[ite + 1][0] = y[ite + 1][0] - c[ite + 1]

        # ---- evidence estimation ----
        zi = np.zeros(Nite)
        for ite in range(Nite):
            yi = flat(y[ite])
            Li = g[ite][0] + g[ite][1]
            if ite < Nite - 1:
                Li1 = g[ite + 1][0] + g[ite + 1][1]
                w_z = _mmin(logminus(yi[2, :], Li), logminus(Li1, Li))
            else:
                w_z = logminus(yi[2, :], Li)
            zi[ite] = np.real(logmean(w_z, None) + c[ite])
        z = float(np.real(logsum(zi, None)))

        # ---- posterior resampling ----
        w_pos = []
        for ite in range(Nite):
            yi = flat(y[ite])
            with np.errstate(divide="ignore", invalid="ignore"):
                w_tmp = yi[2, :] + yi[3, :] - yi[0, :] - yi[1, :]
                if ite < Nite - 1:
                    w_tmp = w_tmp + np.log((yi[2, :] < g[ite + 1][0] + g[ite + 1][1]).astype(float))
            w_tmp = np.where(np.isnan(w_tmp), -np.inf, w_tmp)
            w_pos.append(w_tmp)

        x_all, y_all = VabCmb(x, y)
        w_pos = np.concatenate(w_pos)
        w_pos = np.exp(w_pos - np.max(w_pos))
        Npos = int(mround(np.sum(w_pos) ** 2 / np.sum(w_pos ** 2)))
        ind_pos = randsample(y_all.shape[1], Npos, w_pos)
        return dict(z=z, x_pos=x_all[:, ind_pos], y_pos=y_all[:, ind_pos])

    def EvlY(self, u):
        """Evaluate the base terms obj.Y = [logL; logPi]."""
        obj = copy.copy(self)
        x = obj.U2X(u)
        logL = np.asarray(obj.LKF.EvlLKF(x), dtype=float).ravel()
        obj.X = u
        obj.Y = np.vstack([logL, logGauss(u)])
        obj.Ncal = u.shape[1]
        return obj

    def UpdY(self, y):
        """Prepend the intermediate terms: y := [logLi; logPi; base terms]."""
        obj = copy.copy(self)
        y = flat(y)[2:, :]
        obj.Y = y
        return np.vstack([obj.EvlLKF(), obj.EvlPDF(), y])

    def U2X(self, u):
        """Nataf transformation from U-space to X-space."""
        return self.LKF.Pdis.U2X(u)
