import copy

import numpy as np

from ._utils import flat, mround, randsample
from .log_operations import logmean, logsum
from .support_functions.MisInt import MisInt
from .support_functions.ResRat import ResRat
from .support_functions.VabCmb import VabCmb
from .support_functions.WgtMis_Blc import WgtMis_Blc
from .support_functions.WgtPos import WgtPos


class MIS:
    """Posterior simulation and evidence estimation with multiple importance sampling.

    obj = MIS(out)                  # out: output of SeMCS.RunIte
    u_pos, y_pos, u_all, y_all = obj.PosSim()

    Properties
        Wmis : weights of multiple importance sampling (balance heuristic)
        Wpos : weights of the posterior distribution
        Wres : resampling weights between levels
        C    : log normalizing constants of the intermediate distributions
        Z    : log-evidence estimated with all generated samples
        Zi   : evidence contribution of each iteration

    Reference: Handley & Lemos (2019), Physical Review D 100(2): 023512
    """

    def __init__(self, inp):
        self.X = inp["x"]
        self.Y = inp["y"]
        self.G = inp["g"]
        self.NumIte = inp["Nite"]
        self.NumSam = inp["Nsam"]
        self.SamSze = inp["Nsze"]
        self.IntBay = inp["intBay"]
        self.H = None
        Nite = self.NumIte
        y = [np.array(v, copy=True) for v in self.Y]
        g = self.G
        intBay = copy.copy(self.IntBay)

        # normalize y
        w_res = [None] * Nite
        c = np.zeros(Nite + 1)
        for ite in range(Nite):
            w_res[ite], _ = ResRat(intBay, y[ite], g[ite + 1])
            c[ite + 1] = c[ite] + np.real(logmean(w_res[ite], None))
            if ite < Nite - 1:
                y[ite + 1][0] = y[ite + 1][0] - c[ite + 1]
        self.C = c

        # weights of samples (balance heuristic)
        w_mis = WgtMis_Blc(intBay, g, c, y)

        # evidence
        w_pos = WgtPos(y)
        z_i = MisInt(w_mis, w_pos)
        z = float(np.real(logsum(z_i, None)))

        # normalization of the posterior weights
        w_pos = [w - z for w in w_pos]

        self.Wmis = w_mis
        self.Wpos = w_pos
        self.Wres = w_res
        self.Z = z
        self.Zi = z_i

    def PosSim(self):
        """Resample posterior samples from all levels with the MIS weights."""
        u_all, y_all = VabCmb(self.X, self.Y)
        w_mis_flt, w_pos_flt = VabCmb(self.Wmis, self.Wpos)
        w_pos_mis = (w_mis_flt + w_pos_flt).ravel(order="F")
        Npos = int(mround(np.exp(2 * logsum(w_pos_mis, None) - logsum(2 * w_pos_mis, None))))
        Nall = y_all.shape[1]
        w_norm = np.exp(w_pos_mis - np.max(w_pos_mis))
        ind_pos = randsample(Nall, Npos, w_norm)
        return u_all[:, ind_pos], y_all[:, ind_pos], u_all, y_all

    def UnqSam(self):
        """Number of unique samples at each iteration."""
        Nunq = np.zeros(self.NumIte, dtype=int)
        for ite in range(self.NumIte):
            Nunq[ite] = np.unique(flat(self.X[ite]).T, axis=0).shape[0]
        return Nunq
