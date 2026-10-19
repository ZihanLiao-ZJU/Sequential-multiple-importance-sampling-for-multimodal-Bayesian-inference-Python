"""
Sequential Monte Carlo sampler for Bayesian sequential schemes

Developers: Zihan Liao, Binbin Li, Hua-Ping Wan, Xiao He (Zhejiang University)
Init: Jan 2023 | Last modified: Apr 2025 | Python port: Oct 2026
"""
import copy

import numpy as np

from ._utils import mround
from .support_functions.DspRst import DspRst
from .support_functions.SedSlt import SedSlt
from .support_functions.VabSrk import VabSrk


class SeMCS:
    """Sequential Monte Carlo loop shared by SeMIS and SuS.

    Properties
        IntBay  : intermediate Bayesian state (updated each iteration)
        Sampler : sampling engine (requires .BayStc and .SamGen)
        ConPrb  : target conditional probability (default 0.1)
        NumSam  : samples per iteration (scalar or per-iteration vector)
        G       : optional predefined intermediate parameters
        Verbose : print the progress of each iteration (default True)
    """

    MaxIte = 10000

    def __init__(self, Sampler):
        if Sampler is None:
            raise ValueError("Sampler is required and must contain a valid BayStc field.")
        self.Sampler = Sampler
        self.IntBay = Sampler.BayStc
        self.ConPrb = 0.1
        self.NumSam = 1000
        self.G = []
        self.X = []
        self.Y = []
        self.Verbose = True

    def RunIte(self):
        """Run sequential iterations until convergence or termination."""
        intBay = self.IntBay
        sampler = copy.copy(self.Sampler)
        p = self.ConPrb
        maxi = self.MaxIte
        NumSam = np.atleast_1d(self.NumSam)
        use_vecN = NumSam.size > 1
        use_G = len(self.G) > 0

        Nsam = np.zeros(maxi + 1, dtype=int)
        Nsze = np.zeros((maxi + 1, 2))
        Ncal = np.zeros(maxi + 1, dtype=int)
        x = [None] * (maxi + 1)
        y = [None] * (maxi + 1)
        g = [None] * (maxi + 1)

        if use_vecN:
            Nsam[:NumSam.size] = NumSam
        else:
            Nsam[:] = NumSam[0]
        Nsze[0, :] = [Nsam[0], 1]

        if use_G:
            Nite_int = len(self.G)
            g[:Nite_int] = list(self.G)
        else:
            Nite_int = 0
            g[0] = intBay.G

        x_sed = y_sed = None
        flg_cvg = False
        for ite in range(1, maxi + 1):
            i = ite - 1
            # 1) generate samples (first iteration: from prior; later: from seeds)
            if np.all(Nsze[i, :] > 0) and not np.isnan(Nsze[i, 1]):
                if ite == 1:
                    x[i], y[i], Ncal[i], intBay = sampler.SamGen(int(Nsze[i, 0]), int(Nsze[i, 1]))
                else:
                    x[i], y[i], Ncal[i], intBay = sampler.SamGen(x_sed, y_sed, int(Nsze[i, 1]))
            Nsam[i] = int(np.prod(Nsze[i, :]))

            # 2) update intermediate distribution parameter g
            if ite + 1 <= Nite_int:
                intBay = copy.copy(intBay)
                intBay.G = g[i + 1]
            else:
                g[i + 1], intBay = intBay.UpdObj(y[i], p)
            sampler.BayStc = intBay

            # 3) select seeds and set the chain configuration for the next iteration
            x_sed, y_sed = SedSlt(intBay, x[i], y[i], int(Nsam[i + 1]))
            Nsze[i + 1, 0] = x_sed.shape[1]
            Nsze[i + 1, 1] = mround(Nsam[i + 1] / Nsze[i + 1, 0]) if Nsze[i + 1, 0] > 0 else np.inf

            # 4) console summary
            if self.Verbose:
                DspRst(ite, Nsam[i], Nsze[i, :], Ncal[i])

            # 5) termination conditions
            if Nsze[i + 1, 0] <= 0 or ite + 1 == maxi:
                flg_cvg = False
                break
            elif intBay.FlgCvg:
                flg_cvg = True
                break

        Nite = ite
        x, y, Nsam, Nsze, Ncal = VabSrk(Nite, x, y, Nsam, Nsze, Ncal)
        g = VabSrk(Nite + 1, g)
        return dict(x=x, y=y, g=g, Nsam=Nsam, Nsze=Nsze.astype(int), Ncal=Ncal,
                    Nite=Nite, flg_cvg=flg_cvg, intBay=intBay)
