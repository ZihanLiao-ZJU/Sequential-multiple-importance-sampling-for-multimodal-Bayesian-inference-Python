import copy

import numpy as np

from .._utils import flat, reshape_F, tail_shape
from ..log_operations import logsum


def WgtMis_Blc(intBay, g, c, y):
    """Weights of multiple importance sampling (balance heuristic), log scale."""
    beta = 1
    Nite = len(y)
    Nsze = [tail_shape(v) for v in y]
    Nsam = np.array([a * b for a, b in Nsze], dtype=float)
    intBay = copy.copy(intBay)
    w_mis = [None] * Nite
    for ite in range(Nite):
        yi = flat(y[ite])
        pdf_crs = np.zeros((Nite, yi.shape[1]))
        for ite1 in range(Nite):
            intBay.G = g[ite1]
            intBay.Y = yi[2:, :]
            pdf_crs[ite1, :] = intBay.EvlLKF() + intBay.EvlPDF() - c[ite1]
        # balance heuristic
        w = (beta * np.log(Nsam[ite]) + beta * pdf_crs[ite, :]
             - logsum(beta * pdf_crs + beta * np.log(Nsam)[:, None], 0).ravel())
        w_mis[ite] = reshape_F(w, Nsze[ite])
    return w_mis
