import copy

import numpy as np

from .._utils import flat, nanmax, reshape_F, tail_shape


def ResRat(intBay, y, g):
    """Log ratio between the intermediate distribution with parameters ``g`` and
    the distribution the samples ``y`` were drawn from.

    Returns ``lograt`` [Nc x Ns] (shifted so that its maximum is 0) and the shift ``c``.
    """
    y2 = flat(y)
    intBay = copy.copy(intBay)
    intBay.G = g
    intBay.Y = y2[2:, :]
    Li = intBay.EvlLKF()
    pi = intBay.EvlPDF()
    with np.errstate(invalid="ignore"):
        lograt_tmp = Li + pi - y2[0, :] - y2[1, :]
        c = nanmax(lograt_tmp)
        lograt_tmp = lograt_tmp - c
    lograt_tmp[np.isnan(lograt_tmp)] = -np.inf
    return reshape_F(lograt_tmp, tail_shape(y)), c
