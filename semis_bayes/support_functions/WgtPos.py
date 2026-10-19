import numpy as np

from .._utils import flat, reshape_F, tail_shape


def WgtPos(y):
    """Log-weights of the posterior distribution at every level."""
    w_pos = []
    for yi_full in y:
        yi = flat(yi_full)
        with np.errstate(invalid="ignore"):
            w = yi[2, :] + yi[3, :] - yi[0, :] - yi[1, :]
        w = reshape_F(w, tail_shape(yi_full)).astype(float)
        w[np.isnan(w)] = -np.inf
        w_pos.append(w)
    return w_pos
