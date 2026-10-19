import numpy as np

from ..log_operations import logmean


def MisInt(w_mis, w_tag):
    """Per-level MIS integral in log scale: logmean(w_mis + w_tag)."""
    return np.array([np.real(logmean(a + b, None)) for a, b in zip(w_mis, w_tag)])
