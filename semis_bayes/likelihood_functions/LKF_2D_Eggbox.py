import numpy as np

from ..Nataf import Nataf
from ..distributions import makedist
from ..log_operations import logsum
from ._marginals import ks_statistic, plot_2d


class LKF_2D_Eggbox:
    """2D eggbox-shaped likelihood.

    1. Dimension: 2
    2. Prior: Uniform(0, 10*pi) in each dimension, independent
    3. logz = 235.856

    Reference: Feroz, F., Hobson, M. P., & Bridges, M. (2009). MultiNest: an
    efficient and robust Bayesian inference tool for cosmology and particle
    physics. MNRAS, 398(4), 1601-1614.
    """

    Ndim = 2
    logz_ref = 235.856

    def __init__(self):
        self.Pdis = Nataf([makedist("Uniform", lower=0, upper=10 * np.pi)] * 2, np.eye(2))
        self._marg = None

    def _marginals(self):
        # Reference marginal PDFs/CDFs on the MATLAB grid. They are computed on
        # first use (KStest) because the double integral is expensive.
        if self._marg is None:
            dx_ref, dx_itg = 1e-3, 5e-4
            x_ref = np.arange(0, 10 * np.pi + dx_ref / 2, dx_ref)
            x_itg = np.arange(0, 10 * np.pi + dx_itg / 2, dx_itg)
            # the eggbox is symmetric in its two arguments: both marginals are equal
            pdf = np.zeros(x_ref.size)
            ci = np.cos(x_itg / 2)
            for s in range(0, x_ref.size, 256):
                cr = np.cos(x_ref[s:s + 256] / 2)[:, None]
                L = (2 + cr * ci[None, :]) ** 5
                pdf[s:s + 256] = np.exp(logsum(L + np.log(dx_itg) - 2 * np.log(10 * np.pi), 1).ravel() - self.logz_ref)
            pdf_marg = np.vstack([pdf, pdf])
            self._marg = (np.vstack([x_ref, x_ref]), pdf_marg, np.cumsum(pdf_marg * dx_ref, axis=1))
        return self._marg

    @property
    def X_marg(self):
        return self._marginals()[0]

    @property
    def PDF_marg(self):
        return self._marginals()[1]

    @property
    def CDF_marg(self):
        return self._marginals()[2]

    def EvlLKF(self, theta):
        return (2 + np.cos(theta[0] / 2) * np.cos(theta[1] / 2)) ** 5

    def PltFun(self):
        x_ref = np.arange(0, 10 * np.pi, 0.1)
        return plot_2d(self.EvlLKF, x_ref, x_ref)

    def KStest(self, x_pos, ind_dim):
        return ks_statistic(x_pos, self.X_marg, self.CDF_marg, ind_dim)
