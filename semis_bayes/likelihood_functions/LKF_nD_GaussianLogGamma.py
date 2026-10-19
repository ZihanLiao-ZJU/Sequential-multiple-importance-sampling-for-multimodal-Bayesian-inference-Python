import numpy as np
from scipy.special import gammaln

from ..Nataf import Nataf
from ..distributions import makedist
from ._marginals import ks_statistic, plot_2d


class LKF_nD_GaussianLogGamma:
    """Gaussian + LogGamma likelihood in n dimensions.

    Log-evidence: -8.19 (Ndim = 2), -20.47 (Ndim = 5), -40.94 (Ndim = 10), -81.887 (Ndim = 20)

    Reference: Beaujean, F., & Caldwell, A. (2013). Initializing adaptive
    importance sampling with Markov chains. arXiv:1304.7808.
    """

    def __init__(self, Ndim):
        self.Ndim = Ndim
        self.Pdis = Nataf([makedist("Uniform", lower=-30, upper=30)] * Ndim, np.eye(Ndim))
        self.logz_ref = -Ndim * np.log(60)

        # marginal distributions
        dx_ref = 1e-3
        x_ref = np.linspace(-30, 30, 60001)
        x_marg = np.tile(x_ref, (Ndim, 1))
        pdf_marg = np.zeros((Ndim, x_ref.size))
        cdf_marg = np.zeros((Ndim, x_ref.size))
        a, b = self._log_gamma_mix(x_ref), self._log_normal_mix(x_ref)
        for idim in range(1, Ndim + 1):
            if idim == 1:
                logp = a
            elif idim == 2:
                logp = b
            elif idim <= (Ndim + 2) / 2:
                logp = (x_ref - 10) - np.exp(x_ref - 10) - gammaln(1)
            else:
                logp = -0.5 * (x_ref - 10) ** 2 - np.log(np.sqrt(2 * np.pi))
            pdf_marg[idim - 1] = np.exp(logp)
            cdf_marg[idim - 1] = np.cumsum(pdf_marg[idim - 1] * dx_ref)
        self.PDF_marg = pdf_marg
        self.CDF_marg = cdf_marg
        self.X_marg = x_marg

    @staticmethod
    def _log_gamma_mix(t):
        beta_a, alpha_a, c_a = 1, 1, 10
        beta_b, alpha_b, c_b = 1, 1, -10
        with np.errstate(over="ignore", invalid="ignore"):
            ga = alpha_a * (t - c_a) - np.exp(t - c_a) / beta_a - gammaln(alpha_a) - alpha_a * np.log(beta_a)
            gb = alpha_b * (t - c_b) - np.exp(t - c_b) / beta_b - gammaln(alpha_b) - alpha_b * np.log(beta_b)
            gab = np.maximum(ga, gb)
            return np.log(0.5) + gab + np.log(np.exp(ga - gab) + np.exp(gb - gab))

    @staticmethod
    def _log_normal_mix(t):
        mu_c, sd_c, mu_d, sd_d = 10, 1, -10, 1
        nc = -0.5 * ((t - mu_c) / sd_c) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sd_c)
        nd = -0.5 * ((t - mu_d) / sd_d) ** 2 - np.log(np.sqrt(2 * np.pi)) - np.log(sd_d)
        ncd = np.maximum(nc, nd)
        return np.log(0.5) + ncd + np.log(np.exp(nc - ncd) + np.exp(nd - ncd))

    def EvlLKF(self, theta):
        theta = np.asarray(theta, dtype=float)
        L = self._log_gamma_mix(theta[0]) + self._log_normal_mix(theta[1])
        if self.Ndim >= 3:
            with np.errstate(over="ignore"):
                for i in range(3, self.Ndim + 1):
                    ti = theta[i - 1]
                    if i <= (self.Ndim + 2) / 2:
                        L = L + (ti - 10) - np.exp(ti - 10) - gammaln(1)
                    else:
                        L = L + (-0.5 * (ti - 10) ** 2 - np.log(np.sqrt(2 * np.pi)))
        return L

    def PltFun(self):
        x_ref = np.arange(-30, 30.05, 0.1)
        return plot_2d(lambda th: self.EvlLKF(np.vstack([th, np.zeros((self.Ndim - 2, th.shape[1]))]))
                       if self.Ndim > 2 else self.EvlLKF(th), x_ref, x_ref, zfloor=15)

    def KStest(self, x_pos, ind_dim):
        return ks_statistic(x_pos, self.X_marg, self.CDF_marg, ind_dim)
