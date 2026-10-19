import numpy as np

from .._utils import rand, randn


class ES:
    """Elliptical slice sampling with Nataf transformation.

    Constants
        SedSav : save the seeds or not (default: False)
        Ntry   : maximum number of shrinking steps (default: 20)
    """

    SedSav = False
    Ntry = 20

    def __init__(self, BayStc):
        self.BayStc = BayStc
        self.Nfun = BayStc.Nfun
        self.Ndim = BayStc.Ndim

    def SamGen(self, *args):
        """Generate samples distributed as prior * intermediate likelihood.

        SamGen(Nc, Ns)              : start from Nc prior samples (no seeds)
        SamGen(x_sed, y_sed, Ns)    : Markov chains started from seeds

        Returns u [Ndim x Nc x Ns], y [Nfun x Nc x Ns], Ncal and the updated Bayesian structure.
        """
        baystc = self.BayStc
        sedsav = self.SedSav or len(args) <= 2
        nTry = self.Ntry
        nDim, nFun = self.Ndim, self.Nfun
        if len(args) <= 2:
            Nc, Ns = int(args[0]), int(args[1])
            u_sed = randn(nDim, Nc)
            baystc = baystc.EvlY(u_sed)
            y_sed = np.vstack([baystc.EvlLKF(), baystc.EvlPDF(), baystc.Y])
            Ncal = baystc.Ncal
        else:
            u_sed = np.array(args[0], dtype=float, copy=True)
            y_sed = np.array(args[1], dtype=float, copy=True)
            Ns = int(args[2])
            Nc = u_sed.shape[1]
            Ncal = 0

        u = np.zeros((nDim, Nc, Ns))
        y = np.zeros((nFun, Nc, Ns))

        du = randn(nDim, Nc, Ns)
        p = rand(1, Nc, Ns)

        for k in range(Ns):
            if k == 0 and sedsav:
                u[:, :, 0] = u_sed
                y[:, :, 0] = y_sed
            else:
                Rho = np.sqrt(u_sed ** 2 + du[:, :, k] ** 2)            # [Ndim, Nc]
                with np.errstate(invalid="ignore"):
                    alpha_appro = np.arccos(u_sed / Rho)                 # [Ndim, Nc]
                ind_sam = np.arange(Nc)
                Nrst = Nc
                alphamin = np.zeros((nDim, Nc))
                alphamax = np.pi * np.ones((nDim, Nc))
                for _ in range(nTry):
                    alpha = rand(nDim, Nrst) * (alphamax - alphamin) + alphamin
                    u[:, ind_sam, k] = np.cos(alpha) * Rho[:, ind_sam]
                    baystc = baystc.EvlY(u[:, ind_sam, k])
                    y[:, ind_sam, k] = np.vstack([baystc.EvlLKF(), baystc.EvlPDF(), baystc.Y])
                    Ncal += baystc.Ncal
                    with np.errstate(invalid="ignore"):
                        ind_rej = np.exp(y[0, ind_sam, k] - y_sed[0, ind_sam]) < p[0, ind_sam, k]
                    Nrst = int(np.sum(ind_rej))
                    if Nrst > 0:
                        ind_sam = ind_sam[ind_rej]
                        u[:, ind_sam, k] = u_sed[:, ind_sam]
                        y[:, ind_sam, k] = y_sed[:, ind_sam]
                        alpha = alpha[:, ind_rej]
                        alphamin = alphamin[:, ind_rej]
                        alphamax = alphamax[:, ind_rej]
                        ind_alpha = alpha > alpha_appro[:, ind_sam]
                        alphamin[~ind_alpha] = alpha[~ind_alpha]
                        alphamax[ind_alpha] = alpha[ind_alpha]
                    else:
                        break
                u_sed = u[:, :, k].copy()
                y_sed = y[:, :, k].copy()
        return u, y, Ncal, baystc
