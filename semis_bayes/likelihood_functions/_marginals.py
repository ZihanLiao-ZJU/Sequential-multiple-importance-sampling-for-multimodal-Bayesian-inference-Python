import numpy as np


def ks_statistic(x_pos, X_marg, CDF_marg, ind_dim):
    """Kolmogorov-Smirnov distance between the empirical marginal CDFs of the
    posterior samples and the reference marginal CDFs.

    x_pos   : posterior samples in the parameter space [Ndim x Npos]
    ind_dim : dimensions to test (1-based, as in MATLAB)
    """
    ind_dim = np.atleast_1d(ind_dim)
    ks = np.zeros(ind_dim.size)
    for j, idim in enumerate(ind_dim):
        xs = np.sort(np.asarray(x_pos)[idim - 1, :])
        x_emp = np.unique(xs)
        cnt = np.searchsorted(xs, x_emp, side="right")
        cdf_emp = cnt / xs.size
        x_ref = X_marg[idim - 1, :]
        cdf_ref = CDF_marg[idim - 1, :]
        x_all = np.unique(np.concatenate([x_emp, x_ref]))

        def prev(xp, fp, xq):
            idx = np.searchsorted(xp, xq, side="right") - 1
            out = np.where(idx >= 0, fp[np.clip(idx, 0, None)], 0.0)
            return out

        ks[j] = np.max(np.abs(prev(x_emp, cdf_emp, x_all) - prev(x_ref, cdf_ref, x_all)))
    return ks


def plot_2d(EvlLKF, x_ref, y_ref, zfloor=None):
    """Contour plot of a 2D log-likelihood (matplotlib)."""
    import matplotlib.pyplot as plt
    xx, yy = np.meshgrid(x_ref, y_ref)
    zz = np.vstack([EvlLKF(np.vstack([xx[i], yy[i]])) for i in range(xx.shape[0])])
    zmax = np.max(zz)
    zmin = np.min(zz) if zfloor is None else zmax - zfloor
    fig, ax = plt.subplots()
    cs = ax.contourf(xx, yy, zz, levels=np.linspace(zmin, zmax, 6), extend="min")
    fig.colorbar(cs, ax=ax)
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    plt.show()
    return fig
