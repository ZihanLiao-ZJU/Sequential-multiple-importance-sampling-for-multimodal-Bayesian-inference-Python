"""Independent Python weight invariants, not recorded MATLAB weight parity.

Use the raw level samples and the soft/hard truncation definitions directly;
do not call the production ResRat/WgtMis_Blc/WgtPos implementations.
"""
import numpy as np
from scipy.special import logsumexp


def _levels(out, soft):
    y = [np.asarray(v).reshape(4, -1, order="F") for v in out["y"]]
    c = np.zeros(len(y) + 1)
    for i, yi in enumerate(y if soft else y[:-1]):
        ri, rmax = out["g"][i + 1]
        if np.isinf(ri):
            li = np.zeros(yi.shape[1])
        elif soft:
            li = np.minimum(yi[2] - rmax - ri, 0)
        else:
            li = np.where(yi[2] > rmax + ri, 0.0, -np.inf)
        lograt = li - yi[0]
        lograt -= np.max(lograt)
        c[i + 1] = c[i] + logsumexp(lograt) - np.log(lograt.size)
    return y, c


def semis_probabilities(out, mis):
    """Check MIS weights/evidence, then independently derive posterior probabilities."""
    y, c = _levels(out, soft=True)
    np.testing.assert_allclose(mis.C, c, rtol=1e-12, atol=1e-10)
    logn = np.log([v.shape[1] for v in y])
    combined, integrals = [], []
    for i, yi in enumerate(y):
        q = []
        for j, g in enumerate(out["g"][:-1]):
            li = np.zeros(yi.shape[1]) if np.isinf(g[0]) else np.minimum(yi[2] - g[1] - g[0], 0)
            q.append(li + yi[3] - c[j])
        q = np.asarray(q)
        wm = logn[i] + q[i] - logsumexp(logn[:, None] + q, axis=0)
        wp = yi[2] + yi[3] - yi[0] - yi[1] + c[i]
        np.testing.assert_allclose(np.asarray(mis.Wmis[i]).ravel(order="F"), wm, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(np.asarray(mis.Wpos[i]).ravel(order="F"), wp - mis.Z, rtol=1e-12, atol=1e-10)
        combined.append(wm + wp)
        integrals.append(logsumexp(wm + wp) - logn[i])
    np.testing.assert_allclose(mis.Z, logsumexp(integrals), rtol=1e-12, atol=1e-10)
    w = np.concatenate(combined)
    return np.exp(w - logsumexp(w))


def sus_probabilities(out):
    """Derive SuS posterior probabilities directly from raw level densities."""
    y, c = _levels(out, soft=False)
    weights = []
    for i, yi in enumerate(y):
        w = yi[2] + yi[3] - yi[0] - yi[1] + c[i]
        if i < len(y) - 1:
            threshold = np.sum(out["g"][i + 1])
            w = np.where(yi[2] < threshold, w, -np.inf)
        weights.append(w)
    w = np.concatenate(weights)
    return np.exp(w - logsumexp(w))
