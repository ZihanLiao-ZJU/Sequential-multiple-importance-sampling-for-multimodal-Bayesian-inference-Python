"""Replay Octave draws/selected indices; legacy fixtures do not record weights.

Index replay alone cannot establish cross-language resampling-weight parity.
``check_last_probabilities`` checks a separate Python density invariant instead.
"""
import numpy as np
from scipy.io import loadmat

KIND = {1: "randn", 2: "rand", 3: "randperm", 4: "randi", 5: "randsample"}


class ReplayRNG:
    def __init__(self, fname):
        d = loadmat(fname)
        kinds = d["kinds"].ravel().astype(int)
        lens = d["lens"].ravel().astype(int)
        vals = d["vals"].ravel()
        offs = np.concatenate([[0], np.cumsum(lens)])
        self.entries = [(KIND[k], vals[offs[i]:offs[i + 1]]) for i, k in enumerate(kinds)]
        self.pos = 0
        self.last_probabilities = None

    def _pop(self, kind, n=None):
        if self.pos >= len(self.entries):
            raise RuntimeError(f"replay stream exhausted at {self.pos}: expected {kind}({n})")
        k, v = self.entries[self.pos]
        if k != kind or (n is not None and v.size != n):
            raise RuntimeError(f"replay mismatch at {self.pos}: expected {kind}({n}), recorded {k}({v.size})")
        self.pos += 1
        return v

    def randn(self, *shape):
        return self._pop("randn", int(np.prod(shape))).reshape(shape, order="F")

    def rand(self, *shape):
        return self._pop("rand", int(np.prod(shape))).reshape(shape, order="F")

    def randperm(self, n, k):
        return self._pop("randperm", int(k)).astype(int) - 1

    def randi(self, n, k):
        return self._pop("randi", int(k)).astype(int) - 1

    def randsample(self, n, k, w):
        w = np.asarray(w, dtype=float).ravel(order="F")
        if w.size != n or not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("invalid resampling probabilities or population size")
        scaled = w / np.max(w)
        self.last_probabilities = scaled / np.sum(scaled)
        recorded = self._pop("randsample", int(k))
        if not np.all(np.isfinite(recorded)) or np.any(recorded != np.floor(recorded)):
            raise RuntimeError("recorded resampling indices must be finite integers")
        indices = recorded.astype(int) - 1
        if np.any(indices < 0) or np.any(indices >= n):
            raise RuntimeError("recorded resampling index is outside the supplied population")
        if np.any(w[indices] == 0):
            raise RuntimeError("recorded resampling index has zero supplied probability")
        return indices

    def check_last_probabilities(self, expected):
        """Compare actual input probabilities to an independent Python reference."""
        if self.last_probabilities is None:
            raise AssertionError("no resampling call was observed")
        np.testing.assert_allclose(self.last_probabilities, expected, rtol=1e-10, atol=1e-14)

    @property
    def done(self):
        return self.pos == len(self.entries)
