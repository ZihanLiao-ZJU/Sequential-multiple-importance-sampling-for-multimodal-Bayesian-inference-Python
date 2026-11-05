from .SeMCS import state_property
import copy

import numpy as np

from ._utils import flat, mround, randi, randperm, randsample
from .operations_on_logarithmic_values import logmean, logminus, logsum
from .support_functions.ResRat import ResRat
from .support_functions.VabCmb import VabCmb
from .support_functions.logGauss import logGauss


def _mmin(a, b):
    """MATLAB min of two arrays (complex values are compared by magnitude)."""
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        return np.where(np.abs(b) < np.abs(a), b, a)
    return np.fmin(a, b)


from .SeMCS import prescribed_state
from ._utils import mround
from .support_functions.DspRst import DspRst
from .support_functions.SedSlt import SedSlt

class SuS_Bay_Nataf:
    Nsam = state_property("Nsam",1000)
    """Subset simulation (SuS) with Nataf transformation for Bayesian updating.

    Naming convention (consistent with SeMIS_Bay_Nataf):
        logLi : log-domain HARD truncation term (indicator)
        logPi : log prior density
        G = [r_i, r_max]
            r_i   : adaptive threshold (level parameter)
            r_max : running maximum of the log-likelihood
    """

    def __init__(self, LKF):
        self.LKF = LKF
        self.Nfun = 4
        self.Ndim = LKF.Ndim
        self.Ncal = None
        self.X = None
        self.Y = None
        self.G = {"parameters": np.array([-np.inf, -np.inf]), "lambda": 0.6, "L": np.eye(self.Ndim), "p": 0.1, "Nsam": 1000,
                  "completion_threshold": -1e-1}

    def EvlLKF(self):
        """HARD truncation term: logLi = log(I(logL > r_max + r_i))."""
        logL = self.Y[0, :]
        r_i, r_max = self.G["parameters"][0], self.G["parameters"][1]
        if np.isinf(r_i):
            return np.zeros_like(logL)
        with np.errstate(divide="ignore"):
            return np.log((logL > r_max + r_i).astype(float))

    def EvlPDF(self):
        """Prior log-density logPi."""
        return self.Y[1, :]

    def UpdObj(self, y, p):
        """Update G = [r_i, r_max] with the standard SuS quantile rule."""
        obj = copy.copy(self)
        logL = flat(y)[2, :]
        Nsam = logL.size
        l_sorted = np.sort(logL)[::-1]
        Nsed = int(mround(Nsam * p))
        r_max = max(np.nanmax(logL), obj.G["parameters"][1])
        fth = (l_sorted[Nsed - 1] + l_sorted[Nsed]) / 2 if Nsed > 0 else l_sorted[0]
        G = np.array([fth - r_max, r_max])
        obj.G = {**obj.G, "parameters": G}
        return obj.G, obj

    def SltSed(self, u, y, Nsam):
        """Select seeds satisfying the hard truncation condition."""
        u = flat(u)[:self.Ndim, :]
        y = flat(y)[:self.Nfun, :]
        logL = y[2, :]
        Lth = self.G["parameters"][1] + self.G["parameters"][0]
        Ns = mround(Nsam / np.arange(1, Nsam + 1))
        Nsed = mround(Nsam / Ns)
        ind_sed = logL > Lth
        u_can = u[:, ind_sed]
        y_can = y[:, ind_sed]
        Ncan = int(np.sum(ind_sed))
        ok = np.nonzero(Ncan - Nsed >= 0)[0]
        Nsed = int(Nsed[ok[-1]]) if ok.size else 0
        if Nsed > 0 and Ncan >= Nsed:
            randind = randperm(Ncan, Nsed)
        elif Nsed > 0 and Ncan > 0:
            randind = randi(Ncan, Nsed)
        else:
            randind = np.array([], dtype=int)
        u_sed = u_can[:, randind]
        y_sed = y_can[:, randind]
        if y_sed.size:
            y_sed = self.UpdY(y_sed)
        return u_sed, y_sed

    @property
    def FlgCvg(self):
        return bool(self.G["parameters"][0] >= self.G.get("completion_threshold", -1e-1))

    def BayInf(self, inp):
        """Evidence estimation and posterior resampling from the SuS levels."""
        x = inp["x"]
        y = [np.array(v, copy=True) for v in inp["y"]]
        g = inp["g"]
        Nite = inp["Nite"]

        # ---- normalize weights across levels ----
        w_res = [None] * Nite
        c = np.zeros(Nite)
        for ite in range(Nite - 1):
            w_res[ite], _ = ResRat(self, y[ite], g[ite + 1])
            c[ite + 1] = c[ite] + logmean(w_res[ite], None)
            y[ite + 1][0] = y[ite + 1][0] - c[ite + 1]

        # ---- evidence estimation ----
        zi = np.zeros(Nite)
        for ite in range(Nite):
            yi = flat(y[ite])
            Li = g[ite]["parameters"][0] + g[ite]["parameters"][1]
            if ite < Nite - 1:
                Li1 = g[ite + 1]["parameters"][0] + g[ite + 1]["parameters"][1]
                w_z = _mmin(logminus(yi[2, :], Li), logminus(Li1, Li))
            else:
                w_z = logminus(yi[2, :], Li)
            zi[ite] = np.real(logmean(w_z, None) + c[ite])
        z = float(np.real(logsum(zi, None)))

        # ---- posterior resampling ----
        w_pos = []
        for ite in range(Nite):
            yi = flat(y[ite])
            with np.errstate(divide="ignore", invalid="ignore"):
                w_tmp = yi[2, :] + yi[3, :] - yi[0, :] - yi[1, :]
                if ite < Nite - 1:
                    w_tmp = w_tmp + np.log((yi[2, :] < g[ite + 1]["parameters"][0] + g[ite + 1]["parameters"][1]).astype(float))
            w_tmp = np.where(np.isnan(w_tmp), -np.inf, w_tmp)
            w_pos.append(w_tmp)

        x_all, y_all = VabCmb(x, y)
        w_pos = np.concatenate(w_pos)
        w_pos = np.exp(w_pos - np.max(w_pos))
        Npos = int(mround(np.sum(w_pos) ** 2 / np.sum(w_pos ** 2)))
        ind_pos = randsample(y_all.shape[1], Npos, w_pos)
        return dict(z=z, x_pos=x_all[:, ind_pos], y_pos=y_all[:, ind_pos])

    def EvlY(self, u):
        """Evaluate the base terms obj.Y = [logL; logPi]."""
        obj = copy.copy(self)
        x = obj.U2X(u)
        logL = np.asarray(obj.LKF.EvlLKF(x), dtype=float).ravel()
        obj.X = u
        obj.Y = np.vstack([logL, logGauss(u)])
        obj.Ncal = u.shape[1]
        return obj

    def UpdY(self, y):
        """Prepend the intermediate terms: y := [logLi; logPi; base terms]."""
        obj = copy.copy(self)
        y = flat(y)[2:, :]
        obj.Y = y
        return np.vstack([obj.EvlLKF(), obj.EvlPDF(), y])

    def U2X(self, u):
        """Nataf transformation from U-space to X-space."""
        return self.LKF.Pdis.U2X(u)

    def run_defaults(self):
        return dict(Verbose=True)

    def start_run(self, owner, sampler):
        owner.IntBay.G = dict(owner.IntBay.G, p=owner.ConPrb)
        sampler = copy.copy(sampler)
        model = owner.IntBay
        counts = np.atleast_1d(owner.NumSam)
        limit = owner.MaxIte
        # Preserve the historical max-iteration and schedule conventions.
        ns = np.zeros(limit + 1, dtype=int)
        if counts.size > 1:
            ns[: counts.size] = counts
        else:
            ns[:] = counts[0]
        schedule = [
            (
                copy.deepcopy(v)
                if isinstance(v, dict)
                else dict(parameters=np.asarray(v).copy(), **{"lambda": 0.6})
            )
            for v in owner.G
        ]
        if len(schedule):
            model = copy.copy(model)
            model.G = prescribed_state(model.G, schedule[0], initial=True)
            sampler.BayStc = model
        return dict(
            owner=owner,
            model=model,
            sampler=sampler,
            limit=limit,
            targets=ns,
            schedule=schedule,
            nc=int(ns[0]),
            steps=1,
            xseed=None,
            yseed=None,
            x=[],
            y=[],
            g=[copy.deepcopy(model.G)],
            Nsam=[],
            Nsze=[],
            Ncal=[],
            complete=False,
        )

    def sample_level(self, context, level):
        if level == 0:
            x, y, calls, model = context["sampler"].SamGen(context["nc"], context["steps"])
        else:
            x, y, calls, model = context["sampler"].SamGen(
                context["xseed"], context["yseed"], context["steps"]
            )
        context["model"] = model
        record = dict(
            x=x,
            y=y,
            calls=calls,
            count=context["nc"] * context["steps"],
            shape=(context["nc"], context["steps"]),
        )
        context["x"].append(x)
        context["y"].append(y)
        context["Ncal"].append(calls)
        context["Nsam"].append(record["count"])
        context["Nsze"].append(record["shape"])
        return record

    def advance_level(self, context, record, level):
        model = context["model"]
        if level + 1 < len(context["schedule"]):
            model = copy.copy(model)
            model.G = prescribed_state(model.G, context["schedule"][level + 1])
        else:
            _, model = model.UpdObj(record["y"], context["owner"].ConPrb)
        context["model"] = model
        context["sampler"].BayStc = model
        context["g"].append(copy.deepcopy(model.G))

    def select_run_seeds(self, context, record, level):
        target = int(context["targets"][level + 1])
        xs, ys = SedSlt(context["model"], record["x"], record["y"], target)
        context["xseed"], context["yseed"] = xs, ys
        context["nc"] = xs.shape[1]
        context["steps"] = mround(target / context["nc"]) if context["nc"] else np.inf
        if context["owner"].Verbose:
            DspRst(level + 1, record["count"], np.asarray(record["shape"]), record["calls"])

    def run_complete(self, context, record, phase):
        if phase != "after_seeds":
            return False
        if context["nc"] <= 0 or len(context["x"]) + 1 == context["limit"]:
            return True
        context["complete"] = bool(context["model"].FlgCvg)
        return context["complete"]

    def finish_run(self, context):
        return dict(
            x=context["x"],
            y=context["y"],
            g=context["g"],
            Nsam=np.asarray(context["Nsam"]),
            Nsze=np.asarray(context["Nsze"], dtype=int),
            Ncal=np.asarray(context["Ncal"]),
            Nite=len(context["x"]),
            flg_cvg=context["complete"],
            intBay=context["model"],
        )
