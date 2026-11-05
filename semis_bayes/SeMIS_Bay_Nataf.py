from .SeMCS import state_property
import copy

import numpy as np
from scipy.optimize import minimize_scalar

from ._utils import flat
from .operations_on_logarithmic_values import logmean
from .support_functions.ResRat import ResRat
from .support_functions.logGauss import logGauss


def fminbnd(fun, a, b, options=None):
    """MATLAB fminbnd (golden section + parabolic interpolation, TolX = 1e-4).

    scipy's bounded method implements the same algorithm with the same
    stopping rule. Returns (x, fval).
    """
    res = minimize_scalar(fun, bounds=(a, b), method="bounded", options={"xatol": 1e-4, "maxiter": 500} if options is None else options)
    return res.x, res.fun


from .SeMCS import prescribed_state
from ._utils import mround
from .support_functions.DspRst import DspRst
from .support_functions.SedSlt import SedSlt

class SeMIS_Bay_Nataf:
    Nsam = state_property("Nsam",1000)
    """SeMIS Bayesian updating in Nataf-transformed (standard normal) space.

    Paper-aligned naming:
        r_i   (implementation) <-> lambda_i (theory)
        r_max (implementation) <-> r^{max}_{0:i-1} (running max of log-likelihood)

    EvlLKF() computes the SOFT TRUNCATION term
        logLi(theta) = log(min(L(theta) / exp(r_max + r_i), 1))
                     = min(logL(theta) - r_max - r_i, 0)
    The ratio beta_i(theta) in Eq. (24) is built in ResRat from two logLi terms.
    """

    def __init__(self, LKF):
        self.LKF = LKF
        self.Nfun = 4
        self.Ndim = LKF.Ndim
        self.Ncal = None
        self.X = None
        self.Y = None
        # intermediate parameters G = [r_i, r_max]
        self.G = {"parameters": np.array([-np.inf, -np.inf]), "lambda": 0.6, "L": np.eye(self.Ndim), "p": 0.1, "Nsam": 1000,
                  "completion_threshold": -1e-4,
                  "search": dict(subintervals=100, lower=-1e12, upper=0.0, xatol=1e-4, maxiter=500)}

    def EvlLKF(self):
        """Soft truncation term logLi = min(logL - r_max - r_i, 0)."""
        logL = self.Y[0, :]
        r_i, r_max = self.G["parameters"][0], self.G["parameters"][1]
        if np.isinf(r_i):
            return np.zeros_like(logL)
        return np.fmin(logL - r_max - r_i, 0)

    def EvlPDF(self):
        """Prior log-density logPi."""
        return self.Y[1, :]

    def UpdObj(self, y, p):
        """Update G = [r_i, r_max].

        r_i solves Eq. (25), logmean(beta_i) = log(p), by a 1D bounded search
        over 100 sub-intervals.
        """
        obj = copy.copy(self)
        logL = flat(y)[2, :]
        r_i = obj.G["parameters"][0]
        r_max = max(np.nanmax(logL), obj.G["parameters"][1])
        dr_max = r_max - obj.G["parameters"][1]

        maxi = int(obj.G["search"]["subintervals"])
        err_min = np.inf

        def fun(ri):
            return abs(logmean(ResRat(obj, y, np.array([ri, r_max]))[0], None) - np.log(p))

        rmin = max(obj.G["search"]["lower"], r_i - dr_max)
        rmax = obj.G["search"]["upper"]
        rbnd = (rmax - rmin) / maxi
        for i in range(1, maxi + 1):
            r_tmp, err = fminbnd(fun, rmin + (i - 1) * rbnd, rmin + i * rbnd,
                                    {k:obj.G["search"][k] for k in ("xatol","maxiter")})
            if err < err_min:
                err_min = err
                r_i = r_tmp

        G = np.array([r_i, r_max])
        obj.G = {**obj.G, "parameters": G}
        return obj.G, obj

    @property
    def FlgCvg(self):
        return bool(self.G["parameters"][0] >= self.G.get("completion_threshold", -1e-4))

    def EvlY(self, u):
        """Evaluate and cache the base terms obj.Y = [logL; logPi] at u (U-space)."""
        obj = copy.copy(self)
        x = obj.U2X(u)
        logL = np.asarray(obj.LKF.EvlLKF(x), dtype=float).ravel()
        obj.X = u
        obj.Y = np.vstack([logL, logGauss(u)])
        obj.Ncal = u.shape[1]
        return obj

    def UpdY(self, y):
        """Refresh and prepend the intermediate terms: y := [logLi; logPi; base terms]."""
        obj = copy.copy(self)
        y = flat(y)[2:, :]
        obj.Y = y
        return np.vstack([obj.EvlLKF(), obj.EvlPDF(), y])

    def U2X(self, u):
        """Map U-space samples to X-space via the Nataf transformation."""
        return self.LKF.Pdis.U2X(u)

    def run_defaults(self):
        return dict(Verbose=True)

    def start_run(self,owner,sampler):
        owner.IntBay.G=dict(owner.IntBay.G,p=owner.ConPrb)
        sampler=copy.copy(sampler); model=owner.IntBay
        counts=np.atleast_1d(owner.NumSam)
        limit=owner.MaxIte
        # Preserve the historical max-iteration and schedule conventions.
        ns=np.zeros(limit+1,dtype=int)
        if counts.size>1: ns[:counts.size]=counts
        else: ns[:]=counts[0]
        schedule=[copy.deepcopy(v) if isinstance(v,dict) else
                  dict(parameters=np.asarray(v).copy(), **{"lambda":0.6})
                  for v in owner.G]
        if len(schedule):
            model=copy.copy(model)
            model.G=prescribed_state(model.G,schedule[0],initial=True)
            sampler.BayStc=model
        return dict(owner=owner,model=model,sampler=sampler,limit=limit,
                    targets=ns,schedule=schedule,nc=int(ns[0]),steps=1,
                    xseed=None,yseed=None,x=[],y=[],g=[copy.deepcopy(model.G)],
                    Nsam=[],Nsze=[],Ncal=[],complete=False)

    def sample_level(self,c,level):
        if level==0:
            x,y,calls,model=c["sampler"].SamGen(c["nc"],c["steps"])
        else:
            x,y,calls,model=c["sampler"].SamGen(c["xseed"],c["yseed"],c["steps"])
        c["model"]=model
        record=dict(x=x,y=y,calls=calls,count=c["nc"]*c["steps"],shape=(c["nc"],c["steps"]))
        c["x"].append(x);c["y"].append(y);c["Ncal"].append(calls)
        c["Nsam"].append(record["count"]);c["Nsze"].append(record["shape"])
        return record

    def advance_level(self,c,r,level):
        model=c["model"]
        if level+1<len(c["schedule"]):
            model=copy.copy(model)
            model.G=prescribed_state(model.G,c["schedule"][level+1])
        else:
            _,model=model.UpdObj(r["y"],c["owner"].ConPrb)
        c["model"]=model;c["sampler"].BayStc=model
        c["g"].append(copy.deepcopy(model.G))

    def select_run_seeds(self,c,r,level):
        target=int(c["targets"][level+1])
        xs,ys=SedSlt(c["model"],r["x"],r["y"],target)
        c["xseed"],c["yseed"]=xs,ys
        c["nc"]=xs.shape[1]
        c["steps"]=mround(target/c["nc"]) if c["nc"] else np.inf
        if c["owner"].Verbose: DspRst(level+1,r["count"],np.asarray(r["shape"]),r["calls"])

    def run_complete(self,c,r,phase):
        if phase!="after_seeds": return False
        if c["nc"]<=0 or len(c["x"])+1==c["limit"]: return True
        c["complete"]=bool(c["model"].FlgCvg)
        return c["complete"]

    def finish_run(self,c):
        return dict(x=c["x"],y=c["y"],g=c["g"],Nsam=np.asarray(c["Nsam"]),
                    Nsze=np.asarray(c["Nsze"],dtype=int),Ncal=np.asarray(c["Ncal"]),
                    Nite=len(c["x"]),flg_cvg=c["complete"],intBay=c["model"])
