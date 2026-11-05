"""Method policies for native Bayesian algorithms; no orchestration loop."""
import copy
import numpy as np
from semcs_core import Method
from semcs_core.engine import prescribed_state
from ._utils import mround
from .support_functions.DspRst import DspRst
from .support_functions.SedSlt import SedSlt

class BayesMethod(Method):
    def start(self,owner,rng):
        sampler=copy.copy(owner.Sampler); model=owner.IntBay
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
    def sample(self,c,level):
        if level==0:
            x,y,calls,model=c["sampler"].SamGen(c["nc"],c["steps"])
        else:
            x,y,calls,model=c["sampler"].SamGen(c["xseed"],c["yseed"],c["steps"])
        c["model"]=model
        record=dict(x=x,y=y,calls=calls,count=c["nc"]*c["steps"],shape=(c["nc"],c["steps"]))
        c["x"].append(x);c["y"].append(y);c["Ncal"].append(calls)
        c["Nsam"].append(record["count"]);c["Nsze"].append(record["shape"])
        return record
    def advance(self,c,r,level):
        model=c["model"]
        if level+1<len(c["schedule"]):
            model=copy.copy(model)
            model.G=prescribed_state(model.G,c["schedule"][level+1])
        else:
            _,model=model.UpdObj(r["y"],c["owner"].ConPrb)
        c["model"]=model;c["sampler"].BayStc=model
        c["g"].append(copy.deepcopy(model.G))
    def select_seeds(self,c,r,level):
        target=int(c["targets"][level+1])
        xs,ys=SedSlt(c["model"],r["x"],r["y"],target)
        c["xseed"],c["yseed"]=xs,ys
        c["nc"]=xs.shape[1]
        c["steps"]=mround(target/c["nc"]) if c["nc"] else np.inf
        if c["owner"].Verbose: DspRst(level+1,r["count"],np.asarray(r["shape"]),r["calls"])
    def is_complete(self,c,r,phase):
        if phase!="after_seeds": return False
        if c["nc"]<=0 or len(c["x"])+1==c["limit"]: return True
        c["complete"]=bool(c["model"].FlgCvg)
        return c["complete"]
    def finish(self,c):
        return dict(x=c["x"],y=c["y"],g=c["g"],Nsam=np.asarray(c["Nsam"]),
                    Nsze=np.asarray(c["Nsze"],dtype=int),Ncal=np.asarray(c["Ncal"]),
                    Nite=len(c["x"]),flg_cvg=c["complete"],intBay=c["model"])


"""Legacy entry point backed by the one semcs_core.Engine."""
from semcs_core import LegacyFacade

class SeMCS(LegacyFacade):
    default_verbose=True
    def __init__(self, Sampler):
        super().__init__(Sampler, BayesMethod())
