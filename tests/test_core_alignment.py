"""Read-only audit checks. MATLAB formulas are translated references, not MATLAB runtime."""
import sys
import pathlib
import unittest
import numpy as np
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import semis_bayes as sb
from semis_bayes import _utils
from semis_bayes.support_functions.ResRat import ResRat
from semis_bayes.support_functions.logGauss import logGauss


class Model:
    Ndim=2
    Pdis=sb.Nataf([sb.makedist('Normal')]*2,np.eye(2))
    def EvlLKF(self,x): return -np.sum(x*x,axis=0)


class AlignmentChecks(unittest.TestCase):
    def test_matlab_round(self):
        np.testing.assert_array_equal(_utils.mround([-2.5,-1.5,-.5,0,.5,1.5,2.5]),[-3,-2,-1,0,1,2,3])

    def test_column_major(self):
        x=np.arange(24).reshape(2,3,4)
        want=np.column_stack([x[:,j,k] for k in range(4) for j in range(3)])
        np.testing.assert_array_equal(_utils.flat(x),want)

    def test_gauss(self):
        x=np.array([[-2,0,1],[1,2,-1.]])
        cov=np.array([[2,.3],[.3,1.]])
        np.testing.assert_allclose(logGauss(x,[.5,-.5],cov),stats.multivariate_normal([.5,-.5],cov).logpdf(x.T))

    def test_eggbox_formula(self):
        x=np.array([[0,1,np.pi,10*np.pi],[2,3,4,5.]])
        np.testing.assert_allclose(sb.LKF_2D_Eggbox().EvlLKF(x),[(2+np.cos(x[0,k]/2)*np.cos(x[1,k]/2))**5 for k in range(4)])

    def test_shells_formula(self):
        for n in [2,5,10,20]:
            x=np.linspace(-5,5,n*7).reshape(n,7)
            expected=[]
            for j in range(7):
                c1=np.zeros(n); c1[0]=3.5
                c2=-c1
                a=-(np.linalg.norm(x[:,j]-c1)-2)**2/2/.1**2
                b=-(np.linalg.norm(x[:,j]-c2)-2)**2/2/.1**2
                expected.append(-np.log(np.sqrt(2*np.pi*.1**2))+max(a,b)+np.log(np.exp(a-max(a,b))+np.exp(b-max(a,b))))
            np.testing.assert_allclose(sb.LKF_nD_GaussianShells(n).EvlLKF(x),expected,rtol=1e-13)

    def test_loggamma_formula(self):
        for n in [2,5,10,20]:
            x=np.linspace(-15,15,n*7).reshape(n,7)
            expected=[]
            for j in range(7):
                a=x[0,j]-10-np.exp(x[0,j]-10)
                b=x[0,j]+10-np.exp(x[0,j]+10)
                c=-.5*(x[1,j]-10)**2-np.log(np.sqrt(2*np.pi))
                d=-.5*(x[1,j]+10)**2-np.log(np.sqrt(2*np.pi))
                v=np.log(.5)+max(a,b)+np.log(np.exp(a-max(a,b))+np.exp(b-max(a,b)))
                v+=np.log(.5)+max(c,d)+np.log(np.exp(c-max(c,d))+np.exp(d-max(c,d)))
                for i in range(3,n+1):
                    z=x[i-1,j]
                    v += z-10-np.exp(z-10) if i<=(n+2)/2 else -.5*(z-10)**2-np.log(np.sqrt(2*np.pi))
                expected.append(v)
            np.testing.assert_allclose(sb.LKF_nD_GaussianLogGamma(n).EvlLKF(x),expected,rtol=1e-13)

    def test_normal_nataf(self):
        R=np.array([[1,-.3],[-.3,1.]])
        obj=sb.Nataf([sb.makedist('Normal',mu=1,sigma=2),sb.makedist('Normal',mu=-1,sigma=.5)],R)
        u=np.array([[1,2,-1.],[0,-2,2.]])
        z=np.linalg.cholesky(R)@u
        x=np.array([1+2*z[0],-1+.5*z[1]])
        np.testing.assert_allclose(obj.U2X(u),x)
        np.testing.assert_allclose(obj.X2U(x),u,atol=1e-13)

    def test_lognormal_nataf(self):
        R=np.array([[1,.25],[.25,1.]])
        d=[sb.makedist('Lognormal',mu=.2,sigma=.4),sb.makedist('Lognormal',mu=-.1,sigma=.6)]
        obj=sb.Nataf(d,R)
        cv=[np.sqrt(np.expm1(.4**2)),np.sqrt(np.expm1(.6**2))]
        rho=np.log1p(.25*cv[0]*cv[1])/(.4*.6)
        np.testing.assert_allclose(obj.Rho_U,[[1,rho],[rho,1]])
        u=np.array([[1.,-1],[2,-2]])
        z=np.linalg.cholesky(obj.Rho_U)@u
        np.testing.assert_allclose(obj.U2X(u),np.exp(np.array([[.2],[-.1]])+np.array([[.4],[.6]])*z))

    def test_soft_and_hard_truncation(self):
        for cls in [sb.SeMIS_Bay_Nataf,sb.SuS_Bay_Nataf]:
            obj=cls(Model());obj.Y=np.array([[-5,-3,-1],[0,0,0.]])
            np.testing.assert_array_equal(obj.EvlLKF(),[0,0,0])
            obj.G=np.array([-3.,0])
            want=np.array([-2,0,0]) if cls==sb.SeMIS_Bay_Nataf else np.array([-np.inf,-np.inf,0])
            np.testing.assert_array_equal(obj.EvlLKF(),want)

    def test_sus_quantile_and_convergence(self):
        obj=sb.SuS_Bay_Nataf(Model()); y=np.zeros((4,10));y[2]=np.arange(10)
        g,o=obj.UpdObj(y,.2)
        np.testing.assert_array_equal(g,[-1.5,9]);self.assertFalse(o.FlgCvg)
        o.G=np.array([-.1,9]);self.assertTrue(o.FlgCvg)
        s=sb.SeMIS_Bay_Nataf(Model());s.G=np.array([-1e-4,9]);self.assertTrue(s.FlgCvg)

    def test_mis_deterministic_density_reference(self):
        obj=sb.SeMIS_Bay_Nataf(Model())
        gs=[np.array([-np.inf,-np.inf]),np.array([-2.,0]),np.array([-5e-5,0.])]
        xs=[np.linspace(-1,1,12).reshape(2,3,2),np.linspace(-1.5,1.5,12).reshape(2,2,3)]
        ys=[]
        for x,g in zip(xs,gs):
            xf=_utils.flat(x);ll=Model().EvlLKF(xf);lp=logGauss(xf)
            li=np.zeros(6) if np.isinf(g[0]) else np.minimum(ll-g.sum(),0)
            ys.append(np.vstack([li,lp,ll,lp]).reshape((4,)+x.shape[1:],order='F'))
        inp=dict(x=xs,y=ys,g=gs,Nite=2,Nsam=np.array([6,6]),Nsze=np.array([[3,2],[2,3]]),intBay=obj)
        got=sb.MIS(inp)
        y=[_utils.flat(v).copy() for v in ys];cs=np.zeros(3)
        for i in range(2):
            rat=np.minimum(y[i][2]-gs[i+1].sum(),0)-y[i][0]
            rat-=rat.max(); cs[i+1]=cs[i]+np.log(np.exp(rat).mean())
            if i<1:y[i+1][0]-=cs[i+1]
        np.testing.assert_allclose(got.C,cs)
        zi=[]
        for i,yi in enumerate(y):
            qs=[]
            for j in range(2):
                li=np.zeros(6) if j==0 else np.minimum(yi[2]-gs[j].sum(),0)
                qs.append(np.exp(li+yi[1]-cs[j]))
            wm=qs[i]/np.sum(qs,axis=0)
            wp=np.exp(yi[2]+yi[3]-yi[0]-yi[1])
            np.testing.assert_allclose(np.exp(got.Wmis[i]).ravel(order='F'),wm)
            zi.append(np.mean(wm*wp))
        np.testing.assert_allclose(got.Z,np.log(sum(zi)))
        for i,yi in enumerate(y):
            np.testing.assert_allclose(got.Wpos[i].ravel(order='F'),yi[2]+yi[3]-yi[0]-yi[1]-got.Z)

    def test_acs_with_literal_matlab_loop(self):
        obj=sb.SeMIS_Bay_Nataf(Model());obj.G=np.array([-1.,0])
        seeds=np.linspace(-1.5,1.5,14).reshape(2,7);e=obj.EvlY(seeds)
        sy=np.vstack([e.EvlLKF(),e.EvlPDF(),e.Y]);nc,ns=7,4
        rg=np.random.default_rng(81);du=rg.normal(size=(2,nc,ns));p=rg.uniform(size=(1,nc,ns))
        class Tape:
            def randn(self,*shape):assert shape==du.shape;return du.copy()
            def rand(self,*shape):assert shape==p.shape;return p.copy()
        _utils.set_rng(Tape())
        try: got=sb.aCS(obj).SamGen(seeds,sy,ns)
        finally:_utils.set_rng(None)
        u=np.zeros((2,nc,ns));y=np.zeros((4,nc,ns));seed=seeds.copy();yy=sy.copy()
        sig0=np.maximum(np.std(seed,axis=1,ddof=1,keepdims=True),np.finfo(float).eps);lam=.6;cost=0
        for it in range(1,6):
            idx=np.arange(min(1+(it-1)*2,nc)-1,min(2+(it-1)*2,nc))
            sig=np.minimum(lam*sig0,1);rho=np.sqrt(1-sig**2);acc=0;ncal=0
            for k in range(ns):
                proposal=rho*seed[:,idx]+sig*du[:,idx,k]
                ll=-np.sum(proposal**2,axis=0);lp=-np.log(2*np.pi)-.5*np.sum(proposal**2,axis=0)
                val=np.vstack([np.minimum(ll+1,0),lp,ll,lp])
                reject=np.exp(val[0]-yy[0,idx])<p[0,idx,k]
                proposal[:,reject]=seed[:,idx[reject]];val[:,reject]=yy[:,idx[reject]]
                u[:,idx,k]=proposal;y[:,idx,k]=val
                seed[:,idx]=proposal;yy[:,idx]=val;acc+=len(idx)-reject.sum();ncal+=len(idx)
            lam=np.exp(np.log(lam)+(acc/ncal-.44)/np.sqrt(it));cost+=ncal
        np.testing.assert_allclose(got[0],u,atol=1e-14);np.testing.assert_allclose(got[1],y,atol=1e-14);self.assertEqual(got[2],cost)
