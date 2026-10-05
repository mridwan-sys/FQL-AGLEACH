import numpy as np, fqlsim as S
NAMES={S.LEACH:"LEACH",S.FUZZY:"Fuzzy-LEACH",S.GRID:"Grid-LEACH",S.AGL:"AGLEACH",S.KOPTGRID:"kopt-Grid",S.FQL:"FQL-v2",S.FQL2:"FQL2",S.HEUR:"HEUR",S.FQL3:"FQL3"}
def run(proto,n,R,seed,P=None,qinit=None,dep='uniform'):
    xs,ys=S.make_nodes(n,seed,dep)
    P=S.pvec() if P is None else P
    out=S.simulate(proto,xs,ys,R,10000+seed,P,S.base_grid_size(n),np.zeros((0,2)) if qinit is None else qinit)
    keys=["alive","avgE","ch","deliv","ctrlE","totE","reass","jain","G","rew","div","chcnt","q","xd","xb","xe"]
    return dict(zip(keys,out))
def summ(res,n,R):
    a=res["alive"]
    def first(cond):
        idx=np.where(cond)[0]; return int(idx[0])+1 if len(idx) else R
    fnd=first(a<n); hnd=first(a<=n/2); lnd=first(a==0)
    h=max(hnd,1)
    cc=res["chcnt"]; jch=(cc.sum()**2)/(n*(cc**2).sum()) if cc.sum()>0 else 0
    return dict(FND=fnd,HND=hnd,LND=lnd,alive=a.mean(),nr=a.sum(),deliv=res["deliv"].sum(),
                ctrl=res["ctrlE"].sum()/max(res["totE"].sum(),1e-12),
                jain=res["jain"][:h].mean(),jch=jch,reass=res["reass"][1:h].mean(),
                ch=res["ch"][:h].mean(),div=res["div"][:h].mean())
