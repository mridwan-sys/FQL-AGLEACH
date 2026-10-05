import numpy as np, pickle, fqlsim as S, runlib as L
from multiprocessing import Pool
from experiments import PROP, TEST, SCEN
def job(a):
    sc, kw, s = a
    c = SCEN[sc]; k = dict(c["kw"]); k.update(kw)
    r = L.run(S.FQL2, c["n"], c["R"], s, S.pvec(**k), dep=c["dep"])
    return r["xd"], r["xb"], r["xe"], r["alive"]
out = {}
for sc in ["S1", "S2"]:
    for nm, kw in [("full", PROP), ("frozen", dict(PROP, alpha=0.0))]:
        with Pool(2) as p: res = p.map(job, [(sc, kw, s) for s in TEST])
        out[(sc, nm)] = [np.array(z) for z in zip(*res)]
        print(sc, nm, flush=True)
pickle.dump(out, open("results/behaviour.pkl", "wb"))
