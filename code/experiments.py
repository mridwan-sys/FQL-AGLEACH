"""Final experiment harness for the FQL-AGLEACH revision (test seeds 100-129)."""
import numpy as np, json, time, pickle, sys, os
import fqlsim as S, runlib as L
from multiprocessing import Pool
from scipy.stats import wilcoxon

TEST = list(range(100, 130))
PROP = dict(grid_mode=3, w1=1.0, w5=1.0, kappa=5.0, gamma=0.99)   # FQL-AGLEACH defaults
SCEN = {"S1": dict(n=100, R=4000, dep="uniform", kw={}),
        "S2": dict(n=500, R=5000, dep="uniform", kw={}),
        "S3": dict(n=100, R=4000, dep="clustered", kw={}),
        "S4": dict(n=100, R=4000, dep="uniform", kw=dict(bsx=50.0, bsy=150.0))}
MAIN = [("LEACH", S.LEACH, {}), ("Fuzzy-LEACH", S.FUZZY, {}), ("Grid-LEACH", S.GRID, {}),
        ("AGLEACH", S.AGL, {}), ("FQL-AGLEACH", S.FQL2, PROP)]
ABL = [("AGLEACH (linear grid, max-E)", S.AGL, {}),
       ("+ k_opt grid [v2]", S.KOPTGRID, dict(grid_mode=2)),
       ("+ tile-aligned k_opt grid", S.KOPTGRID, dict(grid_mode=3)),
       ("+ FQL, learning frozen", S.FQL2, dict(PROP, alpha=0.0)),
       ("Hand-tuned phase switch", S.HEUR, dict(grid_mode=3, tau=0.3, l1=0, l2=1, l3=0)),
       ("FQL-AGLEACH (full)", S.FQL2, PROP)]

def one(args):
    proto, sc, kw, seed, keep = args
    c = SCEN[sc]; k = dict(c["kw"]); k.update(kw)
    t = time.perf_counter()
    res = L.run(proto, c["n"], c["R"], seed, S.pvec(**k), dep=c["dep"])
    dt = time.perf_counter() - t
    s = L.summ(res, c["n"], c["R"])
    a = res["alive"]; n = c["n"]
    idx = np.where(a <= 0.1 * n)[0]; s["T90"] = int(idx[0]) + 1 if len(idx) else c["R"]
    s["ms_round"] = 1000 * dt / max(s["LND"], 1)
    s = {kk: float(v) for kk, v in s.items()}
    ts = None
    if keep:
        ts = {kk: res[kk].astype(float) for kk in ("alive", "avgE", "ch", "deliv", "G", "rew", "div")}
        ts["q"] = res["q"]
    return s, ts

def run_set(sc, proto, kw, keep=False, seeds=TEST):
    with Pool(2) as p:
        out = p.map(one, [(proto, sc, kw, s, keep) for s in seeds])
    summ = {k: [o[0][k] for o in out] for k in out[0][0]}
    ts = [o[1] for o in out] if keep else None
    return summ, ts

def holm(ps):
    m = len(ps); order = np.argsort(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(run, 1.0)
    return adj

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    # warm-up compile
    for p in range(9): L.run(p, 50, 20, 0, S.pvec(grid_mode=3))
    what = sys.argv[1] if len(sys.argv) > 1 else "main"
    if what == "main":
        R = {}
        for sc in SCEN:
            for name, p, kw in MAIN:
                summ, ts = run_set(sc, p, kw, keep=True)
                mean_ts = {k: np.mean([t[k] for t in ts], axis=0) for k in ("alive", "avgE", "ch", "deliv", "G", "rew", "div")}
                R[(sc, name)] = dict(summ=summ, ts=mean_ts, q=[t["q"] for t in ts] if name == "FQL-AGLEACH" else None)
                print(sc, name, {k: round(np.mean(v), 3) for k, v in summ.items()}, flush=True)
        pickle.dump(R, open("results/main.pkl", "wb"))
    elif what == "abl":
        R = {}
        for sc in SCEN:
            for name, p, kw in ABL:
                summ, _ = run_set(sc, p, kw)
                R[(sc, name)] = summ
                print(sc, name, {k: round(np.mean(v), 3) for k, v in summ.items()}, flush=True)
        pickle.dump(R, open("results/abl.pkl", "wb"))
    elif what == "sens":
        grid = {"alpha": [0.05, 0.1, 0.2], "gamma": [0.9, 0.95, 0.99], "eps0": [0.05, 0.15, 0.3],
                "kappa": [0.0, 5.0, 20.0], "w5": [0.0, 1.0, 5.0], "pct": [50.0, 75.0, 90.0],
                "nmf": [3.0, 5.0], "C": [1.0, 1.5, 2.0], "eta_ref": [1000.0, 2000.0, 4000.0],
                "dens_r": [0.3, 0.5, 0.7]}
        R = {}
        for sc in ["S1", "S2"]:
            for par, vals in grid.items():
                for v in vals:
                    kw = dict(PROP); kw[par] = v
                    summ, _ = run_set(sc, S.FQL2, kw)
                    R[(sc, par, v)] = summ
                    print(sc, par, v, {k: round(np.mean(summ[k]), 1) for k in ("FND", "HND", "LND", "T90", "deliv")}, flush=True)
        pickle.dump(R, open("results/sens.pkl", "wb"))
