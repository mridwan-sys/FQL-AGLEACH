import pickle, numpy as np
from experiments import PROP
import fqlsim as S
R = pickle.load(open("results/sens.pkl", "rb"))
D = dict(S.DEFAULT); D.update(PROP)
NAME = {"alpha": "$\\alpha$", "gamma": "$\\gamma$", "eps0": "$\\epsilon_0$", "kappa": "$\\kappa$", "w5": "$w_D$",
        "pct": "Eligibility percentile $P$", "nmf": "MFs per input $n_{MF}$", "C": "Grid constant $C$",
        "eta_ref": "$\\eta_{ref}$ (readings/J)", "dens_r": "Density radius $r_\\rho/G$"}
rows = []
pars = []
for (sc, par, v) in R:
    if par not in pars: pars.append(par)
for par in pars:
    vals = sorted({v for (sc, p, v) in R if p == par})
    for i, v in enumerate(vals):
        cells = []
        for sc in ["S1", "S2"]:
            s = R[(sc, par, v)]
            cells += [f"{np.mean(s['FND']):.0f}", f"{np.mean(s['HND']):.0f}", f"{np.mean(s['LND']):.0f}", f"{np.mean(s['deliv'])/1e3:.1f}"]
        vs = f"{v:g}"
        if abs(v - D[par]) < 1e-9:
            vs = "\\textbf{" + vs + "}"
        rows.append((NAME[par] if i == 0 else "") + " & " + vs + " & " + " & ".join(cells) + "\\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
open("results/tab_sens.tex", "w").write("\n".join(rows))
print("\n".join(rows))
# ranges for text
for sc in ["S1","S2"]:
    for k in ["HND","LND","deliv","FND"]:
        vals=[np.mean(R[(sc,p,v)][k]) for (s,p,v) in R if s==sc and p not in ("C",)]
        print(sc,k,min(vals),max(vals))
