import pickle, numpy as np, json
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from experiments import SCEN, MAIN, ABL, holm
M = pickle.load(open("results/main.pkl", "rb"))
A = pickle.load(open("results/abl.pkl", "rb"))
PN = [m[0] for m in MAIN]
BASE = PN[:-1]
MET = [("FND", "FND", 0), ("HND", "HND", 0), ("LND", "LND", 0), ("deliv", "Delivered ($\\times10^3$)", 1), ("nr", "Node-rounds", 1)]
out = {}

def rb(x, y):  # matched-pairs rank-biserial
    d = np.array(x) - np.array(y); d = d[d != 0]
    if len(d) == 0: return 0.0
    from scipy.stats import rankdata
    r = rankdata(np.abs(d)); return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()

stats = {}
for sc in SCEN:
    tests = []
    for b in BASE:
        for k in ["FND", "HND", "T90", "LND", "deliv"]:
            x = M[(sc, "FQL-AGLEACH")]["summ"][k]; y = M[(sc, b)]["summ"][k]
            try: p = wilcoxon(x, y).pvalue
            except ValueError: p = 1.0
            tests.append((b, k, p, rb(x, y), np.mean(x), np.mean(y)))
    adj = holm(np.array([t[2] for t in tests]))
    for t, pa in zip(tests, adj):
        stats[(sc, t[0], t[1])] = dict(p=t[2], padj=pa, r=t[3], fql=t[4], base=t[5], pct=100 * (t[4] - t[5]) / t[5])
# ablation stats: full vs others
astats = {}
ANAMES = [a[0] for a in ABL]
for sc in SCEN:
    tests = []
    for nm in ANAMES[:-1]:
        for k in ["FND", "HND", "LND", "T90", "deliv"]:
            x = A[(sc, ANAMES[-1])][k]; y = A[(sc, nm)][k]
            try: p = wilcoxon(x, y).pvalue
            except ValueError: p = 1.0
            tests.append((nm, k, p, rb(x, y)))
    adj = holm(np.array([t[2] for t in tests]))
    for t, pa in zip(tests, adj):
        astats[(sc, t[0], t[1])] = dict(p=t[2], padj=pa, r=t[3])
pickle.dump((stats, astats), open("results/stats.pkl", "wb"))

def star(p):
    return "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else "$^{\\mathrm{ns}}$"

SCN = {"S1": "S1: 100 nodes, uniform, BS at centre", "S2": "S2: 500 nodes, uniform, BS at centre",
       "S3": "S3: 100 nodes, clustered, BS at centre", "S4": "S4: 100 nodes, uniform, BS outside (50,150)"}
# ---------- main table (per scenario)
rows = []
for sc in SCEN:
    rows.append("\\multicolumn{9}{l}{\\textit{%s}}\\\\" % SCN[sc])
    for p in PN:
        s = M[(sc, p)]["summ"]
        def f(k, dec=0, scale=1):
            v = np.array(s[k]) * scale
            txt = f"{v.mean():.{dec}f}$\\pm${v.std(ddof=1):.{dec}f}"
            return txt
        cells = [f("FND"), f("HND"), f("T90"), f("LND"), f("deliv", 1, 1e-3),
                 f"{100*np.mean(s['ctrl']):.1f}", f"{np.mean(s['jain']):.3f}", f"{np.mean(s['ch']):.1f}"]
        if p == "FQL-AGLEACH":
            # bold best of FND/HND/LND/deliv/alive among protocols
            pass
        rows.append(f"{p} & " + " & ".join(cells) + "\\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
open("results/tab_main.tex", "w").write("\n".join(rows))

# ---------- significance table: % change vs each baseline with Holm stars
rows = []
for sc in SCEN:
    for b in BASE:
        cells = []
        for k in ["FND", "HND", "T90", "LND", "deliv"]:
            st = stats[(sc, b, k)]
            cells.append(f"{st['pct']:+.1f}{star(st['padj'])}")
        rows.append(f"{sc} & {b} & " + " & ".join(cells) + "\\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
open("results/tab_sig.tex", "w").write("\n".join(rows))

# ---------- ablation table
rows = []
for sc in SCEN:
    ref = A[(sc, ANAMES[0])]
    rows.append("\\multicolumn{7}{l}{\\textit{%s}}\\\\" % SCN[sc])
    for nm in ANAMES:
        s = A[(sc, nm)]
        cells = []
        for k in ["FND", "HND", "LND", "T90"]:
            v = np.mean(s[k]); cells.append(f"{v:.0f}")
        cells.append(f"{np.mean(s['deliv'])/1e3:.1f}")
        mark = ""
        if nm != ANAMES[-1]:
            st = astats[(sc, nm, "LND")]
            mark = star(st["padj"])
        rows.append(f"{nm} & " + " & ".join(cells) + f" & {mark}\\\\")
    rows.append("\\midrule")
rows[-1] = "\\bottomrule"
open("results/tab_abl.tex", "w").write("\n".join(rows))

# ---------- numbers for text
num = {}
for (sc, b, k), st in stats.items():
    num[f"{sc}|{b}|{k}"] = st
for (sc, nm, k), st in astats.items():
    num[f"ABL|{sc}|{nm}|{k}"] = st
for (sc, p), v in M.items():
    for k, arr in v["summ"].items():
        num[f"MEAN|{sc}|{p}|{k}"] = float(np.mean(arr))
for (sc, nm), v in A.items():
    for k, arr in v.items():
        num[f"AMEAN|{sc}|{nm}|{k}"] = float(np.mean(arr))
json.dump(num, open("results/numbers.json", "w"), indent=0, default=float)
print(open("results/tab_main.tex").read()); print(open("results/tab_sig.tex").read()); print(open("results/tab_abl.tex").read())
