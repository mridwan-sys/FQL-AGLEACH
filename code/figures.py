import pickle, numpy as np, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from experiments import SCEN, MAIN, ABL, PROP
import fqlsim as S
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": 0.3,
                     "savefig.bbox": "tight", "savefig.dpi": 300})
M = pickle.load(open("results/main.pkl", "rb"))
A = pickle.load(open("results/abl.pkl", "rb"))
PN = [m[0] for m in MAIN]
COL = {"LEACH": "#9e9e9e", "Fuzzy-LEACH": "#8e44ad", "Grid-LEACH": "#2a78d6", "AGLEACH": "#1a8f5f", "FQL-AGLEACH": "#d9480f"}
LS = {"LEACH": ":", "Fuzzy-LEACH": "-.", "Grid-LEACH": "--", "AGLEACH": (0, (5, 1.5, 1, 1.5)), "FQL-AGLEACH": "-"}
TIT = {"S1": "(a) S1: 100 nodes, uniform", "S2": "(b) S2: 500 nodes, uniform",
       "S3": "(c) S3: 100 nodes, clustered", "S4": "(d) S4: 100 nodes, BS outside"}
XL = {"S1": 3600, "S2": 3600, "S3": 3600, "S4": 1800}
OUT = "figs/"
import os; os.makedirs(OUT, exist_ok=True)

def curves(key, fname, ylab, scs=("S1", "S2", "S3", "S4"), cum=False, scale=1.0):
    nc = 2; nr = int(math.ceil(len(scs) / 2))
    fig, axs = plt.subplots(nr, nc, figsize=(7.2, 2.6 * nr))
    axs = np.atleast_1d(axs).ravel()
    for ax, sc in zip(axs, scs):
        for p in PN:
            y = M[(sc, p)]["ts"][key] * scale
            if cum: y = np.cumsum(y)
            ax.plot(np.arange(1, len(y) + 1), y, color=COL[p], ls=LS[p], lw=1.6 if p == "FQL-AGLEACH" else 1.1, label=p)
        ax.set_xlim(0, XL[sc]); ax.set_title(TIT[sc], fontsize=9); ax.set_xlabel("Round"); ax.set_ylabel(ylab)
    axs[0].legend(fontsize=7.5, loc="best", frameon=False)
    fig.tight_layout(); fig.savefig(OUT + fname); plt.close(fig)

curves("alive", "fig_alive.pdf", "Alive nodes")
curves("avgE", "fig_energy.pdf", "Avg. residual energy (J)", scs=("S1", "S2"))
curves("deliv", "fig_deliv.pdf", "Delivered readings ($\\times10^3$)", scs=("S1", "S2"), cum=True, scale=1e-3)

# CH count + number of grid cells
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
for ax, sc in zip(axs, ("S1", "S2")):
    for p in PN:
        y = M[(sc, p)]["ts"]["ch"]
        k = 25; ys = np.convolve(y, np.ones(k) / k, mode="same")
        ax.plot(ys, color=COL[p], ls=LS[p], lw=1.6 if p == "FQL-AGLEACH" else 1.1, label=p)
    ax.set_xlim(0, XL[sc]); ax.set_title(TIT[sc], fontsize=9); ax.set_xlabel("Round"); ax.set_ylabel("Cluster heads per round")
axs[1].legend(fontsize=7.5, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig_ch.pdf"); plt.close(fig)

# learning dynamics: reward and divergence
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5))
for sc, c in (("S1", "#d9480f"), ("S2", "#2a78d6")):
    ts = M[(sc, "FQL-AGLEACH")]["ts"]
    hnd = int(np.mean(M[(sc, "FQL-AGLEACH")]["summ"]["HND"]))
    k = 50
    r = np.convolve(ts["rew"], np.ones(k) / k, mode="valid")
    axs[0].plot(np.arange(len(r)) + k // 2, r, color=c, label=sc)
    d = np.convolve(ts["div"], np.ones(k) / k, mode="valid")
    axs[1].plot(np.arange(len(d)) + k // 2, 100 * d, color=c, label=sc)
    for ax in axs: ax.axvline(hnd, color=c, lw=0.7, ls=":")
axs[0].set_xlim(0, 3700); axs[1].set_xlim(0, 3700)
axs[0].set_title("(a) Mean per-CH reward (50-round moving avg.)", fontsize=9); axs[0].set_xlabel("Round"); axs[0].set_ylabel("Reward")
axs[1].set_title("(b) CHs differing from max-energy choice", fontsize=9); axs[1].set_xlabel("Round"); axs[1].set_ylabel("% of cells")
axs[0].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT + "fig_learning.pdf"); plt.close(fig)

# learned policy heat map (S1, mean over 30 seeds)
Q = np.mean([q[:, 1].reshape(3, 3, 3) for q in M[("S1", "FQL-AGLEACH")]["q"]], axis=0)  # [E, D, rho]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
vmin, vmax = Q.min(), Q.max()
for i, (ax, lab) in enumerate(zip(axs, ["Low", "Medium", "High"])):
    im = ax.imshow(Q[i], cmap="viridis", vmin=vmin, vmax=vmax, origin="lower")
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{Q[i,a,b]:.0f}", ha="center", va="center", color="w" if Q[i, a, b] < (vmin + vmax) / 2 else "k", fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels(["Sparse", "Moderate", "Dense"]); ax.set_yticks(range(3)); ax.set_yticklabels(["Near", "Medium", "Far"])
    ax.set_title(f"Residual energy: {lab}", fontsize=9); ax.grid(False); ax.set_xlabel("Local density $\\hat\\rho_i$")
axs[0].set_ylabel("Distance to BS $\\hat D_i$")
fig.colorbar(im, ax=axs, shrink=0.85, label="$q_r(\\mathrm{CH})$")
fig.savefig(OUT + "fig_policy.pdf"); plt.close(fig)

# ablation chart
ANAMES = [a[0] for a in ABL]
LAB = ["AGLEACH", "+ $k_{opt}$ grid", "+ tile alignment", "+ FQL (frozen)", "Hand-tuned switch", "FQL-AGLEACH"]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
mets = ["FND", "HND", "T90", "LND"]
cols = ["#bdbdbd", "#9ecae1", "#4292c6", "#08519c", "#fdae6b", "#d9480f"]
for ax, sc in zip(axs, SCEN):
    base = {m: np.mean(A[(sc, ANAMES[0])][m]) for m in mets}
    w = 0.13
    for j, nm in enumerate(ANAMES):
        vals = [100 * (np.mean(A[(sc, nm)][m]) / base[m] - 1) for m in mets]
        ax.bar(np.arange(4) + (j - 2.5) * w, vals, w, color=cols[j], label=LAB[j])
    ax.set_xticks(range(4)); ax.set_xticklabels(["FND", "HND", "T$_{90}$", "LND"]); ax.set_title(sc, fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
axs[0].set_ylabel("Change vs. AGLEACH (%)")
hs=[plt.Rectangle((0,0),1,1,color=c) for c in cols]
fig.legend(hs, LAB, loc="lower center", ncol=6, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(OUT + "fig_ablation.pdf"); plt.close(fig)
print("ok")

# behaviour of selected CHs: full vs frozen
B = pickle.load(open("results/behaviour.pkl", "rb"))
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex="row")
for row, sc in enumerate(("S1", "S2")):
    for nm, c, ls in (("frozen", "#08519c", "--"), ("full", "#d9480f", "-")):
        xd, xb, xe, al = B[(sc, nm)]
        n = SCEN[sc]["n"]
        for j, (arr, lab) in enumerate(((xd, "Density advantage $\\hat\\rho_{CH}-\\overline{\\hat\\rho}$"),
                                        (xb, "BS-distance offset / $G$"), (xe, "$E_{CH}/E^{max}_g$"))):
            m = np.where(al > 0, arr, np.nan)
            cnt = np.sum(al > 0, axis=0)
            y = np.nanmean(m, axis=0); k = 50
            ys = np.convolve(np.nan_to_num(y), np.ones(k) / k, mode="same")
            ys[cnt < 15] = np.nan
            ax = axs[row, j]
            ax.plot(ys, color=c, ls=ls, lw=1.3, label="FQL-AGLEACH" if nm == "full" else "Learning frozen")
            ax.set_title(f"{sc}: {lab}", fontsize=8)
            hnd = np.mean([next(i for i, a in enumerate(r) if a <= n / 2) for r in al])
            ax.axvline(hnd, color="gray", lw=0.6, ls=":")
            ax.set_xlim(0, 3000)
            if row == 1: ax.set_xlabel("Round")
axs[0, 0].legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT + "fig_behaviour.pdf"); plt.close(fig)
