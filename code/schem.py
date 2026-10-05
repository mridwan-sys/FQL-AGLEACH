import numpy as np, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import fqlsim as S
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "savefig.bbox": "tight", "savefig.dpi": 300})
# ---------------- network snapshot
xs, ys = S.make_nodes(100, 100)
d = np.hypot(xs - 50, ys - 50)
k = 1.5 * (100 / d.mean()) * math.sqrt(100 / (2 * math.pi)); m = max(1, int(round(math.sqrt(k)))); G = 100 / m
cx = np.minimum((xs // G).astype(int), m - 1); cy = np.minimum((ys // G).astype(int), m - 1)
cell = cx * m + cy
rng = np.random.default_rng(3)
E = rng.uniform(0.55, 0.75, 100)   # illustrative mid-life residual energies
chs = []
for c in np.unique(cell):
    idx = np.where(cell == c)[0]
    thr = np.percentile(E[idx], 75); el = idx[E[idx] >= thr]
    dens = np.array([np.sum(np.hypot(xs[idx] - xs[i], ys[idx] - ys[i]) <= 0.5 * G) for i in el])
    chs.append(el[np.argmax(dens + 0.01 * E[el])])
chs = np.array(chs)
fig, ax = plt.subplots(figsize=(4.6, 4.6))
for i in range(m + 1):
    ax.plot([i * G, i * G], [0, 100], color="#2a78d6", lw=0.8, ls="--"); ax.plot([0, 100], [i * G, i * G], color="#2a78d6", lw=0.8, ls="--")
for i in range(100):
    if i in chs: continue
    j = chs[np.argmin(np.hypot(xs[chs] - xs[i], ys[chs] - ys[i]))]
    ax.plot([xs[i], xs[j]], [ys[i], ys[j]], color="#9aa0a6", lw=0.6, zorder=1)
for j in chs:
    ax.annotate("", xy=(50, 50), xytext=(xs[j], ys[j]), arrowprops=dict(arrowstyle="->", color="#d9480f", lw=0.8, alpha=0.8), zorder=2)
el_mask = np.zeros(100, bool)
for c in np.unique(cell):
    idx = np.where(cell == c)[0]; el_mask[idx[E[idx] >= np.percentile(E[idx], 75)]] = True
ax.scatter(xs[~el_mask], ys[~el_mask], s=14, facecolor="white", edgecolor="k", lw=0.7, zorder=3, label="Sensor node")
ax.scatter(xs[el_mask], ys[el_mask], s=16, facecolor="#ffe8a3", edgecolor="k", lw=0.7, zorder=3, label="Eligible candidate ($E_i\\geq P_{75}$)")
ax.scatter(xs[chs], ys[chs], s=42, facecolor="#1c5cab", edgecolor="k", zorder=4, label="Cluster head")
ax.scatter([50], [50], marker="^", s=120, color="#d9480f", edgecolor="k", zorder=5, label="Base station (50,50)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
ax.set_title(f"$k_{{opt}}$={k:.1f} $\\rightarrow$ m={m}, G={G:.0f} m (tile-aligned)", fontsize=9)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=7.5, frameon=False)
fig.savefig("figs/fig_network.pdf"); plt.close(fig)

# ---------------- flowchart
fig, ax = plt.subplots(figsize=(7.2, 6.4)); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
def box(x, y, w, h, txt, fc="#f7f8fa", ec="#5b616b", fs=7.6, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else None, linespacing=1.35)
def arr(x1, y1, x2, y2, c="#5b616b", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, color=c, lw=0.9, ls=ls))
X, W, H = 6, 60, 7.4
steps = [
 ("1. Status collection: residual energy $E_i(t)$ of each alive node\n(16-bit field piggy-backed on the previous data packet)", "#f7f8fa"),
 ("2. Tile-aligned adaptive grid: $k_{opt}(t)\\rightarrow m(t)=\\mathrm{round}(\\sqrt{k_{opt}(t)})$,\n$G(t)=L/m(t)$; assign alive nodes to the $m^2$ cells", "#eaf2fc"),
 ("3. Eligibility per cell: $\\mathcal{E}_g(t)=\\{i\\in\\mathcal{V}_g: E_i(t)\\geq P_{75}\\}$", "#eaf2fc"),
 ("4. Fuzzify $(\\hat E_i,\\hat D_i,\\hat\\rho_i)$ $\\rightarrow$ 27 firing strengths $\\bar w_r(s_i)$\n$Q(s_i)=\\sum_r \\bar w_r(s_i)\\,q_r$", "#fdefe8"),
 ("5. $\\epsilon$-greedy CH choice: $CH_g=\\arg\\max_{i\\in\\mathcal{E}_g} Q(s_i)$\n(a random eligible node with probability $\\epsilon_t$)", "#fdefe8"),
 ("6. Broadcast CH assignment ($64+16k$ bits);\nnon-CH nodes join the nearest CH", "#f7f8fa"),
 ("7. Steady state: members $\\rightarrow$ CH, aggregation, CH $\\rightarrow$ BS", "#f7f8fa"),
 ("8. Per-CH reward $r_g=\\eta_g/\\eta_{ref}-\\kappa\\,\\bar e_t^{\\,2}\\,\\Delta_g-w_D D_g$", "#fdefe8"),
 ("9. Per-rule TD update $q_r\\leftarrow q_r+\\alpha\\,\\delta_g\\,\\bar w_r(s_{CH_g})$; decay $\\epsilon_t$", "#fdefe8"),
]
y = 92
ys_ = []
for t, fc in steps:
    box(X, y - H, W, H, t, fc=fc); ys_.append(y); y -= H + 2.6
for i in range(len(steps) - 1):
    arr(X + W / 2, ys_[i] - H - 0.3, X + W / 2, ys_[i + 1] + 0.3)
# loop back
ax.annotate("", xy=(X - 0.6, ys_[0] - H / 2), xytext=(X - 0.6, ys_[-1] - H / 2),
            arrowprops=dict(arrowstyle="-|>", connectionstyle="bar,fraction=-0.06", color="#1a8f5f", lw=1.0))
ax.text(X - 7.5, (ys_[0] + ys_[-1]) / 2 - H / 2, "next round", rotation=90, color="#1a8f5f", fontsize=8, va="center")
# side panels
box(71, 66, 28, 24, "Base station\n(all learning)\n\n27 rule values $q_r$\nO($N\\cdot 27$) per round\nno extra sensor-side\ncomputation or memory", fc="#eaf2fc", ec="#1c5cab")
box(71, 36, 28, 24, "Sensor nodes\n(no learning)\n\nsense, transmit,\nreceive one CH-assignment\nbroadcast per round", fc="#f7f8fa")
box(71, 8, 28, 22, "Legend\nblue: grid structure\norange: fuzzy Q-learning\ngrey: data/control plane", fc="white", fs=7.6)
fig.savefig("figs/fig_flow.pdf"); plt.close(fig)
print(k, m, G)
